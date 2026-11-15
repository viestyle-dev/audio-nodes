"""PolyVoice — polyphonic percussive voices (pluck / bell) driven by note events."""

from __future__ import annotations

import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc
from nonoha_core.port import PortValue

from ._common import F64, SAMPLE_RATE_PROPERTY, add_input, add_output, apply_numeric, frames_type

NOTES_TYPE = PortValueType(DataType.Float32, (-1, 3))

# Modal recipes: partial frequency ratios, relative weights, relative decays.
_VOICES = {
    "pluck": (
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        np.array([1.0, 0.5, 0.33, 0.25, 0.2, 0.16]),
        np.array([1.0, 0.7, 0.5, 0.38, 0.3, 0.25]),
    ),
    "bell": (
        np.array([1.0, 2.756, 5.404, 8.933]),
        np.array([1.0, 0.6, 0.35, 0.2]),
        np.array([1.0, 0.7, 0.45, 0.3]),
    ),
    "chime": (
        np.array([1.0, 2.0, 4.16, 6.95]),
        np.array([1.0, 0.3, 0.25, 0.12]),
        np.array([1.0, 0.8, 0.55, 0.4]),
    ),
}
_ATTACK_SECONDS = 0.003


class PolyVoiceNode(BaseNode):
    """Percussive polyphonic synth: each note event starts a decaying voice.

    Voices are modal (a few exponentially decaying partials), rendered for all
    active voices at once with numpy — no per-sample Python loops.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Polyphonic percussive voices (pluck, bell, chime). Each row of "
                "'notes' starts a voice at the next block; voices decay on "
                "their own and are panned individually. Oldest voices are "
                "stolen beyond max_voices."
            ),
            category="Audio",
            inputs={
                "time": "Block clock (f64 per-sample time; only its length is used)",
                "notes": "Note events f32[N x 3]: frequency Hz, velocity 0-1, pan -1..1",
            },
            outputs={
                "frames": "Stereo frames f32[2]",
                "active": "Number of sounding voices",
            },
        )

    PROPERTIES = {
        "voice": {"description": "pluck, bell or chime", "data_type": "str", "default_value": "bell"},
        "decay": {"description": "Decay time of the fundamental in seconds", "default_value": 2.0},
        "brightness": {"description": "0-1: weight of the upper partials", "default_value": 0.6},
        "gain": {"description": "Output gain per voice", "default_value": 0.25},
        "max_voices": {"description": "Maximum simultaneous voices", "default_value": 16},
        "sample_rate": SAMPLE_RATE_PROPERTY,
    }

    def __init__(self, name: str):
        super().__init__(name, "PolyVoice")
        add_input(self, "time", F64, self._on_time)
        add_input(self, "notes", NOTES_TYPE, self._on_notes)
        self._out = add_output(self, "frames", frames_type(2))
        self._out_active = add_output(self, "active", PortValueType(DataType.Int32))
        self._pending: list[np.ndarray] = []
        # Active voices: frequency, velocity, pan, age (samples).
        self._freq = np.zeros(0)
        self._vel = np.zeros(0)
        self._pan = np.zeros(0)
        self._age = np.zeros(0, dtype=np.int64)
        self._ramp: np.ndarray | None = None  # cached 0..n-1
        self._silence: np.ndarray | None = None  # cached read-only zeros
        self._mirror()

    def _mirror(self):
        """Derive the per-block constants from the properties."""
        p = self.properties
        ratios, weights, decays = _VOICES[str(p["voice"])]
        brightness = min(1.0, max(0.0, float(p["brightness"])))
        self._ratios = ratios
        self._weights = weights * (brightness ** np.arange(len(ratios)))  # (P,)
        self._tau = max(0.05, float(p["decay"])) * decays  # (P,)
        self._gain = float(p["gain"])
        self._max_voices = max(1, int(p["max_voices"]))
        sr = float(p["sample_rate"])
        self._inv_sr = 1.0 / sr
        self._nyquist_guard = sr * 0.45
        self._life = float(self._tau[0]) * 6.0 * sr  # samples to ~ -52 dB on the fundamental

    def _on_notes(self, values: list[PortValue]):
        for events in values[0]:
            if len(events):
                self._pending.append(events.copy())  # kept past the callback

    def _start_pending(self):
        if not self._pending:
            return
        events = np.concatenate(self._pending)
        self._pending.clear()
        valid = (events[:, 0] > 0) & (events[:, 1] > 0)
        events = events[valid]
        self._freq = np.concatenate([self._freq, events[:, 0]])
        self._vel = np.concatenate([self._vel, np.clip(events[:, 1], 0, 1)])
        self._pan = np.concatenate([self._pan, np.clip(events[:, 2], -1, 1)])
        self._age = np.concatenate([self._age, np.zeros(len(events), dtype=np.int64)])
        limit = self._max_voices
        if len(self._freq) > limit:  # steal the oldest (they come first)
            keep = slice(len(self._freq) - limit, None)
            self._freq, self._vel = self._freq[keep], self._vel[keep]
            self._pan, self._age = self._pan[keep], self._age[keep]

    def _on_time(self, values: list[PortValue]):
        n = len(values[0])
        if n == 0:
            return
        self._start_pending()
        if len(self._freq):
            out = self._render(n)
        else:
            out = self._silence
            if out is None or len(out) != n:
                out = self._silence = np.zeros((n, 2), dtype=np.float32)
                out.setflags(write=False)
        self._out.dispatch([out])
        self._out_active.dispatch([np.array([len(self._freq)], dtype=np.int32)])

    def _render(self, n: int) -> np.ndarray:
        ramp = self._ramp
        if ramp is None or len(ramp) != n:
            ramp = self._ramp = np.arange(n, dtype=np.float64)
        t = self._age[:, None] + ramp[None, :]  # (V, n), float64
        t *= self._inv_sr
        voice = np.zeros_like(t)
        partial = np.empty_like(t)
        envelope = np.empty_like(t)
        limit = self._nyquist_guard
        # One partial at a time keeps temporaries at (V, n) instead of (V, P, n).
        for ratio, weight, decay in zip(self._ratios, self._weights, self._tau):
            freqs = self._freq * ratio
            audible = freqs < limit
            if not audible.any():
                continue
            np.multiply(t, (2.0 * np.pi) * freqs[:, None], out=partial)
            np.sin(partial, out=partial)
            np.multiply(t, -1.0 / decay, out=envelope)
            np.exp(envelope, out=envelope)
            partial *= envelope
            partial *= (weight * audible)[:, None]
            voice += partial
        np.multiply(t, -1.0 / _ATTACK_SECONDS, out=envelope)
        np.exp(envelope, out=envelope)
        voice *= 1.0 - envelope
        voice *= (self._vel * self._gain)[:, None]

        theta = (self._pan + 1.0) * (np.pi / 4.0)
        stereo = np.matmul(voice.T, np.stack([np.cos(theta), np.sin(theta)], axis=1), dtype=np.float32)

        self._age += n
        alive = self._age < self._life
        self._freq, self._vel = self._freq[alive], self._vel[alive]
        self._pan, self._age = self._pan[alive], self._age[alive]
        return stereo

    def set_properties(self, properties: dict[str, PropertyDataType]):
        if "voice" in properties:
            voice = str(properties["voice"])
            if voice not in _VOICES:
                raise ValueError(f"voice must be one of {sorted(_VOICES)}")
            self.properties["voice"] = voice
        apply_numeric(
            self,
            properties,
            {"decay": float, "brightness": float, "gain": float, "max_voices": int, "sample_rate": int},
        )
        self._mirror()
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
