"""Reverb — stereo Freeverb (8 damped combs + 4 allpasses per channel).

Ported from the audio_synth workspace's SchroederReverb: ring buffers with
numpy slices and scipy's lfilter for the damping filter, so a block costs no
per-sample Python loop. Delay lengths scale with the sample rate. The ring
buffers are updated in place; a block allocates only its output arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from nonoha_core.data_type import PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc
from nonoha_core.port import PortValue

from ._common import F32, SAMPLE_RATE_PROPERTY, add_input, add_output, apply_numeric, frames_type, last_value

# Freeverb delay lengths, tuned for 44.1 kHz.
_COMB_DELAYS = (1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116)
_ALLPASS_DELAYS = (556, 441, 341, 225)
_STEREO_SPREAD = 23
_INPUT_GAIN = 0.015
_ALLPASS_FEEDBACK = 0.5


class _CombBank:
    """The damped combs of both channels, run together.

    Every comb reads its ring buffer ``delay`` samples back, so a chunk of up
    to the shortest delay never reads what it writes: each chunk copies all
    combs' delayed samples into one array, runs one lfilter call for all of
    them (rows), and writes the new values back. (One comb at a time needed a
    filter call per comb and per wrap: ~60 calls per 0.1 s block.)
    """

    __slots__ = ("rings", "index", "store", "chunk")

    def __init__(self, delays: list[int]):
        self.rings = [np.zeros(d, dtype=np.float64) for d in delays]
        self.index = [0] * len(delays)
        self.store = np.zeros((len(delays), 1), dtype=np.float64)  # damping filter state
        self.chunk = min(delays)

    def process(self, x: np.ndarray, feedback: float, damp: float, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Feed ``x`` to every comb; returns the combs' outputs (combs, n)."""
        n = len(x)
        rings, index = self.rings, self.index
        out = np.empty((len(rings), n))
        pos = 0
        while pos < n:
            m = min(self.chunk, n - pos)
            delayed = out[:, pos : pos + m]
            for r, ring in enumerate(rings):
                i = index[r]
                first = min(m, len(ring) - i)  # up to the end of the ring, then wrap
                delayed[r, :first] = ring[i : i + first]
                if first < m:
                    delayed[r, first:] = ring[: m - first]
            # One-pole damping (1 - damp) / (1 - damp z^-1); its state is damp * last output.
            # lfilter returns (y, zf) when zi is given; the stubs type it as a union.
            filtered, _ = lfilter(b, a, delayed, axis=1, zi=self.store * damp)  # type: ignore
            self.store = filtered[:, -1:].copy()
            filtered *= feedback
            filtered += x[pos : pos + m]
            for r, ring in enumerate(rings):
                i, size = index[r], len(ring)
                first = min(m, size - i)
                ring[i : i + first] = filtered[r, :first]
                if first < m:
                    ring[: m - first] = filtered[r, first:]
                i += m
                index[r] = i - size if i >= size else i
            pos += m
        return out


class _Allpass:
    __slots__ = ("buffer", "index")

    def __init__(self, delay: int):
        self.buffer = np.zeros(delay, dtype=np.float64)
        self.index = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        n, buffer = len(x), self.buffer
        size = len(buffer)
        out = np.empty(n)
        pos, idx = 0, self.index
        while pos < n:
            chunk = min(n - pos, size - idx)
            ring = buffer[idx : idx + chunk]
            xs = x[pos : pos + chunk]
            np.subtract(ring, xs, out=out[pos : pos + chunk])  # delayed - x
            ring *= _ALLPASS_FEEDBACK  # x + delayed * feedback, in place
            ring += xs
            idx += chunk
            if idx == size:
                idx = 0
            pos += chunk
        self.index = idx
        return out


class ReverbNode(BaseNode):
    """Stereo reverb with room size, damping, width and wet/dry mix."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Stereo Freeverb reverb (8 damped combs + 4 allpasses per "
                "channel). Stereo in, stereo out; 'mix' can be modulated."
            ),
            category="Audio",
            inputs={"frames": "Stereo frames f32[2]", "mix": "Wet/dry mix 0 (dry) - 1 (wet)"},
            outputs={"frames": "Stereo frames f32[2]"},
        )

    PROPERTIES = {
        "room_size": {"description": "Room size 0-1 (decay length)", "default_value": 0.8},
        "damping": {"description": "High-frequency damping 0-1", "default_value": 0.5},
        "width": {"description": "Stereo width 0-1", "default_value": 1.0},
        "mix": {"description": "Wet/dry mix 0-1", "default_value": 0.3},
        "sample_rate": SAMPLE_RATE_PROPERTY,
    }

    def __init__(self, name: str):
        super().__init__(name, "Reverb")
        add_input(self, "frames", frames_type(2), self._on_frames)
        add_input(self, "mix", F32, self._on_mix)
        self._out = add_output(self, "frames", frames_type(2))
        self._sr = int(self.properties["sample_rate"])
        self._build(self._sr)
        self._mirror()

    def _build(self, sample_rate: int):
        scale = sample_rate / 44100.0
        spread = max(1, round(_STEREO_SPREAD * scale))
        # Left combs, then right combs (one bank, see _CombBank).
        self._combs = _CombBank([max(1, round(d * scale)) + offset for offset in (0, spread) for d in _COMB_DELAYS])
        self._allpasses = [
            [_Allpass(max(1, round(d * scale)) + offset) for d in _ALLPASS_DELAYS] for offset in (0, spread)
        ]

    def _mirror(self):
        p = self.properties
        self._feedback = float(p["room_size"]) * 0.28 + 0.7
        damp = float(p["damping"]) * 0.4
        self._damp = damp
        self._b = np.array([1.0 - damp])
        self._a = np.array([1.0, -damp])
        self._width = float(p["width"])
        self._mix = min(1.0, max(0.0, float(p["mix"])))

    def _on_mix(self, values: list[PortValue]):
        self.properties["mix"] = self._mix = min(1.0, max(0.0, last_value(values)))

    def _on_frames(self, values: list[PortValue]):
        x = values[0]
        n = len(x)
        if n == 0:
            return
        feedback, damp, b, a = self._feedback, self._damp, self._b, self._a
        mono = x[:, 0] + x[:, 1]  # float32 pair -> one new array
        mono *= _INPUT_GAIN

        combs = self._combs.process(mono, feedback, damp, b, a)
        sums = combs.reshape(2, len(_COMB_DELAYS), n).sum(axis=1)  # per channel
        wet = []
        for acc, allpasses in zip(sums, self._allpasses):
            for ap in allpasses:
                acc = ap.process(acc)
            wet.append(acc)

        mix, width = self._mix, self._width
        wet1 = mix * (width / 2.0 + 0.5)
        wet2 = mix * ((1.0 - width) / 2.0)
        dry = 1.0 - mix
        wet_l, wet_r = wet
        out = np.empty((n, 2), dtype=np.float32)
        out[:, 0] = wet_l * wet1 + wet_r * wet2
        out[:, 1] = wet_r * wet1 + wet_l * wet2
        if dry > 0.0:
            out += x * np.float32(dry)
        self._out.dispatch([out])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        apply_numeric(
            self, properties, {"room_size": float, "damping": float, "width": float, "mix": float, "sample_rate": int}
        )
        sr = int(self.properties["sample_rate"])
        if sr != self._sr:
            self._sr = sr
            self._build(sr)
        self._mirror()
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
