"""Block processors: ModFilter, Panner, Mixer, Envelope, SoftClip.

Each node mirrors the properties it uses per block into typed attributes in
``set_properties``, so the processing path does no dictionary lookups or
conversions.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from nonoha_core.data_type import PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc, PropertyDescription, replace_node_output_port
from nonoha_core.port import PortValue, StaticTypeInputPort, StaticTypeOutputPort

from ._common import (
    F32,
    SAMPLE_RATE_PROPERTY,
    ChannelsNode,
    Glide,
    add_input,
    add_output,
    apply_numeric,
    frames_type,
    last_value,
    steps,
)

_CHANNELS_PROPERTY: PropertyDescription = {"description": "Channel count of the frames ports", "data_type": "i32", "default_value": 1}

# --- ModFilter -------------------------------------------------------------------

_FILTER_MODES = ("lowpass", "highpass", "bandpass")
_SUBBLOCK = 128  # cutoff is re-evaluated this often within a block


def _biquads(mode: str, fc: np.ndarray, q: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray]:
    """RBJ cookbook biquad coefficients for each (fc, q): b (K, 3), a (K, 3), a[:, 0] = 1."""
    w0 = (2.0 * math.pi / sr) * fc
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    k = len(fc)
    b = np.empty((k, 3))
    if mode == "lowpass":
        b[:, 0] = (1 - cw) / 2
        b[:, 1] = 1 - cw
        b[:, 2] = b[:, 0]
    elif mode == "highpass":
        b[:, 0] = (1 + cw) / 2
        b[:, 1] = -(1 + cw)
        b[:, 2] = b[:, 0]
    else:  # bandpass, 0 dB peak gain
        b[:, 0] = alpha
        b[:, 1] = 0.0
        b[:, 2] = -alpha
    a0 = 1 + alpha
    b /= a0[:, None]
    a = np.empty((k, 3))
    a[:, 0] = 1.0
    a[:, 1] = -2 * cw / a0
    a[:, 2] = (1 - alpha) / a0
    return b, a


class ModFilterNode(ChannelsNode):
    """Biquad filter whose cutoff and Q can be modulated while it runs."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Low/high/band-pass biquad whose cutoff and Q follow their "
                "inputs (gliding, re-evaluated every 128 samples). Filter "
                "state carries across blocks, so sweeps are continuous."
            ),
            category="Audio",
            inputs={
                "frames": "Audio frames f32[channels]",
                "cutoff": "Cutoff / center frequency in Hz",
                "q": "Resonance (0.5-20; 0.707 = flat)",
            },
            outputs={"frames": "Filtered frames f32[channels]"},
        )

    PROPERTIES = {
        "mode": {"description": "lowpass, highpass or bandpass", "data_type": "str", "default_value": "lowpass"},
        "cutoff": {"description": "Initial cutoff in Hz", "data_type": "f64", "default_value": 1000.0},
        "q": {"description": "Initial Q", "data_type": "f64", "default_value": 0.707},
        "glide": {"description": "Seconds to reach a new cutoff/Q", "data_type": "f64", "default_value": 0.05},
        "channels": _CHANNELS_PROPERTY,
        "sample_rate": SAMPLE_RATE_PROPERTY,
    }

    def __init__(self, name: str):
        super().__init__(name, "ModFilter")
        self._channels = int(self.properties["channels"])
        add_input(self, "frames", frames_type(self._channels), self._on_frames)
        add_input(self, "cutoff", F32, lambda v: self._cutoff.set(last_value(v)))
        add_input(self, "q", F32, lambda v: self._q.set(last_value(v)))
        self._out = add_output(self, "frames", frames_type(self._channels))
        self._frames_inputs = {"frames": self._on_frames}
        self._frames_outputs = ("frames",)
        self._cutoff = Glide(float(self.properties["cutoff"]))
        self._q = Glide(float(self.properties["q"]))
        self._zi: np.ndarray | None = None
        self._mode = str(self.properties["mode"])
        self._glide = float(self.properties["glide"])
        self._sr = float(self.properties["sample_rate"])

    def _on_frames(self, values: list[PortValue]):
        x = values[0]
        n, ch = x.shape
        if n == 0:
            return
        sr, glide, mode = self._sr, self._glide, self._mode
        cutoff = self._cutoff.ramp(n, sr, glide)
        q = self._q.ramp(n, sr, glide)
        zi = self._zi
        if zi is None:  # reset when the channel count changes
            zi = np.zeros((2, ch))
        nyquist_guard = sr * 0.45
        # lfilter returns (y, zf) when zi is given; the stubs type it as a union.
        if cutoff[0] == cutoff[-1] and q[0] == q[-1]:
            # Not gliding (ramps are monotonic): one filter call for the block.
            b, a = _biquads(mode, np.clip(cutoff[:1], 20.0, nyquist_guard), np.maximum(q[:1], 0.1), sr)
            out, zi = lfilter(b[0], a[0], x, axis=0, zi=zi)  # type: ignore
            self._zi = zi
            self._out.dispatch([out.astype(x.dtype, copy=False)])
            return
        # Gliding: coefficients at the middle of each sub-block, all at once.
        starts = np.arange(0, n, _SUBBLOCK)
        mids = (starts + np.minimum(starts + _SUBBLOCK, n) - 1) >> 1
        bs, as_ = _biquads(mode, np.clip(cutoff[mids], 20.0, nyquist_guard), np.maximum(q[mids], 0.1), sr)
        out = np.empty_like(x)
        for i, start in enumerate(starts.tolist()):
            end = start + _SUBBLOCK
            out[start:end], zi = lfilter(bs[i], as_[i], x[start:end], axis=0, zi=zi)  # type: ignore
        self._zi = zi
        self._out.dispatch([out])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        if "mode" in properties:
            mode = str(properties["mode"])
            if mode not in _FILTER_MODES:
                raise ValueError(f"mode must be one of {_FILTER_MODES}")
            self.properties["mode"] = self._mode = mode
        apply_numeric(self, properties, {"cutoff": float, "q": float, "glide": float, "sample_rate": int})
        if "cutoff" in properties:
            self._cutoff.set(float(self.properties["cutoff"]))
        if "q" in properties:
            self._q.set(float(self.properties["q"]))
        self._glide = float(self.properties["glide"])
        self._sr = float(self.properties["sample_rate"])
        if "channels" in properties:
            self._set_channels(int(properties["channels"]))
            self._out = self.outputs["frames"]
            self._zi = None
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


# --- Panner ----------------------------------------------------------------------


class PannerNode(BaseNode):
    """Equal-power stereo panner: mono in, stereo out."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Equal-power panner: places a mono signal between left (-1) and right (+1)",
            category="Audio",
            inputs={"frames": "Mono frames f32[1]", "pan": "Position -1 (left) to +1 (right)"},
            outputs={"frames": "Stereo frames f32[2]"},
        )

    PROPERTIES = {
        "pan": {"description": "Initial position -1..+1", "data_type": "f64", "default_value": 0.0},
        "glide": {"description": "Seconds to reach a new position", "data_type": "f64", "default_value": 0.05},
        "sample_rate": SAMPLE_RATE_PROPERTY,
    }

    def __init__(self, name: str):
        super().__init__(name, "Panner")
        add_input(self, "frames", frames_type(1), self._on_frames)
        add_input(self, "pan", F32, lambda v: self._pan.set(last_value(v)))
        self._out = add_output(self, "frames", frames_type(2))
        self._pan = Glide(float(self.properties["pan"]))
        self._glide = float(self.properties["glide"])
        self._sr = float(self.properties["sample_rate"])

    def _on_frames(self, values: list[PortValue]):
        x = values[0][:, 0]
        n = len(x)
        if n == 0:
            return
        theta = np.clip(self._pan.ramp(n, self._sr, self._glide), -1.0, 1.0)  # fresh array
        theta += 1.0
        theta *= math.pi / 4.0
        out = np.empty((n, 2), dtype=np.float32)
        np.multiply(x, np.cos(theta), out=out[:, 0], casting="same_kind")
        np.multiply(x, np.sin(theta), out=out[:, 1], casting="same_kind")
        self._out.dispatch([out])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        apply_numeric(self, properties, {"pan": float, "glide": float, "sample_rate": int})
        if "pan" in properties:
            self._pan.set(float(self.properties["pan"]))
        self._glide = float(self.properties["glide"])
        self._sr = float(self.properties["sample_rate"])
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


# --- Mixer -----------------------------------------------------------------------


class MixerNode(BaseNode):
    """Sum N inputs once every input has delivered (order independent)."""

    @classmethod
    def describe(cls, properties=None):
        count = int((properties or {}).get("inputs", cls.PROPERTIES["inputs"].get("default_value", 4)))
        return NodeDoc(
            description=(
                "Sums in0..in{N-1}. Waits until every input has delivered "
                "audio for the block, so the result does not depend on the "
                "order in which upstream branches run. All inputs must be "
                "connected."
            ),
            category="Audio",
            inputs={f"in{i}": f"Audio frames input {i}" for i in range(max(1, min(16, count)))},
            outputs={"frames": "Sum of the inputs x gain"},
        )

    PROPERTIES = {
        "inputs": {"description": "Number of inputs (1-16)", "data_type": "i32", "default_value": 4},
        "gain": {"description": "Output gain", "data_type": "f64", "default_value": 1.0},
        "channels": {"description": "Channel count of all frames ports", "data_type": "i32", "default_value": 2},
    }

    def __init__(self, name: str):
        super().__init__(name, "Mixer")
        self._channels = int(self.properties["channels"])
        self._gain = float(self.properties["gain"])
        # Inputs of the current block. They are referenced, not copied: every
        # branch is driven by the same clock dispatch, so upstream nodes do
        # not produce the next block before this one is mixed.
        self._pending: list[np.ndarray | None] = []
        self._filled = 0
        self._out = add_output(self, "frames", frames_type(self._channels))
        self._resize_inputs(int(self.properties["inputs"]))

    def _make_input(self, i: int) -> StaticTypeInputPort:
        port = StaticTypeInputPort(f"in{i}", [frames_type(self._channels)])
        port.subscribe(id(self), lambda v, i=i: self._on_input(i, v))
        return port

    def _resize_inputs(self, count: int):
        count = max(1, min(16, count))
        current = len(self._pending)
        for i in range(count, current):
            name = f"in{i}"
            if self.parent_graph is not None:
                self.parent_graph.disconnect_all_from_node_input(self.name, name)
            del self.inputs[name]
        for i in range(current, count):
            self.inputs[f"in{i}"] = self._make_input(i)
        self._pending = [None] * count
        self._filled = 0
        self.properties["inputs"] = count

    def _on_input(self, i: int, values: list[PortValue]):
        pending = self._pending
        prev = pending[i]
        if prev is None:
            pending[i] = values[0]
            self._filled += 1
        else:  # a second block before the others: keep it contiguous
            pending[i] = np.concatenate((prev, values[0]))
        if self._filled < len(pending):
            return

        # Every slot is filled here (_filled == len), so no element is None.
        ready: list[np.ndarray] = pending  # type: ignore
        n = min([len(p) for p in ready])
        mix = np.array(ready[0][:n], dtype=np.float32)  # copy of the first input
        for p in ready[1:]:
            mix += p[:n]
        filled = 0
        for k, p in enumerate(ready):
            if len(p) > n:
                pending[k] = p[n:]
                filled += 1
            else:
                pending[k] = None
        self._filled = filled
        if self._gain != 1.0:
            mix *= np.float32(self._gain)
        self._out.dispatch([mix])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        if "gain" in properties:
            self.properties["gain"] = self._gain = float(properties["gain"])
        channels = int(properties.get("channels", self._channels))
        if channels != self._channels:
            # Frames ports of every input change type: drop them all and
            # let _resize_inputs recreate them.
            for i in range(len(self._pending)):
                if self.parent_graph is not None:
                    self.parent_graph.disconnect_all_from_node_input(self.name, f"in{i}")
                self.inputs.pop(f"in{i}", None)
            self._channels = channels
            self.properties["channels"] = channels
            self._pending = []
            new_out = StaticTypeOutputPort("frames", [frames_type(channels)])
            replace_node_output_port(self, self._out, new_out)
            self._out = new_out
        self._resize_inputs(int(properties.get("inputs", self.properties["inputs"])))
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


# --- Envelope --------------------------------------------------------------------

_IDLE, _ATTACK, _DECAY, _SUSTAIN, _RELEASE = range(5)


class EnvelopeNode(ChannelsNode):
    """ADSR amplitude envelope (VCA): shapes frames by a gate."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "ADSR envelope applied to audio (a VCA). Gate > 0.5 starts "
                "the attack; gate <= 0.5 starts the release. Linear segments, "
                "computed per block without per-sample Python loops."
            ),
            category="Audio",
            inputs={"frames": "Audio frames f32[channels]", "gate": "Gate (> 0.5 = on)"},
            outputs={"frames": "Frames x envelope", "level": "Envelope level at the end of the block"},
        )

    PROPERTIES = {
        "attack": {"description": "Attack seconds", "data_type": "f64", "default_value": 0.01},
        "decay": {"description": "Decay seconds", "data_type": "f64", "default_value": 0.2},
        "sustain": {"description": "Sustain level 0-1", "data_type": "f64", "default_value": 0.7},
        "release": {"description": "Release seconds", "data_type": "f64", "default_value": 0.5},
        "channels": _CHANNELS_PROPERTY,
        "sample_rate": SAMPLE_RATE_PROPERTY,
    }

    def __init__(self, name: str):
        super().__init__(name, "Envelope")
        self._channels = int(self.properties["channels"])
        add_input(self, "frames", frames_type(self._channels), self._on_frames)
        add_input(self, "gate", F32, self._on_gate)
        self._out = add_output(self, "frames", frames_type(self._channels))
        self._out_level = add_output(self, "level", F32)
        self._frames_inputs = {"frames": self._on_frames}
        self._frames_outputs = ("frames",)
        self._stage = _IDLE
        self._level = 0.0
        self._gate = False
        self._mirror()

    def _mirror(self):
        p = self.properties
        self._attack = float(p["attack"])
        self._decay = float(p["decay"])
        self._sustain = min(1.0, max(0.0, float(p["sustain"])))
        self._release = float(p["release"])
        self._sr = float(p["sample_rate"])

    def _on_gate(self, values: list[PortValue]):
        gate = last_value(values) > 0.5
        if gate and not self._gate:
            self._stage = _ATTACK
        elif not gate and self._gate:
            self._stage = _RELEASE
        self._gate = gate

    def _render(self, n: int) -> np.ndarray:
        sr, sustain = self._sr, self._sustain
        env = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            stage, level = self._stage, self._level
            if stage == _IDLE or stage == _SUSTAIN:
                env[i:] = 0.0 if stage == _IDLE else sustain
                self._level = float(env[-1])
                break
            if stage == _ATTACK:
                target, seconds, nxt = 1.0, self._attack, _DECAY
            elif stage == _DECAY:
                target, seconds, nxt = sustain, self._decay, _SUSTAIN
            else:
                target, seconds, nxt = 0.0, self._release, _IDLE
            span = abs(target - level)
            if seconds <= 0.0 or span < 1e-9:
                self._level, self._stage = target, nxt
                continue
            # Rate: a full 0-1 swing takes `seconds`.
            rate = 1.0 / (seconds * sr)
            count = min(n - i, max(1, int(math.ceil(span / rate))))
            direction = 1.0 if target > level else -1.0
            seg = env[i : i + count]
            np.multiply(steps(count), direction * rate, out=seg)
            seg += level
            if direction > 0:
                np.minimum(seg, target, out=seg)
            else:
                np.maximum(seg, target, out=seg)
            i += count
            self._level = float(seg[-1])
            if self._level == target:
                self._stage = nxt
        return env

    def _on_frames(self, values: list[PortValue]):
        x = values[0]
        n = len(x)
        if n == 0:
            return
        env = self._render(n)
        self._out.dispatch([np.multiply(x, env[:, None], dtype=np.float32)])
        self._out_level.dispatch([np.array([self._level], dtype=np.float32)])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        apply_numeric(
            self,
            properties,
            {"attack": float, "decay": float, "sustain": float, "release": float, "sample_rate": int},
        )
        self._mirror()
        if "channels" in properties:
            self._set_channels(int(properties["channels"]))
            self._out = self.outputs["frames"]
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


# --- SoftClip --------------------------------------------------------------------


class SoftClipNode(ChannelsNode):
    """tanh soft clipper: keeps a busy mix within -1..1 without hard clipping."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Soft clipping with tanh(drive * x): small signals get gain 'drive', the output never exceeds +-1",
            category="Audio",
            inputs={"frames": "Audio frames f32[channels]"},
            outputs={"frames": "Clipped frames"},
        )

    PROPERTIES = {
        "drive": {"description": "Drive (> 0); higher = earlier saturation", "data_type": "f64", "default_value": 1.0},
        "channels": {"description": "Channel count of the frames ports", "data_type": "i32", "default_value": 2},
    }

    def __init__(self, name: str):
        super().__init__(name, "SoftClip")
        self._channels = int(self.properties["channels"])
        add_input(self, "frames", frames_type(self._channels), self._on_frames)
        self._out = add_output(self, "frames", frames_type(self._channels))
        self._frames_inputs = {"frames": self._on_frames}
        self._frames_outputs = ("frames",)
        self._drive = np.float32(max(1e-3, float(self.properties["drive"])))

    def _on_frames(self, values: list[PortValue]):
        y = np.multiply(values[0], self._drive)
        np.tanh(y, out=y)
        self._out.dispatch([y])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        apply_numeric(self, properties, {"drive": float})
        self._drive = np.float32(max(1e-3, float(self.properties["drive"])))
        if "channels" in properties:
            self._set_channels(int(properties["channels"]))
            self._out = self.outputs["frames"]
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
