"""Oscillator and Noise — block generators driven by an AudioClock."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from nonoha_core.data_type import PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc
from nonoha_core.port import PortValue

from ._common import F32, F64, Glide, add_input, add_output, apply_numeric, frames_type, last_value

_WAVEFORMS = ("sine", "triangle", "saw", "square")


def _polyblep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Polynomial band-limited step correction (reduces aliasing of edges)."""
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    x = (t[m] - 1.0) / dt[m]
    out[m] = x * x + x + x + 1.0
    return out


class OscillatorNode(BaseNode):
    """Phase-accumulating oscillator: changing pitch never resets the phase."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Oscillator (sine, triangle, saw, square; saw/square are "
                "band-limited). Keeps its own phase, so frequency changes "
                "glide without clicks. One block out per time block in."
            ),
            category="Audio",
            inputs={
                "time": "Block clock (f64 per-sample time; only its length is used)",
                "frequency": "Frequency in Hz (glides over 'glide' seconds)",
                "amplitude": "Peak amplitude (glides over 'glide' seconds)",
            },
            outputs={"frames": "Mono audio frames f32[1]"},
        )

    PROPERTIES = {
        "waveform": {"description": "sine, triangle, saw or square", "data_type": "str", "default_value": "sine"},
        "frequency": {"description": "Initial frequency in Hz", "data_type": "f64", "default_value": 220.0},
        "amplitude": {"description": "Initial peak amplitude", "data_type": "f64", "default_value": 0.2},
        "glide": {"description": "Seconds to reach a new frequency/amplitude", "data_type": "f64", "default_value": 0.05},
        "sample_rate": {"description": "Sample rate in Hz", "data_type": "i32", "default_value": 44100},
    }

    def __init__(self, name: str):
        super().__init__(name, "Oscillator")
        add_input(self, "time", F64, self._on_time)
        add_input(self, "frequency", F32, lambda v: self._freq.set(last_value(v)))
        add_input(self, "amplitude", F32, lambda v: self._amp.set(last_value(v)))
        self._out = add_output(self, "frames", frames_type(1))
        self._freq = Glide(float(self.properties["frequency"]))
        self._amp = Glide(float(self.properties["amplitude"]))
        self._phase = 0.0
        # Mirrors of the properties used per block (no lookups or casts there).
        self._waveform = str(self.properties["waveform"])
        self._glide = float(self.properties["glide"])
        self._sr = float(self.properties["sample_rate"])

    def _on_time(self, values: list[PortValue]):
        n = len(values[0])
        if n == 0:
            return
        sr = self._sr
        freq = np.clip(self._freq.ramp(n, sr, self._glide), 0.0, sr * 0.45)
        amp = self._amp.ramp(n, sr, self._glide)

        inc = freq / sr
        steps = np.cumsum(inc)
        phase = (self._phase + steps - inc) % 1.0  # phase of each sample
        self._phase = float((self._phase + steps[-1]) % 1.0)

        wave = self._waveform
        if wave == "sine":
            y = np.sin(2.0 * np.pi * phase)
        elif wave == "triangle":
            y = 4.0 * np.abs(phase - 0.5) - 1.0
        elif wave == "saw":
            y = 2.0 * phase - 1.0 - _polyblep(phase, np.maximum(inc, 1e-9))
        else:  # square
            dt = np.maximum(inc, 1e-9)
            y = np.where(phase < 0.5, 1.0, -1.0) + _polyblep(phase, dt) - _polyblep((phase + 0.5) % 1.0, dt)
        self._out.dispatch([np.multiply(y, amp, dtype=np.float32).reshape(-1, 1)])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        if "waveform" in properties:
            wave = str(properties["waveform"])
            if wave not in _WAVEFORMS:
                raise ValueError(f"waveform must be one of {_WAVEFORMS}")
            self.properties["waveform"] = self._waveform = wave
        apply_numeric(self, properties, {"frequency": float, "amplitude": float, "glide": float, "sample_rate": int})
        if "frequency" in properties:
            self._freq.set(float(self.properties["frequency"]))
        if "amplitude" in properties:
            self._amp.set(float(self.properties["amplitude"]))
        self._glide = float(self.properties["glide"])
        self._sr = float(self.properties["sample_rate"])
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


# Pinking filter (Paul Kellet's economy coefficients, -3 dB/octave).
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
# Leaky integrator for brown (-6 dB/octave) noise.
_BROWN_B = np.array([1.0])
_BROWN_A = np.array([1.0, -0.995])
_COLORS = ("white", "pink", "brown")


def _color_gain(b: np.ndarray, a: np.ndarray) -> float:
    """Scale so the colored noise has the same RMS as unit white noise."""
    x = np.random.default_rng(0).standard_normal(1 << 16)
    y = np.asarray(lfilter(b, a, x))[4096:]  # runs once at import; asarray narrows the stub's union
    return float(1.0 / np.sqrt(np.mean(y * y)))


_GAINS = {"pink": _color_gain(_PINK_B, _PINK_A), "brown": _color_gain(_BROWN_B, _BROWN_A)}


class NoiseNode(BaseNode):
    """White, pink or brown noise."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Noise generator: white, pink (-3 dB/oct, wind/rain-like) or "
                "brown (-6 dB/oct, rumble/surf-like). RMS level = amplitude / 3."
            ),
            category="Audio",
            inputs={
                "time": "Block clock (f64 per-sample time; only its length is used)",
                "amplitude": "Level (glides over 'glide' seconds)",
            },
            outputs={"frames": "Mono audio frames f32[1]"},
        )

    PROPERTIES = {
        "color": {"description": "white, pink or brown", "data_type": "str", "default_value": "pink"},
        "amplitude": {"description": "Initial level", "data_type": "f64", "default_value": 0.1},
        "glide": {"description": "Seconds to reach a new amplitude", "data_type": "f64", "default_value": 0.1},
        "seed": {"description": "Random seed (0 = different every run)", "data_type": "i32", "default_value": 0},
        "sample_rate": {"description": "Sample rate in Hz", "data_type": "i32", "default_value": 44100},
    }

    def __init__(self, name: str):
        super().__init__(name, "Noise")
        add_input(self, "time", F64, self._on_time)
        add_input(self, "amplitude", F32, lambda v: self._amp.set(last_value(v)))
        self._out = add_output(self, "frames", frames_type(1))
        self._amp = Glide(float(self.properties["amplitude"]))
        self._rng = np.random.default_rng()
        self._zi: np.ndarray | None = None
        self._color = str(self.properties["color"])
        self._glide = float(self.properties["glide"])
        self._sr = float(self.properties["sample_rate"])

    def _on_time(self, values: list[PortValue]):
        n = len(values[0])
        if n == 0:
            return
        white = self._rng.standard_normal(n)
        color = self._color
        if color == "white":
            y = white
        else:
            b, a = (_PINK_B, _PINK_A) if color == "pink" else (_BROWN_B, _BROWN_A)
            if self._zi is None or len(self._zi) != len(a) - 1:
                self._zi = np.zeros(len(a) - 1)
            y, self._zi = lfilter(b, a, white, zi=self._zi)
            y = y * _GAINS[color]
        amp = self._amp.ramp(n, self._sr, self._glide)
        out = np.multiply(y, amp, dtype=np.float32)
        out *= np.float32(1.0 / 3.0)
        self._out.dispatch([out.reshape(-1, 1)])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        if "color" in properties:
            color = str(properties["color"])
            if color not in _COLORS:
                raise ValueError(f"color must be one of {_COLORS}")
            if color != self._color:
                self._zi = None
            self.properties["color"] = self._color = color
        apply_numeric(self, properties, {"amplitude": float, "glide": float, "seed": int, "sample_rate": int})
        if "amplitude" in properties:
            self._amp.set(float(self.properties["amplitude"]))
        self._glide = float(self.properties["glide"])
        self._sr = float(self.properties["sample_rate"])
        if "seed" in properties:
            seed = int(self.properties["seed"])
            self._rng = np.random.default_rng(seed if seed else None)
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
