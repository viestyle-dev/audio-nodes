"""Shared helpers for the block-based DSP nodes.

Conventions (same as the device nodes):

- Audio is "frames": port type f32[channels], arrays of shape
  (samples, channels). The batch dimension is the sample index.
- Block clocks carry f64 per-sample time, shape (samples,). Generators use
  only its length; they keep their own phase so pitch changes never click.
- Controls (frequency, cutoff, pan, ...) are f32 scalars. A node keeps the
  last control value and glides toward it within each block.
"""

from __future__ import annotations

import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core.node import BaseNode, PropertyDescription, replace_node_input_port, replace_node_output_port
from nonoha_core.port import PortCallback, StaticTypeInputPort, StaticTypeOutputPort

F32 = PortValueType(DataType.Float32)
F64 = PortValueType(DataType.Float64)
BOOL = PortValueType(DataType.Bool)
ANY = PortValueType(DataType.Any)


def frames_type(channels: int) -> PortValueType:
    return PortValueType(DataType.Float32, (channels,))


def add_input(node: BaseNode, name: str, vtype: PortValueType, callback: PortCallback) -> StaticTypeInputPort:
    port = StaticTypeInputPort(name, [vtype])
    port.subscribe(id(node), callback)
    node.inputs[name] = port
    return port


def add_output(node: BaseNode, name: str, vtype: PortValueType) -> StaticTypeOutputPort:
    port = StaticTypeOutputPort(name, [vtype])
    node.outputs[name] = port
    return port


def last_value(values: list) -> float:
    return float(values[0][-1])


_STEPS: dict[int, np.ndarray] = {}


def steps(n: int) -> np.ndarray:
    """Cached read-only 1..n (blocks have a fixed size, so this is reused)."""
    arr = _STEPS.get(n)
    if arr is None:
        arr = np.arange(1, n + 1, dtype=np.float64)
        arr.setflags(write=False)
        _STEPS[n] = arr
    return arr


class Glide:
    """A control value that glides linearly to its target within ``seconds``.

    Each block ramps from the current value toward the target and remembers
    where it stopped, so changes are smooth across block boundaries. A
    settled value returns a cached constant block (no allocation).
    """

    __slots__ = ("current", "target", "_const", "_const_value")

    def __init__(self, value: float):
        self.current = float(value)
        self.target = float(value)
        self._const: np.ndarray | None = None
        self._const_value = float("nan")

    def set(self, value: float) -> None:
        self.target = float(value)

    def jump(self, value: float) -> None:
        self.current = self.target = float(value)

    def ramp(self, n: int, sample_rate: float, seconds: float) -> np.ndarray:
        cur, tgt = self.current, self.target
        if cur == tgt or seconds <= 0.0 or n == 0:
            self.current = tgt
            const = self._const
            if const is None or len(const) != n or self._const_value != tgt:
                const = np.full(n, tgt, dtype=np.float64)
                const.setflags(write=False)
                self._const, self._const_value = const, tgt
            return const
        span = max(1.0, seconds * sample_rate)
        out = steps(n) * (1.0 / span)
        np.minimum(out, 1.0, out=out)
        out *= tgt - cur
        out += cur
        self.current = float(out[-1])
        return out


SAMPLE_RATE_PROPERTY: PropertyDescription = {"description": "Sample rate in Hz", "data_type": "i32", "default_value": 44100}


class ChannelsNode(BaseNode):
    """Base for nodes whose f32[channels] frames ports follow a ``channels`` property.

    The node lists its frames ports in ``_frames_inputs`` (name -> callback)
    and ``_frames_outputs`` (names); call ``_set_channels(n)``.
    """

    _channels: int
    _frames_inputs: dict[str, PortCallback]
    _frames_outputs: tuple[str, ...]

    def _set_channels(self, channels: int) -> None:
        channels = int(channels)
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if channels == self._channels:
            return
        for name, callback in self._frames_inputs.items():
            new = StaticTypeInputPort(name, [frames_type(channels)])
            replace_node_input_port(self, self.inputs[name], new, callback)
        for name in self._frames_outputs:
            new = StaticTypeOutputPort(name, [frames_type(channels)])
            replace_node_output_port(self, self.outputs[name], new)
        self._channels = channels
        self.properties["channels"] = channels


def apply_numeric(node: BaseNode, properties: dict[str, PropertyDataType], spec: dict[str, type]) -> None:
    """Copy properties listed in ``spec`` (name -> int/float/bool/str) with casting.

    Nodes then mirror the values they use per block into typed attributes,
    so the processing path never looks properties up or converts them.
    """
    for key, kind in spec.items():
        if key in properties:
            node.properties[key] = kind(properties[key])
