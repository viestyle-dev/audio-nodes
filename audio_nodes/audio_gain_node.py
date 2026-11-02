import logging

import numpy as np

from nonoha_core.data_type import (
    DataType,
    PortValueType,
    PropertyDataType,
)
from nonoha_core.port import (
    PortValue,
    StaticTypeInputPort,
    StaticTypeOutputPort,
)
from nonoha_core.node import (
    BaseNode,
    NodeDoc,
    replace_node_input_port,
    replace_node_output_port,
)

logger = logging.getLogger(__name__)


class AudioGainNode(BaseNode):
    """Per-sample gain stage with constant-slew glide toward a target gain.

    The gain input port sets a TARGET; the node glides from the current gain
    toward it at a fixed slew rate (ramp_seconds per gain difference of 1.0),
    sample by sample across the frames passing through. A short ramp_seconds
    (default 0.05) acts as click-free volume smoothing; a long one (e.g. 3.0)
    implements fade in/out driven by a single target change.

    Gain starts at 0.0, so the first audible frames always fade in from
    silence.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Apply gain to audio frames, gliding toward the target gain "
                "at a constant slew rate (no clicks). Short ramp_seconds = "
                "smoothed volume control, long ramp_seconds = fade in/out."
            ),
            category="Audio",
            inputs={
                "frames": "Audio frames input. Always 2D (count, channels).",
                "gain": "Target gain (linear). The node glides toward it.",
            },
            outputs={
                "frames": "Gain-applied audio frames. Same shape as input.",
            },
        )

    PROPERTIES = {
        "channels": {
            "description": "Channel count of the frames ports. Must match up/downstream.",
            "data_type": "i32",
            "default_value": 2,
        },
        "sample_rate": {
            "description": "Sample rate used to convert ramp_seconds to a per-frame step.",
            "data_type": "i32",
            "default_value": 44100,
        },
        "ramp_seconds": {
            "description": (
                "Seconds to traverse a gain difference of 1.0 (slew rate). "
                "0 or less = jump instantly."
            ),
            "data_type": "f64",
            "default_value": 0.05,
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "AudioGain")

        channels = 2

        input_frames = StaticTypeInputPort(
            "frames",
            [PortValueType(DataType.Float32, (channels,))],
        )
        input_frames.subscribe(id(self), self._on_input_frames)
        input_gain = StaticTypeInputPort(
            "gain",
            [PortValueType(DataType.Float32)],
        )
        input_gain.subscribe(id(self), self._on_input_gain)

        output_frames = StaticTypeOutputPort(
            "frames",
            [PortValueType(DataType.Float32, (channels,))],
        )

        self._input_frames = input_frames
        self._input_gain = input_gain
        self._output_frames = output_frames

        self.inputs[input_frames.name] = input_frames
        self.inputs[input_gain.name] = input_gain
        self.outputs[output_frames.name] = output_frames

        # Properties
        self._channels = channels
        self._sample_rate = 44100
        self._ramp_seconds = 0.05

        # Gain state (main thread only)
        self._current = 0.0
        self._target = 0.0
        self._step = self._compute_step()

    def _compute_step(self) -> float:
        """Per-frame gain increment. 0.0 means jump instantly."""
        if self._ramp_seconds <= 0.0 or self._sample_rate <= 0:
            return 0.0
        return 1.0 / (self._ramp_seconds * self._sample_rate)

    def set_properties(self, properties: dict[str, PropertyDataType]):
        channels = self._channels
        if "channels" in properties:
            channels = int(properties["channels"])
        if "sample_rate" in properties:
            self._sample_rate = int(properties["sample_rate"])
        if "ramp_seconds" in properties:
            self._ramp_seconds = float(properties["ramp_seconds"])
        self._step = self._compute_step()

        # Handle channels change - recreate both frames ports if needed
        if channels != self._channels:
            new_input_frames = StaticTypeInputPort(
                "frames",
                [PortValueType(DataType.Float32, (channels,))],
            )
            replace_node_input_port(
                self, self._input_frames, new_input_frames, self._on_input_frames
            )
            self._input_frames = new_input_frames

            new_output_frames = StaticTypeOutputPort(
                "frames",
                [PortValueType(DataType.Float32, (channels,))],
            )
            replace_node_output_port(self, self._output_frames, new_output_frames)
            self._output_frames = new_output_frames

            self._channels = channels

        self.properties["channels"] = self._channels
        self.properties["sample_rate"] = self._sample_rate
        self.properties["ramp_seconds"] = self._ramp_seconds

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_input_gain(self, values: list[PortValue]):
        # Use the last value in the batch (most recent target)
        self._target = float(values[0][-1])

    def _on_input_frames(self, values: list[PortValue]):
        frames = values[0]
        if frames.ndim == 1:
            frames = frames.reshape(-1, 1)
        n = len(frames)
        if n == 0:
            return

        cur = self._current
        tgt = self._target

        if cur == tgt:
            if cur == 1.0:
                out = frames  # unity gain: pass through bit-exact
            else:
                out = frames * np.float32(cur)
        elif self._step <= 0.0:
            # Instant mode (ramp_seconds <= 0)
            self._current = tgt
            out = frames * np.float32(tgt)
        else:
            # Constant-slew linear ramp across this batch, clamped at target
            ramp = np.arange(1, n + 1, dtype=np.float32)
            if tgt > cur:
                ramp *= np.float32(self._step)
                ramp += np.float32(cur)
                np.minimum(ramp, np.float32(tgt), out=ramp)
            else:
                ramp *= np.float32(-self._step)
                ramp += np.float32(cur)
                np.maximum(ramp, np.float32(tgt), out=ramp)
            self._current = float(ramp[-1])
            out = frames * ramp[:, None]

        self._output_frames.dispatch([out])
