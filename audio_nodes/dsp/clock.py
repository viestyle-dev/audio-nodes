"""AudioClock — a demand-driven block clock for synthesis graphs."""

from __future__ import annotations

import numpy as np

from nonoha_core.data_type import PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc
from nonoha_core.port import PortValue

from ._common import ANY, BOOL, F64, SAMPLE_RATE_PROPERTY, add_input, add_output, apply_numeric


class AudioClockNode(BaseNode):
    """Emit blocks of per-sample time when the player asks for more audio.

    Nothing here runs on a timer: ``start`` fills the output buffer with
    ``prefill_blocks`` blocks, then every ``demand`` (wire it to
    AudioDevicePlayStream.buffer_low, or to the host app's player) produces
    ``blocks_per_demand`` more. Generation therefore follows playback exactly
    and never runs ahead or falls behind, whatever the host is.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Demand-driven block clock. 'start' prefills the player's "
                "buffer; each 'demand' (e.g. from buffer_low) emits more "
                "blocks. Emits per-sample time in seconds (f64)."
            ),
            category="Audio",
            inputs={
                "start": "Any value: reset time to 0, emit the prefill and start answering demand",
                "demand": "Any value: emit blocks_per_demand blocks (after start)",
                "stop": "Any value: stop answering demand",
            },
            outputs={
                "time": "Per-sample time in seconds, one dispatch per request",
                "primed": "True once the prefill has been emitted (e.g. to start playback)",
            },
        )

    PROPERTIES = {
        "sample_rate": SAMPLE_RATE_PROPERTY,
        "block_size": {"description": "Samples per block (2205 = 50 ms at 44.1 kHz)", "data_type": "i32", "default_value": 2205},
        "prefill_blocks": {"description": "Blocks emitted on start", "data_type": "i32", "default_value": 12},
        "blocks_per_demand": {"description": "Blocks emitted per demand", "data_type": "i32", "default_value": 2},
    }

    def __init__(self, name: str):
        super().__init__(name, "AudioClock")
        add_input(self, "start", ANY, self._on_start)
        add_input(self, "demand", ANY, self._on_demand)
        add_input(self, "stop", ANY, self._on_stop)
        self._out_time = add_output(self, "time", F64)
        self._out_primed = add_output(self, "primed", BOOL)
        self._sample_index = 0
        self._running = False
        self._primed = np.array([True])
        self._primed.setflags(write=False)
        self._mirror()

    def _mirror(self):
        p = self.properties
        self._block_size = max(0, int(p["block_size"]))
        self._prefill = max(0, int(p["prefill_blocks"]))
        self._per_demand = max(0, int(p["blocks_per_demand"]))
        self._inv_sr = 1.0 / float(p["sample_rate"])

    def _emit(self, blocks: int):
        n = self._block_size * blocks
        if n == 0:
            return
        start = self._sample_index
        t = np.arange(start, start + n, dtype=np.float64)
        t *= self._inv_sr
        self._sample_index = start + n
        self._out_time.dispatch([t])

    def _on_start(self, values: list[PortValue]):
        self._sample_index = 0
        self._running = True
        self._emit(self._prefill)
        self._out_primed.dispatch([self._primed])

    def _on_demand(self, values: list[PortValue]):
        if self._running:
            self._emit(self._per_demand)

    def _on_stop(self, values: list[PortValue]):
        self._running = False

    def set_properties(self, properties: dict[str, PropertyDataType]):
        apply_numeric(
            self,
            properties,
            {"sample_rate": int, "block_size": int, "prefill_blocks": int, "blocks_per_demand": int},
        )
        self._mirror()
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
