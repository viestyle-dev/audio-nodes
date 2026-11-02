from unittest.mock import Mock

import numpy as np
import pytest

from nonoha_core.data_type import DataType
from audio_nodes.audio_gain_node import AudioGainNode


# ── Helpers ──


def _setup_node(**props):
    node = AudioGainNode("test")
    node.parent_graph = Mock()
    if props:
        node.set_properties(props)

    dispatched = []
    # Observe the output through the port's subscription API (ports use
    # __slots__, so dispatch cannot be replaced on the instance).
    node._output_frames.subscribe(
        id(dispatched), lambda values: dispatched.append(values[0])
    )
    return node, dispatched


def _ones(n, channels=2):
    return np.ones((n, channels), dtype=np.float32)


# ── Tests ──


def test_ports_and_defaults():
    node = AudioGainNode("test")
    assert node.type == "AudioGain"
    assert "frames" in node.inputs
    assert "gain" in node.inputs
    assert "frames" in node.outputs

    frames_port = node.inputs["frames"]
    assert frames_port.value_type_list[0].data_type == DataType.Float32
    assert frames_port.value_type_list[0].shape == (2,)

    assert node._ramp_seconds == 0.05
    assert node._current == 0.0
    assert node._target == 0.0


def test_unity_gain_passthrough_bitexact():
    node, out = _setup_node()
    node._current = 1.0
    node._target = 1.0

    frames = np.random.randn(64, 2).astype(np.float32)
    node._on_input_frames([frames])

    assert len(out) == 1
    assert out[0] is frames  # unity: passed through without copying


def test_steady_gain_applied():
    node, out = _setup_node()
    node._current = 0.5
    node._target = 0.5

    frames = _ones(64)
    node._on_input_frames([frames])

    np.testing.assert_allclose(out[0], frames * 0.5, rtol=1e-6)


def test_ramp_linear_and_continuous_across_batches():
    """step = 1/(ramp_seconds*sample_rate); the ramp continues across batches."""
    node, out = _setup_node(sample_rate=1000, ramp_seconds=1.0)  # step = 0.001
    node._on_input_gain([np.array([1.0], dtype=np.float32)])

    node._on_input_frames([_ones(100)])
    expected1 = (np.arange(1, 101, dtype=np.float32) * 0.001)[:, None]
    np.testing.assert_allclose(out[0], np.broadcast_to(expected1, (100, 2)), rtol=1e-5)
    assert node._current == pytest.approx(0.1, rel=1e-5)

    node._on_input_frames([_ones(100)])
    expected2 = (0.1 + np.arange(1, 101, dtype=np.float32) * 0.001)[:, None]
    np.testing.assert_allclose(out[1], np.broadcast_to(expected2, (100, 2)), rtol=1e-5)
    assert node._current == pytest.approx(0.2, rel=1e-5)


def test_ramp_clamps_at_target():
    node, out = _setup_node(sample_rate=1000, ramp_seconds=1.0)  # step = 0.001
    node._current = 0.95
    node._target = 1.0

    node._on_input_frames([_ones(100)])
    # Reaches 1.0 at frame 50, then stays clamped
    assert node._current == 1.0
    result = out[0][:, 0]
    assert result[49] == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(result[50:], 1.0, rtol=1e-6)
    assert np.all(np.diff(result) >= -1e-7)  # monotonic non-decreasing


def test_target_change_mid_ramp_is_continuous():
    node, out = _setup_node(sample_rate=1000, ramp_seconds=1.0)
    node._on_input_gain([np.array([1.0], dtype=np.float32)])
    node._on_input_frames([_ones(100)])  # current -> 0.1

    node._on_input_gain([np.array([0.05], dtype=np.float32)])  # new lower target
    node._on_input_frames([_ones(100)])
    result = out[1][:, 0]
    # Starts gliding DOWN from current (0.1), no jump
    assert result[0] == pytest.approx(0.099, rel=1e-4)
    assert node._current == pytest.approx(0.05, rel=1e-5)  # clamped at new target


def test_ramp_seconds_zero_is_instant():
    node, out = _setup_node(ramp_seconds=0.0)
    node._on_input_gain([np.array([0.7], dtype=np.float32)])

    node._on_input_frames([_ones(10)])
    np.testing.assert_allclose(out[0], _ones(10) * np.float32(0.7), rtol=1e-6)
    assert node._current == pytest.approx(0.7)


def test_gain_batch_uses_last_value():
    node, _ = _setup_node()
    node._on_input_gain([np.array([0.1, 0.5, 0.9], dtype=np.float32)])
    assert node._target == pytest.approx(0.9, rel=1e-6)


def test_channels_property_rebuilds_ports():
    node = AudioGainNode("test")
    node.parent_graph = Mock()
    node.set_properties({"channels": 4})

    assert node.inputs["frames"].value_type_list[0].shape == (4,)
    assert node.outputs["frames"].value_type_list[0].shape == (4,)
