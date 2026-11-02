import asyncio
import sys
import time
import threading
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from nonoha_core.data_type import DataType
from audio_nodes.audio_device_nodes import AudioDevicePlayStreamNode


# ── Helpers ──


def _wait_for_state(node, expected_state, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if node._state == expected_state:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"State did not reach '{expected_state}' within {timeout}s "
        f"(current: '{node._state}')"
    )


@pytest.fixture
def mock_sd():
    """Replace sounddevice in sys.modules so worker threads get the mock."""
    mock_module = MagicMock()
    mock_stream = MagicMock()
    mock_module.OutputStream.return_value = mock_stream
    mock_module.CallbackStop = type("CallbackStop", (Exception,), {})
    with patch.dict(sys.modules, {"sounddevice": mock_module}):
        yield mock_module


def _setup_node(mock_sd_module):
    """Create, configure, and activate a PlayStreamNode with mocked sounddevice."""
    node = AudioDevicePlayStreamNode("test")
    mock_graph = Mock()
    node.parent_graph = mock_graph

    # Run event loop in background so call_soon_threadsafe callbacks execute
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    node._main_loop = loop
    node._worker_thread = threading.Thread(
        target=node._worker_loop, daemon=True, name="test_worker"
    )
    node._worker_thread.start()

    # Trigger stream creation
    node._cmd_queue.put((node._CMD_CREATE_STREAM, node._stream_params()))
    return node, loop


def _shutdown_loop(loop):
    loop.call_soon_threadsafe(loop.stop)
    time.sleep(0.1)
    loop.close()


# ── Tests ──


def test_node_creation_ports():
    """Verify port presence and default types."""
    node = AudioDevicePlayStreamNode("test")
    assert node.type == "AudioDevicePlayStream"
    assert "frames" in node.inputs
    assert "play" in node.inputs
    assert "stop" in node.inputs
    assert "buffer_low" in node.outputs
    assert "state" in node.outputs
    assert "last_error" in node.outputs

    frames_port = node.inputs["frames"]
    assert frames_port.value_type_list[0].data_type == DataType.Float32
    assert frames_port.value_type_list[0].shape == (2,)


@pytest.mark.parametrize("channels", [1, 2, 4])
def test_set_properties_port_type(channels):
    """Port type is always f32(channels,) after set_properties."""
    node = AudioDevicePlayStreamNode("test")
    mock_graph = Mock()
    node.parent_graph = mock_graph
    node.set_properties({"channels": channels, "sample_rate": 44100, "buffer_duration": 0.1})

    frames_port = node.inputs["frames"]
    assert frames_port.value_type_list[0].shape == (channels,)


def test_input_frames_1d_reshape():
    """1D input is reshaped to 2D before writing to ring buffer."""
    node = AudioDevicePlayStreamNode("test")
    mock_graph = Mock()
    node.parent_graph = mock_graph
    node.set_properties({"channels": 1, "sample_rate": 44100, "buffer_duration": 1.0})

    data_1d = np.random.randn(100).astype(np.float32)
    node._on_input_frames([data_1d])

    assert node._ring_buffer.available() == 100
    result = node._ring_buffer.read(100)
    assert result.shape == (100, 1)


def test_input_frames_2d():
    """2D input is written directly to ring buffer."""
    node = AudioDevicePlayStreamNode("test")
    mock_graph = Mock()
    node.parent_graph = mock_graph
    node.set_properties({"channels": 2, "sample_rate": 44100, "buffer_duration": 1.0})

    data_2d = np.random.randn(100, 2).astype(np.float32)
    node._on_input_frames([data_2d])

    assert node._ring_buffer.available() == 100
    result = node._ring_buffer.read(100)
    assert result.shape == (100, 2)
    np.testing.assert_array_equal(result, data_2d)


def test_play_nonblocking(mock_sd):
    """_on_input_play returns immediately without blocking."""
    node, loop = _setup_node(mock_sd)
    _wait_for_state(node, "stopped")

    t0 = time.perf_counter()
    node._on_input_play([np.array([1], dtype=np.float32)])
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, f"_on_input_play blocked for {elapsed:.3f}s"

    _wait_for_state(node, "playing")

    node.finalize()
    _shutdown_loop(loop)


def test_play_stop_cycle(mock_sd):
    """Play then stop transitions correctly."""
    node, loop = _setup_node(mock_sd)
    _wait_for_state(node, "stopped")

    node._on_input_play([np.array([1], dtype=np.float32)])
    _wait_for_state(node, "playing")

    node._on_input_stop([np.array([1], dtype=np.float32)])
    _wait_for_state(node, "stopped")

    node.finalize()
    _shutdown_loop(loop)


def test_state_guard_play_when_not_stopped():
    """Play is ignored when state is not 'stopped'."""
    node = AudioDevicePlayStreamNode("test")
    node._state = "idle"
    node._on_input_play([np.array([1], dtype=np.float32)])
    assert node._cmd_queue.empty()


def test_play_idempotent_while_playing():
    """Play while playing/starting is a silent no-op (level-driven play)."""
    node = AudioDevicePlayStreamNode("test")
    for state in ("playing", "starting"):
        node._state = state
        node._on_input_play([np.array([1], dtype=np.float32)])
        assert node._cmd_queue.empty()


def test_play_from_error_recreates_stream():
    """Play in the error state enqueues CREATE_STREAM then PLAY (self-recovery)."""
    node = AudioDevicePlayStreamNode("test")
    node._state = "error"
    node._on_input_play([np.array([1], dtype=np.float32)])

    cmd1, payload1 = node._cmd_queue.get_nowait()
    cmd2, _ = node._cmd_queue.get_nowait()
    assert cmd1 == node._CMD_CREATE_STREAM
    assert payload1 == node._stream_params()
    assert cmd2 == node._CMD_PLAY
    assert node._cmd_queue.empty()


def test_device_id_parsing():
    """device_id mirrors sounddevice: empty=default, number=index, else name query."""
    node = AudioDevicePlayStreamNode("test")
    node.parent_graph = Mock()

    node.set_properties({"device_id": "Speakers wasapi"})
    assert node._device_id == "Speakers wasapi"
    assert node.properties["device_id"] == "Speakers wasapi"

    node.set_properties({"device_id": "3"})
    assert node._device_id == 3

    node.set_properties({"device_id": "0"})
    assert node._device_id == 0
    assert node.properties["device_id"] == "0"  # index 0 must not collapse to ""

    node.set_properties({"device_id": ""})
    assert node._device_id is None
    assert node.properties["device_id"] == ""


def test_state_guard_stop_when_not_playing():
    """Stop is ignored when state is not 'playing' or 'starting'."""
    node = AudioDevicePlayStreamNode("test")
    node._state = "stopped"
    node._on_input_stop([np.array([1], dtype=np.float32)])
    assert node._cmd_queue.empty()


def test_stream_latency_property():
    """stream_latency property is configurable."""
    node = AudioDevicePlayStreamNode("test")
    mock_graph = Mock()
    node.parent_graph = mock_graph

    node.set_properties({"stream_latency": 0.02})
    assert node._stream_latency == 0.02

    node.set_properties({"stream_latency": "low"})
    assert node._stream_latency == "low"


def test_finalize_stops_worker(mock_sd):
    """finalize() shuts down worker thread."""
    node, loop = _setup_node(mock_sd)
    _wait_for_state(node, "stopped")

    worker = node._worker_thread
    node.finalize()
    assert not worker.is_alive()

    _shutdown_loop(loop)


def test_stream_creation_error_dispatches_last_error(mock_sd):
    """A stream creation failure surfaces its message on the last_error output."""
    mock_sd.OutputStream.side_effect = RuntimeError(
        "No output device matching 'Surface Omnisonic waspi'"
    )
    node, loop = _setup_node(mock_sd)
    _wait_for_state(node, "error")

    values = node.outputs["last_error"]._current_values
    assert values is not None
    assert "No output device matching" in str(values[0][0])

    node.finalize()
    _shutdown_loop(loop)
