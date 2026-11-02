import asyncio
import os
import sys
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from audio_nodes.audio_device_nodes import AudioDevicePlayFileNode, RingBuffer


# ── Helpers ──


def _create_test_wav(channels=1, duration=0.5, sample_rate=44100):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if channels == 1:
        data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    else:
        data = np.column_stack(
            [np.sin(2 * np.pi * 440 * t).astype(np.float32) for _ in range(channels)]
        )
    sf.write(tmp.name, data, sample_rate)
    return tmp.name


def _cleanup_file(path):
    time.sleep(0.1)
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass


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


_TRIGGER = [np.array([1], dtype=np.float32)]


@pytest.fixture
def mock_sd():
    """Replace sounddevice in sys.modules so worker threads get the mock."""
    mock_module = MagicMock()
    mock_stream = MagicMock()
    mock_module.OutputStream.return_value = mock_stream
    mock_module.CallbackStop = type("CallbackStop", (Exception,), {})
    with patch.dict(sys.modules, {"sounddevice": mock_module}):
        yield mock_module


def _setup_node(path, mock_sd_module):
    """Create node, start worker, set file_path. Returns (node, loop).

    Does NOT trigger load — caller must do node._on_load(_TRIGGER).
    """
    node = AudioDevicePlayFileNode("test")

    mock_persist = Mock()
    mock_persist.data_dir_path = Path(path).parent
    mock_graph = Mock()
    mock_graph.persist = mock_persist
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

    # Set file_path property (no auto-load)
    node.set_properties({"file_path": Path(path).name})
    return node, loop


def _setup_loaded_node(path, mock_sd_module):
    """Create node, start worker, load file. Returns (node, loop) in 'stopped' state."""
    node, loop = _setup_node(path, mock_sd_module)
    node._on_load(_TRIGGER)
    _wait_for_state(node, "stopped")
    return node, loop


def _shutdown_loop(loop):
    loop.call_soon_threadsafe(loop.stop)
    time.sleep(0.1)
    loop.close()


# ── Tests ──


def test_node_ports():
    """Verify port structure with load/unload/play/stop + state/last_error."""
    node = AudioDevicePlayFileNode("test")
    assert node.type == "AudioDevicePlayFile"
    assert "load" in node.inputs
    assert "unload" in node.inputs
    assert "play" in node.inputs
    assert "stop" in node.inputs
    assert "state" in node.outputs
    assert "last_error" in node.outputs


def test_initial_state():
    """Initial state is 'unloaded'."""
    node = AudioDevicePlayFileNode("test")
    assert node._state == "unloaded"


# ── State guard tests (no worker needed) ──


def test_state_guard_load_when_not_allowed():
    """Load is ignored when state is not 'unloaded' or 'stopped'."""
    node = AudioDevicePlayFileNode("test")
    node._state = "playing"
    node._on_load(_TRIGGER)
    assert node._cmd_queue.empty()


def test_state_guard_load_requires_file_path():
    """Load is ignored when file_path is not set."""
    node = AudioDevicePlayFileNode("test")
    node._state = "unloaded"
    node._on_load(_TRIGGER)
    assert node._cmd_queue.empty()


def test_state_guard_unload_when_not_stopped():
    """Unload is ignored when state is not 'stopped'."""
    node = AudioDevicePlayFileNode("test")
    node._state = "unloaded"
    node._on_unload(_TRIGGER)
    assert node._cmd_queue.empty()


def test_state_guard_play_when_not_stopped():
    """Play is ignored when state is not 'stopped'."""
    node = AudioDevicePlayFileNode("test")
    node._state = "unloaded"
    node._on_play(_TRIGGER)
    assert node._cmd_queue.empty()


def test_state_guard_stop_when_not_playing():
    """Stop is ignored when state is not 'playing'."""
    node = AudioDevicePlayFileNode("test")
    node._state = "stopped"
    node._on_stop(_TRIGGER)
    assert node._cmd_queue.empty()


# ── Worker integration tests ──


def test_load_via_worker(mock_sd):
    """Load trigger → loading → stopped, ring buffer filled, OutputStream created."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)

        assert node._ring_buffer is not None
        assert node._ring_buffer.available() > 0
        mock_sd.OutputStream.assert_called_once()

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_load_nonblocking(mock_sd):
    """_on_load returns immediately without blocking."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_node(path, mock_sd)

        t0 = time.perf_counter()
        node._on_load(_TRIGGER)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, f"_on_load blocked for {elapsed:.3f}s"

        _wait_for_state(node, "stopped")

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_play_nonblocking(mock_sd):
    """_on_play returns immediately without blocking."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)

        t0 = time.perf_counter()
        node._on_play(_TRIGGER)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, f"_on_play blocked for {elapsed:.3f}s"

        _wait_for_state(node, "playing")
        mock_sd.OutputStream.return_value.start.assert_called()

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_play_stop_cycle(mock_sd):
    """Play then stop transitions correctly."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")

        node._on_stop(_TRIGGER)
        _wait_for_state(node, "stopped")

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_play_stop_play_reuse(mock_sd):
    """Stream is reused across stop/play (OutputStream created once during load)."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")

        node._on_stop(_TRIGGER)
        _wait_for_state(node, "stopped")

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")

        assert mock_sd.OutputStream.call_count == 1

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_unload(mock_sd):
    """Unload releases stream and file, returns to 'unloaded'."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)

        node._on_unload(_TRIGGER)
        _wait_for_state(node, "unloaded")

        assert node._ring_buffer is None
        mock_sd.OutputStream.return_value.close.assert_called()

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def _wait_for_condition(predicate, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"Condition not met within {timeout}s")


def test_reload_from_stopped(mock_sd):
    """Load from stopped state destroys and recreates stream."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)
        assert mock_sd.OutputStream.call_count == 1

        # Re-load from stopped
        node._on_load(_TRIGGER)
        _wait_for_condition(lambda: mock_sd.OutputStream.call_count == 2)
        _wait_for_state(node, "stopped")

        # Old stream was closed
        mock_sd.OutputStream.return_value.close.assert_called()

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_state_output_port(mock_sd):
    """State output port dispatches transitions."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_node(path, mock_sd)
        states = []
        node._output_state.subscribe(999, lambda v: states.append(v[0].item()))

        node._on_load(_TRIGGER)
        _wait_for_state(node, "stopped")
        time.sleep(0.05)

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")
        time.sleep(0.05)

        assert "loading" in states
        assert "stopped" in states
        assert "playing" in states

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_load_error(mock_sd):
    """Invalid file path dispatches last_error and returns to 'unloaded'."""
    node = AudioDevicePlayFileNode("test")

    mock_persist = Mock()
    mock_persist.data_dir_path = Path("/nonexistent")
    mock_graph = Mock()
    mock_graph.persist = mock_persist
    node.parent_graph = mock_graph

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    node._main_loop = loop
    node._worker_thread = threading.Thread(
        target=node._worker_loop, daemon=True, name="test_worker"
    )
    node._worker_thread.start()

    errors = []
    node._output_last_error.subscribe(999, lambda v: errors.append(v[0].item()))

    node.set_properties({"file_path": "nonexistent.wav"})
    node._on_load(_TRIGGER)
    _wait_for_state(node, "unloaded")
    time.sleep(0.05)

    assert len(errors) > 0

    node.finalize()
    _shutdown_loop(loop)


def test_finalize_stops_worker(mock_sd):
    """finalize() shuts down worker thread."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)

        worker = node._worker_thread
        node.finalize()
        assert not worker.is_alive()

        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_stereo_load(mock_sd):
    """Stereo file creates 2-channel ring buffer."""
    path = _create_test_wav(channels=2)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)

        assert node._ring_buffer._channels == 2

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_set_properties_no_auto_load(mock_sd):
    """set_properties does not trigger load — state stays 'unloaded'."""
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_node(path, mock_sd)
        time.sleep(0.1)

        assert node._state == "unloaded"
        mock_sd.OutputStream.assert_not_called()

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_natural_completion_then_replay(mock_sd):
    """Regression: after _on_finished fires, stream.abort() must be called
    during CMD_PLAYBACK_ENDED to reset PortAudio state. Without this, on
    Windows/WASAPI a subsequent stream.start() succeeds but produces silent
    playback.
    """
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)
        stream_mock = mock_sd.OutputStream.return_value

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")

        # Simulate PortAudio finishing naturally (file ran to end).
        node._on_finished()
        _wait_for_state(node, "stopped")

        # Fix requires abort() during CMD_PLAYBACK_ENDED.
        assert stream_mock.abort.call_count >= 1, (
            "stream.abort() must be called during CMD_PLAYBACK_ENDED to reset "
            "PortAudio state (Windows/WASAPI silent-replay bug)"
        )
        prior_abort = stream_mock.abort.call_count
        prior_start = stream_mock.start.call_count

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")
        assert stream_mock.start.call_count == prior_start + 1
        # No additional abort needed before start (already aborted in PLAYBACK_ENDED).
        assert stream_mock.abort.call_count == prior_abort

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_natural_completion_abort_swallows_exception(mock_sd):
    """If stream.abort() raises (e.g. paStreamIsStopped on some backends), the
    handler must swallow it and complete the cleanup, reaching state='stopped'.
    """
    path = _create_test_wav(channels=1)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)
        stream_mock = mock_sd.OutputStream.return_value
        stream_mock.abort.side_effect = RuntimeError("paStreamIsStopped")

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")
        node._on_finished()
        # Must still reach 'stopped' even if abort() raised.
        _wait_for_state(node, "stopped")

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


# ── Loop playback (gapless) ──


def test_loop_property():
    node = AudioDevicePlayFileNode("test")
    node.parent_graph = Mock()
    assert node._loop is False
    node.set_properties({"loop": True})
    assert node._loop is True
    assert node.properties["loop"] is True


def test_loop_reader_wraps_without_eof(mock_sd):
    """loop=True: the reader wraps past EOF and never signals it; the buffer
    accumulates more frames than the file contains."""
    path = _create_test_wav(channels=1, duration=0.5)  # 22050 frames
    try:
        node, loop = _setup_loaded_node(path, mock_sd)
        node.set_properties({"loop": True})

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")

        file_frames = int(44100 * 0.5)
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if node._ring_buffer.available() > file_frames:
                break
            time.sleep(0.01)
        assert node._ring_buffer.available() > file_frames, (
            "reader should wrap past EOF and keep filling the buffer"
        )
        assert not node._eof_reached.is_set()

        node._on_stop(_TRIGGER)
        _wait_for_state(node, "stopped")
        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


def test_no_loop_reader_sets_eof(mock_sd):
    """loop=False: the reader signals EOF when the file is exhausted."""
    path = _create_test_wav(channels=1, duration=0.5)
    try:
        node, loop = _setup_loaded_node(path, mock_sd)

        node._on_play(_TRIGGER)
        _wait_for_state(node, "playing")

        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if node._eof_reached.is_set():
                break
            time.sleep(0.01)
        assert node._eof_reached.is_set()

        node._on_stop(_TRIGGER)
        _wait_for_state(node, "stopped")
        # _refill_buffer re-arms the file for replay
        assert not node._eof_reached.is_set()

        node.finalize()
        _shutdown_loop(loop)
    finally:
        _cleanup_file(path)


# ── Audio callback: underrun vs EOF ──


def _callback_node():
    """Node with just enough state to call _audio_callback directly."""
    node = AudioDevicePlayFileNode("test")
    node._sd = MagicMock()
    node._sd.CallbackStop = type("CallbackStop", (Exception,), {})
    node._ring_buffer = RingBuffer(1000, 1)
    return node


def test_callback_underrun_does_not_stop():
    """Regression: a transient short read is an underrun, NOT end-of-file.
    The old implementation raised CallbackStop on any short read, silently
    killing playback forever on a single buffer starvation."""
    node = _callback_node()
    outdata = np.ones((128, 1), dtype=np.float32)

    node._audio_callback(outdata, 128, None, None)  # buffer empty, no EOF

    np.testing.assert_array_equal(outdata, 0)
    assert node._underrun_count == 1


def test_callback_partial_read_pads_and_counts():
    node = _callback_node()
    node._ring_buffer.write(np.full((50, 1), 0.5, dtype=np.float32))
    outdata = np.ones((128, 1), dtype=np.float32)

    node._audio_callback(outdata, 128, None, None)

    np.testing.assert_array_equal(outdata[:50], 0.5)
    np.testing.assert_array_equal(outdata[50:], 0)
    assert node._partial_read_count == 1
    assert node._underrun_count == 0


def test_callback_eof_and_drained_raises_stop():
    """Playback finishes only when EOF is signalled AND the buffer is empty."""
    node = _callback_node()
    node._eof_reached.set()
    outdata = np.ones((128, 1), dtype=np.float32)

    with pytest.raises(node._sd.CallbackStop):
        node._audio_callback(outdata, 128, None, None)
    np.testing.assert_array_equal(outdata, 0)


def test_callback_eof_but_not_drained_keeps_playing():
    """EOF signalled but buffer still has a full block: playback continues."""
    node = _callback_node()
    node._eof_reached.set()
    node._ring_buffer.write(np.full((200, 1), 0.5, dtype=np.float32))
    outdata = np.zeros((128, 1), dtype=np.float32)

    node._audio_callback(outdata, 128, None, None)  # no exception

    np.testing.assert_array_equal(outdata, 0.5)


def test_callback_status_counted():
    node = _callback_node()
    node._ring_buffer.write(np.full((200, 1), 0.5, dtype=np.float32))
    outdata = np.zeros((128, 1), dtype=np.float32)

    node._audio_callback(outdata, 128, None, "output underflow")

    assert node._status_event_count == 1
    assert node._last_status == "output underflow"


# ── device_id parsing ──


def test_device_id_parsing():
    """device_id mirrors sounddevice: empty=default, number=index, else name query."""
    node = AudioDevicePlayFileNode("test")
    node.parent_graph = Mock()

    node.set_properties({"device_id": "Speakers wasapi"})
    assert node._device_id == "Speakers wasapi"
    assert node.properties["device_id"] == "Speakers wasapi"

    node.set_properties({"device_id": "0"})
    assert node._device_id == 0
    assert node.properties["device_id"] == "0"  # index 0 must not collapse to ""

    node.set_properties({"device_id": ""})
    assert node._device_id is None
    assert node.properties["device_id"] == ""
