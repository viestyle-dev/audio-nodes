import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from audio_nodes.audio_file_stream_nodes import AudioFileStreamNode


# ── Helpers ──


def _create_ramp_wav(frames, channels=1, sample_rate=44100):
    """WAV with per-frame identifiable values (float subtype: bit-exact)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    ramp = (np.arange(frames, dtype=np.float32) / max(frames, 1)) * 0.9
    if channels == 1:
        data = ramp
    else:
        data = np.column_stack([ramp for _ in range(channels)])
    sf.write(tmp.name, data, sample_rate, subtype="FLOAT")
    return tmp.name, ramp


def _cleanup_file(path):
    time.sleep(0.05)
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass


def _setup_node(path, channels=1, loop=False):
    """Create a node with mocked parent graph, load the file via set_properties."""
    node = AudioFileStreamNode("test")
    mock_persist = Mock()
    mock_persist.data_dir_path = Path(path).parent
    mock_graph = Mock()
    mock_graph.persist = mock_persist
    node.parent_graph = mock_graph

    node.set_properties(
        {"file_path": Path(path).name, "channels": channels, "loop": loop}
    )
    assert node._sound_file is not None, "file failed to load"

    # Subscribe AFTER set_properties: a channels change recreates the frames
    # port. Ports use __slots__, so dispatch cannot be replaced on the instance.
    frames_dispatched = []
    eof_dispatched = []
    node._output_frames.subscribe(
        id(frames_dispatched), lambda values: frames_dispatched.append(values[0])
    )
    node._output_eof.subscribe(
        id(eof_dispatched), lambda values: eof_dispatched.append(bool(values[0][0]))
    )
    return node, frames_dispatched, eof_dispatched


def _readout(node, count):
    node._on_input_readout_count([np.array([count], dtype=np.int32)])


# ── Tests ──


def test_loop_property_and_ports():
    node = AudioFileStreamNode("test")
    assert "eof" in node.outputs
    assert node._loop is False
    assert node._eof_reached is False

    mock_graph = Mock()
    mock_graph.persist = None
    node.parent_graph = mock_graph
    node.set_properties({"loop": True})
    assert node._loop is True
    assert node.properties["loop"] is True


def test_no_loop_eof_dispatched_once():
    """loop=False: partial frames at EOF, then eof=True exactly once."""
    path, ramp = _create_ramp_wav(frames=100)
    try:
        node, frames, eofs = _setup_node(path, channels=1, loop=False)

        _readout(node, 150)  # over-read: 100 frames + EOF
        assert len(frames) == 1
        assert frames[0].shape == (100, 1)
        np.testing.assert_array_equal(frames[0][:, 0], ramp)
        assert eofs == [True]

        _readout(node, 50)  # at EOF: no frames, no second eof
        assert len(frames) == 1
        assert eofs == [True]

        node.finalize()
    finally:
        _cleanup_file(path)


def test_loop_gapless_wrap():
    """loop=True: a read crossing EOF returns tail + head, sample-exact."""
    path, ramp = _create_ramp_wav(frames=100)
    try:
        node, frames, eofs = _setup_node(path, channels=1, loop=True)

        _readout(node, 90)  # leaves 10 frames before EOF
        _readout(node, 30)  # 10 tail + 20 head
        assert len(frames) == 2
        wrapped = frames[1][:, 0]
        assert wrapped.shape == (30,)
        np.testing.assert_array_equal(wrapped[:10], ramp[90:])
        np.testing.assert_array_equal(wrapped[10:], ramp[:20])
        assert eofs == []  # loop mode never dispatches eof
        assert node._eof_reached is False

        node.finalize()
    finally:
        _cleanup_file(path)


def test_loop_tiny_file_multiwrap():
    """loop=True: a file shorter than the readout wraps multiple times."""
    path, ramp = _create_ramp_wav(frames=8)
    try:
        node, frames, eofs = _setup_node(path, channels=1, loop=True)

        _readout(node, 30)
        assert len(frames) == 1
        out = frames[0][:, 0]
        assert out.shape == (30,)
        expected = np.tile(ramp, 4)[:30]
        np.testing.assert_array_equal(out, expected)

        node.finalize()
    finally:
        _cleanup_file(path)


def test_seek_clears_eof_state():
    """After EOF, seek(0) re-arms reading and the one-shot eof output."""
    path, ramp = _create_ramp_wav(frames=100)
    try:
        node, frames, eofs = _setup_node(path, channels=1, loop=False)

        _readout(node, 150)
        assert eofs == [True]
        assert node._eof_reached is True

        node._on_input_seek([np.array([0], dtype=np.int32)])
        assert node._eof_reached is False

        _readout(node, 150)
        assert len(frames) == 2
        np.testing.assert_array_equal(frames[1][:, 0], ramp)
        assert eofs == [True, True]

        node.finalize()
    finally:
        _cleanup_file(path)


def test_channel_mismatch_gate_with_loop():
    """Channel mismatch still blocks frame dispatch when loop is enabled."""
    path, _ = _create_ramp_wav(frames=100, channels=2)
    try:
        node, frames, eofs = _setup_node(path, channels=1, loop=True)
        assert node._channels_mismatch is True

        _readout(node, 50)
        assert frames == []
        assert eofs == []

        node.finalize()
    finally:
        _cleanup_file(path)


def test_enabling_loop_clears_eof_state():
    path, _ = _create_ramp_wav(frames=100)
    try:
        node, frames, eofs = _setup_node(path, channels=1, loop=False)
        _readout(node, 150)
        assert node._eof_reached is True

        node.set_properties({"loop": True})
        assert node._eof_reached is False

        _readout(node, 50)  # file is at EOF; loop wraps to the start
        assert len(frames) == 2
        assert frames[1].shape == (50, 1)

        node.finalize()
    finally:
        _cleanup_file(path)
