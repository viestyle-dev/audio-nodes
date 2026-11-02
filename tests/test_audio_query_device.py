import sys
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from nonoha_core.data_type import DataType
from audio_nodes.audio_device_nodes import AudioQueryDeviceNode


_FAKE_DEVICES = [
    {
        "name": "Speakers (Realtek)",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_low_input_latency": 0.09,
        "default_low_output_latency": 0.12,
        "default_high_input_latency": 0.18,
        "default_high_output_latency": 0.24,
        "default_samplerate": 48000.0,
    },
    {
        "name": "Microphone (USB)",
        "hostapi": 1,
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_low_input_latency": 0.01,
        "default_low_output_latency": 0.02,
        "default_high_input_latency": 0.03,
        "default_high_output_latency": 0.04,
        "default_samplerate": 44100.0,
    },
]
_FAKE_HOSTAPIS = [{"name": "Windows WASAPI"}, {"name": "MME"}]

_DEVICE_VALUE_NAMES = [
    "index",
    "name",
    "hostapi",
    "max_input_channels",
    "max_output_channels",
    "default_low_input_latency",
    "default_low_output_latency",
    "default_high_input_latency",
    "default_high_output_latency",
    "default_samplerate",
]
_DEVICE_VALUE_TYPES = [
    DataType.Int32,
    DataType.String,
    DataType.String,
    DataType.Int32,
    DataType.Int32,
    DataType.Float64,
    DataType.Float64,
    DataType.Float64,
    DataType.Float64,
    DataType.Float64,
]


@pytest.fixture
def mock_sd():
    mock_module = MagicMock()
    mock_module.query_devices.return_value = _FAKE_DEVICES
    mock_module.query_hostapis.return_value = _FAKE_HOSTAPIS
    mock_module.default.device = {"input": 1, "output": 0}
    with patch.dict(sys.modules, {"sounddevice": mock_module}):
        yield mock_module


def test_query_device_node_creation():
    """Ports exist with the full PaDeviceInfo mirror and named values."""
    node = AudioQueryDeviceNode("test_query")
    assert node.type == "AudioQueryDevice"
    assert "dispatch" in node.inputs

    devices_port = node.outputs["devices"]
    assert [vt.data_type for vt in devices_port.value_type_list] == _DEVICE_VALUE_TYPES
    assert devices_port.value_name_list == _DEVICE_VALUE_NAMES

    for port_name in ("default_output", "default_input"):
        port = node.outputs[port_name]
        assert [vt.data_type for vt in port.value_type_list] == [
            DataType.String,
            DataType.Int32,
            DataType.String,
        ]
        assert port.value_name_list == ["name", "index", "hostapi"]


def test_query_device_nonblocking(mock_sd):
    """Dispatch triggers device query on background thread, doesn't block."""
    node = AudioQueryDeviceNode("test_query")

    t0 = time.perf_counter()
    node._on_input_dispatch([np.array([1], dtype=np.float32)])
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.05, f"_on_input_dispatch blocked for {elapsed:.3f}s"

    # Wait for background thread to complete
    time.sleep(0.3)


def test_query_device_dispatches_results(mock_sd):
    """_query_devices marshals parallel device arrays and both defaults."""
    node = AudioQueryDeviceNode("test_query")
    node._main_loop = Mock()

    node._query_devices()  # run synchronously in the test thread

    node._main_loop.call_soon_threadsafe.assert_called_once()
    args = node._main_loop.call_soon_threadsafe.call_args[0]
    assert args[0] == node._dispatch_results
    device_arrays, default_out, default_in = args[1], args[2], args[3]

    assert len(device_arrays) == 10
    np.testing.assert_array_equal(device_arrays[0], np.array([0, 1], dtype=np.int32))
    assert device_arrays[0].dtype == np.int32
    assert list(device_arrays[1]) == ["Speakers (Realtek)", "Microphone (USB)"]
    assert list(device_arrays[2]) == ["Windows WASAPI", "MME"]
    np.testing.assert_array_equal(device_arrays[3], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(device_arrays[4], np.array([2, 0], dtype=np.int32))
    np.testing.assert_allclose(device_arrays[5], [0.09, 0.01])
    np.testing.assert_allclose(device_arrays[6], [0.12, 0.02])
    np.testing.assert_allclose(device_arrays[7], [0.18, 0.03])
    np.testing.assert_allclose(device_arrays[8], [0.24, 0.04])
    np.testing.assert_allclose(device_arrays[9], [48000.0, 44100.0])
    for arr in device_arrays[5:]:
        assert arr.dtype == np.float64

    assert str(default_out[0][0]) == "Speakers (Realtek)"
    assert int(default_out[1][0]) == 0
    assert str(default_out[2][0]) == "Windows WASAPI"

    assert str(default_in[0][0]) == "Microphone (USB)"
    assert int(default_in[1][0]) == 1
    assert str(default_in[2][0]) == "MME"


def test_query_device_empty_list_and_no_defaults(mock_sd):
    """Zero devices: 10 empty arrays with correct dtypes; defaults = ('', -1, '')."""
    mock_sd.query_devices.return_value = []
    mock_sd.default.device = {"input": -1, "output": -1}

    node = AudioQueryDeviceNode("test_query")
    node._main_loop = Mock()

    node._query_devices()

    args = node._main_loop.call_soon_threadsafe.call_args[0]
    device_arrays, default_out, default_in = args[1], args[2], args[3]

    assert len(device_arrays) == 10
    for arr, data_type in zip(device_arrays, _DEVICE_VALUE_TYPES):
        assert len(arr) == 0
        if data_type == DataType.Int32:
            assert arr.dtype == np.int32
        elif data_type == DataType.Float64:
            assert arr.dtype == np.float64

    for default in (default_out, default_in):
        assert str(default[0][0]) == ""
        assert int(default[1][0]) == -1
        assert str(default[2][0]) == ""


def test_query_device_output_dispatch(mock_sd):
    """_dispatch_results pushes all three outputs to the graph."""
    node = AudioQueryDeviceNode("test_query")
    node._is_active = True
    node.parent_graph = Mock()

    dispatched = {}
    for port_name in ("devices", "default_output", "default_input"):
        port = node.outputs[port_name]
        port.subscribe(
            id(dispatched),
            lambda values, key=port_name: dispatched.__setitem__(key, values),
        )

    # Real arrays: nonoha-core's pytest plugin validates the shape of every
    # dispatched value against the port's value types.
    str_dtype = np.dtypes.StringDType()
    device_arrays = [
        np.array([0], dtype=np.int32),
        np.array(["Speakers (Realtek)"], dtype=str_dtype),
        np.array(["Windows WASAPI"], dtype=str_dtype),
        np.array([0], dtype=np.int32),
        np.array([2], dtype=np.int32),
    ] + [np.array([0.0], dtype=np.float64) for _ in range(5)]
    default_out = [
        np.array(["Speakers (Realtek)"], dtype=str_dtype),
        np.array([0], dtype=np.int32),
        np.array(["Windows WASAPI"], dtype=str_dtype),
    ]
    default_in = [
        np.array(["Microphone (USB)"], dtype=str_dtype),
        np.array([1], dtype=np.int32),
        np.array(["MME"], dtype=str_dtype),
    ]
    node._dispatch_results(device_arrays, default_out, default_in)

    assert dispatched["devices"] is device_arrays
    assert dispatched["default_output"] is default_out
    assert dispatched["default_input"] is default_in
    node.parent_graph.flush_outputs.assert_called_once()
