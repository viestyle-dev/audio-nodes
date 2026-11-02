from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    import sounddevice as sd

from nonoha_core.data_type import (
    DataType,
    PortValueType,
    PropertyDataType,
    name_to_port_value_type,
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

# All PortAudio operations (stream open/close/start/stop) must be serialized
# across threads. PortAudio is NOT thread-safe for concurrent stream operations.
_portaudio_lock = threading.Lock()


def _import_sounddevice():
    """Import sounddevice, skipping unnecessary stderr fd manipulation on Windows.

    sounddevice._initialize() redirects fd 2 (stderr) to devnull via os.dup2
    during Pa_Initialize() to suppress PortAudio debug messages. This causes:
    - Windows: os.dup2 (UCRT _dup2) is NOT atomic — concurrent logging gets
      OSError: Bad file descriptor
    - All platforms: log messages written during Pa_Initialize() are lost to devnull

    Fix: temporarily make os.dup2 a no-op during the first import.
    """
    if "sounddevice" not in sys.modules:
        import os

        original_dup2 = os.dup2
        os.dup2 = lambda fd, fd2: None
        try:
            import sounddevice as sd
        finally:
            os.dup2 = original_dup2
    else:
        import sounddevice as sd
    return sd


def _init_worker_thread():
    """Initialize COM for the current thread (required for WASAPI/DirectSound on Windows)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED


def _cleanup_worker_thread():
    """Release COM for the current thread."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.ole32.CoUninitialize()


def _parse_device_id(device_id_str) -> int | str | None:
    """Convert the device_id property to sounddevice's device parameter.

    The plugin does not interpret the value beyond type conversion; matching
    is delegated to sounddevice:
    - empty string -> None (system default device)
    - numeric string -> PortAudio device index (int)
    - anything else -> sounddevice query string: space-separated substrings
      matched case-insensitively, in order, against
      "<device name>, <host API name>"
    """
    if device_id_str is None:
        return None
    s = str(device_id_str)
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return s


class RingBuffer:
    """Thread-safe ring buffer (fixed size, no memory reallocation).

    Internal data is always held as a 2D array (size, channels).
    Even with channels=1 (mono) the shape is (size, 1).
    """

    def __init__(self, size: int, channels: int, dtype=np.float32):
        self._buffer = np.zeros((size, channels), dtype=dtype)
        self._size = size
        self._channels = channels
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """Write data into the buffer. Returns the number of frames actually written."""
        with self._lock:
            frames = len(data)
            available_space = self._size - self._available

            if frames > available_space:
                frames = available_space

            if frames == 0:
                return 0

            end_pos = self._write_pos + frames

            if end_pos <= self._size:
                self._buffer[self._write_pos : end_pos] = data[:frames]
            else:
                first_part = self._size - self._write_pos
                self._buffer[self._write_pos :] = data[:first_part]
                self._buffer[: frames - first_part] = data[first_part:frames]

            self._write_pos = (self._write_pos + frames) % self._size
            self._available += frames

            return frames

    def read(self, frames: int) -> np.ndarray:
        """Read data from the buffer. Always returns a 2D (N, channels) array."""
        with self._lock:
            if frames > self._available:
                frames = self._available

            if frames == 0:
                return np.zeros((0, self._channels), dtype=self._buffer.dtype)

            end_pos = self._read_pos + frames

            if end_pos <= self._size:
                data = self._buffer[self._read_pos : end_pos].copy()
            else:
                first_part = self._size - self._read_pos
                data = np.concatenate(
                    [
                        self._buffer[self._read_pos :],
                        self._buffer[: frames - first_part],
                    ]
                )

            self._read_pos = (self._read_pos + frames) % self._size
            self._available -= frames

            return data

    def read_into(self, out: np.ndarray) -> int:
        """Copy directly into out without allocating (for the realtime thread).
        Returns the number of frames actually read.
        """
        with self._lock:
            frames = min(len(out), self._available)
            if frames == 0:
                return 0

            end_pos = self._read_pos + frames

            if end_pos <= self._size:
                out[:frames] = self._buffer[self._read_pos : end_pos]
            else:
                first = self._size - self._read_pos
                out[:first] = self._buffer[self._read_pos :]
                out[first:frames] = self._buffer[: frames - first]

            self._read_pos = (self._read_pos + frames) % self._size
            self._available -= frames
            return frames

    def available(self) -> int:
        """Return the number of frames available for reading"""
        with self._lock:
            return self._available

    def reset(self):
        """Reset the buffer"""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._available = 0


class AudioQueryDeviceNode(BaseNode):

    # Complete mirror of PortAudio's per-device info (PaDeviceInfo); there is
    # no further per-device information available through sounddevice.
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

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Query available audio devices on trigger. Logs them and "
            "dispatches the device list as named parallel arrays.",
            category="Audio",
            inputs={
                "dispatch": "Trigger to query audio devices.",
            },
            outputs={
                "devices": "Parallel arrays, one element per device: index, name, "
                "hostapi, max_input_channels, max_output_channels, "
                "default low/high input/output latencies, default_samplerate. "
                "Output devices have max_output_channels > 0.",
                "default_output": "Default output device: name, index, hostapi.",
                "default_input": "Default input device: name, index, hostapi.",
            },
        )

    def __init__(self, name: str):
        super().__init__(name, "AudioQueryDevice")

        input_dispatch = StaticTypeInputPort(
            "dispatch",
            [PortValueType(DataType.Any)],
        )
        input_dispatch.subscribe(id(self), self._on_input_dispatch)

        output_devices = StaticTypeOutputPort(
            "devices",
            [
                PortValueType(DataType.Int32),  # index
                PortValueType(DataType.String),  # name
                PortValueType(DataType.String),  # hostapi
                PortValueType(DataType.Int32),  # max_input_channels
                PortValueType(DataType.Int32),  # max_output_channels
                PortValueType(DataType.Float64),  # default_low_input_latency
                PortValueType(DataType.Float64),  # default_low_output_latency
                PortValueType(DataType.Float64),  # default_high_input_latency
                PortValueType(DataType.Float64),  # default_high_output_latency
                PortValueType(DataType.Float64),  # default_samplerate
            ],
            value_name_list=list(self._DEVICE_VALUE_NAMES),
        )
        output_default_output = StaticTypeOutputPort(
            "default_output",
            [
                PortValueType(DataType.String),
                PortValueType(DataType.Int32),
                PortValueType(DataType.String),
            ],
            value_name_list=["name", "index", "hostapi"],
        )
        output_default_input = StaticTypeOutputPort(
            "default_input",
            [
                PortValueType(DataType.String),
                PortValueType(DataType.Int32),
                PortValueType(DataType.String),
            ],
            value_name_list=["name", "index", "hostapi"],
        )

        self._input_dispatch = input_dispatch
        self._output_devices = output_devices
        self._output_default_output = output_default_output
        self._output_default_input = output_default_input

        self.inputs[input_dispatch.name] = input_dispatch
        self.outputs[output_devices.name] = output_devices
        self.outputs[output_default_output.name] = output_default_output
        self.outputs[output_default_input.name] = output_default_input

        self._main_loop: asyncio.AbstractEventLoop | None = None

    def activate(self):
        self._main_loop = asyncio.get_running_loop()

    def _on_input_dispatch(self, values: list[PortValue]):
        """Query devices on a background thread to avoid blocking the main thread."""
        threading.Thread(target=self._query_devices, daemon=True).start()

    def _dispatch_results(
        self,
        device_arrays: list[np.ndarray],
        default_out: list[np.ndarray],
        default_in: list[np.ndarray],
    ):
        """Dispatch query results (called on main thread via call_soon_threadsafe)."""
        if not self.is_active:
            return
        self._output_devices.dispatch(device_arrays)
        self._output_default_output.dispatch(default_out)
        self._output_default_input.dispatch(default_in)
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _query_devices(self):
        sd = _import_sounddevice()
        _init_worker_thread()
        try:
            with _portaudio_lock:
                devices = sd.query_devices()
                hostapis = sd.query_hostapis()
                default_output = sd.default.device["output"]
                default_input = sd.default.device["input"]
            # logger.info("=== SoundDevice Query Devices ===")
            # logger.info(f"\n{devices}")
            # logger.info("=================================")

            str_dtype = np.dtypes.StringDType()  # type: ignore[attr-defined]
            hostapi_names = [api["name"] for api in hostapis]

            # Parallel columns, one element per device (built in one loop so
            # all arrays are guaranteed the same length)
            columns: list[list] = [[] for _ in self._DEVICE_VALUE_NAMES]
            for i, info in enumerate(devices):
                columns[0].append(i)
                columns[1].append(info["name"])
                columns[2].append(hostapi_names[info["hostapi"]])
                columns[3].append(info["max_input_channels"])
                columns[4].append(info["max_output_channels"])
                columns[5].append(info["default_low_input_latency"])
                columns[6].append(info["default_low_output_latency"])
                columns[7].append(info["default_high_input_latency"])
                columns[8].append(info["default_high_output_latency"])
                columns[9].append(info["default_samplerate"])

            device_arrays = [
                np.array(columns[0], dtype=np.int32),
                np.array(columns[1], dtype=str_dtype),
                np.array(columns[2], dtype=str_dtype),
                np.array(columns[3], dtype=np.int32),
                np.array(columns[4], dtype=np.int32),
                np.array(columns[5], dtype=np.float64),
                np.array(columns[6], dtype=np.float64),
                np.array(columns[7], dtype=np.float64),
                np.array(columns[8], dtype=np.float64),
                np.array(columns[9], dtype=np.float64),
            ]

            def _default_values(default_index) -> list[np.ndarray]:
                """(name, index, hostapi) of a default device; ("", -1, "") if absent."""
                if isinstance(default_index, int) and 0 <= default_index < len(
                    devices
                ):
                    info = devices[default_index]
                    return [
                        np.array([info["name"]], dtype=str_dtype),
                        np.array([default_index], dtype=np.int32),
                        np.array([hostapi_names[info["hostapi"]]], dtype=str_dtype),
                    ]
                return [
                    np.array([""], dtype=str_dtype),
                    np.array([-1], dtype=np.int32),
                    np.array([""], dtype=str_dtype),
                ]

            default_out = _default_values(default_output)
            default_in = _default_values(default_input)

            if self._main_loop is not None:
                self._main_loop.call_soon_threadsafe(
                    self._dispatch_results, device_arrays, default_out, default_in
                )
        except Exception as e:
            logger.error(f"{self.name}: Query error: {e}")
        finally:
            _cleanup_worker_thread()


class AudioDevicePlayFileNode(BaseNode):

    _CMD_LOAD = "load"
    _CMD_UNLOAD = "unload"
    _CMD_PLAY = "play"
    _CMD_STOP = "stop"
    _CMD_PLAYBACK_ENDED = "ended"
    _CMD_SHUTDOWN = "shutdown"

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Play an audio file through the audio device. "
                "Use the load port to acquire the file and audio device, "
                "then play/stop to control playback. "
                "All sounddevice operations run on a dedicated worker thread."
            ),
            category="Audio",
            inputs={
                "load": "Trigger to load file and acquire audio device.",
                "unload": "Trigger to release file and audio device.",
                "play": "Trigger to start playback (must be in stopped state).",
                "stop": "Trigger to stop playback.",
            },
            outputs={
                "state": "Playback state: unloaded/loading/stopped/starting/playing/stopping.",
                "last_error": "Last error message. Dispatched when an error occurs.",
            },
        )

    PROPERTIES = {
        "file_path": {
            "description": "Audio file path relative to the flow's data directory.",
            "data_type": "str",
            "default_value": "",
        },
        "device_id": {
            "description": "Output device. Empty = system default. Number = PortAudio "
            "index. Otherwise a name query, e.g. 'Speakers wasapi': space-separated "
            "substrings matched in order against '<device name>, <host API>' "
            "(case-insensitive). List devices with AudioQueryDevice.",
            "data_type": "str",
            "default_value": "",
        },
        "loop": {
            "description": "Loop playback: seek to start on EOF (gapless).",
            "data_type": "bool",
            "default_value": False,
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "AudioDevicePlayFile")

        # State (read/written only on main thread)
        self._state = "unloaded"

        # Input ports
        input_load = StaticTypeInputPort("load", [PortValueType(DataType.Any)])
        input_load.subscribe(id(self), self._on_load)
        input_unload = StaticTypeInputPort("unload", [PortValueType(DataType.Any)])
        input_unload.subscribe(id(self), self._on_unload)
        input_play = StaticTypeInputPort("play", [PortValueType(DataType.Any)])
        input_play.subscribe(id(self), self._on_play)
        input_stop = StaticTypeInputPort("stop", [PortValueType(DataType.Any)])
        input_stop.subscribe(id(self), self._on_stop)
        self.inputs[input_load.name] = input_load
        self.inputs[input_unload.name] = input_unload
        self.inputs[input_play.name] = input_play
        self.inputs[input_stop.name] = input_stop

        # Output ports
        self._output_state = StaticTypeOutputPort(
            "state", [PortValueType(DataType.String)]
        )
        self._output_last_error = StaticTypeOutputPort(
            "last_error", [PortValueType(DataType.String)]
        )
        self.outputs[self._output_state.name] = self._output_state
        self.outputs[self._output_last_error.name] = self._output_last_error

        # Ring buffer (shared: worker writes during load, audio callback reads)
        self._ring_buffer: RingBuffer | None = None

        # File reader stop signal (shared: worker sets/clears, file reader checks)
        self._stop_file_reader = threading.Event()

        # EOF signal (set by file reader, read by audio callback, cleared by worker).
        # Lets the callback distinguish true end-of-file from a transient underrun.
        self._eof_reached = threading.Event()

        # Worker thread command queue
        self._cmd_queue: queue.Queue = queue.Queue()
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._worker_thread: threading.Thread | None = None

        # sounddevice module (set by worker before any callback can run)
        self._sd = None

        # Underrun stats (written from audio callback, read from worker)
        self._underrun_count = 0
        self._partial_read_count = 0
        self._status_event_count = 0
        self._last_status = ""
        self._stats_lock = threading.Lock()

        # Properties
        self._file_path = ""
        self._device_id: int | str | None = None
        self._loop = False
        self._buffer_duration = 30.0
        self._read_block_size = 8192
        self._stream_blocksize = 2048

    def activate(self):
        self._main_loop = asyncio.get_running_loop()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name=f"{self.name}_worker"
        )
        self._worker_thread.start()

    def set_properties(self, properties: dict[str, PropertyDataType]):
        if "file_path" in properties:
            self._file_path = str(properties["file_path"])
            self.properties["file_path"] = self._file_path
        if "device_id" in properties:
            self._device_id = _parse_device_id(properties["device_id"])
            self.properties["device_id"] = (
                "" if self._device_id is None else str(self._device_id)
            )
        if "loop" in properties:
            # Read by the file reader thread at each EOF; safe to flip mid-play.
            self._loop = bool(properties["loop"])
            self.properties["loop"] = self._loop
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _resolve_file_path(self):
        """Resolve file path using parent_graph. Must be called on main thread."""
        parent_graph = self.parent_graph
        if parent_graph is None or parent_graph.persist is None:
            logger.error(f"{self.name}: parent_graph or persist is None")
            return None
        return parent_graph.persist.data_dir_path / self._file_path

    # ── Main thread: port callbacks (non-blocking) ──

    def _on_load(self, values: list[PortValue]):
        if self._state not in ("unloaded", "stopped"):
            logger.warning(f"{self.name}: Cannot load (state={self._state})")
            return
        if not self._file_path:
            logger.warning(f"{self.name}: Cannot load — file_path not set")
            return
        resolved = self._resolve_file_path()
        if resolved is None:
            return
        self._cmd_queue.put((self._CMD_LOAD, {
            "file_path": resolved, "device_id": self._device_id
        }))

    def _on_unload(self, values: list[PortValue]):
        if self._state != "stopped":
            return
        self._cmd_queue.put((self._CMD_UNLOAD, None))

    def _on_play(self, values: list[PortValue]):
        if self._state != "stopped":
            if self._state not in ("unloaded", "loading"):
                logger.warning(f"{self.name}: Cannot play (state={self._state})")
            return
        self._cmd_queue.put((self._CMD_PLAY, None))

    def _on_stop(self, values: list[PortValue]):
        if self._state != "playing":
            return
        self._cmd_queue.put((self._CMD_STOP, None))

    # ── Main thread: state + error notification (called via call_soon_threadsafe) ──

    def _notify_state(self, new_state: str):
        self._state = new_state
        self._output_state.dispatch([np.array([new_state])])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _notify_state_threadsafe(self, new_state: str):
        if self._main_loop is not None:
            self._main_loop.call_soon_threadsafe(self._notify_state, new_state)

    def _notify_error(self, message: str):
        self._output_last_error.dispatch([np.array([message])])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _notify_error_threadsafe(self, message: str):
        if self._main_loop is not None:
            self._main_loop.call_soon_threadsafe(self._notify_error, message)

    def _report_stats(self):
        with self._stats_lock:
            underruns = self._underrun_count
            partials = self._partial_read_count
            status_events = self._status_event_count
            last_status = self._last_status
            self._underrun_count = 0
            self._partial_read_count = 0
            self._status_event_count = 0
        if underruns > 0 or partials > 0 or status_events > 0:
            msg = (
                f"{self.name}: Buffer health: "
                f"{underruns} underruns, {partials} partial reads, "
                f"{status_events} status events"
            )
            if last_status:
                msg += f" (last: {last_status})"
            logger.warning(msg)

    # ── Audio callback thread ──

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Audio callback — allocation-free read from ring buffer. No file I/O.

        A short read is an underrun (pad with silence and keep playing), NOT
        end-of-file. Playback stops only when the file reader has signalled
        EOF and the buffer is fully drained.
        """
        if status:
            with self._stats_lock:
                self._status_event_count += 1
                self._last_status = str(status)

        rb = self._ring_buffer
        if rb is None:
            outdata[:] = 0
            raise self._sd.CallbackStop

        read_frames = rb.read_into(outdata)
        if read_frames < frames:
            outdata[read_frames:] = 0
            if self._eof_reached.is_set() and rb.available() == 0:
                raise self._sd.CallbackStop
            with self._stats_lock:
                if read_frames == 0:
                    self._underrun_count += 1
                else:
                    self._partial_read_count += 1

    def _on_finished(self):
        """Playback finished — called from audio thread."""
        self._cmd_queue.put((self._CMD_PLAYBACK_ENDED, None))

    # ── File reader thread (started/stopped by worker) ──

    def _file_reader_worker(self, sound_file: sf.SoundFile):
        """Background thread that fills the ring buffer from the audio file.

        With loop enabled, wraps to the file start on EOF so the buffer never
        starves at the loop boundary (gapless). Otherwise signals EOF so the
        audio callback can finish playback once the buffer drains.
        """
        try:
            while not self._stop_file_reader.is_set():
                if self._ring_buffer is None:
                    break
                available_space = self._ring_buffer._size - self._ring_buffer.available()
                if available_space >= self._read_block_size:
                    data = sound_file.read(self._read_block_size, dtype="float32")
                    if len(data) > 0:
                        if data.ndim == 1:
                            data = data.reshape(-1, 1)
                        self._ring_buffer.write(data)
                    if len(data) < self._read_block_size:
                        if self._loop and sound_file.frames > 0:
                            sound_file.seek(0)
                            continue
                        self._eof_reached.set()
                        break
                else:
                    self._stop_file_reader.wait(0.01)
        except Exception as e:
            logger.error(f"{self.name}: File reader error: {e}")
            # Treat a dead reader as EOF so the callback can finish playback
            # instead of underrunning forever.
            self._eof_reached.set()

    # ── Worker thread: all sounddevice + file I/O operations ──

    def _worker_loop(self):
        sd = _import_sounddevice()
        self._sd = sd  # used by the audio callback (no import inside callback)
        _init_worker_thread()

        stream: sd.OutputStream | None = None
        sound_file: sf.SoundFile | None = None
        file_reader: threading.Thread | None = None
        worker_playing = False

        def _stop_file_reader_thread():
            nonlocal file_reader
            self._stop_file_reader.set()
            if file_reader is not None and file_reader.is_alive():
                file_reader.join(timeout=1.0)
            file_reader = None

        def _close_stream():
            nonlocal stream
            if stream is not None:
                with _portaudio_lock:
                    try:
                        stream.abort()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
                stream = None

        def _close_file():
            nonlocal sound_file
            if sound_file is not None:
                sound_file.close()
                sound_file = None

        def _refill_buffer():
            if sound_file is not None and self._ring_buffer is not None:
                sound_file.seek(0)
                self._ring_buffer.reset()
                self._eof_reached.clear()
                buf_frames = int(sound_file.samplerate * self._buffer_duration)
                data = sound_file.read(buf_frames, dtype="float32")
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                self._ring_buffer.write(data)

        try:
            while True:
                cmd, payload = self._cmd_queue.get()

                if cmd == self._CMD_SHUTDOWN:
                    break

                elif cmd == self._CMD_LOAD:
                    # Clean up existing resources
                    _stop_file_reader_thread()
                    _close_stream()
                    _close_file()

                    self._notify_state_threadsafe("loading")

                    try:
                        file_path = payload["file_path"]
                        device_id = payload.get("device_id")

                        sound_file = sf.SoundFile(file_path)

                        # Create ring buffer and fill initial data
                        self._eof_reached.clear()
                        buf_frames = int(sound_file.samplerate * self._buffer_duration)
                        self._ring_buffer = RingBuffer(
                            buf_frames, sound_file.channels, dtype=np.float32
                        )
                        initial_data = sound_file.read(buf_frames, dtype="float32")
                        if initial_data.ndim == 1:
                            initial_data = initial_data.reshape(-1, 1)
                        self._ring_buffer.write(initial_data)

                        # Create stream (the slow part: ~12s on Windows)
                        t0 = time.perf_counter()
                        with _portaudio_lock:
                            stream = sd.OutputStream(
                                samplerate=sound_file.samplerate,
                                channels=sound_file.channels,
                                device=device_id,
                                callback=self._audio_callback,
                                blocksize=self._stream_blocksize,
                                finished_callback=self._on_finished,
                            )
                        logger.info(
                            f"{self.name}: [PERF] load sd.OutputStream(): "
                            f"{time.perf_counter() - t0:.3f}s"
                        )

                        logger.info(
                            f"{self.name}: Loaded {file_path} "
                            f"({sound_file.samplerate}Hz, {sound_file.channels}ch, "
                            f"buffer={self._buffer_duration}s)"
                        )
                        self._notify_state_threadsafe("stopped")

                    except Exception as e:
                        logger.error(f"{self.name}: Load error: {e}")
                        _close_stream()
                        _close_file()
                        self._ring_buffer = None
                        self._notify_error_threadsafe(str(e))
                        self._notify_state_threadsafe("unloaded")

                elif cmd == self._CMD_UNLOAD:
                    _stop_file_reader_thread()
                    _close_stream()
                    _close_file()
                    self._ring_buffer = None
                    self._notify_state_threadsafe("unloaded")

                elif cmd == self._CMD_PLAY:
                    if stream is None or sound_file is None:
                        self._notify_error_threadsafe(
                            "Cannot play: file or stream not loaded"
                        )
                        continue

                    self._notify_state_threadsafe("starting")

                    try:
                        # Start file reader thread
                        self._stop_file_reader.clear()
                        file_reader = threading.Thread(
                            target=self._file_reader_worker,
                            args=(sound_file,),
                            daemon=True,
                        )
                        file_reader.start()

                        with _portaudio_lock:
                            stream.start()
                        worker_playing = True
                        self._notify_state_threadsafe("playing")

                    except Exception as e:
                        logger.error(f"{self.name}: Play error: {e}")
                        _stop_file_reader_thread()
                        self._notify_error_threadsafe(str(e))
                        self._notify_state_threadsafe("stopped")

                elif cmd == self._CMD_STOP:
                    worker_playing = False
                    self._notify_state_threadsafe("stopping")
                    _stop_file_reader_thread()

                    if stream is not None:
                        with _portaudio_lock:
                            try:
                                stream.abort()
                            except Exception:
                                pass

                    self._report_stats()
                    _refill_buffer()
                    self._notify_state_threadsafe("stopped")

                elif cmd == self._CMD_PLAYBACK_ENDED:
                    if not worker_playing:
                        logger.info(f"{self.name}: Playback ended (already stopped)")
                        continue
                    worker_playing = False
                    logger.info(f"{self.name}: Playback finished")
                    self._notify_state_threadsafe("stopping")
                    _stop_file_reader_thread()

                    # Reset PortAudio stream state — required on Windows/WASAPI to
                    # avoid silent replay after natural completion. Mirrors CMD_STOP.
                    if stream is not None:
                        with _portaudio_lock:
                            try:
                                stream.abort()
                            except Exception:
                                pass

                    self._report_stats()
                    _refill_buffer()
                    self._notify_state_threadsafe("stopped")

        except Exception as e:
            logger.error(f"{self.name}: Worker error: {e}")
        finally:
            _stop_file_reader_thread()
            _close_stream()
            _close_file()
            _cleanup_worker_thread()

    def finalize(self):
        self._cmd_queue.put((self._CMD_SHUTDOWN, None))
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self._report_stats()


class AudioDevicePlayStreamNode(BaseNode):

    _CMD_CREATE_STREAM = "create_stream"
    _CMD_PLAY = "play"
    _CMD_STOP = "stop"
    _CMD_SHUTDOWN = "shutdown"

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Real-time audio playback from streaming input. "
                "Incoming frames are buffered in a ring buffer and played through the audio device callback. "
                "Supports buffer_low feedback for demand-driven generation. "
                "All sounddevice operations run on a dedicated worker thread."
            ),
            category="Audio",
            inputs={
                "frames": "Audio frames to play. Shape (batch_size, channels).",
                "play": "Trigger to start audio playback.",
                "stop": "Trigger to stop audio playback.",
            },
            outputs={
                "buffer_low": "Dispatches True when buffer level drops below threshold.",
                "state": "Playback state: idle/stopped/starting/playing/error.",
                "last_error": "Last error message. Dispatched when an error occurs.",
            },
        )

    PROPERTIES = {
        "sample_rate": {
            "description": "Audio sample rate in Hz.",
            "data_type": "i32",
            "default_value": 44100,
        },
        "channels": {
            "description": "Number of audio channels.",
            "data_type": "i32",
            "default_value": 2,
        },
        "device_id": {
            "description": "Output device. Empty = system default. Number = PortAudio "
            "index. Otherwise a name query, e.g. 'Speakers wasapi': space-separated "
            "substrings matched in order against '<device name>, <host API>' "
            "(case-insensitive). List devices with AudioQueryDevice.",
            "data_type": "str",
            "default_value": "",
        },
        "buffer_duration": {
            "description": "Ring buffer duration in seconds.",
            "data_type": "f64",
            "default_value": 1.0,
        },
        "buffer_low_threshold": {
            "description": "Buffer level ratio (0-1) that triggers buffer_low signal.",
            "data_type": "f64",
            "default_value": 0.3,
        },
        "stream_latency": {
            "description": "Sounddevice output latency. Float seconds or 'low'/'high'.",
            "data_type": "str",
            "default_value": "high",
        },
    }

    _BUFFER_LOW_TRUE = np.array([True], dtype=bool)

    def __init__(self, name: str):
        super().__init__(name, "AudioDevicePlayStream")

        sample_rate = 44100
        channels = 2
        buffer_duration = 1.0

        # Input ports
        input_frames = StaticTypeInputPort(
            "frames", [PortValueType(DataType.Float32, (channels,))]
        )
        input_frames.subscribe(id(self), self._on_input_frames)
        input_play = StaticTypeInputPort("play", [PortValueType(DataType.Any)])
        input_play.subscribe(id(self), self._on_input_play)
        input_stop = StaticTypeInputPort("stop", [PortValueType(DataType.Any)])
        input_stop.subscribe(id(self), self._on_input_stop)

        # Output ports
        output_buffer_low = StaticTypeOutputPort(
            "buffer_low", [PortValueType(DataType.Bool)]
        )
        self._output_state = StaticTypeOutputPort(
            "state", [PortValueType(DataType.String)]
        )
        self._output_last_error = StaticTypeOutputPort(
            "last_error", [PortValueType(DataType.String)]
        )

        self._input_frames = input_frames
        self._input_play = input_play
        self._input_stop = input_stop
        self._output_buffer_low = output_buffer_low

        self.inputs[input_frames.name] = input_frames
        self.inputs[input_play.name] = input_play
        self.inputs[input_stop.name] = input_stop
        self.outputs[output_buffer_low.name] = output_buffer_low
        self.outputs[self._output_state.name] = self._output_state
        self.outputs[self._output_last_error.name] = self._output_last_error

        # State (read/written only on main thread)
        self._state = "idle"
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._worker_thread: threading.Thread | None = None
        self._cmd_queue: queue.Queue = queue.Queue()

        # Configuration
        self._sample_rate = sample_rate
        self._channels = channels
        self._buffer_duration = buffer_duration
        self._stream_blocksize = 0
        self._stream_latency: float | str = "high"
        self._device_id: int | str | None = None

        # Buffer low notification
        self._buffer_low_threshold = 0.3
        self._buffer_low_event = threading.Event()

        # Ring buffer (shared: main thread writes frames, audio callback reads)
        buffer_frames = int(sample_rate * buffer_duration)
        self._ring_buffer = RingBuffer(buffer_frames, channels, dtype=np.float32)

        # Underrun stats (written from audio callback, read from main thread)
        self._underrun_count = 0
        self._partial_read_count = 0
        self._status_event_count = 0
        self._last_status = ""
        self._stats_lock = threading.Lock()

    def activate(self):
        self._main_loop = asyncio.get_running_loop()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name=f"{self.name}_worker"
        )
        self._worker_thread.start()
        # Create stream with current settings
        self._cmd_queue.put((self._CMD_CREATE_STREAM, self._stream_params()))

    def _stream_params(self):
        return {
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "device_id": self._device_id,
            "blocksize": self._stream_blocksize,
            "latency": self._stream_latency,
        }

    def set_properties(self, properties: dict[str, PropertyDataType]):
        sample_rate = self._sample_rate
        channels = self._channels
        buffer_duration = self._buffer_duration
        device_id = self._device_id
        buffer_low_threshold = self._buffer_low_threshold

        if "sample_rate" in properties:
            sample_rate = int(properties["sample_rate"])
        if "channels" in properties:
            channels = int(properties["channels"])
        if "buffer_duration" in properties:
            buffer_duration = float(properties["buffer_duration"])
        if "device_id" in properties:
            device_id = _parse_device_id(properties["device_id"])
        if "buffer_low_threshold" in properties:
            buffer_low_threshold = float(properties["buffer_low_threshold"])
        if "stream_latency" in properties:
            val = properties["stream_latency"]
            if isinstance(val, str) and val in ("low", "high"):
                self._stream_latency = val
            else:
                self._stream_latency = float(val)
            self.properties["stream_latency"] = self._stream_latency

        # Replace input port if channels changed
        new_input_frames = self._input_frames
        if channels != self._channels:
            new_input_frames = StaticTypeInputPort(
                "frames", [PortValueType(DataType.Float32, (channels,))]
            )
        if new_input_frames != self._input_frames:
            replace_node_input_port(
                self, self._input_frames, new_input_frames, self._on_input_frames
            )
            self._input_frames = new_input_frames

        # Recreate ring buffer if needed
        need_update_buffer = (
            sample_rate != self._sample_rate
            or channels != self._channels
            or buffer_duration != self._buffer_duration
        )
        if need_update_buffer:
            buffer_frames = int(sample_rate * buffer_duration)
            self._ring_buffer = RingBuffer(buffer_frames, channels, dtype=np.float32)
            logger.info(
                f"{self.name}: Buffer recreated "
                f"({sample_rate}Hz, {channels}ch, {buffer_duration}s = {buffer_frames} frames)"
            )

        # Check if stream needs recreation
        need_recreate_stream = (
            sample_rate != self._sample_rate
            or channels != self._channels
            or device_id != self._device_id
        )

        # Update state
        self._sample_rate = sample_rate
        self._channels = channels
        self._buffer_duration = buffer_duration
        self._device_id = device_id
        self._buffer_low_threshold = buffer_low_threshold

        self.properties["sample_rate"] = sample_rate
        self.properties["channels"] = channels
        self.properties["device_id"] = "" if device_id is None else str(device_id)
        self.properties["buffer_duration"] = buffer_duration
        self.properties["buffer_low_threshold"] = buffer_low_threshold

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

        if need_recreate_stream and self._worker_thread is not None:
            self._cmd_queue.put((self._CMD_CREATE_STREAM, self._stream_params()))

    # ── Main thread: port callbacks (non-blocking) ──

    def _on_input_frames(self, values: list[PortValue]):
        """Write incoming frames to ring buffer (main thread, no blocking)."""
        frames_batch = values[0]
        if frames_batch.ndim == 1:
            frames_batch = frames_batch.reshape(-1, 1)

        written = self._ring_buffer.write(frames_batch)

        if self._buffer_low_event.is_set():
            self._buffer_low_event.clear()

        if written < len(frames_batch):
            logger.warning(
                f"{self.name}: Buffer overflow - wrote {written}/{len(frames_batch)} frames"
            )

    def _on_input_play(self, values: list[PortValue]):
        if self._state in ("playing", "starting"):
            return  # already playing: idempotent, silent (play may be level-driven)
        if self._state == "error":
            # Self-recovery: recreate the stream, then play. The worker handles
            # commands serially, so create -> play is guaranteed in order.
            logger.info(f"{self.name}: Recovering from error state")
            self._buffer_low_event.clear()
            self._cmd_queue.put((self._CMD_CREATE_STREAM, self._stream_params()))
            self._cmd_queue.put((self._CMD_PLAY, None))
            return
        if self._state != "stopped":
            return  # "idle": stream creation in progress; a later play will land
        self._buffer_low_event.clear()
        self._cmd_queue.put((self._CMD_PLAY, None))

    def _on_input_stop(self, values: list[PortValue]):
        if self._state not in ("playing", "starting"):
            return
        self._cmd_queue.put((self._CMD_STOP, None))

    # ── Main thread: state + buffer_low notification ──

    def _notify_state(self, new_state: str):
        self._state = new_state
        self._output_state.dispatch([np.array([new_state])])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _notify_state_threadsafe(self, new_state: str):
        if self._main_loop is not None:
            self._main_loop.call_soon_threadsafe(self._notify_state, new_state)

    def _notify_error(self, message: str):
        self._output_last_error.dispatch([np.array([message])])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _notify_error_threadsafe(self, message: str):
        if self._main_loop is not None:
            self._main_loop.call_soon_threadsafe(self._notify_error, message)

    def _dispatch_buffer_low(self):
        """Dispatch buffer_low signal (called on main thread via call_soon_threadsafe)."""
        self._output_buffer_low.dispatch([self._BUFFER_LOW_TRUE])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()
        self._report_stats()

    def _report_stats(self):
        with self._stats_lock:
            underruns = self._underrun_count
            partials = self._partial_read_count
            status_events = self._status_event_count
            last_status = self._last_status
            self._underrun_count = 0
            self._partial_read_count = 0
            self._status_event_count = 0
        if underruns > 0 or partials > 0 or status_events > 0:
            msg = (
                f"{self.name}: Buffer health: "
                f"{underruns} underruns, {partials} partial reads, "
                f"{status_events} status events"
            )
            if last_status:
                msg += f" (last: {last_status})"
            logger.warning(msg)

    # ── Audio callback thread ──

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Audio callback — allocation-free read from ring buffer."""
        if status:
            with self._stats_lock:
                self._status_event_count += 1
                self._last_status = str(status)

        read_frames = self._ring_buffer.read_into(outdata)

        if read_frames == 0:
            outdata[:] = 0
            with self._stats_lock:
                self._underrun_count += 1
        elif read_frames < frames:
            outdata[read_frames:] = 0
            with self._stats_lock:
                self._partial_read_count += 1

        if not self._buffer_low_event.is_set():
            available = self._ring_buffer.available()
            buffer_level = available / self._ring_buffer._size
            if buffer_level <= self._buffer_low_threshold:
                self._buffer_low_event.set()
                self._main_loop.call_soon_threadsafe(self._dispatch_buffer_low)

    def _on_finished(self):
        """Stream finished — called from audio thread."""
        logger.info(f"{self.name}: Stream finished")
        self._cmd_queue.put((self._CMD_STOP, None))

    # ── Worker thread ──

    def _worker_loop(self):
        sd = _import_sounddevice()
        _init_worker_thread()

        stream: sd.OutputStream | None = None

        try:
            while True:
                cmd, payload = self._cmd_queue.get()

                if cmd == self._CMD_SHUTDOWN:
                    break

                elif cmd == self._CMD_CREATE_STREAM:
                    # Close existing stream
                    if stream is not None:
                        with _portaudio_lock:
                            try:
                                stream.abort()
                            except Exception:
                                pass
                            try:
                                stream.close()
                            except Exception:
                                pass
                        stream = None

                    try:
                        t0 = time.perf_counter()
                        with _portaudio_lock:
                            stream = sd.OutputStream(
                                samplerate=payload["sample_rate"],
                                channels=payload["channels"],
                                device=payload["device_id"],
                                callback=self._audio_callback,
                                blocksize=payload["blocksize"],
                                latency=payload["latency"],
                                finished_callback=self._on_finished,
                            )
                        logger.info(
                            f"{self.name}: [PERF] create sd.OutputStream(): "
                            f"{time.perf_counter() - t0:.3f}s"
                        )
                        self._notify_state_threadsafe("stopped")
                    except Exception as e:
                        logger.error(f"{self.name}: Stream creation error: {e}")
                        stream = None
                        self._notify_error_threadsafe(str(e))
                        self._notify_state_threadsafe("error")

                elif cmd == self._CMD_PLAY:
                    if stream is None:
                        self._notify_error_threadsafe(
                            "Cannot play: stream not created"
                        )
                        self._notify_state_threadsafe("error")
                        continue

                    self._notify_state_threadsafe("starting")
                    try:
                        with _portaudio_lock:
                            stream.start()
                        logger.info(
                            f"{self.name}: Playback started "
                            f"({self._sample_rate}Hz, {self._channels}ch, "
                            f"device={self._device_id})"
                        )
                        self._notify_state_threadsafe("playing")
                    except Exception as e:
                        logger.error(f"{self.name}: Failed to start: {e}")
                        self._notify_error_threadsafe(str(e))
                        self._notify_state_threadsafe("error")

                elif cmd == self._CMD_STOP:
                    if stream is not None:
                        with _portaudio_lock:
                            try:
                                stream.abort()
                            except Exception:
                                pass
                    self._ring_buffer.reset()
                    self._report_stats()
                    self._notify_state_threadsafe("stopped")

        except Exception as e:
            logger.error(f"{self.name}: Worker error: {e}")
        finally:
            if stream is not None:
                with _portaudio_lock:
                    try:
                        stream.abort()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
            _cleanup_worker_thread()

    def finalize(self):
        self._cmd_queue.put((self._CMD_SHUTDOWN, None))
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self._report_stats()
