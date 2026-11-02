import numpy as np
import soundfile as sf
import logging
import threading
import time
from enum import Enum

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


class AudioFileStreamStatus(Enum):
    NOT_LOADED = 0
    LOADING = 1
    LOADED = 2


class AudioFileStreamNode(BaseNode):

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Stream audio frames from a file on demand. "
                "Reads frames when readout_count is received. "
                "Outputs loading status, audio info, and EOF detection."
            ),
            category="Audio",
            inputs={
                "readout_count": "Number of frames to read per dispatch.",
                "seek": "Seek to frame position in the audio file.",
            },
            outputs={
                "frames": "Audio frames output. Always 2D (count, channels).",
                "loading_status": "File loading state: 0=NOT_LOADED, 1=LOADING, 2=LOADED.",
                "audio_info": "File metadata: [channels, sample_rate, total_frames].",
                "eof": "True, dispatched once when end of file is reached (loop=False only).",
            },
        )

    PROPERTIES = {
        "file_path": {
            "description": "Audio file path relative to the flow's data directory.",
            "data_type": "str",
            "default_value": "",
        },
        "channels": {
            "description": "Expected channel count. Must match the audio file.",
            "data_type": "i32",
            "default_value": 2,
        },
        "loop": {
            "description": "Loop playback: seek to start and continue reading on EOF (gapless).",
            "data_type": "bool",
            "default_value": False,
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "AudioFileStream")

        channels = 2

        input_readout_count = StaticTypeInputPort(
            "readout_count",
            [PortValueType(DataType.Int32)],
        )
        input_readout_count.subscribe(id(self), self._on_input_readout_count)
        input_seek = StaticTypeInputPort(
            "seek",
            [PortValueType(DataType.Int32)],
        )
        input_seek.subscribe(id(self), self._on_input_seek)

        output_frames = StaticTypeOutputPort(
            "frames",
            [PortValueType(DataType.Float32, (channels,))],
        )
        output_loading_status = StaticTypeOutputPort(
            "loading_status",
            [PortValueType(DataType.Uint8)],
        )

        output_audio_info = StaticTypeOutputPort(
            "audio_info",
            [
                PortValueType(DataType.Int32),
                PortValueType(DataType.Int32),
                PortValueType(DataType.Int32),
            ],
            value_name_list=["channels", "sample_rate", "total_frames"],
        )

        output_eof = StaticTypeOutputPort(
            "eof",
            [PortValueType(DataType.Bool)],
        )

        self._input_readout_count = input_readout_count
        self._input_seek = input_seek
        self._output_frames = output_frames
        self._output_loading_status = output_loading_status
        self._output_audio_info = output_audio_info
        self._output_eof = output_eof

        self._file_path: str = ""
        self._sound_file: sf.SoundFile | None = None
        self._channels = channels
        self._channels_mismatch = False  # True if file channels != configured channels
        self._loop = False
        self._eof_reached = False  # one-shot guard for the eof output (loop=False only)

        self.inputs[input_readout_count.name] = input_readout_count
        self.inputs[input_seek.name] = input_seek

        self.outputs[output_loading_status.name] = output_loading_status
        self.outputs[output_audio_info.name] = output_audio_info
        self.outputs[output_frames.name] = output_frames
        self.outputs[output_eof.name] = output_eof

        # Initial status dispatch (with batch dimension)
        output_loading_status.dispatch(
            [np.array([AudioFileStreamStatus.NOT_LOADED.value], dtype=np.uint8)]
        )

    def set_properties(self, properties: dict[str, PropertyDataType]):
        file_path = self._file_path

        if "file_path" in properties:
            file_path = str(properties["file_path"])

        channels = self._channels
        if "channels" in properties:
            channels = int(properties["channels"])

        if "loop" in properties:
            new_loop = bool(properties["loop"])
            if new_loop and not self._loop:
                self._eof_reached = False
            self._loop = new_loop

        need_update_file = file_path != self._file_path

        # Handle channels change - recreate output_frames port if needed
        new_output_frames = self._output_frames
        if channels != self._channels:
            new_output_frames = StaticTypeOutputPort(
                "frames",
                [PortValueType(DataType.Float32, (channels,))],
            )

        if new_output_frames != self._output_frames:
            replace_node_output_port(self, self._output_frames, new_output_frames)
            self._output_frames = new_output_frames

        # Update state
        self._file_path = file_path
        self._channels = channels

        # Update channels mismatch state if file is loaded
        if self._sound_file is not None:
            old_mismatch = self._channels_mismatch
            self._channels_mismatch = self._sound_file.channels != channels
            if self._channels_mismatch and not old_mismatch:
                logger.warning(
                    f"{self.name}: Channel mismatch - file has {self._sound_file.channels} channels, "
                    f"but node configured for {channels} channels. Frames will not be dispatched."
                )
            elif not self._channels_mismatch and old_mismatch:
                logger.info(f"{self.name}: Channel mismatch resolved.")
        self.properties["file_path"] = file_path
        self.properties["channels"] = channels
        self.properties["loop"] = self._loop

        # Notify graph (REQUIRED)
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

        # Load audio file if path changed
        if need_update_file:
            if file_path == "":
                # Empty path: unload file
                if self._sound_file is not None:
                    self._sound_file.close()
                    self._sound_file = None
                self._channels_mismatch = False
                # Dispatch NOT_LOADED status (with batch dimension)
                self._output_loading_status.dispatch(
                    [np.array([AudioFileStreamStatus.NOT_LOADED.value], dtype=np.uint8)]
                )
            else:
                # Non-empty path: load file
                self._load_audio_file(file_path)

    def _load_audio_file(self, file_path: str):
        """Load audio file and update status"""
        try:
            # Dispatch LOADING status (with batch dimension)
            self._output_loading_status.dispatch(
                [np.array([AudioFileStreamStatus.LOADING.value], dtype=np.uint8)]
            )

            # Close previous file if exists
            if self._sound_file is not None:
                self._sound_file.close()

            # Resolve file path using parent_graph.persist.data_dir_path
            parent_graph = self.parent_graph
            if parent_graph is None:
                raise RuntimeError(f"{self.name}: parent_graph is None")

            persist = parent_graph.persist
            if persist is None:
                raise RuntimeError(f"{self.name}: persist is None")

            # Get data directory and resolve absolute path
            data_dir_path = persist.data_dir_path
            resolved_file_path = data_dir_path / file_path

            # Open new sound file and measure time
            start_time = time.perf_counter()
            self._sound_file = sf.SoundFile(resolved_file_path)
            load_time = time.perf_counter() - start_time

            # Dispatch LOADED status (with batch dimension)
            self._output_loading_status.dispatch(
                [np.array([AudioFileStreamStatus.LOADED.value], dtype=np.uint8)]
            )

            # Dispatch audio info (channels, sample_rate, frames)
            # Each value wrapped in array for batch dimension
            self._output_audio_info.dispatch(
                [
                    np.array([self._sound_file.channels], dtype=np.int32),  # channels
                    np.array(
                        [self._sound_file.samplerate], dtype=np.int32
                    ),  # sample_rate
                    np.array([self._sound_file.frames], dtype=np.int32),  # frames
                ]
            )

            # Fresh file: reset EOF state
            self._eof_reached = False

            # Check channel mismatch
            self._channels_mismatch = self._sound_file.channels != self._channels
            if self._channels_mismatch:
                logger.warning(
                    f"{self.name}: Channel mismatch - file has {self._sound_file.channels} channels, "
                    f"but node configured for {self._channels} channels. Frames will not be dispatched."
                )

            logger.info(
                f"{self.name}: Loaded audio file: {resolved_file_path}, "
                f"channels={self._sound_file.channels}, "
                f"samplerate={self._sound_file.samplerate}, "
                f"frames={self._sound_file.frames}, "
                f"load_time={load_time:.4f}s"
            )

        except Exception as e:
            logger.error(f"{self.name}: Failed to load audio file {file_path}: {e}")
            self._sound_file = None
            self._channels_mismatch = False
            # Dispatch NOT_LOADED status on error (with batch dimension)
            self._output_loading_status.dispatch(
                [np.array([AudioFileStreamStatus.NOT_LOADED.value], dtype=np.uint8)]
            )

        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _on_input_readout_count(self, values: list[PortValue]):
        """Read audio frames based on readout_count values

        Dispatches separately for each count in readout_count_array.
        Buffer is reused when shape matches previous iteration.
        Also dispatches eof status for each read.
        """
        readout_count_array = values[0]

        if self._sound_file is None or self._channels_mismatch:
            return

        try:
            for count in readout_count_array:
                count = int(count)
                if count <= 0:
                    continue

                # Read frames (soundfile returns 1D for mono, 2D for stereo+)
                frames_data = self._sound_file.read(count, dtype="float32")

                # Always output 2D (count, channels)
                if frames_data.ndim == 1:
                    frames_data = frames_data.reshape(-1, 1)

                actual_count = len(frames_data)

                if self._loop and actual_count < count:
                    # Gapless loop: wrap to the start and fill the remainder of
                    # this batch (tiny files may need multiple wraps).
                    parts = [frames_data] if actual_count > 0 else []
                    remaining = count - actual_count
                    while remaining > 0:
                        self._sound_file.seek(0)
                        chunk = self._sound_file.read(remaining, dtype="float32")
                        if chunk.ndim == 1:
                            chunk = chunk.reshape(-1, 1)
                        if len(chunk) == 0:
                            break  # empty file: avoid infinite loop
                        parts.append(chunk)
                        remaining -= len(chunk)
                    if parts:
                        frames_data = (
                            parts[0]
                            if len(parts) == 1
                            else np.concatenate(parts, axis=0)
                        )
                    actual_count = len(frames_data)

                # Dispatch frames (frames_data is already the valid portion)
                if actual_count > 0:
                    self._output_frames.dispatch([frames_data])

                # One-shot EOF notification (loop=False only)
                if not self._loop and actual_count < count and not self._eof_reached:
                    self._eof_reached = True
                    self._output_eof.dispatch([np.array([True])])

        except Exception as e:
            logger.error(f"{self.name}: Error reading audio frames: {e}")

    def _on_input_seek(self, values: list[PortValue]):
        """Seek to a specific frame position in the audio file

        seek_position is the frame index to seek to (Int32)
        """
        seek_position_array = values[0]  # Shape: (batch_size,)

        # Check if file is loaded
        if self._sound_file is None:
            return

        try:
            # Use the last value in the batch (most recent seek request)
            seek_position = int(seek_position_array[-1])

            # Clamp to valid range
            if seek_position < 0:
                seek_position = 0
            elif seek_position >= self._sound_file.frames:
                seek_position = self._sound_file.frames - 1

            # Seek to position
            self._sound_file.seek(seek_position)
            self._eof_reached = False

        except Exception as e:
            logger.error(f"{self.name}: Error seeking: {e}")

    def finalize(self):
        """Clean up resources when node is destroyed"""
        if self._sound_file is not None:
            self._sound_file.close()
            self._sound_file = None
