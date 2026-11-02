# audio-nodes

A nonoha-engine plugin for audio playback and streaming, providing audio device control and file streaming capabilities within the nonoha-engine visual programming environment.

## Overview

This plugin provides specialized nodes for audio playback using sounddevice and soundfile libraries. It enables real-time audio streaming, file playback with preloading, and audio device management.

## Features

### Nodes Provided

- **AudioQueryDevice**: Queries available audio devices
  - Logs all detected audio devices with details
  - Dispatches the device list as named parallel arrays (`devices` output,
    one element per device): index, name, hostapi, max_input_channels,
    max_output_channels, default low/high input/output latencies,
    default_samplerate — a complete mirror of PortAudio's per-device info.
    Output devices are the entries with `max_output_channels > 0`
  - Dispatches the default output and input devices (`default_output` /
    `default_input` outputs: name, index, hostapi)
  - Useful for finding device IDs / name queries for other audio nodes

- **AudioDevicePlayFile**: Plays audio from a file with ring buffer
  - Automatic file preloading for low-latency playback
  - Background file reading with ring buffer (30 second buffer by default)
  - Gapless loop playback (`loop` property)
  - Underrun-tolerant playback (transient buffer starvation pads with
    silence and is counted, instead of terminating playback)
  - Configurable device ID selection
  - Play/Stop control via input ports

- **AudioDevicePlayStream**: Real-time streaming audio playback
  - Accepts streaming audio frames via input port
  - Thread-safe ring buffer for smooth playback
  - Buffer low notification output for flow control
  - Configurable sample rate, channels, and buffer duration
  - Play/Stop control via input ports (play is idempotent and may be
    level-driven; play from the error state recreates the stream)
  - Error messages (device not found, rate mismatch, ...) are dispatched
    on the `last_error` output

- **AudioFileStream**: Streams audio data from a file
  - On-demand frame reading with configurable readout count
  - Gapless loop playback (`loop` property): wraps to the file start on EOF
  - Seek support for random access
  - EOF detection output (dispatched once at EOF when loop is disabled)
  - Audio info output (channels, sample rate, total frames)
  - Loading status output (NOT_LOADED, LOADING, LOADED)
  - Buffer reuse optimization for efficient memory usage

- **AudioGain**: Gain stage with constant-slew glide (no clicks)
  - `gain` input port sets a target; the node glides toward it per sample
  - `ramp_seconds` property sets the slew rate (seconds per gain
    difference of 1.0): short (default 0.05) = smoothed volume control,
    long (e.g. 3.0) = fade in/out from a single target change
  - Gain starts at 0.0, so playback always fades in from silence
  - Chain two instances for independent fade and volume stages:
    `AudioFileStream → AudioGain(fade) → AudioGain(volume) → AudioDevicePlayStream`

## Installation

### From the Repository

```sh
pip install git+https://github.com/viestyle-dev/audio-nodes.git
```

Requires Python 3.11+. `nonoha-core` is provided by the host runtime (the
nonoha engine), so it is not a runtime dependency of this package.

### As a Dependency in nonoha-engine Workspace

1. Add `audio-nodes` to your workspace's `pyproject.toml`:

```toml
[project]
dependencies = [
    "audio-nodes",
    # other dependencies...
]

[tool.uv.sources]
audio-nodes = { path = "./plugins/audio-nodes", editable = false }
```

2. Install dependencies using uv:

```bash
uv sync
```

### As a Built-in Plugin

To include audio-nodes as a built-in plugin in your nonoha-engine instance:

1. Import the plugin in your `src/main.py`:

```python
import audio_nodes
```

2. Add to RuntimeConfig:

```python
from nonoha_engine.runtime_config import RuntimeConfig

config = RuntimeConfig(
    builtin_plugins=[
        audio_nodes,
        # other built-in plugins...
    ]
)
```

## Usage

Once installed, the audio nodes will appear in the nonoha-engine node palette. You can:

1. Drag and drop audio nodes into your visual programming workspace
2. Configure node properties (file path, device ID, sample rate, channels)
3. Connect data flows between audio nodes and other processing nodes
4. Run the graph to enable audio playback

### Example: File Playback

1. Add an `AudioDevicePlayFile` node
2. Set the `file_path` property to your audio file (relative to graph data directory)
3. Connect a trigger to the `play` input port
4. Audio will be preloaded automatically and play with minimal latency

### Example: Streaming Playback

1. Add an `AudioFileStream` node to read audio data
2. Add an `AudioDevicePlayStream` node for playback
3. Connect the `frames` output to the `frames` input
4. Use `buffer_low` output to trigger more frame reads

Insert `AudioGain` nodes between them for volume control and fades. Note
that gain applied upstream of the output buffer becomes audible only after
the buffered audio drains (≈ `buffer_duration` seconds of latency).

## Device Selection (`device_id`)

The `device_id` property of the playback nodes accepts, mirroring
sounddevice's `device` parameter:

- empty string: system default output device
- a number: PortAudio device index (**not stable** across reboots or device
  changes — prefer name queries for permanent installations)
- anything else: a sounddevice name query — space-separated substrings
  matched case-insensitively, in order, against
  `"<device name>, <host API name>"`

Platform notes:

- **Windows**: every physical device is enumerated once per host API
  (MME / DirectSound / WASAPI / WDM-KS), so a bare device name often matches
  multiple entries. Append the host API to disambiguate, e.g.
  `"Speakers wasapi"`. MME truncates device names to ~31 characters.
- **macOS**: Core Audio only; names are stable and unique.
- **Linux / Raspberry Pi**: ALSA enumerates real and virtual devices;
  `"default"` is usually what you want.

Use `AudioQueryDevice` to list the available devices and their host APIs.

## Device Sharing and Startup Time

All playback nodes open their own PortAudio output stream in **shared mode**
(the OS mixes multiple streams), so any number of audio nodes — in one graph
or across graphs in the same engine process — can output to the same device
concurrently without conflicts. Exclusive-mode (WASAPI exclusive) and ASIO
single-client setups are not supported.

Stream creation is slow on Windows (~12 s per stream) and is serialized
process-wide, so expect roughly `12 s × (number of playback nodes)` of
startup time before all nodes reach the `stopped` state.

## Requirements

- Python >= 3.11
- nonoha-core (nonoha-engine core library)
- sounddevice (audio I/O)
- soundfile (audio file reading)
- numpy (array processing)

### Platform-Specific Requirements

**Linux (including Raspberry Pi):**

On Linux systems, you need to install PortAudio before using the audio nodes:

```bash
sudo apt install libportaudio2
```

This provides the necessary audio backend for sounddevice to function properly.

## Development

To contribute or modify the plugin:

```bash
uv sync        # also installs the dev dependency group (pytest, nonoha-core)
uv run pytest
```

On Windows, run pytest through the venv's Python instead of `uv run`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## API Reference

### sounddevice Documentation

<https://python-sounddevice.readthedocs.io/>

### soundfile Documentation

<https://python-soundfile.readthedocs.io/>

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Pull requests require signing the [nonoha CLA](CLA.md) (a one-time
comment on your first PR, shared across the nonoha repositories).

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
