"""audio-nodes: audio playback, streaming and block-based synthesis.

Two registration functions:

- ``register_nodes``: every node, including the audio-device and file nodes.
  This is the entry point plugin hosts call.
- ``register_portable_nodes``: synthesis and processing only (no audio
  device, no files), for runtimes where the host app owns audio output and
  plays the frames a graph produces.

Node modules are imported inside the functions, so importing the package
never loads the audio-device library.
"""

import logging

from nonoha_core.node_registry import NodeRegistry

logger = logging.getLogger(__name__)


def register_portable_nodes(registry: NodeRegistry):
    """Register the synthesis and processing nodes (no device or file I/O)."""
    from .audio_gain_node import AudioGainNode
    from .dsp import NODES

    registry.register_node("AudioGain", AudioGainNode)
    for type_name, factory in NODES.items():
        registry.register_node(type_name, factory)


def register_nodes(registry: NodeRegistry):
    """Register all audio nodes, including device playback and file streaming."""
    from .audio_device_nodes import (
        AudioDevicePlayStreamNode,
        AudioQueryDeviceNode,
        AudioDevicePlayFileNode,
    )
    from .audio_file_stream_nodes import AudioFileStreamNode

    logger.info("Registering audio-nodes plugin")
    register_portable_nodes(registry)
    registry.register_node("AudioQueryDevice", AudioQueryDeviceNode)
    registry.register_node("AudioDevicePlayFile", AudioDevicePlayFileNode)
    registry.register_node("AudioDevicePlayStream", AudioDevicePlayStreamNode)
    registry.register_node("AudioFileStream", AudioFileStreamNode)


async def finalize():
    """Optional cleanup function."""
    logger.info("Finalizing audio-nodes plugin")
