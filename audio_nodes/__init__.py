import logging
from nonoha_core.node_registry import NodeRegistry

logger = logging.getLogger(__name__)


def register_nodes(registry: NodeRegistry):
    """Register audio device nodes with the engine."""
    from .audio_device_nodes import (
        AudioDevicePlayStreamNode,
        AudioQueryDeviceNode,
        AudioDevicePlayFileNode,
    )
    from .audio_file_stream_nodes import AudioFileStreamNode
    from .audio_gain_node import AudioGainNode

    logger.info("Registering audio-nodes plugin")
    registry.register_node("AudioQueryDevice", AudioQueryDeviceNode)
    registry.register_node("AudioDevicePlayFile", AudioDevicePlayFileNode)
    registry.register_node("AudioDevicePlayStream", AudioDevicePlayStreamNode)
    registry.register_node("AudioFileStream", AudioFileStreamNode)
    registry.register_node("AudioGain", AudioGainNode)


async def finalize():
    """Optional cleanup function."""
    logger.info("Finalizing audio-nodes plugin")