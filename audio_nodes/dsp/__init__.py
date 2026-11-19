"""Block-based synthesis and processing nodes (numpy + scipy only).

These nodes touch no audio device and no file, so they run wherever the
graph runs; the host decides how the resulting frames are played.
"""

from .clock import AudioClockNode
from .generators import NoiseNode, OscillatorNode
from .processors import EnvelopeNode, MixerNode, ModFilterNode, PannerNode, SoftClipNode
from .reverb import ReverbNode
from .voices import PolyVoiceNode

NODES = {
    "AudioClock": AudioClockNode,
    "Oscillator": OscillatorNode,
    "Noise": NoiseNode,
    "ModFilter": ModFilterNode,
    "Panner": PannerNode,
    "Mixer": MixerNode,
    "Envelope": EnvelopeNode,
    "SoftClip": SoftClipNode,
    "PolyVoice": PolyVoiceNode,
    "Reverb": ReverbNode,
}
