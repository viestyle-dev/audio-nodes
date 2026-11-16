"""Tests for the block-based synthesis nodes (audio_nodes.dsp)."""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from nonoha_core._testing import assert_all_ports_documented
from nonoha_core.node_registry import NodeRegistry

import audio_nodes
from audio_nodes.dsp import NODES
from audio_nodes.dsp.clock import AudioClockNode
from audio_nodes.dsp.generators import NoiseNode, OscillatorNode
from audio_nodes.dsp.processors import EnvelopeNode, MixerNode, ModFilterNode, PannerNode, SoftClipNode
from audio_nodes.dsp.reverb import ReverbNode
from audio_nodes.dsp.voices import PolyVoiceNode

SR = 44100
BLOCK = 2205
ANY = np.array([True])


def clock_block(n=BLOCK, start=0):
    return [np.arange(start, start + n, dtype=np.float64) / SR]


def collect(port):
    got = []
    port.subscribe(id(got), lambda v: got.append(v[0]))
    return got


# --- registration -----------------------------------------------------------------

def test_portable_registration_excludes_device_nodes():
    registry = NodeRegistry()
    audio_nodes.register_portable_nodes(registry)
    names = set(registry.node_definitions)
    assert set(NODES) | {"AudioGain"} == names
    sub = NodeRegistry()
    for name, factory in NODES.items():
        sub.register_node(name, factory)
    assert_all_ports_documented(sub)


def test_portable_registration_never_imports_sounddevice():
    code = (
        "import sys, audio_nodes\n"
        "from nonoha_core.node_registry import NodeRegistry\n"
        "audio_nodes.register_portable_nodes(NodeRegistry())\n"
        "assert 'sounddevice' not in sys.modules, 'sounddevice was imported'\n"
    )
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- AudioClock -----------------------------------------------------------------------

def test_clock_prefill_then_demand():
    clock = AudioClockNode("c")
    clock.set_properties({"block_size": 100, "prefill_blocks": 3, "blocks_per_demand": 2})
    times, primed = collect(clock.outputs["time"]), collect(clock.outputs["primed"])

    clock.inputs["demand"].dispatch([ANY])
    assert times == []  # not started yet

    clock.inputs["start"].dispatch([ANY])
    assert len(times[0]) == 300 and primed and bool(primed[0][0])
    clock.inputs["demand"].dispatch([ANY])
    assert len(times[1]) == 200
    np.testing.assert_allclose(times[1][0], 300 / SR)  # continuous time

    clock.inputs["stop"].dispatch([ANY])
    clock.inputs["demand"].dispatch([ANY])
    assert len(times) == 2


# --- Oscillator / Noise --------------------------------------------------------------

def test_oscillator_frequency_and_phase_continuity():
    osc = OscillatorNode("o")
    osc.set_properties({"frequency": 441.0, "amplitude": 1.0, "glide": 0.0})
    out = collect(osc.outputs["frames"])
    osc.inputs["time"].dispatch(clock_block(SR))
    y = out[0][:, 0]
    spectrum = np.abs(np.fft.rfft(y))
    assert np.argmax(spectrum) == 441  # 1 Hz bins over one second

    # Change pitch between blocks: the waveform must not jump.
    osc.inputs["frequency"].dispatch([np.array([660.0], np.float32)])
    osc.inputs["time"].dispatch(clock_block(BLOCK))
    joined = np.concatenate([out[0][-200:, 0], out[1][:200, 0]])
    max_step = 2 * np.pi * 660 / SR
    assert np.max(np.abs(np.diff(joined))) <= max_step * 1.01


@pytest.mark.parametrize("wave", ["saw", "square", "triangle"])
def test_oscillator_waveforms_bounded(wave):
    osc = OscillatorNode("o")
    osc.set_properties({"waveform": wave, "frequency": 3000.0, "amplitude": 0.5})
    out = collect(osc.outputs["frames"])
    osc.inputs["time"].dispatch(clock_block())
    assert out[0].shape == (BLOCK, 1)
    assert np.all(np.isfinite(out[0])) and np.max(np.abs(out[0])) <= 0.5 * 1.1


def test_noise_colors_have_falling_spectrum():
    def band_ratio(color):
        node = NoiseNode("n")
        node.set_properties({"color": color, "amplitude": 1.0, "seed": 7})
        out = collect(node.outputs["frames"])
        for _ in range(20):
            node.inputs["time"].dispatch(clock_block(4096))
        y = np.concatenate(out)[:, 0]
        power = np.abs(np.fft.rfft(y)) ** 2
        freqs = np.fft.rfftfreq(len(y), 1 / SR)
        low = power[(freqs > 200) & (freqs < 400)].mean()
        high = power[(freqs > 6400) & (freqs < 12800)].mean()
        return 10 * np.log10(low / high)

    assert abs(band_ratio("white")) < 2.0
    assert 10.0 < band_ratio("pink") < 20.0  # ~3 dB/oct over 5 octaves
    assert band_ratio("brown") > band_ratio("pink")


# --- ModFilter / Panner / SoftClip --------------------------------------------------

def sine(freq, n=BLOCK, amp=1.0):
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(-1, 1)


def test_lowpass_attenuates_and_sweeps_stay_stable():
    filt = ModFilterNode("f")
    filt.set_properties({"cutoff": 500.0, "glide": 0.0})
    out = collect(filt.outputs["frames"])
    for _ in range(4):
        filt.inputs["frames"].dispatch([sine(8000)])
    rms = np.sqrt(np.mean(out[-1] ** 2))
    assert rms < 0.707 / 20  # > 26 dB down

    rng = np.random.default_rng(3)
    for _ in range(50):
        filt.inputs["cutoff"].dispatch([np.array([rng.uniform(30, 18000)], np.float32)])
        filt.inputs["q"].dispatch([np.array([rng.uniform(0.5, 12)], np.float32)])
        filt.inputs["frames"].dispatch([rng.standard_normal((BLOCK, 1)).astype(np.float32)])
    assert np.all(np.isfinite(out[-1])) and np.max(np.abs(out[-1])) < 50


def test_filter_channels_change_rebuilds_ports():
    filt = ModFilterNode("f")
    filt.set_properties({"channels": 2})
    assert filt.inputs["frames"].value_type_list[0].shape == (2,)
    out = collect(filt.outputs["frames"])
    filt.inputs["frames"].dispatch([np.zeros((10, 2), np.float32)])
    assert out[0].shape == (10, 2)


def test_panner_equal_power():
    pan = PannerNode("p")
    pan.set_properties({"glide": 0.0})
    out = collect(pan.outputs["frames"])
    pan.inputs["frames"].dispatch([np.ones((4, 1), np.float32)])
    np.testing.assert_allclose(out[0][0], [np.sqrt(0.5)] * 2, rtol=1e-5)
    pan.inputs["pan"].dispatch([np.array([-1.0], np.float32)])
    pan.inputs["frames"].dispatch([np.ones((4, 1), np.float32)])
    np.testing.assert_allclose(out[1][-1], [1.0, 0.0], atol=1e-6)


def test_soft_clip_bounds():
    clip = SoftClipNode("s")
    clip.set_properties({"drive": 2.0})
    out = collect(clip.outputs["frames"])
    clip.inputs["frames"].dispatch([np.array([[-10, 10], [0, 0.01]], np.float32)])
    assert np.max(np.abs(out[0])) <= 1.0 + 1e-6 and out[0][1, 0] == 0


# --- Mixer ------------------------------------------------------------------------------

def test_mixer_waits_for_all_inputs_and_ignores_order():
    a = np.full((5, 2), 0.25, np.float32)
    b = np.full((5, 2), 0.5, np.float32)
    results = []
    for order in ((0, 1), (1, 0)):
        mixer = MixerNode("m")
        mixer.set_properties({"inputs": 2})
        out = collect(mixer.outputs["frames"])
        mixer.inputs[f"in{order[0]}"].dispatch([(a, b)[order[0]]])
        assert out == []  # still waiting for the other input
        mixer.inputs[f"in{order[1]}"].dispatch([(a, b)[order[1]]])
        results.append(out[0])
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_allclose(results[0], 0.75)


def test_mixer_input_count_change():
    mixer = MixerNode("m")
    mixer.set_properties({"inputs": 3, "channels": 1})
    assert set(n for n in mixer.inputs if n.startswith("in")) == {"in0", "in1", "in2"}
    assert mixer.inputs["in0"].value_type_list[0].shape == (1,)


# --- Envelope -------------------------------------------------------------------------

def test_envelope_adsr_shape():
    env = EnvelopeNode("e")
    env.set_properties({"attack": 0.01, "decay": 0.02, "sustain": 0.5, "release": 0.05})
    out = collect(env.outputs["frames"])
    ones = np.ones((BLOCK, 1), np.float32)

    env.inputs["frames"].dispatch([ones])
    assert np.max(out[0]) == 0  # gate closed: silence

    env.inputs["gate"].dispatch([np.array([1.0], np.float32)])
    env.inputs["frames"].dispatch([ones])
    y = out[1][:, 0]
    peak_at = int(np.argmax(y))
    assert y[peak_at] == pytest.approx(1.0) and peak_at == pytest.approx(0.01 * SR, abs=2)
    assert y[-1] == pytest.approx(0.5)  # reached sustain

    env.inputs["gate"].dispatch([np.array([0.0], np.float32)])
    env.inputs["frames"].dispatch([ones])
    y = out[2][:, 0]
    assert y[0] < 0.5 and y[-1] == 0.0


# --- PolyVoice / Reverb ------------------------------------------------------------------

def test_polyvoice_plays_decays_and_steals():
    pv = PolyVoiceNode("v")
    pv.set_properties({"decay": 0.2, "max_voices": 2})
    out, active = collect(pv.outputs["frames"]), collect(pv.outputs["active"])
    notes = np.array([[[440, 1, -1], [660, 1, 1], [880, 1, 0]]], np.float32)
    pv.inputs["notes"].dispatch([notes])
    pv.inputs["time"].dispatch(clock_block())
    assert int(active[0][0]) == 2  # third note stole the oldest
    first = out[0]
    assert first.shape == (BLOCK, 2) and np.max(np.abs(first)) > 0.01

    for _ in range(40):  # 2 s of blocks: well past 6 x decay
        pv.inputs["time"].dispatch(clock_block())
    assert int(active[-1][0]) == 0
    assert np.max(np.abs(out[-1])) == 0


def test_reverb_tail_and_dry_passthrough():
    rv = ReverbNode("r")
    out = collect(rv.outputs["frames"])
    impulse = np.zeros((BLOCK, 2), np.float32)
    impulse[0] = 1.0
    rv.inputs["frames"].dispatch([impulse])
    rv.inputs["frames"].dispatch([np.zeros((BLOCK, 2), np.float32)])
    assert np.sqrt(np.mean(out[1] ** 2)) > 1e-4  # tail continues after the impulse

    dry = ReverbNode("d")
    dry.set_properties({"mix": 0.0})
    out = collect(dry.outputs["frames"])
    x = np.random.default_rng(0).standard_normal((BLOCK, 2)).astype(np.float32)
    dry.inputs["frames"].dispatch([x])
    np.testing.assert_allclose(out[0], x, atol=1e-6)


# --- block algorithms match the plain per-sample definitions --------------------------------


def _polyvoice_reference(pv: PolyVoiceNode, n: int) -> np.ndarray:
    """The modal voice formula, evaluated directly (sin and exp per sample)."""
    t = (pv._age[:, None] + np.arange(n)[None, :]) / SR
    voice = np.zeros_like(t)
    for ratio, weight, tau in zip(pv._ratios, pv._weights, pv._tau):
        freqs = pv._freq * ratio
        partial = np.sin(2 * np.pi * freqs[:, None] * t) * np.exp(-t / tau)
        voice += partial * (weight * (freqs < pv._nyquist_guard))[:, None]
    voice *= 1 - np.exp(-t / 0.003)
    voice *= (pv._vel * pv._gain)[:, None]
    theta = (pv._pan + 1) * np.pi / 4
    return voice.T @ np.stack([np.cos(theta), np.sin(theta)], axis=1)


@pytest.mark.parametrize("voice", ["pluck", "bell", "chime"])
def test_polyvoice_matches_the_direct_formula(voice):
    pv = PolyVoiceNode("v")
    pv.set_properties({"voice": voice, "decay": 1.5, "max_voices": 12})
    out = collect(pv.outputs["frames"])
    rng = np.random.default_rng(7)
    for block in range(24):
        if block % 3 == 0:  # new notes, some above the Nyquist guard for the upper partials
            notes = [[rng.uniform(80, 12000), rng.uniform(0.2, 1), rng.uniform(-1, 1)] for _ in range(3)]
            pv.inputs["notes"].dispatch([np.array([notes], np.float32)])
        n = (BLOCK, 1000, 4410, 37)[block % 4]
        pv._start_pending()
        expected = _polyvoice_reference(pv, n) if len(pv._freq) else np.zeros((n, 2))
        pv.inputs["time"].dispatch(clock_block(n))
        np.testing.assert_allclose(out[-1], expected, atol=2e-6)


def _filter_reference(mode, cutoff, q, x, zi):
    """ModFilter as sub-blocks of 128 samples, coefficients at each middle sample."""
    from scipy.signal import lfilter

    out = np.empty_like(x, dtype=np.float64)
    for start in range(0, len(x), 128):
        end = min(len(x), start + 128)
        mid = (start + end - 1) >> 1
        w0 = 2 * np.pi * min(max(cutoff[mid], 20.0), SR * 0.45) / SR
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2 * max(0.1, q[mid]))
        b = {"lowpass": [(1 - cw) / 2, 1 - cw, (1 - cw) / 2],
             "highpass": [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2],
             "bandpass": [alpha, 0.0, -alpha]}[mode]
        a0 = 1 + alpha
        out[start:end], zi = lfilter(np.array(b) / a0, [1.0, -2 * cw / a0, (1 - alpha) / a0], x[start:end], axis=0, zi=zi)
    return out, zi


@pytest.mark.parametrize("mode", ["lowpass", "highpass", "bandpass"])
def test_filter_matches_the_sub_block_definition_gliding_and_steady(mode):
    filt = ModFilterNode("f")
    filt.set_properties({"mode": mode, "cutoff": 900.0, "q": 2.0, "glide": 0.05, "channels": 2})
    out = collect(filt.outputs["frames"])
    rng = np.random.default_rng(5)
    zi = np.zeros((2, 2))
    for block in range(12):
        if block in (0, 6):  # glide to a new cutoff and q, then hold
            filt.inputs["cutoff"].dispatch([np.array([rng.uniform(100, 8000)], np.float32)])
            filt.inputs["q"].dispatch([np.array([rng.uniform(0.5, 8)], np.float32)])
        x = rng.standard_normal((BLOCK, 2)).astype(np.float32)
        cut0, cut_t, q0, q_t = filt._cutoff.current, filt._cutoff.target, filt._q.current, filt._q.target
        filt.inputs["frames"].dispatch([x])
        ramp = lambda c0, c1: np.minimum((np.arange(1, BLOCK + 1)) / (0.05 * SR), 1.0) * (c1 - c0) + c0
        expected, zi = _filter_reference(mode, ramp(cut0, cut_t), ramp(q0, q_t), x, zi)
        assert out[-1].dtype == np.float32
        np.testing.assert_allclose(out[-1], expected, rtol=1e-4, atol=1e-4)


def _freeverb_reference(x: np.ndarray, sr: int, room_size: float, damping: float) -> np.ndarray:
    """Per-sample Freeverb: the wet signal of each channel (before width/mix)."""
    from audio_nodes.dsp import reverb

    scale = sr / 44100.0
    spread = max(1, round(reverb._STEREO_SPREAD * scale))
    feedback = room_size * 0.28 + 0.7
    damp = damping * 0.4
    mono = (x[:, 0].astype(np.float64) + x[:, 1]) * reverb._INPUT_GAIN
    wet = []
    for offset in (0, spread):
        combs = [[np.zeros(max(1, round(d * scale)) + offset), 0, 0.0] for d in reverb._COMB_DELAYS]
        aps = [[np.zeros(max(1, round(d * scale)) + offset), 0] for d in reverb._ALLPASS_DELAYS]
        y = np.zeros(len(mono))
        for k, sample in enumerate(mono):
            acc = 0.0
            for comb in combs:
                buf, i, store = comb
                out = buf[i]
                store = out * (1 - damp) + store * damp
                buf[i] = sample + store * feedback
                comb[1], comb[2] = (i + 1) % len(buf), store
                acc += out
            for ap in aps:
                buf, i = ap
                delayed = buf[i]
                buf[i] = acc + delayed * reverb._ALLPASS_FEEDBACK
                ap[1] = (i + 1) % len(buf)
                acc = delayed - acc
            y[k] = acc
        wet.append(y)
    return np.stack(wet, axis=1)


def test_reverb_matches_per_sample_freeverb():
    # A low sample rate shortens the delays, so the rings wrap many times and
    # blocks of odd sizes cross the chunk boundaries.
    sr = 8000
    rv = ReverbNode("r")
    rv.set_properties({"sample_rate": sr, "room_size": 0.7, "damping": 0.6, "mix": 1.0, "width": 1.0})
    out = collect(rv.outputs["frames"])
    x = np.random.default_rng(11).standard_normal((3000, 2)).astype(np.float32)
    pos = 0
    for n in (1000, 37, 203, 1460, 300):
        rv.inputs["frames"].dispatch([x[pos : pos + n]])
        pos += n
    expected = _freeverb_reference(x, sr, 0.7, 0.6)  # width 1, mix 1: out = wet
    np.testing.assert_allclose(np.concatenate(out), expected, rtol=1e-5, atol=1e-6)


# --- performance budget -----------------------------------------------------------------

def test_block_budget_for_a_demo_sized_patch():
    """8 voices + pad + noise + filter + reverb must render a 50 ms block fast."""
    clock = AudioClockNode("c")
    pad, noise = OscillatorNode("pad"), NoiseNode("n")
    filt, pan, mixer = ModFilterNode("f"), PannerNode("p"), MixerNode("m")
    pv, rv, clip = PolyVoiceNode("v"), ReverbNode("r"), SoftClipNode("s")
    mixer.set_properties({"inputs": 3})
    pv.set_properties({"max_voices": 8, "decay": 3.0})

    pad_st = PannerNode("pp")
    clock.outputs["time"].subscribe(1, pad.inputs["time"].dispatch)
    clock.outputs["time"].subscribe(2, noise.inputs["time"].dispatch)
    clock.outputs["time"].subscribe(3, pv.inputs["time"].dispatch)
    pad.outputs["frames"].subscribe(4, pad_st.inputs["frames"].dispatch)
    noise.outputs["frames"].subscribe(5, filt.inputs["frames"].dispatch)
    filt.outputs["frames"].subscribe(6, pan.inputs["frames"].dispatch)
    pad_st.outputs["frames"].subscribe(7, mixer.inputs["in0"].dispatch)
    pan.outputs["frames"].subscribe(8, mixer.inputs["in1"].dispatch)
    pv.outputs["frames"].subscribe(9, mixer.inputs["in2"].dispatch)
    mixer.outputs["frames"].subscribe(10, rv.inputs["frames"].dispatch)
    rv.outputs["frames"].subscribe(11, clip.inputs["frames"].dispatch)
    final = collect(clip.outputs["frames"])

    clock.set_properties({"prefill_blocks": 1, "blocks_per_demand": 1})
    clock.inputs["start"].dispatch([ANY])
    pv.inputs["notes"].dispatch([np.array([[[220 * (i + 1), 0.8, 0] for i in range(8)]], np.float32)])
    start = time.perf_counter()
    for _ in range(20):
        filt.inputs["cutoff"].dispatch([np.array([np.random.uniform(200, 4000)], np.float32)])
        clock.inputs["demand"].dispatch([ANY])
    per_block = (time.perf_counter() - start) / 20
    assert len(final) == 21 and final[-1].shape == (BLOCK, 2)
    assert per_block < 0.025, f"{per_block * 1000:.1f} ms per 50 ms block"
