"""
Unit tests for Speech Enhancement & Howling Suppression algorithms.
"""
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
import numpy as np
from dataset.synthetic_pipeline import SyntheticAudioDataset
from models.howling_detector import HowlingDetector
from models.afc_neural_filter import AdaptiveNLMS, DynamicNotchFilter
from metrics.evaluator import compute_sisdr, compute_erle, compute_snr
from infer import enhance_audio


def test_synthetic_data_generation():
    dataset = SyntheticAudioDataset(sample_rate=16000, duration=1.0)
    clean = dataset.generate_clean_speech()
    assert len(clean) == 16000
    noisy = dataset.add_noise(clean, snr_db=10.0)
    assert len(noisy) == 16000
    corrupted = dataset.add_howling(noisy, howling_freq=1000.0)
    assert len(corrupted) == 16000


def test_howling_detector():
    detector = HowlingDetector(sample_rate=16000, n_fft=512, hop_length=256)
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    # Pure sinusoidal frame (howling)
    tone = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

    # Feed 3 consecutive frames to satisfy persistence threshold
    detector.detect_frame(tone)
    detector.detect_frame(tone)
    is_howling, freq = detector.detect_frame(tone)

    assert is_howling is True
    assert freq is not None
    assert abs(freq - 1000.0) < 50.0  # Within frequency resolution bin


def test_adaptive_nlms():
    nlms = AdaptiveNLMS(filter_length=64, mu=0.2)
    # Speaker reference
    x = 0.5
    # Echo impulse response: echo = 0.8 * x
    mic_in = 0.4
    e, y = nlms.process_sample(x, mic_in)
    assert isinstance(e, float)
    assert isinstance(y, float)


def test_metrics_computation():
    ref = np.sin(np.linspace(0, 10, 1000))
    est = ref + 0.01 * np.random.randn(1000)
    sisdr = compute_sisdr(ref, est)
    assert sisdr > 15.0  # High fidelity

    erle = compute_erle(est, ref)
    assert isinstance(erle, float)


def test_enhancement_pipeline():
    dataset = SyntheticAudioDataset(sample_rate=16000, duration=1.0)
    clean = dataset.generate_clean_speech()
    corrupted = dataset.add_howling(clean, howling_freq=1500.0, howling_to_signal_ratio_db=6.0)

    enhanced, info = enhance_audio(corrupted, sample_rate=16000)
    assert len(enhanced) == len(corrupted)
    assert info["howling_detected"] is True
