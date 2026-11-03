"""
Synthetic Audio Data Pipeline for Speech Denoising and Howling Simulation.
"""
from __future__ import annotations
import numpy as np


class SyntheticAudioDataset:
    """Generates synthetic audio benchmarks with controllable SNR and acoustic feedback howling."""

    def __init__(self, sample_rate: int = 16000, duration: float = 3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)

    def generate_clean_speech(self) -> np.ndarray:
        """Simulate clean human speech with harmonic formants and pause intervals."""
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        # Formants around 200Hz, 500Hz, 1500Hz, 2500Hz modulated by envelope
        f0 = 150.0 + 30.0 * np.sin(2 * np.pi * 1.5 * t)
        speech = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t) + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)

        # Modulate with speech syllabic envelope (2~4 Hz)
        envelope = np.clip(np.sin(2 * np.pi * 3.0 * t), 0, 1) ** 2
        speech = speech * envelope
        # Normalize
        speech = speech / (np.max(np.abs(speech)) + 1e-8) * 0.5
        return speech.astype(np.float32)

    def add_noise(self, clean: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
        """Mix additive noise at target Signal-to-Noise Ratio (SNR)."""
        noise = np.random.randn(len(clean)).astype(np.float32)
        p_clean = np.mean(clean ** 2) + 1e-12
        p_noise = np.mean(noise ** 2) + 1e-12
        scalar = np.sqrt(p_clean / (p_noise * (10.0 ** (snr_db / 10.0))))
        return clean + scalar * noise

    def add_howling(
        self,
        signal_in: np.ndarray,
        howling_freq: float = 1200.0,
        howling_to_signal_ratio_db: float = 6.0,
    ) -> np.ndarray:
        """Inject acoustic feedback whistling at specified resonant frequency."""
        t = np.linspace(0, len(signal_in) / self.sample_rate, len(signal_in), endpoint=False)
        # Tone starts after 0.5 seconds to simulate onset of feedback buildup
        onset = int(0.5 * self.sample_rate)
        tone = np.zeros_like(signal_in)
        tone[onset:] = np.sin(2 * np.pi * howling_freq * t[onset:])

        p_sig = np.mean(signal_in ** 2) + 1e-12
        p_tone = np.mean(tone[onset:] ** 2) + 1e-12
        scalar = np.sqrt(p_sig * (10.0 ** (howling_to_signal_ratio_db / 10.0)) / p_tone)

        return signal_in + scalar * tone
