"""
Hybrid Acoustic Feedback Cancellation (AFC) with Adaptive NLMS and Notch Filter Bank.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from typing import List, Optional


class AdaptiveNLMS:
    """Normalized Least Mean Squares (NLMS) filter for linear acoustic feedback cancellation."""

    def __init__(self, filter_length: int = 256, mu: float = 0.1, eps: float = 1e-6):
        self.L = filter_length
        self.mu = mu
        self.eps = eps
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.buffer = np.zeros(filter_length, dtype=np.float32)

    def process_sample(self, loudspeaker_ref: float, mic_in: float) -> tuple[float, float]:
        """
        loudspeaker_ref: x(n), reference audio going to speaker
        mic_in: d(n), microphone input containing speech + feedback
        Returns:
            (error_sample e(n), feedback_estimate y(n))
        """
        # Shift input buffer
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = loudspeaker_ref

        # Filter estimation: y(n) = w^T * x
        y = float(np.dot(self.weights, self.buffer))
        # Error: e(n) = d(n) - y(n)
        e = mic_in - y

        # Weight update
        norm_x = float(np.dot(self.buffer, self.buffer)) + self.eps
        self.weights += (self.mu / norm_x) * e * self.buffer

        return e, y


class DynamicNotchFilter:
    """Second-order IIR Notch filter for pinpoint howling frequency attenuation."""

    def __init__(self, sample_rate: int = 16000, Q: float = 30.0):
        self.sample_rate = sample_rate
        self.Q = Q
        self.b = np.array([1.0, 0.0, 0.0])
        self.a = np.array([1.0, 0.0, 0.0])
        self.zi = np.zeros(2)

    def update_frequency(self, freq_hz: float):
        """Update notch center frequency."""
        w0 = freq_hz / (self.sample_rate / 2.0)
        w0 = min(max(w0, 0.01), 0.99)
        self.b, self.a = signal.iirnotch(w0, self.Q)
        self.zi = np.zeros(2)

    def filter(self, x: np.ndarray) -> np.ndarray:
        if np.all(self.b == [1.0, 0.0, 0.0]):
            return x
        y, self.zi = signal.lfilter(self.b, self.a, x, zi=self.zi)
        return y
