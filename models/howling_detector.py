"""
Acoustic Feedback Howling Detector using PAPR and Spectral Flatness.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional


class HowlingDetector:
    """
    Detects acoustic feedback whistling (howling) in audio streams.
    Key criteria:
    1. PAPR (Peak-to-Average Power Ratio) > Threshold
    2. Spectral Peak Sharpness and Persistence across consecutive frames
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        papr_threshold_db: float = 12.0,
        persistence_frames: int = 3,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.papr_threshold_db = papr_threshold_db
        self.persistence_frames = persistence_frames
        self.peak_history: List[int] = []

    def detect_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Processes a single time-domain frame (length n_fft).
        Returns:
            (is_howling, howling_frequency_hz)
        """
        if len(frame) < self.n_fft:
            frame = np.pad(frame, (0, self.n_fft - len(frame)))

        windowed = frame * np.hanning(self.n_fft)
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)

        # Ignore DC and very low frequency bins (<100Hz)
        valid_idx = freqs > 100
        spectrum_valid = spectrum[valid_idx]
        freqs_valid = freqs[valid_idx]

        if len(spectrum_valid) == 0:
            return False, None

        peak_val = max(float(np.max(spectrum_valid)), 1e-12)
        avg_val = max(float(np.mean(spectrum_valid)), 1e-12)

        papr_db = 10.0 * np.log10(peak_val / avg_val)
        peak_idx = np.argmax(spectrum_valid)
        detected_freq = freqs_valid[peak_idx]

        if papr_db > self.papr_threshold_db:
            self.peak_history.append(int(peak_idx))
            if len(self.peak_history) > self.persistence_frames:
                self.peak_history.pop(0)

            # Check if peak frequency stayed consistent
            if len(self.peak_history) == self.persistence_frames and len(set(self.peak_history)) == 1:
                return True, float(detected_freq)
        else:
            self.peak_history.clear()

        return False, None
