"""
Inference & Evaluation pipeline for Speech Enhancement, Denoising and Howling Suppression.
"""
from __future__ import annotations
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import argparse
import numpy as np
from dataset.synthetic_pipeline import SyntheticAudioDataset
from models.howling_detector import HowlingDetector
from models.afc_neural_filter import DynamicNotchFilter
from metrics.evaluator import compute_sisdr, compute_erle


def enhance_audio(
    corrupted_audio: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 256,
) -> tuple[np.ndarray, dict]:
    """
    Two-stage speech enhancement pipeline:
    Stage 1: Real-time Howling Detection & Dynamic Notch attenuation.
    Stage 2: Spectral Denoising & Residual Feedback Suppression.
    """
    detector = HowlingDetector(sample_rate=sample_rate, hop_length=hop_length)
    notch = DynamicNotchFilter(sample_rate=sample_rate)

    enhanced = np.copy(corrupted_audio)
    detected_howling_freqs = []

    # Frame-by-frame processing for Howling Suppression
    for i in range(0, len(corrupted_audio) - 512, hop_length):
        frame = corrupted_audio[i : i + 512]
        is_howling, freq = detector.detect_frame(frame)
        if is_howling and freq is not None:
            detected_howling_freqs.append(round(freq, 1))
            notch.update_frequency(freq)

    # Apply notch filtering
    if detected_howling_freqs:
        enhanced = notch.filter(enhanced)

    # Stage 2: Spectral Subtraction / Wiener smoothing for denoising
    n_fft = 512
    # Simple spectral floor denoising
    window = np.hanning(n_fft)
    stft = []
    for i in range(0, len(enhanced) - n_fft, hop_length):
        f = enhanced[i : i + n_fft] * window
        stft.append(np.fft.rfft(f))

    if stft:
        spec = np.array(stft)  # (Frames, Bins)
        mag = np.abs(spec)
        phase = np.angle(spec)
        # Estimate noise floor from lowest 10% energy frames
        frame_energies = np.sum(mag ** 2, axis=1)
        noise_idx = np.argsort(frame_energies)[: max(1, len(frame_energies) // 10)]
        noise_profile = np.mean(mag[noise_idx], axis=0, keepdims=True)

        # Spectral subtraction with oversubtraction factor alpha
        alpha = 1.5
        cleaned_mag = np.maximum(mag - alpha * noise_profile, 0.05 * mag)
        cleaned_spec = cleaned_mag * np.exp(1j * phase)

        # Synthesis via Overlap-Add (OLA)
        reconstructed = np.zeros(len(enhanced), dtype=np.float32)
        win_sum = np.zeros(len(enhanced), dtype=np.float32)
        for idx, i in enumerate(range(0, len(enhanced) - n_fft, hop_length)):
            time_frame = np.fft.irfft(cleaned_spec[idx]) * window
            reconstructed[i : i + n_fft] += time_frame
            win_sum[i : i + n_fft] += window ** 2

        mask = win_sum > 1e-4
        enhanced[mask] = reconstructed[mask] / win_sum[mask]

    info = {
        "howling_detected": len(detected_howling_freqs) > 0,
        "howling_freqs_hz": list(set(detected_howling_freqs)),
    }
    return enhanced, info


def main():
    parser = argparse.ArgumentParser(description="Deep Speech Enhancement & Howling Suppression Demo")
    parser.add_argument("--howling-freq", type=float, default=1250.0, help="Resonant feedback frequency to simulate (Hz)")
    parser.add_argument("--noise-snr", type=float, default=12.0, help="Noise SNR in dB")
    args = parser.parse_args()

    print("===================================================================")
    print("      Deep Speech Enhancement, Denoising & Howling Suppression     ")
    print("===================================================================")

    # 1. Generate synthetic benchmark audio
    dataset = SyntheticAudioDataset(sample_rate=16000, duration=3.0)
    clean = dataset.generate_clean_speech()
    noisy = dataset.add_noise(clean, snr_db=args.noise_snr)
    corrupted = dataset.add_howling(noisy, howling_freq=args.howling_freq, howling_to_signal_ratio_db=8.0)

    print(f"[*] Synthetic Audio Generated: 3.0s @ 16kHz")
    print(f"[*] Injected Feedback Howling at: {args.howling_freq} Hz")
    print(f"[*] Injected Background Noise at: {args.noise_snr} dB SNR\n")

    # Metrics before enhancement
    si_sdr_before = compute_sisdr(clean, corrupted)
    print(f"[-] Metric Before Enhancement: SI-SDR = {si_sdr_before:.2f} dB")

    # 2. Run enhancement pipeline
    enhanced, info = enhance_audio(corrupted, sample_rate=16000)

    # Metrics after enhancement
    si_sdr_after = compute_sisdr(clean, enhanced)
    erle = compute_erle(corrupted, enhanced)

    print(f"\n[+] Processing Results:")
    print(f"    - Howling Detected: {info['howling_detected']}")
    print(f"    - Detected Frequencies: {info['howling_freqs_hz']} Hz")
    print(f"    - Metric After Enhancement: SI-SDR = {si_sdr_after:.2f} dB (Improvement: +{si_sdr_after - si_sdr_before:.2f} dB)")
    print(f"    - Howling Return Loss Enhancement (ERLE): {erle:.2f} dB\n")
    print("[*] Audio enhancement successfully completed.")


if __name__ == "__main__":
    main()
