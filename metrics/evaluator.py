"""
Objective evaluation metrics for speech enhancement and howling suppression.
"""
from __future__ import annotations
import numpy as np


def compute_sisdr(reference: np.ndarray, estimated: np.ndarray, eps: float = 1e-8) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    """
    ref = reference - np.mean(reference)
    est = estimated - np.mean(estimated)

    # Project est onto ref: s_target = (<est, ref> / ||ref||^2) * ref
    dot = np.dot(est, ref)
    norm_ref = np.dot(ref, ref) + eps
    alpha = dot / norm_ref

    s_target = alpha * ref
    e_noise = est - s_target

    p_target = np.sum(s_target ** 2) + eps
    p_noise = np.sum(e_noise ** 2) + eps

    return float(10.0 * np.log10(p_target / p_noise))


def compute_erle(original_corrupted: np.ndarray, processed: np.ndarray, eps: float = 1e-8) -> float:
    """
    Echo / Howling Return Loss Enhancement (ERLE):
    ERLE = 10 * log10 ( E[d^2(n)] / E[e^2(n)] )
    """
    p_orig = np.mean(original_corrupted ** 2) + eps
    p_proc = np.mean(processed ** 2) + eps
    return float(10.0 * np.log10(p_orig / p_proc))


def compute_snr(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-8) -> float:
    """Signal-to-Noise Ratio (SNR)."""
    noise = noisy - clean
    p_signal = np.sum(clean ** 2) + eps
    p_noise = np.sum(noise ** 2) + eps
    return float(10.0 * np.log10(p_signal / p_noise))
