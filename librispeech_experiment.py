"""
Multipulse-excitation LPC experiment on a LibriSpeech test-clean subset.

Two sweeps, each run for all three greedy pulse-selection methods
(omp, thresholding, thresholding_gram) plus a regular (non-sparse) LPC
predictor as a reference:

  1. pulse sweep   - fix LPC order p, vary number of pulses K = 1..32
  2. order sweep   - fix number of pulses K, vary LPC order p

For every point on every sweep we measure, per utterance and then
averaged over the subset:
  - wall-clock synthesis time
  - PESQ (wideband) against the clean utterance
  - Whisper (base) word error rate, both reference and hypothesis run
    through Whisper's EnglishTextNormalizer before jiwer.wer

A single clean-audio Whisper WER baseline is computed once and drawn as
a horizontal reference line on the WER plots.
"""

import os
import json
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

import librosa
import torchaudio
import jiwer
import whisper
from whisper.normalizers import EnglishTextNormalizer
from pesq import pesq

from greedy import omp, thresholding, thresholding_gram

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR = 16000
FRAME_MS = 10
L = int(SR * FRAME_MS / 1000)          # samples per analysis/synthesis frame

N_SUBSET = 20                          # number of LibriSpeech utterances to use
SUBSET_SEED = 0
DATA_ROOT = "./librispeech_data"
OUT_DIR = "./experiment_results"

DEFAULT_P = 10                         # LPC order held fixed during the pulse sweep
DEFAULT_K = 8                          # pulse count held fixed during the order sweep

PULSE_SWEEP = list(range(1, 33))               # K = 1..32
ORDER_SWEEP = [p for p in range(2, 33, 2) if p < L]   # p = 2..32, stays well under frame length

WHISPER_MODEL_NAME = "base"

METHODS = {
    "omp": omp,
    "thresholding": thresholding,
    "thresholding_gram": thresholding_gram,
}

METRICS = [
    ("time", "Runtime (s / utterance)"),
    ("pesq", "PESQ (wideband)"),
    ("wer", "Word Error Rate"),
]


# ---------------------------------------------------------------------------
# LPC / multipulse core (frame-based, mirrors lpc_demo.py)
# ---------------------------------------------------------------------------

def lpc_matrix(seg, p):
    n = len(seg)
    return np.column_stack([seg[p - k - 1:n - k - 1] for k in range(p)])


def fit_lpc(seg, p):
    """Fit an order-p LPC filter A(z) = 1 + sum a_k z^-k to one frame."""
    H = lpc_matrix(seg, p)
    a = np.linalg.lstsq(H, -seg[p:], rcond=None)[0]
    A = np.concatenate(([1.0], a))
    return A, H, a


def frame_impulse_response(A, L):
    impulse = np.zeros(L)
    impulse[0] = 1.0
    return lfilter([1.0], A, impulse)


def synthesize_mpe(fn, x, p, L, K):
    """Frame-by-frame multipulse-excited reconstruction using greedy method fn."""
    N = len(x)
    n_frames = N // L
    out = np.zeros(N)
    for i in range(n_frames):
        seg = x[i * L:(i + 1) * L]
        if len(seg) <= p or np.sum(seg ** 2) < 1e-12:
            continue
        A, _, _ = fit_lpc(seg, p)
        h = frame_impulse_response(A, L)
        if not np.all(np.isfinite(h)) or np.abs(h).max() > 1e6:
            continue
        m, _, locs = fn(h, seg, K=K)
        e = np.zeros(len(seg))
        e[locs] = m
        out[i * L:(i + 1) * L] = np.convolve(e, h)[:len(seg)]
    return out


def synthesize_regular_lpc(x, p, L):
    """Frame-based non-sparse reference: one-step LPC prediction from the true past samples."""
    N = len(x)
    n_frames = N // L
    out = np.zeros(N)
    for i in range(n_frames):
        seg = x[i * L:(i + 1) * L]
        if len(seg) <= p or np.sum(seg ** 2) < 1e-12:
            continue
        _, H, a = fit_lpc(seg, p)
        out[i * L:(i + 1) * L][p:] = -(H @ a)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_subset(n=N_SUBSET, seed=SUBSET_SEED):
    """Random subset of LibriSpeech test-clean, resampled to SR, as (audio, transcript) pairs."""
    dataset = torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url="test-clean", download=True)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    items = []
    for idx in indices[:n]:
        waveform, sr, transcript, *_ = dataset[idx]
        audio = waveform.squeeze(0).numpy().astype(np.float32)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        items.append((audio, transcript))
    return items


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pesq(ref, deg, sr=SR):
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    if np.max(np.abs(deg)) < 1e-8:
        return np.nan
    try:
        return pesq(sr, ref, deg, "wb")
    except Exception:
        return np.nan


def transcribe(model, audio):
    try:
        result = model.transcribe(audio.astype(np.float32), fp16=False, language="en")
        return result["text"]
    except Exception:
        return ""


def compute_wer(normalizer, reference_text, hyp_text):
    ref_n = normalizer(reference_text)
    hyp_n = normalizer(hyp_text)
    if len(ref_n.strip()) == 0:
        return np.nan
    return jiwer.wer(ref_n, hyp_n)


def compute_baseline_wer(dataset, whisper_model, normalizer):
    wers = []
    for audio, transcript in dataset:
        hyp = transcribe(whisper_model, audio)
        wers.append(compute_wer(normalizer, transcript, hyp))
    return float(np.nanmean(wers))


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def empty_result_block():
    return {"time": [], "pesq": [], "wer": []}


def run_sweep(dataset, param_name, param_values, fixed_params, whisper_model, normalizer):
    """
    param_name is "K" (pulse sweep) or "p" (order sweep); the other parameter is
    held at fixed_params["p"] / fixed_params["K"]. Returns per-method + regular_lpc
    metric lists, one averaged value per entry in param_values.
    """
    results = {name: empty_result_block() for name in METHODS}
    results["regular_lpc"] = empty_result_block()

    for val in param_values:
        p = val if param_name == "p" else fixed_params["p"]
        K = val if param_name == "K" else fixed_params["K"]

        per_method = {name: empty_result_block() for name in METHODS}
        lpc_block = empty_result_block()

        for audio, transcript in dataset:
            for name, fn in METHODS.items():
                t0 = time.perf_counter()
                out = synthesize_mpe(fn, audio, p, L, K)
                dt = time.perf_counter() - t0

                per_method[name]["time"].append(dt)
                per_method[name]["pesq"].append(compute_pesq(audio, out))
                hyp = transcribe(whisper_model, out)
                per_method[name]["wer"].append(compute_wer(normalizer, transcript, hyp))

            t0 = time.perf_counter()
            lpc_out = synthesize_regular_lpc(audio, p, L)
            dt = time.perf_counter() - t0

            lpc_block["time"].append(dt)
            lpc_block["pesq"].append(compute_pesq(audio, lpc_out))
            hyp = transcribe(whisper_model, lpc_out)
            lpc_block["wer"].append(compute_wer(normalizer, transcript, hyp))

        for name in METHODS:
            for metric in ("time", "pesq", "wer"):
                results[name][metric].append(float(np.nanmean(per_method[name][metric])))
        for metric in ("time", "pesq", "wer"):
            results["regular_lpc"][metric].append(float(np.nanmean(lpc_block[metric])))

        print(f"  {param_name}={val} done")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(results, param_values, xlabel, baseline_wer, out_prefix):
    os.makedirs(OUT_DIR, exist_ok=True)
    for metric_key, ylabel in METRICS:
        plt.figure(figsize=(8, 5))
        for name in METHODS:
            plt.plot(param_values, results[name][metric_key], marker="o", label=name)
        plt.plot(param_values, results["regular_lpc"][metric_key],
                  marker="s", linestyle="--", label="regular LPC")
        if metric_key == "wer":
            plt.axhline(baseline_wer, color="black", linestyle=":", label="clean baseline WER")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs {xlabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{out_prefix}_{metric_key}.png"), dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_results(name, param_values, results, baseline_wer):
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "param_values": param_values,
        "results": results,
        "baseline_wer": baseline_wer,
    }
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def main():
    print("Loading LibriSpeech test-clean subset...")
    dataset = load_subset(N_SUBSET, SUBSET_SEED)
    print(f"  {len(dataset)} utterances")

    print(f"Loading Whisper ({WHISPER_MODEL_NAME})...")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    normalizer = EnglishTextNormalizer()

    print("Computing clean-audio baseline WER...")
    baseline_wer = compute_baseline_wer(dataset, whisper_model, normalizer)
    print(f"  baseline WER = {baseline_wer:.4f}")

    fixed_params = {"p": DEFAULT_P, "K": DEFAULT_K}

    print(f"Running pulse sweep (K={PULSE_SWEEP[0]}..{PULSE_SWEEP[-1]}, p={DEFAULT_P})...")
    pulse_results = run_sweep(dataset, "K", PULSE_SWEEP, fixed_params, whisper_model, normalizer)
    save_results("pulse_sweep", PULSE_SWEEP, pulse_results, baseline_wer)
    plot_sweep(pulse_results, PULSE_SWEEP, "Number of pulses (K)", baseline_wer, "pulse_sweep")

    print(f"Running order sweep (p={ORDER_SWEEP[0]}..{ORDER_SWEEP[-1]}, K={DEFAULT_K})...")
    order_results = run_sweep(dataset, "p", ORDER_SWEEP, fixed_params, whisper_model, normalizer)
    save_results("order_sweep", ORDER_SWEEP, order_results, baseline_wer)
    plot_sweep(order_results, ORDER_SWEEP, "LPC order (p)", baseline_wer, "order_sweep")

    print(f"Done. Plots and raw results written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
