"""
Statistical properties of the LSF (line spectral frequency) delta quantizer
used by this project's MPE-LPC pipeline (mpe_common.py / synth_core.py /
lsf.py). Every frame of a LibriSpeech subset is LPC-analyzed at a fixed order
p, converted to LSFs, and run through quantize_lsfs; the transmitted symbol
indices are what the entropies below are measured on.

Plots written to ./properties_results:
  entropy_vs_resolution.png  bits/delta vs quantizer resolution (2,3,...,8 bits),
                             as both a single pooled model and per-index models,
                             against the allocated fixed rate.
  entropy_per_index.png      bits for each individual LSF index, one curve per
                             quantizer resolution. Each curve averages to the
                             corresponding "per-index" point in the plot above.
  lsf_delta_histogram.png    distribution of the raw (unquantized) LSF deltas.
  spectral_envelope.png      log spectral power of the all-pole envelope 1/A(z),
                             quantized vs unquantized: mean envelope over
                             frequency, and the distribution of log power.

No transcription/WER here, so this only needs load_subset -- not Whisper.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpe_common import DEFAULT_P, N_SUBSET, SUBSET_SEED, load_subset
from synth_core import L, SR, LSF_BITS, POLE_MARGIN, fit_lpc
from lsf import lpc_to_lsf, lsf_to_lpc, quantize_lsfs, delta_max_for_order

OUT_DIR = "./properties_results"
DEFAULT_BITS_VALUES = [2, 3, 4, 5, 6, 7, 8]


def collect_frames(dataset, p):
    """LPC coefficients and LSFs for every usable frame of `dataset` at order p.

    Returns (A, lsfs) with shapes (n_frames, p+1) and (n_frames, p). Frames that
    are silent or whose LPC fit is not strictly minimum-phase are skipped, the
    same conditions synthesize_mpe skips on.
    """
    A_list, lsf_list = [], []
    for audio, _ in dataset:
        audio = np.asarray(audio, dtype=float)
        for i in range(len(audio) // L):
            seg = audio[i * L:(i + 1) * L]
            if len(seg) <= p or np.sum(seg ** 2) < 1e-12:
                continue
            A, _, _ = fit_lpc(seg, p)
            try:
                lsf = lpc_to_lsf(A)
            except (RuntimeError, np.linalg.LinAlgError):
                continue
            A_list.append(A)
            lsf_list.append(lsf)
    return np.array(A_list), np.array(lsf_list)


def raw_lsf_deltas(lsfs):
    """Deltas of each LSF row, including the gap from 0 to lsf[0] (the
    convention quantize_lsfs uses for its DPCM predictor)."""
    return np.diff(np.concatenate([np.zeros((len(lsfs), 1)), lsfs], axis=1), axis=1)


def quantizer_indices(lsfs, p, n_bits):
    """Transmitted DPCM symbol index per LSF, shape (n_frames, p)."""
    delta_max = delta_max_for_order(p)
    out = np.empty((len(lsfs), p), dtype=int)
    for i, lsf in enumerate(lsfs):
        _, out[i] = quantize_lsfs(lsf, n_bits=n_bits, delta_max=delta_max,
                                  return_indices=True)
    return out


def entropy_bits(symbols, n_bits):
    """Shannon entropy (bits) of the empirical distribution over the
    2**n_bits-symbol quantizer alphabet."""
    symbols = np.asarray(symbols).ravel()
    if symbols.size == 0:
        return float("nan")
    counts = np.bincount(symbols, minlength=2 ** n_bits)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def per_index_entropy(indices, n_bits):
    """Entropy of each LSF index's own symbol distribution, shape (p,)."""
    return np.array([entropy_bits(indices[:, j], n_bits)
                     for j in range(indices.shape[1])])


def lpc_log_spectra(A, n_fft):
    """Log power spectra (dB) of the all-pole envelopes 1/A(z), one row per
    frame. rfft of the coefficient rows evaluates A(z) on the frequency grid."""
    spec = np.fft.rfft(A, n=n_fft, axis=1)
    return -20.0 * np.log10(np.abs(spec) + 1e-12)


def quantized_lpc(lsfs, p, n_bits, pole_margin=POLE_MARGIN):
    """LPC coefficients the decoder reconstructs from the quantized LSFs,
    including the bandwidth expansion synthesize_mpe applies."""
    delta_max = delta_max_for_order(p)
    decay = pole_margin ** np.arange(p + 1)
    out = np.empty((len(lsfs), p + 1))
    for i, lsf in enumerate(lsfs):
        lsf_hat = quantize_lsfs(lsf, n_bits=n_bits, delta_max=delta_max)
        out[i] = lsf_to_lpc(lsf_hat) * decay
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_entropy_vs_resolution(bits_values, pooled, per_index_mean, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(bits_values, per_index_mean, marker="o", label="per-index models (mean)")
    plt.plot(bits_values, pooled, marker="s", label="single pooled model")
    plt.plot(bits_values, bits_values, color="black", linestyle=":",
             label="allocated (fixed rate)")
    plt.xlabel("Quantizer resolution (bits per delta)")
    plt.ylabel("Entropy (bits per LSF delta)")
    plt.title("Quantized LSF-delta entropy vs quantizer resolution")
    plt.xticks(bits_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_entropy_per_index(bits_values, per_index, p, out_path):
    plt.figure(figsize=(8, 5))
    idx = np.arange(1, p + 1)
    for n_bits, curve in zip(bits_values, per_index):
        plt.plot(idx, curve, marker="o", label=f"{n_bits} bits")
    plt.xlabel("LSF index")
    plt.ylabel("Entropy (bits)")
    plt.title(f"Quantized LSF-delta entropy per index (p={p})")
    plt.xticks(idx)
    plt.legend(title="quantizer resolution", ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_delta_histogram(deltas, p, out_path, bins=60):
    plt.figure(figsize=(8, 5))
    plt.hist(deltas.ravel(), bins=bins, color="tab:blue", edgecolor="none")
    plt.xlabel("LSF delta (rad)")
    plt.ylabel("Count")
    plt.title(f"Histogram of raw LSF deltas (p={p}, n={deltas.size})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_spectral_comparison(log_true, log_quant, n_bits, p, out_path, bins=80):
    freqs = np.fft.rfftfreq(2 * (log_true.shape[1] - 1), d=1.0 / SR)
    sd = float(np.mean(np.sqrt(np.mean((log_true - log_quant) ** 2, axis=1))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(freqs, log_true.mean(axis=0), label="unquantized")
    ax1.plot(freqs, log_quant.mean(axis=0), linestyle="--", label=f"quantized ({n_bits} bits)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Mean log spectral power (dB)")
    ax1.set_title("Mean spectral envelope")
    ax1.legend()

    ax2.hist(log_true.ravel(), bins=bins, histtype="step", density=True, label="unquantized")
    ax2.hist(log_quant.ravel(), bins=bins, histtype="step", density=True,
             label=f"quantized ({n_bits} bits)")
    ax2.set_xlabel("Log spectral power (dB)")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of log spectral power")
    ax2.legend()

    fig.suptitle(f"All-pole envelope 1/A(z), p={p} -- mean spectral distortion {sd:.2f} dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return sd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subset", type=int, default=N_SUBSET,
                         help=f"number of LibriSpeech utterances to use (default: {N_SUBSET})")
    parser.add_argument("--seed", type=int, default=SUBSET_SEED,
                         help=f"random seed for subset selection (default: {SUBSET_SEED})")
    parser.add_argument("--p", type=int, default=DEFAULT_P,
                         help=f"LPC order (even; default: {DEFAULT_P})")
    parser.add_argument("--bits-values", type=str,
                         default=",".join(str(b) for b in DEFAULT_BITS_VALUES),
                         help="comma-separated quantizer resolutions to sweep "
                              f"(default: {','.join(str(b) for b in DEFAULT_BITS_VALUES)})")
    parser.add_argument("--envelope-bits", type=int, default=LSF_BITS,
                         help="quantizer resolution used for the spectral envelope "
                              f"comparison (default: {LSF_BITS})")
    parser.add_argument("--n-fft", type=int, default=512,
                         help="FFT size for the spectral envelopes (default: 512)")
    args = parser.parse_args()

    if args.p % 2:
        parser.error(f"LSF conversion requires an even order; got p={args.p}")
    try:
        args.bits_list = sorted({int(tok) for tok in args.bits_values.split(",") if tok.strip()})
    except ValueError:
        parser.error("--bits-values must be a comma-separated list of integers")
    if not args.bits_list or args.bits_list[0] < 1:
        parser.error("--bits-values must contain positive integers")
    return args


def main():
    args = parse_args()
    p = args.p
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading LibriSpeech test-clean subset...")
    dataset = load_subset(args.n_subset, args.seed)
    print(f"  {len(dataset)} utterances")

    print(f"LPC/LSF analysis at p={p}...")
    A_true, lsfs = collect_frames(dataset, p)
    print(f"  {len(lsfs)} frames")

    print(f"Quantizer resolution sweep {args.bits_list}...")
    pooled, per_index_mean, per_index_curves = [], [], []
    for n_bits in args.bits_list:
        indices = quantizer_indices(lsfs, p, n_bits)
        curve = per_index_entropy(indices, n_bits)
        pooled.append(entropy_bits(indices, n_bits))
        per_index_mean.append(float(curve.mean()))
        per_index_curves.append(curve)
        print(f"  {n_bits} bits: pooled H={pooled[-1]:.3f}, "
              f"per-index mean H={per_index_mean[-1]:.3f} bits/delta")

    plot_entropy_vs_resolution(args.bits_list, pooled, per_index_mean,
                                os.path.join(OUT_DIR, "entropy_vs_resolution.png"))
    plot_entropy_per_index(args.bits_list, per_index_curves, p,
                            os.path.join(OUT_DIR, "entropy_per_index.png"))
    plot_delta_histogram(raw_lsf_deltas(lsfs), p,
                          os.path.join(OUT_DIR, "lsf_delta_histogram.png"))

    print(f"Spectral envelope comparison at {args.envelope_bits} bits...")
    log_true = lpc_log_spectra(A_true, args.n_fft)
    log_quant = lpc_log_spectra(quantized_lpc(lsfs, p, args.envelope_bits), args.n_fft)
    sd = plot_spectral_comparison(log_true, log_quant, args.envelope_bits, p,
                                   os.path.join(OUT_DIR, "spectral_envelope.png"))
    print(f"  mean spectral distortion {sd:.2f} dB")

    print(f"Done. Plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
