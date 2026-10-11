import argparse
import csv
import os

import numpy as np
import torch
import whisper.audio as whisper_audio
from scipy.signal import lfilter

from mpe_common import OUT_DIR, load_subset, SUBSET_SEED, DEFAULT_P, DEFAULT_K, bootstrap_ci
from synth_core import (L, SR, GAMMA, LSF_BITS, EXC_BITS, POLE_MARGIN,
                        fit_lpc, frame_impulse_response, weighting_filter,
                        _zero_input_response)
from lsf import lpc_to_lsf, lsf_to_lpc, quantize_lsfs, quantize_excitation, delta_max_for_order
from algorithms import omp, mpog, lst, gclst

METHODS = {"omp": omp, "mpog": mpog, "lst": lst, "gclst": gclst}


def synthesize_weighted(fn, x, p, L, K, gamma, lsf_bits=LSF_BITS, exc_bits=EXC_BITS,
                        pole_margin=POLE_MARGIN):
    N = len(x)
    n_frames = N // L
    out = np.zeros(N)

    global_scale = np.max(np.abs(x)) + 1e-9
    clamp = 20.0 * global_scale
    signal_energy = 0.0
    noise_energy = 0.0

    for i in range(n_frames):
        start, end = i * L, (i + 1) * L
        seg = x[start:end]
        if len(seg) <= p or np.sum(seg ** 2) < 1e-12:
            continue
        A_true, _, _ = fit_lpc(seg, p)
        hist = np.pad(out[max(0, start - p):start][::-1], (0, max(0, p - start)))

        A_hat = A_true
        if p % 2 == 0:
            try:
                lsf = lpc_to_lsf(A_true)
                candidate = lsf_to_lpc(quantize_lsfs(lsf, n_bits=lsf_bits,
                                                      delta_max=delta_max_for_order(p)))
                candidate = candidate * pole_margin ** np.arange(p + 1)
                ring = _zero_input_response(candidate, hist, len(seg))
                if np.all(np.isfinite(ring)) and np.max(np.abs(ring)) < 50 * global_scale:
                    A_hat = candidate
            except (RuntimeError, np.linalg.LinAlgError):
                pass
        A = A_hat

        h = frame_impulse_response(A, L)
        if not np.all(np.isfinite(h)) or np.abs(h).max() > 1e6:
            continue

        ring = _zero_input_response(A, hist, len(seg))
        target = seg - ring

        b_w, A_w = weighting_filter(A, gamma)
        target_w = lfilter(b_w, A_w, target)
        impulse = np.zeros(len(seg))
        impulse[0] = 1.0
        h_w = lfilter([1.0], A_w, impulse)

        m, locs = fn(h_w, target_w, K=K)
        m = quantize_excitation(m, n_bits=exc_bits)
        e = np.zeros(len(seg))
        e[locs] = m

        recon_w = np.convolve(e, h_w)[:len(seg)]
        signal_energy += np.sum(target_w ** 2)
        noise_energy += np.sum((target_w - recon_w) ** 2)

        frame_out = ring + np.convolve(e, h)[:len(seg)]
        out[start:end] = np.clip(frame_out, -clamp, clamp)
    return out, signal_energy, noise_energy


def snr_from_energies(signal_energy, noise_energy):
    if noise_energy <= 0.0:
        return float("inf")
    if signal_energy <= 0.0:
        return float("-inf")
    return 10.0 * np.log10(signal_energy / noise_energy)


def mel_l2(clean, synth):
    """Raw L2 (Frobenius) norm between the two log-mel spectrograms."""
    clean_mel = whisper_audio.log_mel_spectrogram(clean.astype(np.float32))
    synth_mel = whisper_audio.log_mel_spectrogram(synth.astype(np.float32))
    return float(torch.norm(clean_mel - synth_mel).item())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-utterances", type=int, default=1,
                    help="number of LibriSpeech utterances to pool over (default: 1)")
    ap.add_argument("--seed", type=int, default=SUBSET_SEED)
    ap.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean")
    ap.add_argument("--p", type=int, default=DEFAULT_P, help=f"LPC order (default: {DEFAULT_P})")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"pulses per L={L} frame (default: {DEFAULT_K})")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--algorithms", type=str, default=",".join(METHODS),
                    help=f"comma-separated algorithms (default: all = {','.join(METHODS)})")
    ap.add_argument("--out", type=str,
                    default=os.path.join(OUT_DIR, "region_synthesis_eval.csv"))
    args = ap.parse_args()

    args.algo_list = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    unknown = [a for a in args.algo_list if a not in METHODS]
    if not args.algo_list or unknown:
        ap.error(f"--algorithms must be from {list(METHODS)}; got unknown: {unknown}")
    return args


def main():
    args = parse_args()

    print(f"Loading LibriSpeech {args.split} subset (seed={args.seed}, "
          f"fixed -- reproducible even when n-utterances is smaller than the full split)...")
    dataset = load_subset(args.n_utterances, args.seed, split=args.split)

    mel_l2s = {name: [] for name in args.algo_list}
    snrs = {name: [] for name in args.algo_list}
    total_samples = 0

    for u_idx, (audio, transcript) in enumerate(dataset):
        audio = np.asarray(audio, dtype=float)
        n_frames = len(audio) // L
        clean = audio[:n_frames * L]        # synthesize_weighted only fills whole frames
        if len(clean) == 0:
            print(f"  utterance {u_idx}: too short, skipping")
            continue
        total_samples += len(clean)
        print(f"utterance {u_idx}: \"{transcript[:50]}\" ({len(clean) / SR:.2f} s)")

        for name in args.algo_list:
            fn = METHODS[name]
            synth, sig_e, noise_e = synthesize_weighted(fn, clean, args.p, L, args.k,
                                                        gamma=args.gamma)
            l2 = mel_l2(clean, synth)
            snr = snr_from_energies(sig_e, noise_e)
            mel_l2s[name].append(l2)
            snrs[name].append(snr)
            print(f"    {name:8s} mel_l2={l2:9.3f}  weighted_snr={snr:6.2f} dB")

    n_used = len(next(iter(mel_l2s.values()))) if args.algo_list else 0
    print(f"\n{n_used}/{len(dataset)} utterances used, {total_samples} samples total "
          f"({total_samples / SR:.1f} s)")

    rows = []
    for name in args.algo_list:
        l2_mean, l2_lo, l2_hi = bootstrap_ci(mel_l2s[name])
        snr_finite = [v for v in snrs[name] if np.isfinite(v)]
        snr_mean, snr_lo, snr_hi = bootstrap_ci(snr_finite)
        rows.append({"algorithm": name, "n_utterances": n_used, "n_samples": total_samples,
                     "mel_l2_mean": l2_mean, "mel_l2_lo": l2_lo, "mel_l2_hi": l2_hi,
                     "weighted_snr_db_mean": snr_mean, "weighted_snr_db_lo": snr_lo,
                     "weighted_snr_db_hi": snr_hi})
        print(f"  {name:8s} mel_l2={l2_mean:9.3f} [{l2_lo:.3f}, {l2_hi:.3f}]  "
              f"weighted_snr={snr_mean:6.2f} [{snr_lo:.2f}, {snr_hi:.2f}] dB")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "n_utterances", "n_samples",
                                          "mel_l2_mean", "mel_l2_lo", "mel_l2_hi",
                                          "weighted_snr_db_mean", "weighted_snr_db_lo",
                                          "weighted_snr_db_hi"])
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
