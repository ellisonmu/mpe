import argparse
import csv
import os

import numpy as np

from mpe_common import DEFAULT_P, N_SUBSET, SUBSET_SEED, OUT_DIR, load_subset
from synth_core import GAMMA, L, METHODS, method_fn, synthesize_mpe

DEFAULT_K_VALUES = [5, 10, 20, 30]


def snr_db(x, y):
    """SNR in dB between clean x and coded y (same length). inf when the
    error is exactly zero; -inf (rather than a divide-by-zero error) when x
    itself is silent."""
    noise = np.sum((x - y) ** 2)
    signal = np.sum(x ** 2)
    if noise <= 0.0:
        return float("inf")
    if signal <= 0.0:
        return float("-inf")
    return 10.0 * np.log10(signal / noise)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-subset", type=int, default=N_SUBSET,
                    help=f"number of LibriSpeech utterances to use (default: {N_SUBSET})")
    ap.add_argument("--seed", type=int, default=SUBSET_SEED,
                    help=f"random seed for subset selection (default: {SUBSET_SEED})")
    ap.add_argument("--p", type=int, default=DEFAULT_P,
                    help=f"LPC order held fixed during the sweep (default: {DEFAULT_P})")
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help=f"perceptual error-weighting factor (default: {GAMMA})")
    ap.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean",
                    help="LibriSpeech split (default: test-clean)")
    ap.add_argument("--k-values", type=str,
                    default=",".join(str(k) for k in DEFAULT_K_VALUES),
                    help="comma-separated pulse counts to sweep "
                         f"(default: {','.join(str(k) for k in DEFAULT_K_VALUES)})")
    ap.add_argument("--algorithms", type=str, default=",".join(METHODS),
                    help="comma-separated algorithms to measure, e.g. mp,comp "
                         f"(default: all = {','.join(METHODS)})")
    ap.add_argument("--quantize-lsf", action=argparse.BooleanOptionalAction, default=True,
                    help="quantize the LSFs (default: on; --no-quantize-lsf uses the "
                         "unquantized LPC filter)")
    ap.add_argument("--quantize-exc", action=argparse.BooleanOptionalAction, default=True,
                    help="quantize the excitation pulse amplitudes (default: on; "
                         "--no-quantize-exc keeps the exact solved amplitudes)")
    ap.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "snr.csv"))
    args = ap.parse_args()

    try:
        args.k_list = sorted({int(t) for t in args.k_values.split(",") if t.strip()})
    except ValueError:
        ap.error("--k-values must be a comma-separated list of integers")
    if not args.k_list or args.k_list[0] < 1:
        ap.error("--k-values must contain positive integers")

    args.algo_list = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    unknown = [a for a in args.algo_list if a not in METHODS]
    if not args.algo_list or unknown:
        ap.error(f"--algorithms must be from {list(METHODS)}; got unknown: {unknown}")
    return args


def main():
    args = parse_args()

    print(f"Loading LibriSpeech {args.split} subset...")
    dataset = load_subset(args.n_subset, args.seed, split=args.split)
    print(f"  {len(dataset)} utterances")
    print(f"  quantize_lsf={args.quantize_lsf}, quantize_exc={args.quantize_exc}")

    rows = []
    for K in args.k_list:
        for algo in args.algo_list:
            fn = method_fn(algo)
            snrs = []
            for audio, _ in dataset:
                x = np.asarray(audio, dtype=float)
                n_frames = len(x) // L
                x = x[:n_frames * L]   # synthesize_mpe only fills whole frames
                if len(x) == 0:
                    continue
                y = synthesize_mpe(fn, x, args.p, L, K, gamma=args.gamma,
                                   quantize_lsf=args.quantize_lsf,
                                   quantize_exc=args.quantize_exc)
                snrs.append(snr_db(x, y))

            finite = [v for v in snrs if np.isfinite(v)]
            mean_snr = float(np.mean(finite)) if finite else float("nan")
            rows.append({"K": K, "algorithm": algo, "snr_db": mean_snr,
                        "n_utterances": len(snrs)})
            print(f"  K={K:<3d} {algo:8s} SNR={mean_snr:6.2f} dB  "
                 f"({len(finite)}/{len(snrs)} utterances finite)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "algorithm", "snr_db", "n_utterances"])
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
