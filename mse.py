import argparse
import csv
import os

import numpy as np

from mpe_common import DEFAULT_P, N_SUBSET, SUBSET_SEED, OUT_DIR, load_subset
from synth_core import GAMMA, L, METHODS, synthesize_mpe

DEFAULT_K_VALUES = [5, 10, 20, 35]

OMP = METHODS["omp"]
ALL_VS_OMP_NAMES = [name for name in METHODS if name != "omp"]


def excitation(fn, h_w, target_w, K):
    m, locs = fn(h_w, target_w, K=K)
    e = np.zeros(len(target_w))
    e[locs] = m
    return e


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-subset", type=int, default=N_SUBSET)
    ap.add_argument("--seed", type=int, default=SUBSET_SEED)
    ap.add_argument("--p", type=int, default=DEFAULT_P)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean")
    ap.add_argument("--k-values", type=str,
                    default=",".join(str(k) for k in DEFAULT_K_VALUES))
    ap.add_argument("--algorithms", type=str, default=",".join(ALL_VS_OMP_NAMES),
                    help="comma-separated algorithms to compare against omp, e.g. mpog "
                         f"(default: all = {','.join(ALL_VS_OMP_NAMES)})")
    ap.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "mse.csv"))
    args = ap.parse_args()
    args.k_list = sorted({int(t) for t in args.k_values.split(",") if t.strip()})

    args.algo_list = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    unknown = [a for a in args.algo_list if a not in ALL_VS_OMP_NAMES]
    if not args.algo_list or unknown:
        ap.error(f"--algorithms must be from {ALL_VS_OMP_NAMES} (omp is always the "
                 f"reference, not a selectable comparand); got unknown: {unknown}")
    return args


def main():
    args = parse_args()

    print(f"Loading LibriSpeech {args.split} subset...")
    dataset = load_subset(args.n_subset, args.seed, split=args.split)
    print(f"  {len(dataset)} utterances")

    rows = []
    for K in args.k_list:
        # omp runs closed-loop per utterance at this K, which both fixes the
        # (h_w, target_w) every other algorithm below is scored on and gives
        # omp's own excitation for those frames -- omp is the reference, not
        # dense.
        frames = []
        for audio, _ in dataset:
            capture = []
            synthesize_mpe(OMP, audio, args.p, L, K, gamma=args.gamma, capture=capture)
            frames.extend(capture)

        se = {name: 0.0 for name in args.algo_list}
        n = 0
        for h_w, target_w in frames:
            e_omp = excitation(OMP, h_w, target_w, K)
            for name in args.algo_list:
                e = excitation(METHODS[name], h_w, target_w, K)
                se[name] += np.sum((e - e_omp) ** 2)
            n += len(target_w)

        print(f"  K={K:<3d} {len(frames)} frames")
        for name, total in se.items():
            rows.append({"K": K, "series": f"{name}_vs_omp", "mse": total / n})
            print(f"    {name}_vs_omp: {total / n:.6e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "series", "mse"])
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
