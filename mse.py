"""Pulse sweep scored by excitation MSE instead of WER.

For each K, every algorithm's excitation is compared against the dense
(full-density) least-squares excitation on identical search inputs. Also
reports the MSE between omp's and gclst's excitations.
"""

import argparse
import csv
import os

import numpy as np

from mpe_common import DEFAULT_P, N_SUBSET, SUBSET_SEED, OUT_DIR, load_subset
from synth_core import GAMMA, L, METHODS, dense, synthesize_mpe

DEFAULT_K_VALUES = [1, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35]


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
    ap.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "mse.csv"))
    args = ap.parse_args()
    args.k_list = sorted({int(t) for t in args.k_values.split(",") if t.strip()})
    return args


def main():
    args = parse_args()

    print(f"Loading LibriSpeech {args.split} subset...")
    dataset = load_subset(args.n_subset, args.seed, split=args.split)
    print(f"  {len(dataset)} utterances")

    # One dense pass per utterance captures the (h_w, target_w) each frame
    # presents, so every algorithm below is scored on identical inputs.
    frames = []
    for audio, _ in dataset:
        capture = []
        synthesize_mpe(dense, audio, args.p, L, 0, gamma=args.gamma, capture=capture)
        frames.extend(capture)
    print(f"  {len(frames)} frames")

    rows = []
    for K in args.k_list:
        se = {name: 0.0 for name in METHODS}
        se["omp_vs_gclst"] = 0.0
        n = 0
        for h_w, target_w in frames:
            e_dense = excitation(dense, h_w, target_w, None)
            es = {name: excitation(fn, h_w, target_w, K) for name, fn in METHODS.items()}
            for name, e in es.items():
                se[name] += np.sum((e - e_dense) ** 2)
            se["omp_vs_gclst"] += np.sum((es["omp"] - es["gclst"]) ** 2)
            n += len(target_w)

        for name, total in se.items():
            rows.append({"K": K, "series": name, "mse": total / n})
            print(f"  K={K:<3d} {name:15s} {total / n:.6e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "series", "mse"])
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
