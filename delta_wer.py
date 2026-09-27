"""
Paired per-utterance WER deltas between algorithm pairs, from the
--per-utterance CSV that pulse_sweep.py produces (K,p,algorithm,utterance,wer).

For each of (omp, gclst), (omp, mpog), (mpog, gclst): for every K present in
the file, pairs up each utterance's WER under both algorithms (delta = wer_a -
wer_b) and reports the mean with a 95% bootstrap CI over utterances, plus one
more row pooling every K's utterances together ("all"). `significant` is
true when the 95% CI excludes 0.

Also writes delta_wer_summary.csv: just the pooled ("all") row per pair, with
a plain-language `verdict` (e.g. "significant, gclst beats mpog") -- the
condensed form of the report, not the full per-K breakdown.
"""

import argparse
import csv
import os
from collections import defaultdict

from mpe_common import OUT_DIR, bootstrap_ci

PAIRS = [("omp", "gclst"), ("omp", "mpog"), ("mpog", "gclst")]


def load_per_utterance(path):
    # wer[K][algorithm][utterance] = wer
    wer = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            K = int(row["K"])
            wer[K][row["algorithm"]][row["utterance"]] = float(row["wer"])
    return wer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path",
                    default=os.path.join(OUT_DIR, "pulse_sweep_per_utterance.csv"))
    ap.add_argument("--out", dest="out_path",
                    default=os.path.join(OUT_DIR, "delta_wer.csv"))
    args = ap.parse_args()

    wer = load_per_utterance(args.in_path)
    K_values = sorted(wer)

    rows = []
    summary_rows = []
    for a, b in PAIRS:
        pooled = []
        for K in K_values:
            wa, wb = wer[K].get(a), wer[K].get(b)
            if not wa or not wb:
                continue
            common = sorted(set(wa) & set(wb))
            deltas = [wa[u] - wb[u] for u in common]
            pooled.extend(deltas)
            mean, lo, hi = bootstrap_ci(deltas)
            sig = (lo > 0) or (hi < 0)
            rows.append({"pair": f"{a}-{b}", "K": K, "n": len(deltas),
                         "mean_delta": mean, "ci_lo": lo, "ci_hi": hi,
                         "significant": sig})
            print(f"{a} - {b}, K={K:>3} (n={len(deltas):>4}): "
                  f"mean={mean:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
                  f"{'  *' if sig else ''}")
        mean, lo, hi = bootstrap_ci(pooled)
        sig = (lo > 0) or (hi < 0)
        rows.append({"pair": f"{a}-{b}", "K": "all", "n": len(pooled),
                     "mean_delta": mean, "ci_lo": lo, "ci_hi": hi,
                     "significant": sig})
        print(f"{a} - {b}, K=all (n={len(pooled):>4}): "
              f"mean={mean:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
              f"{'  *' if sig else ''}")
        print()

        if not sig:
            verdict = "not significant (crosses 0)"
        elif mean < 0:
            verdict = f"significant, {a} beats {b}"
        else:
            verdict = f"significant, {b} beats {a}"
        summary_rows.append({"pair": f"{a}-{b}", "n": len(pooled),
                             "mean_delta": mean, "ci_lo": lo, "ci_hi": hi,
                             "significant": sig, "verdict": verdict})

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "K", "n", "mean_delta",
                                          "ci_lo", "ci_hi", "significant"])
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out_path}")

    base, ext = os.path.splitext(args.out_path)
    summary_path = f"{base}_summary{ext}"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "n", "mean_delta", "ci_lo",
                                          "ci_hi", "significant", "verdict"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"-> {summary_path}")
    for r in summary_rows:
        print(f"  {r['pair']}: mean={r['mean_delta']:+.5f}, "
              f"CI [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}] -- {r['verdict']}")


if __name__ == "__main__":
    main()
