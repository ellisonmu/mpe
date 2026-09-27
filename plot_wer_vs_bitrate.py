"""Quick static plot of WER vs coded bitrate, one line per algorithm, no error
bars -- joins experiment_results/pulse_sweep.csv (WER per K) with
experiment_results/pulse_sweep_bitrate.csv (bitrate per K) on (algorithm, K).
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WER_CSV = os.path.join("experiment_results", "pulse_sweep.csv")
BITRATE_CSV = os.path.join("experiment_results", "pulse_sweep_bitrate.csv")
OUT_PNG = os.path.join("experiment_results", "wer_vs_bitrate_plot.png")

COLORS = {
    "omp": "#2a78d6",
    "lst": "#eb6834",
    "gclst": "#1baf7a",
    "mp": "#eda100",
    "comp": "#e87ba4",
    "mpog": "#4a3aa7",
}


def load_wer(path):
    """{(algorithm, K): wer} from the wide per-K checkpoint CSV, ignoring
    the _lo/_hi confidence-interval columns -- this plot carries no error bars."""
    with open(path, newline="") as f:
        lines = f.readlines()
    reader = csv.DictReader(lines[2:])  # skip the #config / #baseline_wer header lines
    wer = {}
    for row in reader:
        K = int(row["K"])
        for name in COLORS:
            col = f"{name}_wer"
            if col in row and row[col] not in (None, ""):
                wer[(name, K)] = float(row[col])
    return wer


def load_bitrate(path):
    """{(algorithm, K): bitrate_bps}."""
    bitrate = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            bitrate[(row["algorithm"], int(row["K"]))] = float(row["bitrate_bps"])
    return bitrate


def main():
    wer = load_wer(WER_CSV)
    bitrate = load_bitrate(BITRATE_CSV)

    by_algo = defaultdict(list)
    for (name, K), bps in bitrate.items():
        if (name, K) in wer:
            by_algo[name].append((bps, wer[(name, K)]))

    for name in by_algo:
        by_algo[name].sort()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for name, color in COLORS.items():
        pts = by_algo.get(name)
        if not pts:
            continue
        xs = [p[0] / 1000 for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=color, label=name)

    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("WER")
    ax.set_title("WER vs bitrate")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
