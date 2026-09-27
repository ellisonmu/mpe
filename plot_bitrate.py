"""Quick static plot of experiment_results/pulse_sweep_bitrate.csv: coded
bitrate vs nominal K, one line per algorithm (see pulse_sweep.py measure_bitrate).
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = os.path.join("experiment_results", "pulse_sweep_bitrate.csv")
OUT_PNG = os.path.join("experiment_results", "pulse_sweep_bitrate_plot.png")

COLORS = {
    "omp": "#2a78d6",
    "lst": "#eb6834",
    "gclst": "#1baf7a",
    "mp": "#eda100",
    "comp": "#e87ba4",
    "mpog": "#4a3aa7",
}


def main():
    by_algo = defaultdict(list)
    with open(IN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            name = row["algorithm"]
            if name not in COLORS:
                continue
            by_algo[name].append((int(row["K"]), float(row["bitrate_bps"])))

    for name in by_algo:
        by_algo[name].sort()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for name, color in COLORS.items():
        pts = by_algo.get(name)
        if not pts:
            continue
        ks = [p[0] for p in pts]
        vals = [p[1] / 1000 for p in pts]
        ax.plot(ks, vals, marker="o", markersize=4, linewidth=2, color=color, label=name)

    ax.set_xlabel("K (nominal pulse count)")
    ax.set_ylabel("Bitrate (kbps)")
    ax.set_title("Coded bitrate vs K")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
