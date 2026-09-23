"""Quick static plot of experiment_results/mse.csv: MSE vs K for lst, gclst,
mp, and comp, each against omp (the reference; see mse.py)."""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = os.path.join("experiment_results", "mse.csv")
OUT_PNG = os.path.join("experiment_results", "mse_plot.png")

COLORS = {
    "lst_vs_omp": "#eb6834",
    "gclst_vs_omp": "#1baf7a",
    "mp_vs_omp": "#eda100",
    "comp_vs_omp": "#e87ba4",
}


def main():
    by_series = defaultdict(list)
    with open(IN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            name = row["series"]
            if name not in COLORS:
                continue
            by_series[name].append((int(row["K"]), float(row["mse"])))

    for name in by_series:
        by_series[name].sort()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for name, color in COLORS.items():
        pts = by_series.get(name)
        if not pts:
            continue
        ks = [p[0] for p in pts]
        vals = [p[1] * 1e5 for p in pts]
        ax.plot(ks, vals, marker="o", markersize=4, linewidth=2, color=color, label=name)

    ax.set_xlabel("K (pulse count)")
    ax.set_ylabel("MSE vs omp (x1e-5)")
    ax.set_title("Excitation MSE vs omp, by K")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
