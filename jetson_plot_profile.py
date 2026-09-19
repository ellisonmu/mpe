import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_core import FRAME_MS          # torch-free; safe to import for this

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FRAME_BUDGET_US = FRAME_MS * 1000.0

# Okabe-Ito colorblind-safe qualitative palette (Okabe & Ito, 2008) -- distinct
# under protanopia/deuteranopia without relying on a red-vs-green contrast.
# Markers differ per algorithm too, so identity never rests on color alone.
ALGO_STYLE = {
    "lst":      ("LST",              "#0072B2", "o"),
    "mp":       ("MP",               "#E69F00", "s"),
    "gclst":    ("GCLST",            "#009E73", "^"),
    "comp":     ("OMP (Cholesky)",   "#CC79A7", "D"),
    "omp":      ("OMP (direct)",     "#D55E00", "v"),
    "dense_lpc": ("Dense (reference)", "#56B4E9", "P"),
}
_FALLBACK_COLORS = ["#000000", "#999999", "#8B4513", "#4B0082"]

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
LW, MS = 1.8, 6.0


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def algo_style(name, idx=0):
    if name in ALGO_STYLE:
        return ALGO_STYLE[name]
    return (name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)], "x")


def _num(v):
    if v >= 1:
        return f"{v:,.0f}"
    return f"{v:g}"


def log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: _num(v)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def k_axis(ax, k_values):
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _to_num(x):
    if x in (None, "", "nan", "None"):
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return x


def _read_rows(path):
    with open(path, newline="") as f:
        return [{k: _to_num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_summary(paths):
    """Load one or more summary CSVs, keyed by (device, algo, K).

    Later paths override earlier ones on a key collision, so a small
    single-algorithm re-profile (e.g. after fixing just `mp`) can be layered
    on top of a full prior run without re-running everything: pass the old
    full summary first, the new partial one last, and only the algorithms
    actually present in the new file get replaced -- everything else (and its
    original CIs) is carried through untouched.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged = {}
    for path in paths:
        rows = _read_rows(path)
        if not rows:
            print(f"  warning: no rows in {path}, skipping")
            continue
        for r in rows:
            merged[(r["device"], r["algo"], r["K"])] = r
        print(f"  loaded {len(rows)} rows from {path}")
    if not merged:
        raise SystemExit(f"no rows in any of: {paths}")
    rows = list(merged.values())
    devices = sorted({r["device"] for r in rows})
    algos = sorted({r["algo"] for r in rows},
                   key=lambda a: list(ALGO_STYLE).index(a) if a in ALGO_STYLE else 99)
    return rows, devices, algos


def series_for(rows, device, algo, ycol):
    sel = sorted((r for r in rows if r["device"] == device and r["algo"] == algo),
                 key=lambda r: r["K"])
    if not sel:
        return None
    x = np.array([r["K"] for r in sel], dtype=float)
    y = np.array([r[ycol] for r in sel], dtype=float)
    lo_col, hi_col = f"{ycol}_lo", f"{ycol}_hi"
    lo = np.array([r.get(lo_col, float("nan")) for r in sel], dtype=float)
    hi = np.array([r.get(hi_col, float("nan")) for r in sel], dtype=float)
    return x, y, lo, hi, sel


def draw_series(ax, rows, devices, algos, ycol, with_band=True):
    for i, algo in enumerate(algos):
        label, color, marker = algo_style(algo, i)
        for device in devices:
            got = series_for(rows, device, algo, ycol)
            if got is None:
                continue
            x, y, lo, hi = got[0], got[1], got[2], got[3]
            if with_band and not np.isnan(lo).all():
                yerr = np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)])
                ax.errorbar(x, y, yerr=yerr, color=color, lw=LW, marker=marker,
                            ms=MS, mec="white", mew=1.0, label=label,
                            capsize=3, zorder=3)
            else:
                ax.plot(x, y, color=color, lw=LW, marker=marker, ms=MS,
                        mec="white", mew=1.0, label=label, zorder=3)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.png")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_search_latency(rows, devices, algos):
    """Search-only latency vs K, log-log, 95% CI across runs."""
    k_values = sorted({r["K"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 5.2))
    draw_series(ax, rows, devices, algos, "mean_latency_us")
    log_y(ax)
    k_axis(ax, k_values)
    ax.set_xlabel("Pulse count K")
    ax.set_ylabel("Search latency (us/frame)")
    ax.set_title("Sparse-search latency vs K\n"
                 f"{devices[0]}, single core, 95% CI across runs", loc="left")
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, "jetson_search_latency")


def fig_search_budget(rows, devices, algos):
    """Search time as % of the frame budget -- feasibility of the search alone."""
    k_values = sorted({r["K"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 5.2))
    draw_series(ax, rows, devices, algos, "search_pct_of_budget", with_band=False)
    ax.axhline(100, color=INK, lw=1.2, ls=(0, (4, 2)), zorder=2)
    ax.annotate("100% of 10 ms frame budget", (k_values[0], 100),
               xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)
    log_y(ax)
    k_axis(ax, k_values)
    ax.set_xlabel("Pulse count K")
    ax.set_ylabel("Search time (% of frame budget)")
    ax.set_title("Search cost relative to the 10 ms frame budget", loc="left")
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, "jetson_search_budget")


def fig_total_budget(rows, devices, algos):
    """Front end + search as % of budget -- the real-time feasibility question."""
    k_values = sorted({r["K"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 5.2))
    draw_series(ax, rows, devices, algos, "total_pct_of_budget", with_band=False)
    ax.axhline(100, color=INK, lw=1.2, ls=(0, (4, 2)), zorder=2)
    ax.annotate("100% -- real-time boundary", (k_values[0], 100),
               xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)

    frontend_pct = [100.0 * r["frontend_us_per_frame"] / FRAME_BUDGET_US
                    for r in rows if not np.isnan(r["frontend_us_per_frame"])]
    if frontend_pct:
        fe = float(np.mean(frontend_pct))
        ax.axhline(fe, color=MUTED, lw=1.2, ls=(0, (1, 1.5)), zorder=2)
        ax.annotate(f"front end alone: ~{fe:.0f}%", (k_values[-1], fe),
                   xytext=(-4, -10), textcoords="offset points", ha="right",
                   fontsize=8, color=INK_2)

    log_y(ax)
    k_axis(ax, k_values)
    ax.set_xlabel("Pulse count K")
    ax.set_ylabel("Total frame time (% of frame budget)")
    ax.set_title("Front end + search vs the 10 ms frame budget\n"
                 "(front end alone already exceeds it -- see log)", loc="left")
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, "jetson_total_budget")


def fig_energy(rows, devices, algos):
    """Energy per frame vs K, from the TX2's INA3221 rails."""
    if all(np.isnan(r.get("energy_mj_per_frame", float("nan"))) for r in rows):
        print("  no power data (energy_mj_per_frame all nan) -- skipping energy figure")
        return
    k_values = sorted({r["K"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 5.2))
    draw_series(ax, rows, devices, algos, "energy_mj_per_frame", with_band=False)
    log_y(ax)
    k_axis(ax, k_values)
    ax.set_xlabel("Pulse count K")
    ax.set_ylabel("Search energy (mJ/frame)")
    ax.set_title("Search energy vs K (board power, INA3221)", loc="left")
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, "jetson_energy")


def print_findings(rows, devices, algos):
    """Console summary of the two headline results, so they don't get lost in
    a CSV: whether the front end alone clears the frame budget, and where (if
    anywhere) OMP-Cholesky overtakes direct OMP."""
    print("\n=== Key findings ===")
    fe = [r["frontend_us_per_frame"] for r in rows if not np.isnan(r["frontend_us_per_frame"])]
    if fe:
        fe_mean = float(np.mean(fe))
        print(f"  front end alone: {fe_mean:,.0f} us/frame "
              f"({100*fe_mean/FRAME_BUDGET_US:.0f}% of a {FRAME_BUDGET_US:.0f} us frame budget)")

    if "omp" in algos and "comp" in algos:
        omp = {r["K"]: r["mean_latency_us"] for r in rows if r["algo"] == "omp"}
        comp = {r["K"]: r["mean_latency_us"] for r in rows if r["algo"] == "comp"}
        common = sorted(set(omp) & set(comp))
        crossover = None
        for k in common:
            if comp[k] < omp[k]:
                crossover = k
                break
        if crossover is not None:
            print(f"  OMP-Cholesky overtakes direct OMP by K={crossover:.0f} "
                  f"(comp {comp[crossover]:,.0f} us vs omp {omp[crossover]:,.0f} us)")
        else:
            print("  direct OMP stayed faster than OMP-Cholesky across all tested K")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, nargs="+",
                    default=[ROOT / "jetson_mpe_profiling_summary.csv"],
                    help="one or more summary CSVs; later files override "
                         "earlier ones on (device, algo, K) collisions, so a "
                         "small re-profile can be layered on a full prior run "
                         "(default: jetson_mpe_profiling_summary.csv)")
    args = ap.parse_args()

    style()
    rows, devices, algos = load_summary(args.summary)
    print(f"Loaded {len(rows)} rows: devices={devices}  algos={algos}")

    fig_search_latency(rows, devices, algos)
    fig_search_budget(rows, devices, algos)
    fig_total_budget(rows, devices, algos)
    fig_energy(rows, devices, algos)
    print_findings(rows, devices, algos)


if __name__ == "__main__":
    main()
