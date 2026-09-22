"""Jetson TX2 latency/throughput profiling for the MPE sparse-search algorithms.
"""

import argparse
import csv
import glob
import os
import platform
import statistics
import sys
import time

def _pin_threads(default="1"):
    n = default
    for i, tok in enumerate(sys.argv[1:]):
        if tok == "--threads" and i + 2 <= len(sys.argv) - 1:
            n = sys.argv[i + 2]
        elif tok.startswith("--threads="):
            n = tok.split("=", 1)[1]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    return str(n)

THREADS = _pin_threads()

import numpy as np
import soundfile as sf

from synth_core import (SR, FRAME_MS, L, GAMMA, DENSE_NAME, METHODS,
                        method_fn, synthesize_mpe)

DEFAULT_K_VALUES = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35]
DEFAULT_WARMUP = 200
DEFAULT_ITERS = 1000
DEFAULT_REPEATS = 5

STAGES_CSV = "jetson_mpe_profiling.csv"
SUMMARY_CSV = "jetson_mpe_profiling_summary.csv"

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(n_runs):
    if n_runs < 2:
        return float("nan")
    df = n_runs - 1
    return _T95[max(d for d in _T95 if d <= df)]


def get_cupy(device):
    if device != "gpu":
        return None
    try:
        import cupy as cp
    except ImportError:
        print("  cupy not importable; GPU timing unavailable", file=sys.stderr)
        return None
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:                       
        print(f"  cupy present but no usable device ({exc})", file=sys.stderr)
        return None
    return cp


def container_tag():
    """Short description of what this is running on, for the CSV."""
    bits = [platform.machine(), f"py{platform.python_version()}",
            f"numpy{np.__version__}", f"threads={THREADS}"]
    core = os.environ.get("OPENBLAS_CORETYPE")
    if core:
        bits.append(f"OPENBLAS_CORETYPE={core}")
    return " ".join(bits)

_RAIL_GLOBS = [
    "/sys/bus/i2c/drivers/ina3221x/*/iio:device0/in_power*_input",
    "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power*_input",
    "/sys/devices/platform/*/i2c*/*/iio:device0/in_power*_input",
]


def find_power_rails():
    for pattern in _RAIL_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            try:
                for h in hits:
                    with open(h) as f:
                        f.read()
                return hits
            except OSError:
                continue
    return []


def read_power_mw(rails):
    total = 0.0
    for path in rails:
        try:
            with open(path) as f:
                total += float(f.read().strip())
        except (OSError, ValueError):
            pass
    return total


def peak_rss_mb():
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024.0                      # Linux reports KB
    except (ImportError, AttributeError):
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")


def load_audio(patterns, max_seconds=None):
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits and os.path.exists(pat):
            hits = [pat]
        if not hits:
            raise FileNotFoundError(f"no audio matched: {pat}")
        paths.extend(hits)

    chunks = []
    for path in paths:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)                            # mono
        if sr != SR:
            from scipy.signal import resample_poly
            from fractions import Fraction
            frac = Fraction(SR, sr).limit_denominator()
            x = resample_poly(x, frac.numerator, frac.denominator)
        chunks.append(np.asarray(x, dtype=np.float64))

    audio = np.concatenate(chunks)
    if max_seconds is not None:
        audio = audio[:int(max_seconds * SR)]
    return audio, paths


def capture_search_inputs(algo, audio, p, K, gamma):
    capture = []
    t0 = time.perf_counter()
    synthesize_mpe(method_fn(algo), audio, p, L, K, gamma=gamma, capture=capture)
    synth_s = time.perf_counter() - t0
    if not capture:
        raise RuntimeError(f"{algo}: no frames captured -- audio too short or silent")
    return capture, synth_s


def time_search_cpu(fn, frames, K, warmup, iters, rails=()):
    n = len(frames)
    for i in range(warmup):
        h_w, target_w = frames[i % n]
        fn(h_w, target_w, K=K)

    power = []
    lat_us = np.empty(iters, dtype=float)
    for i in range(iters):
        h_w, target_w = frames[i % n]
        t0 = time.perf_counter()
        fn(h_w, target_w, K=K)
        lat_us[i] = (time.perf_counter() - t0) * 1e6
        if rails and i % 64 == 0:
            power.append(read_power_mw(rails))
    return lat_us, (float(np.mean(power)) if power else float("nan"))


def time_search_gpu(cp, fn, frames, K, warmup, iters, rails=()):
    n = len(frames)
    for i in range(warmup):
        h_w, target_w = frames[i % n]
        fn(h_w, target_w, K=K)
    cp.cuda.Device().synchronize()

    start, end = cp.cuda.Event(block=True), cp.cuda.Event(block=True)
    power = []
    lat_us = np.empty(iters, dtype=float)
    for i in range(iters):
        h_w, target_w = frames[i % n]
        start.record()
        fn(h_w, target_w, K=K)
        end.record()
        end.synchronize()
        lat_us[i] = cp.cuda.get_elapsed_time(start, end) * 1e3   # ms -> us
        if rails and i % 64 == 0:
            power.append(read_power_mw(rails))
    return lat_us, (float(np.mean(power)) if power else float("nan"))

FRAME_BUDGET_US = FRAME_MS * 1000.0


def summarize(lat_us, n_samples, power_mw=float("nan")):
    mean = float(np.mean(lat_us))
    p99 = float(np.percentile(lat_us, 99))
    return {
        "mean_latency_us": mean,
        "std_latency_us": float(np.std(lat_us, ddof=1)) if lat_us.size > 1 else 0.0,
        "p99_latency_us": p99,
        "min_latency_us": float(np.min(lat_us)),
        "throughput_samples_per_s": n_samples / (mean * 1e-6),
        "rtf": (n_samples / SR) / (mean * 1e-6),
        "search_pct_of_budget": 100.0 * mean / FRAME_BUDGET_US,
        "p99_pct_of_budget": 100.0 * p99 / FRAME_BUDGET_US,
        "power_mw": power_mw,
        # energy the search spends per frame; the figure that matters on battery
        "energy_mj_per_frame": (power_mw * 1e-3) * (mean * 1e-6) * 1e3,
    }

def write_csv(path, rows, fieldnames):
    if not rows:
        print(f"  nothing to write to {path}")
        return
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=fieldnames).to_csv(path, index=False)
    except ImportError:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    print(f"  -> {path} ({len(rows)} rows)")


STAGE_FIELDS = ["device", "algo", "K", "run_id", "n_frames", "n_samples",
                "mean_latency_us", "std_latency_us", "p99_latency_us",
                "min_latency_us", "throughput_samples_per_s", "rtf",
                "search_pct_of_budget", "p99_pct_of_budget",
                "power_mw", "energy_mj_per_frame",
                "frontend_us_per_frame", "search_share_of_frame",
                "total_pct_of_budget", "realtime_ok", "peak_rss_mb",
                "threads", "iters", "warmup", "p", "gamma", "frame_ms",
                "frame_budget_us", "container"]

SUMMARY_FIELDS = ["device", "algo", "K", "n_runs", "n_samples",
                  "mean_latency_us", "mean_latency_us_lo", "mean_latency_us_hi",
                  "sd_across_runs_us", "p99_latency_us",
                  "throughput_samples_per_s", "throughput_samples_per_s_lo",
                  "throughput_samples_per_s_hi", "rtf",
                  "search_pct_of_budget", "p99_pct_of_budget",
                  "power_mw", "energy_mj_per_frame",
                  "frontend_us_per_frame", "search_share_of_frame",
                  "total_pct_of_budget", "realtime_ok", "threads", "container"]


def summarize_across_runs(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["device"], r["algo"], r["K"]), []).append(r)

    out = []
    for (device, algo, K), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        means = [r["mean_latency_us"] for r in rs]
        n = len(means)
        mean = statistics.mean(means)
        sd = statistics.stdev(means) if n > 1 else 0.0
        half = t95(n) * sd / np.sqrt(n) if n > 1 else float("nan")
        lo, hi = (mean - half, mean + half) if n > 1 else (float("nan"), float("nan"))
        n_samples = rs[0]["n_samples"]
        out.append({
            "device": device, "algo": algo, "K": K, "n_runs": n,
            "n_samples": n_samples,
            "mean_latency_us": mean,
            "mean_latency_us_lo": lo, "mean_latency_us_hi": hi,
            "sd_across_runs_us": sd,
            "p99_latency_us": max(r["p99_latency_us"] for r in rs),
            "throughput_samples_per_s": n_samples / (mean * 1e-6),
            # a slower latency bound is a lower throughput bound, hence the swap
            "throughput_samples_per_s_lo": n_samples / (hi * 1e-6) if hi == hi else float("nan"),
            "throughput_samples_per_s_hi": n_samples / (lo * 1e-6) if lo == lo and lo > 0 else float("nan"),
            "rtf": (n_samples / SR) / (mean * 1e-6),
            "search_pct_of_budget": 100.0 * mean / FRAME_BUDGET_US,
            # worst run's tail, not an average of tails
            "p99_pct_of_budget": max(r["p99_pct_of_budget"] for r in rs),
            "power_mw": statistics.mean([r["power_mw"] for r in rs]),
            "energy_mj_per_frame": statistics.mean([r["energy_mj_per_frame"] for r in rs]),
            "frontend_us_per_frame": statistics.mean([r["frontend_us_per_frame"] for r in rs]),
            "search_share_of_frame": statistics.mean([r["search_share_of_frame"] for r in rs]),
            "total_pct_of_budget": statistics.mean([r["total_pct_of_budget"] for r in rs]),
            # real time only counts if EVERY run cleared the budget
            "realtime_ok": all(r["realtime_ok"] for r in rs),
            "threads": rs[0]["threads"],
            "container": rs[0]["container"],
        })
    return out

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", nargs="+", default=["input/input.wav"],
                    help="audio file(s) or glob(s); concatenated (default: input/input.wav)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="truncate the concatenated audio to this many seconds")
    ap.add_argument("--algos", type=str, default=",".join(METHODS),
                    help=f"comma-separated algorithms (default: {','.join(METHODS)}; "
                         f"add {DENSE_NAME} for the dense reference)")
    ap.add_argument("--k-values", type=str,
                    default=",".join(str(k) for k in DEFAULT_K_VALUES),
                    help=f"comma-separated pulse counts "
                         f"(default: {','.join(str(k) for k in DEFAULT_K_VALUES)})")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help=f"independent full runs, for the across-run CI "
                         f"(default: {DEFAULT_REPEATS})")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                    help=f"timed searches per (algo, K, run) (default: {DEFAULT_ITERS})")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help=f"untimed warmup searches (default: {DEFAULT_WARMUP})")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                    help="cpu = perf_counter; gpu = CUDA events (needs CuPy) (default: cpu)")
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS threads; 1 models a single-core edge budget and "
                         "keeps OMP's dense solves from being flattered (default: 1)")
    ap.add_argument("--no-power", action="store_true",
                    help="skip Jetson INA3221 power sampling even if rails are found")
    ap.add_argument("--p", type=int, default=16, help="LPC order (default: 16)")
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help=f"perceptual weighting factor (default: {GAMMA})")
    ap.add_argument("--out-prefix", type=str, default="jetson_mpe_profiling",
                    help="output CSV prefix (default: jetson_mpe_profiling)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny warmup/iters/repeats for a quick desktop sanity check")
    args = ap.parse_args()

    if args.smoke:
        args.warmup, args.iters, args.repeats = 5, 20, 2
        args.max_seconds = args.max_seconds or 1.0

    args.algo_list = [a.strip() for a in args.algos.split(",") if a.strip()]
    known = set(METHODS) | {DENSE_NAME}
    unknown = [a for a in args.algo_list if a not in known]
    if unknown:
        ap.error(f"unknown --algos: {unknown}; choose from {sorted(known)}")
    try:
        args.k_list = sorted({int(t) for t in args.k_values.split(",") if t.strip()})
    except ValueError:
        ap.error("--k-values must be a comma-separated list of integers")
    if not args.k_list or args.k_list[0] < 1:
        ap.error("--k-values must contain positive integers")
    return args


def main():
    args = parse_args()
    cp = get_cupy(args.device)
    device = "gpu" if cp is not None else "cpu"
    if args.device == "gpu" and cp is None:
        print("  falling back to CPU timing")
    container = container_tag()

    if platform.machine().startswith(("aarch64", "arm")) and not os.environ.get("OPENBLAS_CORETYPE"):
        print("  WARNING: OPENBLAS_CORETYPE is unset on an ARM host; "
              "NumPy can SIGILL on the TX2. export OPENBLAS_CORETYPE=ARMV8",
              file=sys.stderr)

    rails = [] if args.no_power else find_power_rails()
    if rails:
        print(f"Power rails: {len(rails)} INA3221 channel(s), "
              f"idle {read_power_mw(rails):.0f} mW")
    else:
        print("Power rails: none found (not a Jetson, or --no-power); "
              "energy columns will be nan")

    print(f"Device: {device}  ({container})")
    print(f"Frame budget: {FRAME_BUDGET_US:.0f} us ({FRAME_MS} ms frames)")
    print(f"Loading audio: {args.audio}")
    audio, paths = load_audio(args.audio, args.max_seconds)
    print(f"  {len(paths)} file(s), {len(audio)/SR:.1f} s, {len(audio)//L} frames of {FRAME_MS} ms")

    stage_rows = []
    cells_total = len(args.algo_list) * len(args.k_list)
    cell_i = 0
    t_start = time.perf_counter()
    for algo in args.algo_list:
        for K in args.k_list:
            cell_i += 1
            frames, synth_s = capture_search_inputs(algo, audio, args.p, K, args.gamma)
            n_frames = len(frames)
            fn = method_fn(algo)
            k_eff = 0 if algo == DENSE_NAME else K

            for run in range(1, args.repeats + 1):
                if cp is not None:
                    lat_us, pw = time_search_gpu(cp, fn, frames, k_eff,
                                                 args.warmup, args.iters, rails)
                else:
                    lat_us, pw = time_search_cpu(fn, frames, k_eff,
                                                 args.warmup, args.iters, rails)
                stats = summarize(lat_us, L, pw)

                search_s = stats["mean_latency_us"] * 1e-6 * n_frames
                frontend_us = max(0.0, (synth_s - search_s)) / n_frames * 1e6
                frame_us = stats["mean_latency_us"] + frontend_us

                stage_rows.append({
                    "device": device, "algo": algo, "K": K, "run_id": run,
                    "n_frames": n_frames, "n_samples": L,
                    **stats,
                    "frontend_us_per_frame": frontend_us,
                    "search_share_of_frame": (stats["mean_latency_us"] / frame_us
                                              if frame_us > 0 else float("nan")),
                    "total_pct_of_budget": 100.0 * frame_us / FRAME_BUDGET_US,
                    "realtime_ok": frame_us < FRAME_BUDGET_US,
                    "peak_rss_mb": peak_rss_mb(),
                    "threads": THREADS,
                    "iters": args.iters, "warmup": args.warmup,
                    "p": args.p, "gamma": args.gamma, "frame_ms": FRAME_MS,
                    "frame_budget_us": FRAME_BUDGET_US,
                    "container": container,
                })
                e = stats["energy_mj_per_frame"]
                print(f"  {algo:10s} K={K:<3d} run {run}/{args.repeats}  "
                      f"{stats['mean_latency_us']:8.1f} us/frame  "
                      f"({stats['search_pct_of_budget']:5.1f}% budget)  "
                      f"P99 {stats['p99_latency_us']:8.1f}"
                      + (f"  {e:7.3f} mJ" if e == e else ""), flush=True)

            done = time.perf_counter() - t_start
            eta = done / cell_i * (cells_total - cell_i)
            print(f"    [cell {cell_i}/{cells_total}  elapsed {done/60:.1f} min  "
                  f"eta {eta/60:.1f} min]", flush=True)

    print("\nWriting results...")
    write_csv(f"{args.out_prefix}.csv", stage_rows, STAGE_FIELDS)
    summary = summarize_across_runs(stage_rows)
    write_csv(f"{args.out_prefix}_summary.csv", summary, SUMMARY_FIELDS)

    print(f"\nSummary (search us/frame, 95% CI across runs; "
          f"budget {FRAME_BUDGET_US:.0f} us/frame):")
    for r in summary:
        ci = (f"[{r['mean_latency_us_lo']:.1f}, {r['mean_latency_us_hi']:.1f}]"
              if r["mean_latency_us_hi"] == r["mean_latency_us_hi"] else "(n/a, 1 run)")
        e = r["energy_mj_per_frame"]
        print(f"  {r['algo']:10s} K={r['K']:<3d} {r['mean_latency_us']:8.1f} {ci:>20s}  "
              f"search {r['search_pct_of_budget']:5.1f}%  "
              f"total {r['total_pct_of_budget']:6.1f}%  "
              f"{'RT-OK' if r['realtime_ok'] else 'OVER '}"
              + (f"  {e:7.3f} mJ" if e == e else ""))


if __name__ == "__main__":
    main()
