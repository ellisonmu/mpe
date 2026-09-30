"""
Multi-frame excitation comparison: run omp, mpog, lst, and gclst separately
on a 1600-sample (100 ms) region of a single LibriSpeech utterance, made up
of 10 consecutive L=160-sample (10 ms) coding frames -- the project's real
frame size (see synth_core.L). Each method runs its own real closed-loop
synthesis (synth_core.synthesize_mpe, frame-by-frame, each frame's LPC
history built from that method's own already-synthesized output, exactly as
in a real sweep), and the 10 frames' excitation signals are concatenated
into one 1600-sample signal per method.

The region is picked by finding where the audio starts: the first L=160
frame whose energy clears a fixed fraction of the utterance's peak frame
energy (i.e. leading silence is skipped), then taking that frame and the
next 9 frames after it.

Writes each method's excitation signal (one column per method, one row per
sample) to a CSV, and plots them stacked on a shared x-axis with the 10
sub-frame boundaries marked.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpe_common import OUT_DIR, load_subset, SUBSET_SEED, DEFAULT_P, DEFAULT_K
from synth_core import SR, GAMMA, L as FRAME_L, synthesize_mpe
from algorithms import omp, mpog, lst, gclst

N_SUBFRAMES = 10          # 10 * 160 = 1600 samples = 100 ms
REGION_L = FRAME_L * N_SUBFRAMES

METHODS = {"omp": omp, "mpog": mpog, "lst": lst, "gclst": gclst}


def find_audio_start(audio, frame_l=FRAME_L, n_subframes=N_SUBFRAMES, thresh=0.1):
    """Index (in frame_l units) of the first frame whose energy clears
    thresh * peak frame energy, i.e. where leading silence ends and the
    audio starts."""
    n_frames = len(audio) // frame_l
    if n_frames < n_subframes:
        raise ValueError(f"utterance too short for a {n_subframes}-frame region")
    energies = np.array([np.sum(audio[i*frame_l:(i+1)*frame_l] ** 2) for i in range(n_frames)])

    loud_thresh = thresh * energies.max()
    above = np.where(energies >= loud_thresh)[0]
    start = int(above[0]) if len(above) else 0
    return min(start, n_frames - n_subframes)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--utterance-index", type=int, default=0,
                    help="index into the usual LibriSpeech subset (default: 0)")
    ap.add_argument("--seed", type=int, default=SUBSET_SEED)
    ap.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean")
    ap.add_argument("--p", type=int, default=DEFAULT_P, help=f"LPC order (default: {DEFAULT_P})")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"pulses per L={FRAME_L} frame (default: {DEFAULT_K})")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--out-csv", type=str,
                    default=os.path.join(OUT_DIR, "single_frame_excitation.csv"))
    ap.add_argument("--out-png", type=str,
                    default=os.path.join(OUT_DIR, "single_frame_excitation.png"))
    return ap.parse_args()


def main():
    args = parse_args()

    dataset = load_subset(args.utterance_index + 1, args.seed, split=args.split)
    audio, transcript = dataset[args.utterance_index]
    audio = np.asarray(audio, dtype=float)

    region_start_frame = find_audio_start(audio)
    region_start = region_start_frame * FRAME_L
    region_end = region_start + REGION_L
    t0, t1 = region_start / SR * 1000, region_end / SR * 1000
    print(f"utterance {args.utterance_index}: \"{transcript[:60]}\"")
    print(f"selected region: frames {region_start_frame}-{region_start_frame + N_SUBFRAMES - 1} "
          f"({t0:.1f}-{t1:.1f} ms), p={args.p}, K={args.k} per {FRAME_L}-sample frame")

    # each method runs its own real closed-loop synthesis (own synthesized
    # history feeds each subsequent frame's LPC), from utterance start
    # through the end of the region -- frames only depend on the past, so
    # trimming the tail doesn't change anything before region_end.
    excitations = {}
    for name, fn in METHODS.items():
        exc_frames = []
        synthesize_mpe(fn, audio[:region_end], args.p, FRAME_L, args.k,
                       gamma=args.gamma, exc_capture=exc_frames)
        region_frames = exc_frames[region_start_frame:region_start_frame + N_SUBFRAMES]
        e = np.concatenate(region_frames)
        excitations[name] = e
        locs = np.nonzero(e)[0]
        print(f"  {name}: {len(locs)} pulses total across {N_SUBFRAMES} frames")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample"] + list(METHODS))
        for n in range(REGION_L):
            w.writerow([n] + [excitations[m][n] for m in METHODS])
    print(f"-> {args.out_csv}")

    fig, axes = plt.subplots(len(METHODS), 1, figsize=(12, 8), sharex=True)
    for ax, name in zip(axes, METHODS):
        ax.stem(np.arange(REGION_L), excitations[name], basefmt=" ")
        for fb in range(FRAME_L, REGION_L, FRAME_L):
            ax.axvline(fb, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sample (region-local, dashed lines = frame boundaries)")
    fig.suptitle(f"Excitation signals (10 x L={FRAME_L} frames, concatenated) -- "
                f"utterance {args.utterance_index}, {t0:.0f}-{t1:.0f} ms, "
                f"p={args.p}, K={args.k}")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"-> {args.out_png}")


if __name__ == "__main__":
    main()
