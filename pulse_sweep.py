"""
Pulse-count (sparsity) sweep: fix LPC order p, vary the number of pulses K,
for omp/lst/gclst plus a dense (non-sparse) excitation reference run through
the same closed-loop pipeline. See mpe_common.py / synth_core.py.

Results are checkpointed per K value, so re-running skips any K already done.

Optionally (--bitrate, on by default) also measures the coded bitrate per
(algorithm, K):

    R(K) = (fs/L) * (p*b2 + K_eff*b1 + ceil(log2(C(L, K_eff))))

p*b2 is the LSF side info, K_eff*b1 the pulse-magnitude side info, and the
ceil(log2(C(L, K_eff))) term the combinatorial position code. K_eff is the
*actual* mean count of nonzero excitation samples per frame for that
algorithm/K -- not the nominal K -- since quantize_excitation can round a
small pulse to zero and some algorithms (mpog with repeats, comp's early
break) can settle on fewer than K distinct active positions. It's read off
the same synthesis pass run_sweep already does for WER (see bitrate_out in
mpe_common.run_sweep) rather than a second one, so turning it on is nearly
free -- except it means a --no-resume-free rerun of an already-checkpointed
K contributes no bitrate row for that K, since it isn't re-synthesized.
"""

import argparse
import csv
import os

from mpe_common import (
    DEFAULT_P, SUBSET_SEED, N_SUBSET, GAMMA, OUT_DIR,
    METHODS, load_subset, load_whisper, get_or_compute_baseline,
    load_checkpoint, save_checkpoint, sweep_config,
    run_sweep, plot_sweep, default_workers, set_fp16,
)

DEFAULT_K_VALUES = [1, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35]


def save_bitrate(out_name, rows):
    path = os.path.join(OUT_DIR, f"{out_name}_bitrate.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "K", "k_eff", "bitrate_bps"])
        w.writeheader()
        w.writerows(rows)
    return path


def save_per_utterance(out_name, rows):
    path = os.path.join(OUT_DIR, f"{out_name}_per_utterance.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "p", "algorithm", "utterance", "wer"])
        w.writeheader()
        w.writerows(rows)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subset", type=int, default=N_SUBSET,
                         help=f"number of LibriSpeech utterances to use (default: {N_SUBSET})")
    parser.add_argument("--seed", type=int, default=SUBSET_SEED,
                         help=f"random seed for subset selection (default: {SUBSET_SEED})")
    parser.add_argument("--p", type=int, default=DEFAULT_P,
                         help=f"LPC order held fixed during the sweep (default: {DEFAULT_P})")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                         help=f"perceptual error-weighting factor, W(z)=A(z)/A(z/gamma) "
                              f"(default: {GAMMA})")
    parser.add_argument("--k-values", type=str,
                         default=",".join(str(k) for k in DEFAULT_K_VALUES),
                         help="comma-separated pulse counts to sweep "
                              f"(default: {','.join(str(k) for k in DEFAULT_K_VALUES)})")
    parser.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean",
                         help="LibriSpeech split (default: test-clean). test-other results "
                              "are saved to their own file, pulse_sweep_test-other.csv")
    parser.add_argument("--algorithms", type=str, default=",".join(METHODS),
                         help="comma-separated algorithms to run, e.g. mp,comp "
                              f"(default: all = {','.join(METHODS)}). Algorithms already "
                              "in the checkpoint are skipped, new ones are merged in")
    parser.add_argument("--workers", type=int, default=None,
                         help=f"parallel synthesis processes (default: {default_workers()})")
    parser.add_argument("--no-resume", action="store_true",
                         help="ignore any existing checkpoint and recompute every K")
    parser.add_argument("--bitrate", action=argparse.BooleanOptionalAction, default=True,
                         help="measure coded bitrate per (algorithm, K) alongside WER "
                              "(default: on; --no-bitrate skips it and its extra "
                              "synthesis pass)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                         help="Whisper transcription precision (default: auto -- fp16 iff "
                              "a CUDA GPU is available; --no-fp16 forces fp32 even on GPU, "
                              "--fp16 forces fp16 even without a GPU, where it will be slow "
                              "or unsupported)")
    parser.add_argument("--per-utterance", action="store_true",
                         help="also save each utterance's individual WER to "
                              "<out_name>_per_utterance.csv (off by default -- one row "
                              "per algorithm per utterance per K, so this can get large)")
    args = parser.parse_args()

    try:
        args.k_list = sorted({int(tok) for tok in args.k_values.split(",") if tok.strip()})
    except ValueError:
        parser.error("--k-values must be a comma-separated list of integers")
    if not args.k_list or args.k_list[0] < 1:
        parser.error("--k-values must contain positive integers")

    args.algo_list = [tok.strip() for tok in args.algorithms.split(",") if tok.strip()]
    unknown = [a for a in args.algo_list if a not in METHODS]
    if not args.algo_list or unknown:
        parser.error(f"--algorithms must be from {list(METHODS)}; got unknown: {unknown}")
    return args


def main():
    args = parse_args()
    set_fp16(args.fp16)
    # test-clean keeps the original filename so existing results are reused
    out_name = "pulse_sweep" if args.split == "test-clean" else f"pulse_sweep_{args.split}"

    print(f"Loading LibriSpeech {args.split} subset...")
    dataset = load_subset(args.n_subset, args.seed, split=args.split)
    print(f"  {len(dataset)} utterances")

    fixed = {"p": args.p, "K": None, "seed": args.seed}
    config = sweep_config("K", fixed, args.gamma, len(dataset), args.seed)
    ckpt = load_checkpoint(out_name, config, resume=not args.no_resume)

    print("Loading Whisper...")
    whisper_model, normalizer = load_whisper()
    print(f"  device: {next(whisper_model.parameters()).device}, "
          f"fp16: {'auto' if args.fp16 is None else args.fp16}")

    print("Clean-audio baseline WER...")
    baseline_wer = get_or_compute_baseline(ckpt, dataset, whisper_model, normalizer)
    save_checkpoint(out_name, ckpt)

    print(f"Pulse sweep K={args.k_list}, p={args.p}, gamma={args.gamma}, "
          f"algorithms={args.algo_list}, bitrate={'on' if args.bitrate else 'off'}")
    bitrate_rows = [] if args.bitrate else None
    bitrate_save_cb = ((lambda rows: save_bitrate(out_name, rows))
                        if args.bitrate else None)
    per_utterance_rows = [] if args.per_utterance else None
    series, present = run_sweep(dataset, "K", args.k_list, fixed,
                                 whisper_model, normalizer, gamma=args.gamma,
                                 out_name=out_name, workers=args.workers,
                                 resume=not args.no_resume, methods=args.algo_list,
                                 bitrate_out=bitrate_rows,
                                 bitrate_save_cb=bitrate_save_cb,
                                 per_utterance_out=per_utterance_rows)

    plot_sweep(series, present, "Number of pulses (K)", baseline_wer, out_name)

    if args.per_utterance:
        if per_utterance_rows:
            per_utterance_path = save_per_utterance(out_name, per_utterance_rows)
            print(f"  per-utterance WER -> {per_utterance_path}")
        else:
            print("  per-utterance WER: nothing new synthesized this run "
                  "(all K already checkpointed)")

    if args.bitrate:
        if bitrate_rows:
            bitrate_path = save_bitrate(out_name, bitrate_rows)
            print(f"  bitrate -> {bitrate_path}")
        else:
            print("  bitrate: nothing new synthesized this run (all K already "
                  "checkpointed) -- rerun with --no-resume to get bitrate for them")

    print(f"Done. Results and plot written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
