"""
Pulse-count (sparsity) sweep: fix LPC order p, vary the number of pulses K,
for omp/lst/gclst plus a dense (non-sparse) excitation reference run through
the same closed-loop pipeline. See mpe_common.py / synth_core.py.

Results are checkpointed per K value, so re-running skips any K already done.
"""

import argparse

from mpe_common import (
    DEFAULT_P, SUBSET_SEED, N_SUBSET, GAMMA, OUT_DIR,
    METHODS, load_subset, load_whisper, get_or_compute_baseline,
    load_checkpoint, save_checkpoint, sweep_config,
    run_sweep, plot_sweep, default_workers,
)

DEFAULT_K_VALUES = [1, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35]


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

    print("Clean-audio baseline WER...")
    baseline_wer = get_or_compute_baseline(ckpt, dataset, whisper_model, normalizer)
    save_checkpoint(out_name, ckpt)

    print(f"Pulse sweep K={args.k_list}, p={args.p}, gamma={args.gamma}, "
          f"algorithms={args.algo_list}")
    series, present = run_sweep(dataset, "K", args.k_list, fixed,
                                 whisper_model, normalizer, gamma=args.gamma,
                                 out_name=out_name, workers=args.workers,
                                 resume=not args.no_resume, methods=args.algo_list)

    plot_sweep(series, present, "Number of pulses (K)", baseline_wer, out_name)
    print(f"Done. Results and plot written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
