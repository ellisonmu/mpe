import argparse

from mpe_common import (
    DEFAULT_K, SUBSET_SEED, N_SUBSET, L, GAMMA, OUT_DIR,
    load_subset, load_whisper, get_or_compute_baseline,
    load_checkpoint, save_checkpoint, sweep_config,
    run_sweep, plot_sweep, default_workers,
)

DEFAULT_ORDERS = [p for p in range(2, 33, 2) if p < L]   # even, under frame length


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subset", type=int, default=N_SUBSET,
                         help=f"number of LibriSpeech utterances to use (default: {N_SUBSET})")
    parser.add_argument("--seed", type=int, default=SUBSET_SEED,
                         help=f"random seed for subset selection (default: {SUBSET_SEED})")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                         help=f"pulse count held fixed during the sweep (default: {DEFAULT_K})")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                         help=f"perceptual error-weighting factor, W(z)=A(z)/A(z/gamma) "
                              f"(default: {GAMMA})")
    parser.add_argument("--p-values", type=str,
                         default=",".join(str(p) for p in DEFAULT_ORDERS),
                         help="comma-separated LPC orders to sweep (even only; "
                              f"default: {','.join(str(p) for p in DEFAULT_ORDERS)})")
    parser.add_argument("--workers", type=int, default=None,
                         help=f"parallel synthesis processes (default: {default_workers()})")
    parser.add_argument("--no-resume", action="store_true",
                         help="ignore any existing checkpoint and recompute every p")
    args = parser.parse_args()

    try:
        args.p_list = sorted({int(tok) for tok in args.p_values.split(",") if tok.strip()})
    except ValueError:
        parser.error("--p-values must be a comma-separated list of integers")
    if not args.p_list or args.p_list[0] < 2:
        parser.error("--p-values must contain integers >= 2")
    odd = [p for p in args.p_list if p % 2]
    if odd:
        parser.error(f"LSF conversion requires even orders; got odd: {odd}")
    return args


def main():
    args = parse_args()
    out_name = "order_sweep"

    print("Loading LibriSpeech test-clean subset...")
    dataset = load_subset(args.n_subset, args.seed)
    print(f"  {len(dataset)} utterances")

    fixed = {"p": None, "K": args.k, "seed": args.seed}
    config = sweep_config("p", fixed, args.gamma, len(dataset), args.seed)
    ckpt = load_checkpoint(out_name, config, resume=not args.no_resume)

    print("Loading Whisper...")
    whisper_model, normalizer = load_whisper()

    print("Clean-audio baseline WER...")
    baseline_wer = get_or_compute_baseline(ckpt, dataset, whisper_model, normalizer)
    save_checkpoint(out_name, ckpt)

    print(f"Order sweep p={args.p_list}, K={args.k}, gamma={args.gamma}")
    series, present = run_sweep(dataset, "p", args.p_list, fixed,
                                 whisper_model, normalizer, gamma=args.gamma,
                                 out_name=out_name, workers=args.workers,
                                 resume=not args.no_resume)

    plot_sweep(series, present, "LPC order (p)", baseline_wer, out_name)
    print(f"Done. Results and plot written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
