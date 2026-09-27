import argparse
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from synth_core import SR

def uniform_quantization(x, bits):
    if bits < 1:
        raise ValueError("bits must be >= 1")
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)

    n_levels = 2 ** int(bits)
    step = 2.0 / n_levels
    xq = np.floor(x / step) * step + step / 2.0            # mid-rise
    xq = np.clip(xq, -1.0 + step / 2.0, 1.0 - step / 2.0)
    return xq.astype(np.float32)


def _match_length(x, n):
    if len(x) == n:
        return x
    return x[:n] if len(x) > n else np.pad(x, (0, n - len(x)))


def undersample(audio, fs_khz, sr=SR):
    x = np.asarray(audio, dtype=np.float64)
    target = int(round(float(fs_khz) * 1000))
    if target >= sr:
        return x.astype(np.float32)
    frac = Fraction(target, sr).limit_denominator()
    lowered = resample_poly(x, frac.numerator, frac.denominator)
    restored = resample_poly(lowered, frac.denominator, frac.numerator)
    return _match_length(restored, len(x)).astype(np.float32)

def quantize_audio(audio, bits):
    x = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return x.astype(np.float32)
    return (uniform_quantization(x / peak, bits) * peak).astype(np.float32)

DISTORTIONS = {
    "undersample": {
        "values": [1, 2, 4, 8, 12, 16],
        "dtype": int,
        "param_name": "fs_khz",
        "xlabel": "Input sample rate (kHz)",
        "out_name": "distortion_undersample",
    },
    "quantize": {
        "values": [1, 2, 4, 8, 16],
        "dtype": int,
        "param_name": "q_bits",
        "xlabel": "Input quantization (bits)",
        "out_name": "distortion_quantize",
    },
}

def make_distortion(kind, seed):
    """distort(audio, value, idx) -> audio, as run_sweep expects."""
    if kind == "undersample":
        return lambda audio, val, idx: undersample(audio, val)
    if kind == "quantize":
        return lambda audio, val, idx: quantize_audio(audio, val)
    raise ValueError(f"unknown distortion kind: {kind}")

def parse_args():
    from mpe_common import (DEFAULT_P, DEFAULT_K, SUBSET_SEED, N_SUBSET, GAMMA,
                            default_workers, METHODS)

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kinds", type=str, default="undersample,quantize",
                         help="comma-separated distortions to sweep "
                              "(undersample, quantize; default: both)")
    parser.add_argument("--values", type=str, default=None,
                         help="override the swept values (comma-separated); only "
                              "valid when a single --kinds entry is given")
    parser.add_argument("--n-subset", type=int, default=N_SUBSET,
                         help=f"number of LibriSpeech utterances to use (default: {N_SUBSET})")
    parser.add_argument("--seed", type=int, default=SUBSET_SEED,
                         help=f"random seed for subset selection and noise "
                              f"(default: {SUBSET_SEED})")
    parser.add_argument("--p", type=int, default=DEFAULT_P,
                         help=f"LPC order, held fixed (default: {DEFAULT_P})")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                         help=f"pulse count, held fixed (default: {DEFAULT_K})")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                         help=f"perceptual error-weighting factor (default: {GAMMA})")
    parser.add_argument("--algorithms", type=str, default=",".join(METHODS),
                         help="comma-separated algorithms to run, e.g. mp,comp "
                              f"(default: all = {','.join(METHODS)}). Algorithms already "
                              "in the checkpoint are skipped, new ones are merged in")
    parser.add_argument("--workers", type=int, default=None,
                         help=f"parallel synthesis processes (default: {default_workers()})")
    parser.add_argument("--no-resume", action="store_true",
                         help="ignore existing checkpoints and recompute everything")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                         help="Whisper transcription precision (default: auto -- fp16 iff "
                              "a CUDA GPU is available; --no-fp16 forces fp32 even on GPU, "
                              "--fp16 forces fp16 even without a GPU, where it will be slow "
                              "or unsupported)")
    args = parser.parse_args()

    args.algo_list = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    unknown = [a for a in args.algo_list if a not in METHODS]
    if not args.algo_list or unknown:
        parser.error(f"--algorithms must be from {list(METHODS)}; got unknown: {unknown}")

    args.kind_list = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in args.kind_list if k not in DISTORTIONS]
    if unknown:
        parser.error(f"unknown --kinds: {unknown}; choose from {sorted(DISTORTIONS)}")
    if not args.kind_list:
        parser.error("--kinds must name at least one distortion")

    args.value_list = None
    if args.values is not None:
        if len(args.kind_list) != 1:
            parser.error("--values requires exactly one --kinds entry")
        cast = DISTORTIONS[args.kind_list[0]]["dtype"]
        try:
            args.value_list = sorted({cast(tok) for tok in args.values.split(",")
                                      if tok.strip()})
        except ValueError:
            parser.error(f"--values must be {cast.__name__} values for this distortion")
        if not args.value_list:
            parser.error("--values must be non-empty")
    return args


def main():
    from mpe_common import (OUT_DIR, load_subset, load_whisper, get_or_compute_baseline,
                            load_checkpoint, save_checkpoint, sweep_config,
                            run_sweep, plot_sweep, set_fp16)

    args = parse_args()
    set_fp16(args.fp16)

    print("Loading LibriSpeech test-clean subset...")
    dataset = load_subset(args.n_subset, args.seed)
    print(f"  {len(dataset)} utterances")

    print("Loading Whisper...")
    whisper_model, normalizer = load_whisper()
    print(f"  device: {next(whisper_model.parameters()).device}, "
          f"fp16: {'auto' if args.fp16 is None else args.fp16}")
    print(f"  algorithms={args.algo_list}")

    baseline_wer = None
    for kind in args.kind_list:
        spec = DISTORTIONS[kind]
        values = args.value_list if args.value_list is not None else spec["values"]
        out_name = spec["out_name"]

        fixed = {"p": args.p, "K": args.k, "seed": args.seed}
        config = sweep_config(spec["param_name"], fixed, args.gamma,
                              len(dataset), args.seed)
        ckpt = load_checkpoint(out_name, config, resume=not args.no_resume)

        print(f"\n=== {kind} sweep: {spec['param_name']}={values} "
              f"(p={args.p}, K={args.k}, gamma={args.gamma}) ===")
        if baseline_wer is None:
            print("Clean-audio baseline WER...")
            baseline_wer = get_or_compute_baseline(ckpt, dataset, whisper_model, normalizer)
        else:
            ckpt["baseline_wer"] = baseline_wer
        save_checkpoint(out_name, ckpt)

        series, present = run_sweep(dataset, spec["param_name"], values, fixed,
                                     whisper_model, normalizer, gamma=args.gamma,
                                     out_name=out_name, workers=args.workers,
                                     resume=not args.no_resume,
                                     distort=make_distortion(kind, args.seed),
                                     include_input_ref=True, methods=args.algo_list)
        plot_sweep(series, present, spec["xlabel"], baseline_wer, out_name)
        print(f"  -> {out_name}.csv / _wer.png")

    print(f"\nDone. Results written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
