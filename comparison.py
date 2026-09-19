import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

from synth_core import (SR, FRAME_MS, mpe_bitrate_kbps_combinatorial,
                        mpe_frame_bits_combinatorial)


CODECS = {
    "opus": {
        "encoder": "libopus", "ext": "opus", "rate": 16000,
        "bitrates": [6, 8, 12, 16, 24, 32],
    },
    "mp3": {
        "encoder": "libmp3lame", "ext": "mp3", "rate": 16000,
        "bitrates": [8, 16, 24, 32, 48, 64],
    },
    "amr_nb": {   
        "encoder": "libopencore_amrnb", "ext": "amr", "rate": 8000,
        "bitrates": [4.75, 5.9, 7.4, 12.2],
    },
    "amr_wb": {  
        "encoder": "libvo_amrwbenc", "ext": "amr", "rate": 16000,
        "bitrates": [6.6, 12.65, 23.85],
    },
}


def _ffmpeg(cmd):
    return subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + cmd,
                          capture_output=True, text=True)


def codec_roundtrip(audio, spec, br_kbps, tmpdir, tag):
    audio = np.asarray(audio, dtype=np.float32)
    dur = len(audio) / SR
    src = os.path.join(tmpdir, f"{tag}_src.wav")
    enc = os.path.join(tmpdir, f"{tag}.{spec['ext']}")
    dec = os.path.join(tmpdir, f"{tag}_dec.wav")
    sf.write(src, audio, SR)

    r = _ffmpeg(["-i", src, "-c:a", spec["encoder"],
                 "-b:a", str(int(round(br_kbps * 1000))),
                 "-ar", str(spec["rate"]), "-ac", "1", enc])
    if r.returncode != 0:
        raise RuntimeError(f"encode failed ({spec['encoder']} @ {br_kbps}k): "
                           f"{r.stderr.strip()[:200]}")
    nbytes = os.path.getsize(enc)

    r = _ffmpeg(["-i", enc, "-ar", str(SR), "-ac", "1", dec])
    if r.returncode != 0:
        raise RuntimeError(f"decode failed ({spec['encoder']} @ {br_kbps}k): "
                           f"{r.stderr.strip()[:200]}")
    y, _ = sf.read(dec, dtype="float32")

    # AMR pads out to a whole frame; trim/pad back so lengths line up
    if len(y) > len(audio):
        y = y[:len(audio)]
    elif len(y) < len(audio):
        y = np.pad(y, (0, len(audio) - len(y)))
    return y, nbytes * 8 / dur / 1000.0

def comparison_config(args, split):
    return {"split": split, "p": args.p, "gamma": args.gamma,
            "n_subset": args.n_subset, "seed": args.seed, "frame_ms": FRAME_MS}


def load_points(out_dir, out_name, config, resume=True):
    fresh = {"config": config, "baseline_wer": None, "points": {}}
    path = os.path.join(out_dir, f"{out_name}.csv")
    if not resume or not os.path.exists(path):
        return fresh
    try:
        with open(path, newline="") as f:
            lines = f.readlines()
    except OSError as exc:
        print(f"  checkpoint unreadable ({exc}); starting fresh")
        return fresh

    if len(lines) < 2 or not lines[0].startswith("#config ") or not lines[1].startswith("#baseline_wer "):
        return fresh
    try:
        saved_config = json.loads(lines[0][len("#config "):])
        baseline_wer = json.loads(lines[1][len("#baseline_wer "):])
    except json.JSONDecodeError as exc:
        print(f"  checkpoint unreadable ({exc}); starting fresh")
        return fresh

    if saved_config != config:
        print("  checkpoint settings differ from this run; starting fresh")
        return fresh

    import csv as _csv
    reader = _csv.DictReader(lines[2:])
    points = {}
    for row in reader:
        key = f"{row['system']}@{row['setting']}"
        points[key] = {"system": row["system"], "setting": row["setting"],
                       "bitrate_kbps": float(row["bitrate_kbps"]),
                       "wer": float(row["wer"]),
                       "wer_lo": float(row["wer_lo"]),
                       "wer_hi": float(row["wer_hi"])}
    return {"config": config, "baseline_wer": baseline_wer, "points": points}


def save_points(out_dir, out_name, ckpt):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{out_name}.csv")
    import csv as _csv
    rows = sorted(ckpt["points"].values(),
                  key=lambda r: (r["system"], r["bitrate_kbps"]))
    with open(path, "w", newline="") as f:
        f.write(f"#config {json.dumps(ckpt['config'])}\n")
        f.write(f"#baseline_wer {json.dumps(ckpt['baseline_wer'])}\n")
        w = _csv.writer(f)
        w.writerow(["system", "setting", "bitrate_kbps", "wer", "wer_lo", "wer_hi"])
        for r in rows:
            w.writerow([r["system"], r["setting"], r["bitrate_kbps"],
                        r["wer"], r["wer_lo"], r["wer_hi"]])


def plot_rate_distortion(ckpt, split, out_dir, out_name):
    points = list(ckpt["points"].values())
    if not points:
        print("  no points to plot")
        return
    plt.figure(figsize=(8, 5.5))
    systems = sorted({r["system"] for r in points})
    for system in systems:
        rows = sorted((r for r in points if r["system"] == system),
                      key=lambda r: r["bitrate_kbps"])
        x = np.array([r["bitrate_kbps"] for r in rows], dtype=float)
        y = np.array([r["wer"] for r in rows], dtype=float)
        lo = np.array([r["wer_lo"] for r in rows], dtype=float)
        hi = np.array([r["wer_hi"] for r in rows], dtype=float)
        yerr = np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)])
        style = dict(marker="o", linewidth=2) if system == "gclst" else dict(marker="s", alpha=0.85)
        plt.errorbar(x, y, yerr=yerr, label=system, capsize=3, **style)

    if ckpt.get("baseline_wer") is not None:
        plt.axhline(ckpt["baseline_wer"], color="black", linestyle=":",
                    label="clean baseline WER")
    plt.xlabel("Bitrate (kbit/s)")
    plt.ylabel("Word Error Rate")
    plt.title(f"Rate-distortion: MPE vs standard codecs ({split})\n"
              f"error bars: 95% bootstrap CI")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{out_name}_rd.png"), dpi=200)
    plt.close()


def parse_args():
    from mpe_common import DEFAULT_P, SUBSET_SEED, N_SUBSET, GAMMA, default_workers

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=str, default="test-clean,test-other",
                         help="comma-separated LibriSpeech splits (default: both)")
    parser.add_argument("--k-values", type=str, default="2,5,10,20",
                         help="comma-separated MPE pulse counts (default: 2,5,10,20)")
    parser.add_argument("--codecs", type=str, default=",".join(CODECS),
                         help=f"comma-separated codecs (default: {','.join(CODECS)})")
    parser.add_argument("--n-subset", type=int, default=N_SUBSET,
                         help=f"utterances per split (default: {N_SUBSET})")
    parser.add_argument("--seed", type=int, default=SUBSET_SEED,
                         help=f"subset seed (default: {SUBSET_SEED})")
    parser.add_argument("--p", type=int, default=DEFAULT_P,
                         help=f"LPC order for MPE (default: {DEFAULT_P})")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                         help=f"perceptual weighting factor (default: {GAMMA})")
    parser.add_argument("--workers", type=int, default=None,
                         help=f"parallel MPE synthesis processes (default: {default_workers()})")
    parser.add_argument("--no-resume", action="store_true",
                         help="ignore existing checkpoints and recompute everything")
    args = parser.parse_args()

    args.split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    args.k_list = sorted({int(t) for t in args.k_values.split(",") if t.strip()})
    args.codec_list = [c.strip() for c in args.codecs.split(",") if c.strip()]
    unknown = [c for c in args.codec_list if c not in CODECS]
    if unknown:
        parser.error(f"unknown --codecs: {unknown}; choose from {sorted(CODECS)}")
    return args


def main():
    from mpe_common import (OUT_DIR, METHODS, load_subset, load_whisper, transcribe,
                            compute_wer, compute_baseline_wer, bootstrap_ci,
                            default_workers, _synthesis_map, synthesize_utterance_job)

    args = parse_args()
    workers = default_workers() if args.workers is None else args.workers

    print("Loading Whisper...")
    whisper_model, normalizer = load_whisper()

    for split in args.split_list:
        out_name = f"comparison_{split.replace('-', '_')}"
        print(f"\n########## {split} ##########")
        print("Loading LibriSpeech subset (downloads on first use)...")
        dataset = load_subset(args.n_subset, args.seed, split=split)
        print(f"  {len(dataset)} utterances")

        config = comparison_config(args, split)
        ckpt = load_points(OUT_DIR, out_name, config, resume=not args.no_resume)
        points = ckpt["points"]

        if ckpt["baseline_wer"] is None:
            print("Clean-audio baseline WER...")
            ckpt["baseline_wer"] = compute_baseline_wer(dataset, whisper_model, normalizer)
            save_points(OUT_DIR, out_name, ckpt)
        print(f"  baseline WER = {ckpt['baseline_wer']:.4f}")

        gclst = METHODS["gclst"]
        with _synthesis_map(workers) as pmap:
            for K in args.k_list:
                key = f"gclst@K={K}"
                if key in points:
                    print(f"  skip {key} (done)")
                    continue
                jobs = [(i, audio, ["gclst"], args.p, K, args.gamma)
                        for i, (audio, _) in enumerate(dataset)]
                synth = dict(pmap(synthesize_utterance_job, jobs))
                wers = []
                for i, (_, transcript) in enumerate(dataset):
                    hyp = transcribe(whisper_model, synth[i]["gclst"])
                    wers.append(compute_wer(normalizer, transcript, hyp))
                mean, lo, hi = bootstrap_ci(wers)
                br = mpe_bitrate_kbps_combinatorial(args.p, K)
                points[key] = {"system": "gclst", "setting": f"K={K}",
                               "bitrate_kbps": br, "wer": mean,
                               "wer_lo": lo, "wer_hi": hi}
                save_points(OUT_DIR, out_name, ckpt)
                print(f"  {key}: {br:.1f} kbps  WER={mean:.4f}  "
                      f"({mpe_frame_bits_combinatorial(args.p, K)} bits/frame)")

        with tempfile.TemporaryDirectory() as tmpdir:
            for codec in args.codec_list:
                spec = CODECS[codec]
                for br_req in spec["bitrates"]:
                    key = f"{codec}@{br_req}"
                    if key in points:
                        print(f"  skip {key} (done)")
                        continue
                    wers, rates = [], []
                    try:
                        for i, (audio, transcript) in enumerate(dataset):
                            dec, meas = codec_roundtrip(audio, spec, br_req, tmpdir,
                                                        f"{codec}_{br_req}_{i}")
                            rates.append(meas)
                            hyp = transcribe(whisper_model, dec)
                            wers.append(compute_wer(normalizer, transcript, hyp))
                    except RuntimeError as exc:
                        print(f"  {key}: SKIPPED -- {exc}")
                        continue
                    mean, lo, hi = bootstrap_ci(wers)
                    br = float(np.mean(rates))
                    points[key] = {"system": codec, "setting": f"{br_req}k",
                                   "bitrate_kbps": br, "wer": mean,
                                   "wer_lo": lo, "wer_hi": hi}
                    save_points(OUT_DIR, out_name, ckpt)
                    print(f"  {key}: requested {br_req}k -> measured {br:.1f} kbps  "
                          f"WER={mean:.4f}")

        plot_rate_distortion(ckpt, split, OUT_DIR, out_name)
        print(f"  -> {out_name}.csv / _rd.png")

    print(f"\nDone. Results written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
