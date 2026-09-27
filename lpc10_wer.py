"""Isolated test: LPC-10 (2400 bps, FS-1015) scored by WER, as an external
codec reference point alongside the project's own MPE algorithms.

For each utterance: resample to 8 kHz mono, encode/decode through the
reference LPC-10 codec (tools/lpc10/{nuke,unnuke}), resample the decoded
audio back to 16 kHz, transcribe with Whisper, and score WER against the
reference transcript (and against the clean-audio baseline WER, for
comparison). See tools/lpc10/README.md for what these binaries are and how
to rebuild them (e.g. on the Jetson).
"""

import argparse
import csv
import os
import platform
import subprocess

import numpy as np
from scipy.signal import resample_poly

from mpe_common import (SR, N_SUBSET, SUBSET_SEED, OUT_DIR, load_subset,
                        load_whisper, transcribe, compute_wer, bootstrap_ci)

LPC10_SR = 8000
LPC10_FRAME = 180   # samples/frame at 8 kHz; a trailing partial frame is dropped

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "lpc10")


def _tool_path(name):
    exe = name + (".exe" if platform.system() == "Windows" else "")
    path = os.path.join(TOOLS_DIR, exe)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- see tools/lpc10/README.md to build it "
            f"(a Windows .exe is checked in; other platforms rebuild from tools/lpc10/src)")
    return path


def lpc10_roundtrip(audio, sr=SR):
    """16 kHz float audio -> 8 kHz LPC-10 encode -> decode -> 16 kHz float
    audio, same length as the input (padded/trimmed around the codec's
    fixed 180-sample frame)."""
    nuke = _tool_path("nuke")
    unnuke = _tool_path("unnuke")

    x8k = resample_poly(np.asarray(audio, dtype=np.float64), LPC10_SR, sr)

    n_frames = -(-len(x8k) // LPC10_FRAME)   # ceil: pad up, never drop audio
    padded = np.zeros(n_frames * LPC10_FRAME, dtype=np.float64)
    padded[:len(x8k)] = x8k
    pcm_in = np.clip(padded * 32768.0, -32768, 32767).astype("<i2").tobytes()

    coded = subprocess.run(nuke, input=pcm_in, stdout=subprocess.PIPE,
                           check=True).stdout
    pcm_out = subprocess.run(unnuke, input=coded, stdout=subprocess.PIPE,
                             check=True).stdout

    y8k = np.frombuffer(pcm_out, dtype="<i2").astype(np.float64) / 32768.0
    y8k = y8k[:len(x8k)]   # drop the padding back off
    y = resample_poly(y8k, sr, LPC10_SR)
    return y[:len(audio)] if len(y) >= len(audio) else np.pad(y, (0, len(audio) - len(y)))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-subset", type=int, default=N_SUBSET,
                    help=f"number of LibriSpeech utterances to use (default: {N_SUBSET})")
    ap.add_argument("--seed", type=int, default=SUBSET_SEED,
                    help=f"random seed for subset selection (default: {SUBSET_SEED})")
    ap.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean",
                    help="LibriSpeech split (default: test-clean)")
    ap.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "lpc10_wer.csv"))
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Loading LibriSpeech {args.split} subset...")
    dataset = load_subset(args.n_subset, args.seed, split=args.split)
    print(f"  {len(dataset)} utterances")

    print("Loading Whisper...")
    whisper_model, normalizer = load_whisper()

    baseline_wers = []
    lpc10_wers = []
    for i, (audio, transcript) in enumerate(dataset):
        audio = np.asarray(audio, dtype=np.float64)

        hyp_clean = transcribe(whisper_model, audio)
        baseline_wers.append(compute_wer(normalizer, transcript, hyp_clean))

        try:
            decoded = lpc10_roundtrip(audio)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"  [{i}] lpc10 codec failed: {exc}")
            continue

        hyp_lpc10 = transcribe(whisper_model, decoded.astype(np.float32))
        lpc10_wers.append(compute_wer(normalizer, transcript, hyp_lpc10))
        print(f"  [{i}] baseline={baseline_wers[-1]:.3f}  lpc10={lpc10_wers[-1]:.3f}")

    base_mean, base_lo, base_hi = bootstrap_ci(baseline_wers)
    lpc10_mean, lpc10_lo, lpc10_hi = bootstrap_ci(lpc10_wers)
    print(f"\nbaseline WER: {base_mean:.4f}  [{base_lo:.4f}, {base_hi:.4f}]")
    print(f"lpc10 WER:    {lpc10_mean:.4f}  [{lpc10_lo:.4f}, {lpc10_hi:.4f}]")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "wer", "wer_lo", "wer_hi", "n"])
        w.writeheader()
        w.writerow({"series": "baseline", "wer": base_mean,
                   "wer_lo": base_lo, "wer_hi": base_hi, "n": len(baseline_wers)})
        w.writerow({"series": "lpc10", "wer": lpc10_mean,
                   "wer_lo": lpc10_lo, "wer_hi": lpc10_hi, "n": len(lpc10_wers)})
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
