import argparse
import csv
import math
import os

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import EncodecModel, AutoProcessor

from mpe_common import (SR, N_SUBSET, SUBSET_SEED, OUT_DIR, load_subset,
                        load_whisper, transcribe, compute_wer, bootstrap_ci)

MODEL_NAME = "facebook/encodec_24khz"
DEFAULT_BANDWIDTHS = [24.0]   # kbps; the model's own target_bandwidths


def load_encodec():
    model = EncodecModel.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, processor, device


def encodec_roundtrip(model, processor, device, audio, bandwidth, sr=SR):
    """16 kHz float audio -> encoder's own sample rate -> EnCodec encode/decode
    at `bandwidth` kbps -> back to 16 kHz, same length as the input.

    Returns (decoded_audio, measured_kbps). measured_kbps is read off the
    actual number of quantizers/codebooks the encoder used for this call
    (audio_codes.shape), not assumed from the requested `bandwidth` -- the
    model internally floors an arbitrary target to the nearest quantizer
    count, so this is the true rate rather than the nominal one."""
    enc_sr = processor.sampling_rate
    x = resample_poly(np.asarray(audio, dtype=np.float64), enc_sr, sr).astype(np.float32)

    inputs = processor(raw_audio=x, sampling_rate=enc_sr, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    padding_mask = inputs.get("padding_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)

    with torch.no_grad():
        enc_out = model.encode(input_values, padding_mask, bandwidth=bandwidth)
        dec_out = model.decode(enc_out.audio_codes, enc_out.audio_scales, padding_mask)

    # audio_codes: [batch, segments, num_quantizers, num_frames]
    num_quantizers = enc_out.audio_codes.shape[-2]
    bits_per_codebook = math.log2(model.config.codebook_size)
    measured_kbps = num_quantizers * model.config.frame_rate * bits_per_codebook / 1000.0

    y = dec_out.audio_values.squeeze().cpu().numpy()
    y16 = resample_poly(y, sr, enc_sr)
    if len(y16) < len(audio):
        y16 = np.pad(y16, (0, len(audio) - len(y16)))
    return y16[:len(audio)], measured_kbps


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-subset", type=int, default=N_SUBSET,
                    help=f"number of LibriSpeech utterances to use (default: {N_SUBSET})")
    ap.add_argument("--seed", type=int, default=SUBSET_SEED,
                    help=f"random seed for subset selection (default: {SUBSET_SEED})")
    ap.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean",
                    help="LibriSpeech split (default: test-clean)")
    ap.add_argument("--bandwidths", type=str,
                    default=",".join(str(b) for b in DEFAULT_BANDWIDTHS),
                    help="comma-separated EnCodec target bandwidths in kbps "
                         f"(default: {','.join(str(b) for b in DEFAULT_BANDWIDTHS)})")
    ap.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "encodec_wer.csv"))
    args = ap.parse_args()

    try:
        args.bandwidth_list = sorted({float(b) for b in args.bandwidths.split(",") if b.strip()})
    except ValueError:
        ap.error("--bandwidths must be a comma-separated list of numbers")
    if not args.bandwidth_list:
        ap.error("--bandwidths must be non-empty")
    return args


def main():
    args = parse_args()

    print(f"Loading LibriSpeech {args.split} subset...")
    dataset = load_subset(args.n_subset, args.seed, split=args.split)
    print(f"  {len(dataset)} utterances")

    print("Loading Whisper...")
    whisper_model, normalizer = load_whisper()

    print(f"Loading {MODEL_NAME}...")
    encodec_model, processor, device = load_encodec()
    print(f"  device: {device}")

    baseline_wers = []
    for audio, transcript in dataset:
        hyp = transcribe(whisper_model, np.asarray(audio, dtype=np.float32))
        baseline_wers.append(compute_wer(normalizer, transcript, hyp))
    base_mean, base_lo, base_hi = bootstrap_ci(baseline_wers)
    print(f"baseline WER: {base_mean:.4f}  [{base_lo:.4f}, {base_hi:.4f}]")

    rows = [{"bandwidth_kbps": "baseline", "measured_kbps": "", "wer": base_mean,
             "wer_lo": base_lo, "wer_hi": base_hi, "n": len(baseline_wers)}]

    for bandwidth in args.bandwidth_list:
        wers = []
        measured = []
        for i, (audio, transcript) in enumerate(dataset):
            audio = np.asarray(audio, dtype=np.float64)
            decoded, kbps = encodec_roundtrip(encodec_model, processor, device, audio, bandwidth)
            measured.append(kbps)
            hyp = transcribe(whisper_model, decoded.astype(np.float32))
            wers.append(compute_wer(normalizer, transcript, hyp))
            print(f"  [bw={bandwidth} kbps, {i}] measured={kbps:.3f} kbps  wer={wers[-1]:.3f}")

        mean, lo, hi = bootstrap_ci(wers)
        mean_kbps = float(np.mean(measured))
        rows.append({"bandwidth_kbps": bandwidth, "measured_kbps": mean_kbps, "wer": mean,
                     "wer_lo": lo, "wer_hi": hi, "n": len(wers)})
        print(f"bandwidth={bandwidth} kbps (measured {mean_kbps:.3f} kbps) "
             f"WER: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bandwidth_kbps", "measured_kbps",
                                          "wer", "wer_lo", "wer_hi", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
