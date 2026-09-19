"""
Experiment harness shared by sparsity_sweep.py (K) and order_sweep.py (p):
LibriSpeech loading, Whisper/WER metrics, checkpointed sweeps with parallel
synthesis, and plotting.

The DSP itself lives in synth_core.py, which is deliberately torch-free so
worker processes can import it cheaply.
"""

import os
import csv
import json
import random
import contextlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
import soundfile as sf
import torch
import torchaudio
import jiwer
import whisper
from whisper.normalizers import EnglishTextNormalizer

from synth_core import (SR, FRAME_MS, L, GAMMA, LSF_BITS, EXC_BITS, POLE_MARGIN,
                        METHODS, DENSE_NAME, synthesize_mpe, fit_lpc, lpc_matrix,
                        levinson_durbin, frame_impulse_response, weighting_filter,
                        _zero_input_response, synthesize_utterance_job)


def _load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr

torchaudio.load = _load_via_soundfile

N_SUBSET = 20                          # number of LibriSpeech utterances to use
SUBSET_SEED = 0
DATA_ROOT = "./librispeech_data"
OUT_DIR = "./experiment_results"
DEFAULT_P, DEFAULT_K = 16, 8
WHISPER_MODEL_NAME = "base"
N_BOOTSTRAP = 2000                     # resamples for the 95% WER confidence interval

INPUT_NAME = "distorted_input"         # distorted but UNCODED audio (distortion sweeps)
SERIES = list(METHODS) + [DENSE_NAME]


def load_subset(n=N_SUBSET, seed=SUBSET_SEED, split="test-clean"):
    """Random subset of a LibriSpeech split, resampled to SR, as (audio,
    transcript) pairs. `split` is a torchaudio LIBRISPEECH url, e.g.
    "test-clean" or "test-other" (downloaded on first use)."""
    os.makedirs(DATA_ROOT, exist_ok=True)
    dataset = torchaudio.datasets.LIBRISPEECH(root=DATA_ROOT, url=split, download=True)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    items = []
    for idx in indices[:n]:
        waveform, sr, transcript, *_ = dataset[idx]
        audio = waveform.squeeze(0).numpy().astype(np.float32)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        items.append((audio, transcript))
    return items


def load_whisper():
    return whisper.load_model(WHISPER_MODEL_NAME), EnglishTextNormalizer()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def transcribe(model, audio):
    try:
        result = model.transcribe(audio.astype(np.float32), fp16=False, language="en")
        return result["text"]
    except Exception:
        return ""


def compute_wer(normalizer, reference_text, hyp_text):
    ref_n = normalizer(reference_text)
    hyp_n = normalizer(hyp_text)
    if len(ref_n.strip()) == 0:
        return np.nan
    return jiwer.wer(ref_n, hyp_n)


def compute_baseline_wer(dataset, whisper_model, normalizer):
    wers = []
    for audio, transcript in dataset:
        hyp = transcribe(whisper_model, audio)
        wers.append(compute_wer(normalizer, transcript, hyp))
    return float(np.nanmean(wers))


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=95, seed=0):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Checkpointing: results are keyed by param value so a re-run skips finished work
# ---------------------------------------------------------------------------

def results_path(out_name):
    return os.path.join(OUT_DIR, f"{out_name}.csv")


def sweep_config(param_name, fixed_params, gamma, n_subset, seed):
    """Settings a checkpoint must match to be safely resumable -- mixing results
    across different p/gamma/subset would silently produce an incoherent curve."""
    return {
        "param_name": param_name,
        "p": fixed_params.get("p"),
        "K": fixed_params.get("K"),
        "gamma": gamma,
        "n_subset": n_subset,
        "seed": seed,
        "frame_ms": FRAME_MS,
        "lsf_bits": LSF_BITS,
        "exc_bits": EXC_BITS,
        "pole_margin": POLE_MARGIN,
    }


def load_checkpoint(out_name, config, resume=True):
    """Existing checkpoint if it matches `config`, else a fresh one.

    The checkpoint lives entirely in the CSV: two leading `#`-comment lines
    carry the config and baseline WER (as JSON, so they round-trip exactly),
    then the usual header + data rows.
    """
    fresh = {"config": config, "series": list(SERIES),
             "baseline_wer": None, "by_value": {}}
    path = results_path(out_name)
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
        print("  checkpoint settings differ from this run; ignoring it and starting fresh")
        differing = [k for k in config if saved_config.get(k) != config[k]]
        print(f"    differing keys: {differing}")
        return fresh

    param_name = config["param_name"]
    reader = csv.DictReader(lines[2:])
    series_names = [h[:-len("_wer")] for h in (reader.fieldnames or [])
                    if h.endswith("_wer") and h != "baseline_wer"]
    by_value = {}
    for row in reader:
        val = row[param_name]
        block = {}
        for name in series_names:
            def _num(x):
                return float(x) if x not in (None, "", "nan") else float("nan")
            block[name] = {"wer": _num(row.get(f"{name}_wer")),
                            "wer_lo": _num(row.get(f"{name}_wer_lo")),
                            "wer_hi": _num(row.get(f"{name}_wer_hi"))}
        by_value[val] = block

    return {"config": config, "series": series_names or list(SERIES),
            "baseline_wer": baseline_wer, "by_value": by_value}


def save_checkpoint(out_name, ckpt):
    """Write the checkpoint as a single CSV: config/baseline as leading `#`
    comment lines (JSON-encoded, so load_checkpoint round-trips them exactly),
    then one row per swept value, one column per series' wer/wer_lo/wer_hi."""
    param_name = ckpt["config"]["param_name"]
    by_value = ckpt.get("by_value", {})
    baseline_wer = ckpt.get("baseline_wer")
    series_names = ckpt.get("series", SERIES)

    path = results_path(out_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    header = [param_name]
    for name in series_names:
        header += [f"{name}_wer", f"{name}_wer_lo", f"{name}_wer_hi"]

    # numeric sort of the string-keyed values (K/p are ints, noise alpha a float)
    rows = sorted(by_value.items(), key=lambda kv: float(kv[0]))
    with open(path, "w", newline="") as f:
        f.write(f"#config {json.dumps(ckpt['config'])}\n")
        f.write(f"#baseline_wer {json.dumps(baseline_wer)}\n")
        writer = csv.writer(f)
        writer.writerow(header)
        for val, block in rows:
            row = [val]
            for name in series_names:
                cell = block.get(name, {})
                row += [cell.get("wer"), cell.get("wer_lo"), cell.get("wer_hi")]
            writer.writerow(row)


def checkpoint_to_series(ckpt, param_values):
    """Aligned per-series lists for plotting, plus the param values actually present."""
    by_value = ckpt.get("by_value", {})
    series_names = ckpt.get("series", SERIES)
    present = [v for v in param_values if str(v) in by_value]
    series = {n: {"wer": [], "wer_lo": [], "wer_hi": []} for n in series_names}
    for v in present:
        block = by_value[str(v)]
        for n in series_names:
            cell = block.get(n, {})
            for key in ("wer", "wer_lo", "wer_hi"):
                series[n][key].append(cell.get(key, float("nan")))
    return series, present


def get_or_compute_baseline(ckpt, dataset, whisper_model, normalizer):
    """Clean-audio baseline WER, reused from the checkpoint when available."""
    if ckpt.get("baseline_wer") is not None:
        print(f"  baseline WER = {ckpt['baseline_wer']:.4f} (from checkpoint)")
        return ckpt["baseline_wer"]
    baseline = compute_baseline_wer(dataset, whisper_model, normalizer)
    ckpt["baseline_wer"] = baseline
    print(f"  baseline WER = {baseline:.4f}")
    return baseline


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _synthesis_map(workers):
    """map() over synthesis jobs, parallel across processes unless workers<=1."""
    if workers <= 1:
        yield map
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            yield ex.map


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


def run_sweep(dataset, param_name, param_values, fixed_params, whisper_model, normalizer,
              gamma=GAMMA, out_name=None, workers=None, resume=True,
              distort=None, include_input_ref=False):
    """Sweep `param_name` over `param_values`, checkpointing after each value.

    `distort(audio, value, idx) -> audio` optionally transforms the input audio
    per swept value (used by distortion_sweep.py); the coder then runs on the
    distorted signal. With `include_input_ref`, the distorted-but-uncoded audio
    is transcribed too, so damage caused by the distortion itself can be told
    apart from damage the coder adds on top of it.
    """
    series_names = list(METHODS) + [DENSE_NAME]
    if include_input_ref:
        series_names.append(INPUT_NAME)

    config = sweep_config(param_name, fixed_params, gamma,
                          len(dataset), fixed_params.get("seed"))
    ckpt = load_checkpoint(out_name, config, resume) if out_name else {
        "config": config, "series": series_names, "baseline_wer": None, "by_value": {}}
    ckpt["series"] = series_names
    by_value = ckpt["by_value"]

    todo = [v for v in param_values if str(v) not in by_value]
    done = [v for v in param_values if str(v) in by_value]
    if done:
        print(f"  skipping {len(done)} already done: {done}")
    if not todo:
        print("  nothing left to run")
        return checkpoint_to_series(ckpt, param_values)
    print(f"  running {len(todo)}: {todo}")

    # dense_lpc ignores K, so it can be computed once and reused -- but only on
    # a plain K sweep. On a p sweep the filter changes with p, and on a
    # distortion sweep the input audio itself changes, so it must be redone
    # for every value.
    reuse_dense = (param_name == "K") and (distort is None)
    dense_cached = None
    if reuse_dense:
        for block in by_value.values():
            if DENSE_NAME in block:
                dense_cached = block[DENSE_NAME]
                break

    workers = default_workers() if workers is None else workers
    print(f"  synthesis workers: {workers}")

    with _synthesis_map(workers) as pmap:
        for val in todo:
            p = val if param_name == "p" else fixed_params["p"]
            K = val if param_name == "K" else fixed_params["K"]

            audios = [audio if distort is None else distort(audio, val, i)
                      for i, (audio, _) in enumerate(dataset)]

            names = list(METHODS)
            need_dense = (not reuse_dense) or (dense_cached is None)
            if need_dense:
                names.append(DENSE_NAME)

            jobs = [(i, a, names, p, K, gamma) for i, a in enumerate(audios)]
            synth = dict(pmap(synthesize_utterance_job, jobs))

            wers = {n: [] for n in names}
            if include_input_ref:
                wers[INPUT_NAME] = []
            for i, (_, transcript) in enumerate(dataset):
                for n in names:
                    hyp = transcribe(whisper_model, synth[i][n])
                    wers[n].append(compute_wer(normalizer, transcript, hyp))
                if include_input_ref:
                    hyp = transcribe(whisper_model, audios[i])
                    wers[INPUT_NAME].append(compute_wer(normalizer, transcript, hyp))

            block = {}
            for n in METHODS:
                mean, lo, hi = bootstrap_ci(wers[n])
                block[n] = {"wer": mean, "wer_lo": lo, "wer_hi": hi}
            if need_dense:
                mean, lo, hi = bootstrap_ci(wers[DENSE_NAME])
                block[DENSE_NAME] = {"wer": mean, "wer_lo": lo, "wer_hi": hi}
                if reuse_dense:
                    dense_cached = block[DENSE_NAME]
            else:
                block[DENSE_NAME] = dict(dense_cached)
            if include_input_ref:
                mean, lo, hi = bootstrap_ci(wers[INPUT_NAME])
                block[INPUT_NAME] = {"wer": mean, "wer_lo": lo, "wer_hi": hi}

            by_value[str(val)] = block
            if out_name:
                save_checkpoint(out_name, ckpt)
            print(f"  {param_name}={val} done"
                  + (" (saved)" if out_name else ""))

    return checkpoint_to_series(ckpt, param_values)

def plot_sweep(series, param_values, xlabel, baseline_wer, out_prefix):
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name in series:
        if name == DENSE_NAME:
            style = dict(marker="s", linestyle="--")
        elif name == INPUT_NAME:
            style = dict(marker="^", linestyle="-.", color="dimgray")
        else:
            style = dict(marker="o")
        mean = np.asarray(series[name]["wer"], dtype=float)
        lo = np.asarray(series[name]["wer_lo"], dtype=float)
        hi = np.asarray(series[name]["wer_hi"], dtype=float)
        # asymmetric yerr; clip tiny negative gaps from float roundoff
        yerr = np.vstack([np.maximum(0, mean - lo), np.maximum(0, hi - mean)])
        plt.errorbar(param_values, mean, yerr=yerr, label=name, capsize=3, **style)
    if baseline_wer is not None:
        plt.axhline(baseline_wer, color="black", linestyle=":", label="clean baseline WER")
    plt.xlabel(xlabel)
    plt.ylabel("Word Error Rate")
    plt.title(f"Word Error Rate vs {xlabel} (error bars: 95% bootstrap CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{out_prefix}_wer.png"), dpi=200)
    plt.close()
