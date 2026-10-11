# MPE

Code for the paper *Is Joint Re-Orthogonalization Necessary? Excitation Search in Multi-Pulse LPC for ASR*.

Implements several excitation-search algorithms for multi-pulse LPC (MP, OMP, Comp, LST, GCLST, MPOG - see `algorithms.py`), closed-loop frame synthesis (`synth_core.py`), and scripts to evaluate them on LibriSpeech by WER, SNR, excitation MSE, and bitrate.

## Setup

```bash
pip install -r requirements.txt
```

Scripts expect a LibriSpeech split under `librispeech_data/` (e.g. `test-clean`, `test-other`), loaded via `mpe_common.load_subset`.

## Usage

Most scripts share `--p` (LPC order), `--k-values`/`-K` (pulse counts), `--algorithms`, `--split`, and `--n-subset`, and write their output CSV/PNG to `experiment_results/`.

```bash
# WER vs. pulse count, per algorithm
python pulse_sweep.py --k-values 5,10,20,30 --algorithms omp,mp,comp,lst,gclst,mpog

# SNR vs. pulse count
python snr.py --k-values 5,10,20,30

# Excitation MSE against OMP as reference
python mse.py

# Per-utterance paired WER deltas between algorithm pairs (needs pulse_sweep.py --per-utterance output)
python delta_wer.py

# LPC-10 (2400 bps) reference point by WER
python lpc10_wer.py

# Region/single-frame excitation comparison plot
python single_frame_excitation.py

# Jetson inference-time profiling
python jetson_mpe_profile.py
```

Run any script with `--help` for its full option list.

## Layout

- `algorithms.py` — excitation-search methods
- `synth_core.py` — closed-loop MPE frame synthesis, bitrate math
- `lsf.py` — LPC/LSF conversion and quantization
- `mpe_common.py` — shared dataset loading and constants
- `tools/` — external codec references (e.g. LPC-10)
- `experiment_results/` — script outputs

