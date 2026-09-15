import os
import librosa
import numpy as np
from scipy.signal import lfilter
from greedy import thresholding, omp, thresholding_gram
import soundfile as sf
from plotting import visualize

def save(audio, label, sr):
    folder = label
    os.makedirs(folder, exist_ok=True)
    sf.write(os.path.join(folder, f"{label}.wav"), audio, sr)
    visualize(audio, label, sr=sr)

x, sr = librosa.load("Achird.wav", sr=16000)
duration = 3
x = x[:3*sr]
save(x, "input", sr)

N = x.size

p = 10                       # LPC order (Atal & Remde use order ~10 for telephone-band speech)
frame_ms = 10                # analysis/synthesis frame length
L = int(sr * frame_ms / 1000)
n_pulses = 8                 # pulses per frame
n_frames = N // L

H_global = np.column_stack([x[p-k-1:N-k-1] for k in range(p)])
a_global = np.linalg.lstsq(H_global, -x[p:], rcond=None)[0]
x_lpc = -(H_global @ a_global)
save(x_lpc, "lpc", sr)


def lpc_matrix(seg, p):
    n = len(seg)
    H = np.column_stack([seg[p-k-1:n-k-1] for k in range(p)])
    return H

def least_squares(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]

def fit_lpc(seg, p):
    """Fit an order-p LPC filter A(z) = 1 + sum a_k z^-k to one frame."""
    H = lpc_matrix(seg, p)
    a = least_squares(H, -seg[p:])
    return np.concatenate(([1.0], a))

def frame_h(A, L):
    """Impulse response of the synthesis filter 1/A(z), truncated to the frame length."""
    impulse = np.zeros(L)
    impulse[0] = 1.0
    return lfilter([1.0], A, impulse)

def multipulse_frame(fn, h, seg, K):
    """Run a greedy pulse-selection method on one frame and rebuild the excited waveform."""
    m, _, locs = fn(h, seg, K=K)
    e = np.zeros(len(seg))
    e[locs] = m
    return np.convolve(e, h)[:len(seg)]

def synthesize(fn, x, p, L, K):
    out = np.zeros(N)
    for i in range(n_frames):
        seg = x[i*L:(i+1)*L]
        if len(seg) <= p or np.sum(seg**2) < 1e-12:
            continue
        A = fit_lpc(seg, p)
        h = frame_h(A, L)
        if not np.all(np.isfinite(h)) or np.abs(h).max() > 1e6:
            continue
        out[i*L:(i+1)*L] = multipulse_frame(fn, h, seg, K)
    return out

x_omp = synthesize(omp, x, p, L, n_pulses)
x_thresh = synthesize(thresholding, x, p, L, n_pulses)
x_thresh_gram = synthesize(thresholding_gram, x, p, L, n_pulses)

save(x_omp, "omp", sr)
save(x_thresh, "thresh", sr)
save(x_thresh_gram, "thresh_gram", sr)
