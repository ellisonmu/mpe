import os
import librosa
import numpy as np
from scipy.signal import lfilter
from algorithms import omp, lst, gclst
import soundfile as sf
from demo.plotting import visualize

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

def weighting_filter(A, gamma=1):
    p = len(A) - 1
    powers = gamma ** np.arange(p + 1)
    A_gamma = A * powers
    return A, A_gamma

def synthesize(fn, x, p, L, K, gamma=1):
    out = np.zeros(N)
    for i in range(n_frames):
        start, end = i * L, (i + 1) * L
        seg = x[start:end]
        if len(seg) <= p or np.sum(seg**2) < 1e-12:
            continue
        A = fit_lpc(seg, p)
        h = frame_h(A, L)
        if not np.all(np.isfinite(h)) or np.abs(h).max() > 1e6:
            continue
        
        ring = np.zeros(L)
        hist = np.pad(out[max(0, start - p):start][::-1], (0, max(0, p - start)))
        buf = np.concatenate((hist, np.zeros(0)))
        for n in range(L):
                    y = -np.dot(A[1:], buf[:p])
                    ring[n] = y
                    buf = np.concatenate(([y], buf[:-1]))
        target = seg - ring
        
        b_w, a_w = weighting_filter(A, gamma)
        target_w = lfilter(b_w, a_w, target)
        h_w = lfilter(b_w, a_w, h)
        
        e = np.zeros(L)
        m, locs = fn(h_w, target_w, K=K)
        e[locs] = m
        
        out[start:end] = ring + np.convolve(e, h)[:L]
    return out

x_omp = synthesize(omp, x, p, L, n_pulses)
x_lst = synthesize(lst, x, p, L, n_pulses)
x_gclst = synthesize(gclst, x, p, L, n_pulses)

save(x_omp, "omp", sr)
save(x_lst, "lst", sr)
save(x_gclst, "gclst", sr)
