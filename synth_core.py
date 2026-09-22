import math

import numpy as np
from scipy.signal import lfilter

from algorithms import omp, lst, gclst, dense
from lsf import (lpc_to_lsf, lsf_to_lpc, quantize_lsfs, quantize_excitation,
                 delta_max_for_order)

SR = 16000
FRAME_MS = 10
L = int(SR * FRAME_MS / 1000)          

GAMMA = 0.85                           
LSF_BITS = 3                          
EXC_BITS = 3                         
POLE_MARGIN = 0.994                    

METHODS = {
    "omp": omp,
    "lst": lst,
    "gclst": gclst,
}

DENSE_NAME = "dense_lpc"

GAIN_BITS = 6


def mpe_frame_bits(p, K, L=L, lsf_bits=LSF_BITS, exc_bits=EXC_BITS,
                   gain_bits=GAIN_BITS):
    loc_bits = int(np.ceil(np.log2(L)))
    return p * lsf_bits + K * (loc_bits + exc_bits) + gain_bits


def mpe_bitrate_kbps(p, K, L=L, sr=SR, **kw):
    """Fixed transmission rate of the sparse MPE coder, in kbit/s."""
    frames_per_sec = sr / L
    return mpe_frame_bits(p, K, L=L, **kw) * frames_per_sec / 1000.0


def mpe_frame_bits_combinatorial(p, K, L=L, n_sub=1, lsf_bits=LSF_BITS,
                                 exc_bits=EXC_BITS, gain_bits=GAIN_BITS):
    r_lsf = p * lsf_bits
    loc_bits = math.ceil(math.log2(math.comb(L, K)))
    return r_lsf + n_sub * (loc_bits + K * exc_bits + gain_bits)


def mpe_bitrate_kbps_combinatorial(p, K, L=L, sr=SR, n_sub=1, **kw):
    bits = mpe_frame_bits_combinatorial(p, K, L=L, n_sub=n_sub, **kw)
    t_frame_s = n_sub * L / sr
    return bits / t_frame_s / 1000.0

def lpc_matrix(seg, p):
    n = len(seg)
    return np.column_stack([seg[p - k - 1:n - k - 1] for k in range(p)])

def levinson_durbin(seg, p):
    seg = np.asarray(seg, dtype=float)
    full = np.correlate(seg, seg, mode="full")
    r = full[len(seg) - 1:len(seg) - 1 + p + 1]
    if r.size < p + 1 or r[0] <= 0:
        return None
    a = np.zeros(p + 1)
    a[0] = 1.0
    E = r[0]
    for i in range(1, p + 1):
        acc = r[i] + (np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else 0.0)
        k = -acc / E
        new = a.copy()
        new[i] = k
        for j in range(1, i):
            new[j] = a[j] + k * a[i - j]
        a = new
        E *= 1.0 - k * k
        if E <= 0:
            return None
    return a

def fit_lpc(seg, p):
    H = lpc_matrix(seg, p)
    A = levinson_durbin(seg, p)
    if A is None:
        A = np.concatenate(([1.0], np.zeros(p)))  

def frame_impulse_response(A, L):
    impulse = np.zeros(L)
    impulse[0] = 1.0
    return lfilter([1.0], A, impulse)

def weighting_filter(A, gamma):
    p = len(A) - 1
    powers = gamma ** np.arange(p + 1)
    return A, A * powers

def _zero_input_response(A, hist, n_samples):
    p = len(A) - 1
    ring = np.zeros(n_samples)
    buf = hist.copy()
    for n in range(n_samples):
        y = -np.dot(A[1:], buf[:p])
        ring[n] = y
        buf = np.concatenate(([y], buf[:-1]))
    return ring

def synthesize_mpe(fn, x, p, L, K, gamma=GAMMA, lsf_bits=LSF_BITS, exc_bits=EXC_BITS,
                    pole_margin=POLE_MARGIN, capture=None):
    N = len(x)
    n_frames = N // L
    out = np.zeros(N)

    global_scale = np.max(np.abs(x)) + 1e-9
    clamp = 20.0 * global_scale

    for i in range(n_frames):
        start, end = i * L, (i + 1) * L
        seg = x[start:end]
        if len(seg) <= p or np.sum(seg ** 2) < 1e-12:
            continue
        A_true, _, _ = fit_lpc(seg, p)
        hist = np.pad(out[max(0, start - p):start][::-1], (0, max(0, p - start)))

        A_hat = A_true
        if p % 2 == 0:
            try:
                lsf = lpc_to_lsf(A_true)
                candidate = lsf_to_lpc(quantize_lsfs(lsf, n_bits=lsf_bits,
                                                      delta_max=delta_max_for_order(p)))
                candidate = candidate * pole_margin ** np.arange(p + 1)
                ring = _zero_input_response(candidate, hist, len(seg))
                if np.all(np.isfinite(ring)) and np.max(np.abs(ring)) < 50 * global_scale:
                    A_hat = candidate
            except (RuntimeError, np.linalg.LinAlgError):
                pass
        A = A_hat

        h = frame_impulse_response(A, L)
        if not np.all(np.isfinite(h)) or np.abs(h).max() > 1e6:
            continue

        ring = _zero_input_response(A, hist, len(seg))
        target = seg - ring

        b_w, A_w = weighting_filter(A, gamma)
        target_w = lfilter(b_w, A_w, target)
        impulse = np.zeros(len(seg))
        impulse[0] = 1.0
        h_w = lfilter([1.0], A_w, impulse)

        if capture is not None:
            capture.append((h_w.copy(), target_w.copy()))

        m, locs = fn(h_w, target_w, K=K)
        m = quantize_excitation(m, n_bits=exc_bits)
        e = np.zeros(len(seg))
        e[locs] = m

        frame_out = ring + np.convolve(e, h)[:len(seg)]
        out[start:end] = np.clip(frame_out, -clamp, clamp)
    return out


def method_fn(name):
    return dense if name == DENSE_NAME else METHODS[name]


def synthesize_utterance_job(job):
    idx, audio, names, p, K, gamma = job
    audio = np.asarray(audio, dtype=float)
    outs = {}
    for name in names:
        k = 0 if name == DENSE_NAME else K     # dense ignores K
        outs[name] = synthesize_mpe(method_fn(name), audio, p, L, k, gamma=gamma)
    return idx, outs
