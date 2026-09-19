import numpy as np

DELTA_MAX = 0.60
DELTA_MAX_RATIO = DELTA_MAX / (np.pi / 11)

def delta_max_for_order(p):
    """LSF-delta quantizer full-scale, scaled to order p so the p=10 tuning
    (DELTA_MAX) generalizes instead of silently mis-sizing at other orders."""
    return DELTA_MAX_RATIO * np.pi / (p + 1)


def _build_pq(A):
    """Padded P(z), Q(z) coefficients (ascending powers of u = z^-1)."""
    a_padded = np.concatenate((A, [0.0]))       # A(z), degree p+1
    shifted = np.concatenate(([0.0], A[::-1]))  # z^-(p+1) A(z^-1), degree p+1
    return a_padded + shifted, a_padded - shifted


def _deflate(coeffs, sign):
    """Synthetic-divide `coeffs` (ascending powers of u) by (1 + sign*u)."""
    n = len(coeffs) - 1
    q = np.zeros(n)
    q[0] = coeffs[0]
    for i in range(1, n):
        q[i] = coeffs[i] - sign * q[i - 1]
    return q


def _cosine_eval(c, w):
    """Evaluate a palindromic (length p+1, even p) polynomial as a cosine sum."""
    p = len(c) - 1
    half = p // 2
    val = np.full_like(w, c[half], dtype=float)
    for k in range(half):
        val += 2.0 * c[k] * np.cos((half - k) * w)
    return val


def _bisect_roots(c, grid, n_iter=8):
    vals = _cosine_eval(c, grid)

    exact_roots = grid[vals == 0.0]

    sign_change = vals[:-1] * vals[1:] < 0
    idx = np.where(sign_change)[0]
    if idx.size == 0:
        return np.sort(exact_roots)

    lo, hi, flo = grid[idx].copy(), grid[idx + 1].copy(), vals[idx].copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fmid = _cosine_eval(c, mid)
        take_hi = flo * fmid <= 0
        hi = np.where(take_hi, mid, hi)
        lo = np.where(take_hi, lo, mid)
        flo = np.where(take_hi, flo, fmid)

    return np.sort(np.concatenate([exact_roots, 0.5 * (lo + hi)]))


def lpc_to_lsf(A, grid_size=None):
    A = np.asarray(A, dtype=float)
    p = len(A) - 1
    if p % 2 != 0:
        raise ValueError("lpc_to_lsf only supports even LPC order")
    if grid_size is None:
        grid_size = max(256, 16 * (p + 1))

    P, Q = _build_pq(A)
    P1 = _deflate(P, +1.0)   # remove root at z = -1
    Q1 = _deflate(Q, -1.0)   # remove root at z = +1

    grid = np.linspace(1e-4, np.pi - 1e-4, grid_size)
    p_roots = _bisect_roots(P1, grid)
    q_roots = _bisect_roots(Q1, grid)

    half = p // 2
    if len(p_roots) != half or len(q_roots) != half:
        raise RuntimeError(
            f"expected {half} roots each from P/Q, got {len(p_roots)}/{len(q_roots)} "
            "(frame likely not strictly minimum-phase)"
        )

    lsf = np.empty(p)
    lsf[0::2] = p_roots   # interlacing theorem: P-roots occupy even 0-indexed slots
    lsf[1::2] = q_roots   # Q-roots occupy odd 0-indexed slots

    if not np.all(np.diff(lsf) > 0):
        raise RuntimeError("P/Q roots not strictly interlaced (frame likely ill-conditioned)")
    return lsf

def lsf_to_lpc(lsf):
    """Reconstruct LPC coefficients from p LSFs (interlaced P/Q ordering)."""
    lsf = np.sort(np.asarray(lsf, dtype=float))
    p_omegas = lsf[0::2]
    q_omegas = lsf[1::2]

    def quad_product(omegas):
        poly = np.array([1.0])
        for w in omegas:
            poly = np.convolve(poly, [1.0, -2.0 * np.cos(w), 1.0])
        return poly

    P_hat = np.convolve(quad_product(p_omegas), [1.0, 1.0])
    Q_hat = np.convolve(quad_product(q_omegas), [1.0, -1.0])
    A_hat = 0.5 * (P_hat + Q_hat)
    return A_hat[:-1]   # trailing coefficient is ~0 by construction

def quantize_lsfs(lsf, n_bits=3, min_gap=0.02, delta_max=DELTA_MAX, w_max=np.pi,
                  return_indices=False):
    lsf = np.sort(np.asarray(lsf, dtype=float))
    p = lsf.size
    max_index = 2 ** n_bits - 1
    step = delta_max / max_index

    lsf_hat = np.empty(p)
    indices = np.empty(p, dtype=int)
    prev = 0.0
    for i in range(p):
        delta = lsf[i] - prev
        index = min(max(round(delta / step), 0), max_index)
        prev = prev + max(index * step, min_gap)
        lsf_hat[i] = prev
        indices[i] = index

    # top end can still spill past w_max on a large final delta; pull just
    # the tail back down to preserve strict ordering.
    for i in range(p - 1, -1, -1):
        ceiling = w_max - min_gap if i == p - 1 else lsf_hat[i + 1] - min_gap
        if lsf_hat[i] > ceiling:
            lsf_hat[i] = ceiling
    # `indices` are the transmitted symbols; the tail pull-down above is a
    # decoder-side repair, so it does not change them.
    return (lsf_hat, indices) if return_indices else lsf_hat

def quantize_excitation(m, n_bits=3):
    """Symmetric uniform PTQ of pulse magnitudes: per-frame gain + fixed-level code."""
    m = np.asarray(m, dtype=float)
    scale = np.max(np.abs(m)) if m.size else 0.0
    if scale < 1e-12:
        return m.copy()

    levels = 2 ** (n_bits - 1) - 1
    normalized = m / scale
    q_normalized = np.clip(np.round(normalized * levels), -levels, levels) / levels
    return q_normalized * scale
