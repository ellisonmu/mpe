import numpy as np
from scipy.linalg import solve_triangular

def _finalize(H, s, loc):
    if len(loc) == 0:
        loc = [0]
    mag, *_ = np.linalg.lstsq(H, s, rcond=None)
    return mag, loc

def lst(h, s, K=100):
    N = len(s)
    corr = np.correlate(s, h, mode="full")[len(h)-1:]
    loc = list(np.argsort(-np.abs(corr), kind="stable")[:K])
    H = np.zeros((N, len(loc)))
    for i, Omega in enumerate(loc):
        H[Omega:, i] = h[:N-Omega]
    return _finalize(H, s, loc)

def mp(h, s, K=100):
    N = len(s)
    r = s.copy()
    loc = []
    mag = []
    for _ in range(K):
        corr = np.correlate(r, h, mode="full")[len(h)-1:]
        # corr[loc] = 0                      
        Omega = int(np.argmax(np.abs(corr)))
        c = corr[Omega] / np.sum(h[:N-Omega] ** 2)
        r[Omega:] = r[Omega:] - c * h[:N-Omega]
        loc.append(Omega)
        mag.append(c)
    return np.array(mag), loc

def mpog(h, s, K=100):
    N = len(s)
    r = s.copy()
    loc = []
    mag = np.zeros((N, 1))
    for _ in range(K):
        corr = np.correlate(r, h, mode="full")[len(h)-1:]                 
        Omega = int(np.argmax(np.abs(corr)))
        c = corr[Omega] / np.sum(h[:N-Omega] ** 2)
        r[Omega:] = r[Omega:] - c * h[:N-Omega]
        if Omega not in loc:
            loc.append(Omega)
        mag[Omega] += c 
    return mag[mag != 0].tolist(), np.nonzero(mag)[0].tolist()


def omp(h, s, K=100):
    N = len(s)
    err = s.copy()
    loc = []
    H = np.empty((N, 0))
    for _ in range(K):
        corr = np.correlate(err, h, mode="full")[len(h)-1:]
        corr[loc] = 0                      
        Omega = np.argmax(np.abs(corr))
        loc.append(Omega)
        atom = np.zeros(N)
        atom[Omega:] = h[:N-Omega]
        H = np.column_stack((H, atom))
        mag, _, _, _ = np.linalg.lstsq(H, s, rcond=None)
        err = s - H @ mag
    return _finalize(H, s, loc)

def comp(h, s, K=100):
    N = len(s)
    err = s.copy()
    loc = []
    H = np.zeros((N, K))                
    Lc = np.zeros((K, K))               
    rhs = np.zeros(K)                   
    for k in range(K):
        corr = np.correlate(err, h, mode="full")[len(h)-1:]
        # corr[loc] = 0  
        Omega = int(np.argmax(np.abs(corr)))
        atom = np.zeros(N)
        atom[Omega:] = h[:N-Omega]
        if k == 0:
            d2 = atom @ atom
        else:
            g = H[:, :k].T @ atom
            w = solve_triangular(Lc[:k, :k], g, lower=True)
            Lc[k, :k] = w
            d2 = atom @ atom - w @ w
        if not np.isfinite(d2) or d2 <= 1e-12:
            break
        Lc[k, k] = np.sqrt(d2)
        H[:, k] = atom
        rhs[k] = atom @ s
        loc.append(Omega)
        z = solve_triangular(Lc[:k+1, :k+1], rhs[:k+1], lower=True)
        mag = solve_triangular(Lc[:k+1, :k+1].T, z, lower=False)
        err = s - H[:, :k+1] @ mag
    return _finalize(H[:, :len(loc)], s, loc)

def dense(h, s, K=None):
    N = len(s)
    loc = list(range(N))
    H = np.column_stack([
        np.concatenate((np.zeros(n), h[:N - n]))
        for n in loc
    ])
    return _finalize(H, s, loc)

def gclst(h, s, K=100):
    N = len(s)
    loc = []
    H = np.zeros((N, K))    # preallocated: gclst always fills all K columns,
                             # since corr[loc]=0 rules out ever repeating a pick
    corr = np.correlate(s, h, mode="full")[len(h)-1:]
    auto = np.correlate(h, h, mode="full")
    gram = np.zeros(N)
    for k in range(K):
        Omega = int(np.argmax(np.abs(corr)))
        nz = np.where(corr != 0)[0]
        gram[nz] = auto[len(h)-1 + nz - Omega]
        corr = corr - gram * (corr[Omega] / np.sum(h[:N-Omega] ** 2))
        loc.append(Omega)
        H[Omega:, k] = h[:N-Omega]
        corr[loc] = 0
    return _finalize(H, s, loc)

