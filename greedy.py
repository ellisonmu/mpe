import numpy as np

def _finalize(A_curr, b, S_curr):
    if len(S_curr) == 0:
        S_curr = [0]
    x_curr, *_ = np.linalg.lstsq(A_curr, b, rcond=None)
    r_curr = b - A_curr @ x_curr
    return x_curr, r_curr, S_curr

#Orthogonal Matching Pursuit
def omp(h, b, K=100, eps=0.01):
    N = len(b)
    r_curr = b.copy()
    S_curr = []
    A_curr = np.empty((N, 0))
    for _ in range(K):
        corr = np.correlate(r_curr, h, mode="full")[len(h)-1:]
        Omega_curr = np.argmax(np.abs(corr))
        S_curr.append(Omega_curr)
        atom = np.zeros(N)
        atom[Omega_curr:] = h[:N-Omega_curr]
        A_curr = np.column_stack((A_curr, atom))
        corr[S_curr] = 0
        x_curr, _, _, _ = np.linalg.lstsq(A_curr, b, rcond=None)
        r_curr = b - A_curr @ x_curr
    return _finalize(A_curr, b, S_curr)

def thresholding(h, b, K=100, eps=0.01):
    N = len(b)
    r_curr = b.copy()
    S_curr = []
    A_curr = np.empty((N, 0))
    corr = np.correlate(r_curr, h, mode="full")[len(h)-1:]
    for _ in range(K):
        Omega_curr = np.argmax(np.abs(corr))
        S_curr.append(Omega_curr)
        atom = np.zeros(N)
        atom[Omega_curr:] = h[:N-Omega_curr]
        A_curr = np.column_stack((A_curr, atom))
        corr[S_curr] = 0
    return _finalize(A_curr, b, S_curr)

def thresholding_gram(h, b, K=100, eps=0.01):
    N = len(b)
    S_curr = []
    A_curr = np.empty((N, 0))
    corr = np.correlate(b, h, mode="full")[len(h)-1:]
    h_auto = np.correlate(h, h, mode="full")
    gram_col = np.zeros(N)
    for _ in range(K):
        Omega_curr = int(np.argmax(np.abs(corr)))
        for j in np.where(corr != 0)[0]:
            lag = j - Omega_curr
            gram_col[j] = h_auto[len(h)-1 + lag]
        corr = corr - gram_col * (corr[Omega_curr] / np.sum(h**2))
        S_curr.append(Omega_curr)
        atom = np.zeros(N)
        atom[Omega_curr:] = h[:N-Omega_curr]
        A_curr = np.column_stack((A_curr, atom))
        corr[S_curr] = 0
    return _finalize(A_curr, b, S_curr)

def visualize(audio, label):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"{label}_waveform.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram")
    plt.tight_layout()
    plt.savefig(f"{label}_spectrogram.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(audio, bins=100)
    plt.title("Waveform Sample Distribution")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{label}_histogram.png", dpi=300)
    plt.close()