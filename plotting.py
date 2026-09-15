import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def visualize(audio, label, sr=16000, out_dir="."):
    folder = os.path.join(out_dir, label)
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "waveform.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "spectrogram.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(audio, bins=100)
    plt.title("Waveform Sample Distribution")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "histogram.png"), dpi=300)
    plt.close()