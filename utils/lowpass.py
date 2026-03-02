import numpy as np
import librosa

def lowpass(data, cutoff, fs, order=1):
    y_input = np.asarray(data, dtype=np.float64)
    y_filtered = librosa.butter_lp(y=y_input, cutoff=cutoff, fs=fs, order=order)
    return y_filtered

if __name__ == "__main__":
    fs = 22050
    t = np.linspace(0, 1, fs)
    data = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t)
    result = lowpass(data, 200, fs, order=2)
    print(result.shape)