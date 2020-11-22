import numpy as np


def sys3(x, fs):
    """
    System 2

    Parameters
    ----------
    x : ndarray (N,)
        input signal
    fs : int
        sampling frequency

    Returns
    -------
    y : ndarray (N, )
        output signal
    xo : ndarray (N, 2)
        output signal
    """
    stretch = 10
    y = np.arctan(stretch * np.pi * x / 1000) * 1000 * 10**(-30 / 20)
    N = len(y)
    noise = np.random.uniform(size=N)
    noise = noise - np.mean(noise)  # mittelwertfrei
    noise = noise / np.std(noise)  # Effektivwert normieren
    noise = noise * 10**(-96 / 20)
    y = y + noise
    t = np.arange(0, N)
    Hum50 = 10**(-80 / 20) * np.sqrt(0.6) * np.sqrt(2) * np.cos(2 * np.pi * 60 / fs * t)
    Hum100 = 10**(-90 / 20) * np.sqrt(0.6) * np.sqrt(2) * np.cos(2 * np.pi * 120 / fs * t)
    Hum200 = 10**(-100 / 20) * np.sqrt(0.6) * np.sqrt(2) * np.cos(2 * np.pi * 180 / fs * t)
    y = y + Hum50 + Hum100 + Hum200
    return y
