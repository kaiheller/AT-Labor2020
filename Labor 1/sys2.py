import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def sys2(xi, fs):
    """
    System 2

    Parameters
    ----------
    xi : ndarray (N, 2)
        input signal
    fs : int
        sampling frequency

    Returns
    -------
    xosys : ndarray (N, 2)
        output signal
    xo : ndarray (N, 2)
        output signal
    """
    xi = np.asarray(xi)
    fm = 6000
    Q = np.sqrt(2)
    G = 9
    QType = 'HalfPadLossQ'
    FilterType = 'PEQ'
    PreWarpType = 'Cos'

    a = 3 * [None]
    b = 3 * [None]

    a[0] = 1
    g = 10**(G / 20) - 1

    if QType == 'HalfPadLossQ':
        Qbp = Q * 10**(G / 40)
    elif QType == 'Consnp.tantQ':
        Qbp = Q
    elif QType == 'SymmetricalQ':
        Qbp = Q * 10**(G / 20) if G < 0 else Q
    else:
        Qbp = Q * 10 ** (G / 20) if G < 0 else Q
        print('invalid QType, set to default SymmetricalQ')

    if PreWarpType == 'Cos':
        Qbp *= np.cos(np.pi * fm / fs)
    elif PreWarpType == 'np.tan':
        Qbp *= (np.pi * fm / fs) / np.np.tan(np.pi * fm / fs)
    else:
        Qbp *= np.cos(np.pi * fm / fs)
        print('invalid PreWarpType, set to default Cos')

    w = np.tan(np.pi * fm / fs)
    den = Qbp + w + Qbp * w**2

    if FilterType == 'BP':
        b[0] = w / den
        b[1] = 0
        b[2] = -b[1]
        a[1] = 2 * Qbp * (w**2 - 1) / den
        a[2] = (Qbp - w + Qbp * w**2) / den

    elif FilterType == 'PEQ':
        b[0] = ((1 + g) * w + Qbp * (1 + w**2)) / den
        b[1] = 2 * Qbp * (w**2 - 1) / den
        b[2] = (-(1 + g) * w + Qbp * (1 + w**2)) / den
        a[1] = b[1]
        a[2] = (Qbp - w + Qbp * w**2) / den
    else:
        b[0] = ((1 + g) * w + Qbp * (1 + w**2)) / den
        b[1] = 2 * Qbp * (w**2 - 1) / den
        b[2] = (-(1 + g) * w + Qbp * (1 + w**2)) / den
        a[1] = b[1]
        a[2] = (Qbp - w + Qbp * w**2) / den
        print('invalid FilterType, set to default PEQ')

    # FIR-Filter:
    M = 2**15 + 1
    FIR_ORDER = 48

    df = fs / M
    freq = np.arange(0, int((M - 1) / 2) + 1) * df
    GRP_DLY_SMP = FIR_ORDER / 2
    grp_dly_s = GRP_DLY_SMP / fs

    FIR_PEQ = np.zeros((M))
    FIR_PEQ[0] = 1
    FIR_PEQ = signal.lfilter(b, a, FIR_PEQ)
    FIR_PEQ = signal.freqz(FIR_PEQ, 1, M, 'whole')[1]

    dw = 2 * np.pi * df
    phi0 = 0
    phi1 = 0
    nn = M - 1
    FIR_PEQ = np.abs(FIR_PEQ).astype(np.complex)

    for n in range(1, int((M - 1) / 2)):
        phi1 = phi0 - grp_dly_s * dw
        FIR_PEQ[n] = FIR_PEQ[n] * np.exp(1j * phi1)
        FIR_PEQ[nn] = np.conj(FIR_PEQ[n])
        nn -= 1
        phi0 = phi1

    FIR_PEQ = np.fft.ifft(FIR_PEQ)
    np.isreal(FIR_PEQ)
    FIR_PEQ = FIR_PEQ[0:FIR_ORDER + 1]

    nn = FIR_ORDER
    for n in range(0, FIR_ORDER):
        FIR_PEQ[nn] = FIR_PEQ[n]
        nn -= 1

    FIR_PEQ = FIR_PEQ.real
    # Apply filters
    xo = np.zeros(xi.shape)
    xo[:, 0] = signal.lfilter(b, a, xi[:, 0])
    xo[:, 1] = signal.lfilter(FIR_PEQ, 1, xi[:, 1])

    ##############################################
    # System High Pass ###########################
    ##############################################
    b = [None] * 3
    a = [None] * 3
    fg = 20
    G = 0
    c1 = 1 / 0.9
    c2 = 1

    Ts = 1 / fs
    wg = 2 * fs * np.tan(np.pi * fg / fs)     # Prewarping
    g = 10**(G / 20)                    # linear gain
    b[0] = 4 * g
    b[1] = -2 * b[0]
    b[2] = b[0]
    a[0] = 4 + 2 * c1 * Ts * wg + c2 * Ts**2 * wg**2
    a[1] = -8 + 2 * c2 * Ts**2 * wg**2
    a[2] = 4 - 2 * c1 * Ts * wg + c2 * Ts**2 * wg**2
    tmp = a[0]
    # final coeffs, normalized to a0=1
    aHP = a / tmp
    bHP = b / tmp

    ##############################################
    # System Low Pass ############################
    ##############################################
    b = [None] * 3
    a = [None] * 3
    fg = 15000
    G = 0
    c1 = 1 / 0.9
    c2 = 1

    Ts = 1 / fs
    wg = 2 * fs * np.tan(np.pi * fg / fs)     # Prewarping
    g = 10**(G / 20)                    # linear gain

    b[0] = g * Ts**2 * wg**2
    b[1] = 2 * b[0]
    b[2] = b[0]
    a[0] = 4 * c2 + 2 * c1 * Ts * wg + Ts**2 * wg**2
    a[1] = -8 * c2 + 2 * Ts**2 * wg**2
    a[2] = 4 * c2 - 2 * c1 * Ts * wg + Ts**2 * wg**2
    tmp = a[0]
    # final coeffs, normalized to a0=1
    aLP = a / tmp
    bLP = b / tmp

    xosys = np.zeros(xi.shape)
    xosys[:, 0] = signal.lfilter(bHP * 10**(-0 / 20), aHP, xi[:, 0])
    xosys[:, 1] = signal.lfilter(bHP * 10**(+3 / 20), aHP, xi[:, 1])
    xosys[:, 0] = signal.lfilter(bLP, aLP, xosys[:, 0])
    xosys[:, 1] = signal.lfilter(bLP, aLP, xosys[:, 1])

    xo[:, 0] = signal.lfilter(bHP * 10**(-0 / 20), aHP, xo[:, 0])
    xo[:, 1] = signal.lfilter(bHP * 10**(+3 / 20), aHP, xo[:, 1])
    xo[:, 0] = signal.lfilter(bLP, aLP, xo[:, 0])
    xo[:, 1] = signal.lfilter(bLP, aLP, xo[:, 1])

    return xosys, xo
