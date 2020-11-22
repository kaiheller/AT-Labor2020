"""
1st author:
Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
couvreur@thor.fpms.ac.be
Last modification: Aug. 20, 1997, 10:00am.

References:
   [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.

2nd author:
Frank Schultz, FG Audiokommunikation, TU Berlin, 25.06.2009
frank.schultz@tu-berlin.de | +49 175 15 49 763 | Skype: j0shiiv

3rd author - translation into Python:
Kai Jurgeit, FG Audiokommunikation, TU Berlin, 08.11.2020
jurgeit@tu-berlin.de
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def a_weighting(fs=48000):
    """Digital A-Weighting Filter Design

    Parameters
    ----------
    fs : int
        sampling frequency

    Returns
    -------
    z : ndarray
        Numerator of the transformed digital filter transfer function.
    p : ndarray
        Denominator of the transformed digital filter transfer function.
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    NUMs = np.array([(2 * np.pi * f4)**2 * (10**(A1000 / 20)), 0, 0, 0, 0])
    DENs = np.convolve(
        [1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1)**2])
    DENs = np.convolve(
        np.convolve(DENs, [1, 2 * np.pi * f3]),
        [1, 2 * np.pi * f2])
    return signal.bilinear(NUMs, DENs, fs)


def c_weighting(fs=48000):
    """
    Digital C-Weighting Filter Design

    Parameters
    ----------
    fs : int
        sampling frequency

    Returns
    -------
    z : ndarray
        Numerator of the transformed digital filter transfer function.
    p : ndarray
        Denominator of the transformed digital filter transfer function.
    """
    f1 = 20.598997
    f4 = 12194.217
    C1000 = 0.0619
    np.pi = 3.14159265358979
    NUMs = np.array([(2 * np.pi * f4)**2 * (10**(C1000 / 20)), 0, 0])
    DENs = np.convolve(
        [1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1)**2])
    return signal.bilinear(NUMs, DENs, fs)
