"""Generate output signals"""
import numpy as _np
import scipy.signal as _sig

def monochromatic(A, freq, fs, numsample, taper=None):
    """Generate a monochromatic signal.
    
    :param A: amplitude of the signal
    :param freq: frequency of the signal
    :param fs: sampling rate
    :param numsample: number of samples of the signal
    :param taper: tapering ratio between 0 and 1 (0 yields rectangular window)
    :returns: a monochromatic signal and the corresponding time grid
    """
    timegrid = (_np.arange(numsample)/fs).reshape((1, numsample))
    signal = A*_np.cos(2*_np.pi*freq*timegrid)
    if taper is not None:
        signal = signal*_sig.tukey(numsample, taper)
    return signal, timegrid