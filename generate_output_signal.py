"""Generate output signals"""
import numpy as _np

def monochromatic(A, freq, fs, numsample):
    """Generate a monochromatic signal.
    
    :param A: amplitude of the signal
    :param freq: frequency of the signal
    :param fs: sampling rate
    :param numsample: number of samples of the signal
    :returns: a monochromatic signal and the corresponding time grid
    """
    timegrid = (_np.arange(numsample)/fs).reshape((1, numsample))
    signal = A*_np.cos(2*_np.pi*freq*timegrid)
    return signal, timegrid