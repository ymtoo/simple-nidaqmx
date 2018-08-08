"""Generate output signals"""
import numpy as _np
import scipy.signal as _sig

def monochromatic(A, freq, fs, numsamptrans, taper=None):
    """Generate a monochromatic signal.
    
    :param A: amplitude of the signal
    :param freq: frequency of the signal
    :param fs: sampling rate
    :param numsamptrans: number of transmitting samples 
    :param taper: tapering ratio between 0 and 1 (0 yields rectangular window)
    :returns: a monochromatic signal and the corresponding time grid
    """
    timegrid = (_np.arange(numsamptrans)/fs).reshape((1, numsamptrans))
    signal = A*_np.cos(2*_np.pi*freq*timegrid)
    if taper is not None:
        signal = signal*_sig.tukey(numsamptrans, taper)
    return signal, timegrid

def pulse_train(A, freq, fs, numsamptrans, numpulse, numsamppulsenz, taper=None):
    """Generate a pulse train of monochromatic signals.
    
    :param A: amplitude of the signal
    :param freq: frequency of the signal
    :param fs: sampling rate
    :param numsamptrans: number of transmitting samples 
    :numpulse: number of pulses of the signal
    :numsamppulsenz: number of nonzero samples of a pulse
    :param taper: tapering ratio between 0 and 1 (0 yields rectangular window)
    :returns: a pulse train and the corresponding time grid
    """
    timegrid = (_np.arange(numsamptrans)/fs).reshape((1, numsamptrans))
    numsamppulse = numsamptrans//numpulse
    signal = _np.zeros((numpulse, numsamppulse))
    for i in range(numpulse):
        signalpulsenz, _ =  monochromatic(A, freq, fs, numsamppulsenz, taper)
        signal[i, 0:numsamppulsenz] = signalpulsenz
    signal = signal.reshape((1, numsamptrans))
    return signal, timegrid