"""Multiple output voltage transmission and multiple input voltage reception using nidaqmx"""
import numpy as _np
import os as _os
import time

import nidaqmx 
from nidaqmx.constants import Edge
from nidaqmx.utils import flatten_channel_string
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter

def check_driver():
    """Check NI DAQ driver."""
    system = nidaqmx.system.System.local()
    print(system.driver_version)

def momi(signal, fs, aolist, ailist, rangeval, numsamprec, savedirname=None):
    """Perform multiple output voltage transmission and multiple input voltage reception. The recorded data is a 2-D array which each row consists of a channel recording. The implementation is referred to test_read_write.py from nidaqmx.
    
    :params signal: transmitted signals
    :params fs: sampling rate 
    :params aolist: list of analog output channel name
    :params ailist: list of analog input channel name
    :params rangeval: list contains minimum and maximum amplitude values of the transmitted and received signals
    :params numsamprec: number of recording samples 
    :params savedirname: directory and filename to save the recording (None means do not save)
    :params returns: 2-D array with the last column as the reference signal and others as the recorded data
    """
    system = nidaqmx.system.System.local()
    devicename = system.devices[0].name
    outputchannel = [devicename+'/'+ao for ao in aolist]
    inputchannel = [devicename+'/'+ai for ai in ailist]
    numoutputchannel = len(outputchannel)
    numinputchannel = len(inputchannel)
    
    numsignal, numsamptrans = signal.shape
    if numsignal != numoutputchannel:
        raise ValueError('dimension mismatch')
    if numsamptrans > numsamprec:
        print("Number of samples of the transmitted signals is greater than the recordings.")
    print("Device name: {}".format(devicename))
    print("Output channel: {}".format(outputchannel))
    print("Input channel: {}".format(inputchannel))
    
    minval, maxval = rangeval
    numsampclk = max(numsamptrans, numsamprec)
    timeout = _np.ceil(numsamprec/fs)
    with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task, nidaqmx.Task() as sample_clk_task:
        sample_clk_task.co_channels.add_co_pulse_chan_freq(
                '{0}/ctr0'.format(devicename), freq=fs)
        sample_clk_task.timing.cfg_implicit_timing(
                samps_per_chan=numsampclk)
    
        samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(devicename)
    
        write_task.ao_channels.add_ao_voltage_chan(
                flatten_channel_string(outputchannel), max_val=maxval, 
                min_val=minval)
        write_task.timing.cfg_samp_clk_timing(
                fs, source=samp_clk_terminal, active_edge=Edge.RISING,
                samps_per_chan=numsamptrans)
    
        read_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(inputchannel), max_val=maxval, 
                min_val=minval)
        read_task.timing.cfg_samp_clk_timing(
                fs, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=numsamprec)
    
        writer = AnalogMultiChannelWriter(write_task.out_stream)
        reader = AnalogMultiChannelReader(read_task.in_stream)
    
        writer.write_many_sample(signal)
    
        read_task.start()
        write_task.start()
        sample_clk_task.start()
    
        data = _np.zeros((numinputchannel, numsamprec), dtype=_np.float64)
        reader.read_many_sample(
                data, number_of_samples_per_channel=numsamprec,
                timeout=timeout)
    
        if savedirname is not None:
            if _os.path.isfile('.'.join([savedirname, 'npy'])):
                print('The file exists. The current data has not been saved.' )
            else:
                _np.save(savedirname, data)
        return data
    
def repeat_momi(signal, fs, aolist, ailist, rangeval, numsamprec, savedirname=None, numrep=1, pausetime=0, startat=0):
    """Perform repeated momi.
        
    :params signal: transmitted signals
    :params fs: sampling rate 
    :params aolist: list of analog output channel name
    :params ailist: list of analog input channel name
    :params rangeval: list contains minimum and maximum amplitude values of the transmitted and received signals
    :params numsamprec: number of recording samples 
    :params savedirname: directory and filename to save the recording (None means do not save)
    :params numrep: number of repeated recordings (default is 1)
    :params pausetime: puase time in seconds between two consecutive recordings (default is 0)
    :params startat: starting index of repeated recordings
    """
    print("Number of recordings is {}.".format(numrep))
    for i in range(numrep):
        idx = i+startat
        print("{}". format(idx), end=' ')
        if savedirname is not None:
            savedirnameinst = '_'.join([savedirname, '{}'.format(idx)])
        else:
            savedirnameinst = savedirname # None
        momi(signal, fs, aolist, ailist, rangeval, numsamprec, savedirname=savedirnameinst)
        time.sleep(pausetime)