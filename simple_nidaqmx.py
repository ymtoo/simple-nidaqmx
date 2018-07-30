"""Multiple output transmission and multiple input reception using nidaqmx"""
import numpy as _np
import os as _os

import nidaqmx 
from nidaqmx.constants import Edge
from nidaqmx.utils import flatten_channel_string
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter

def check_driver():
    """Check NI DAQ driver."""
    system = nidaqmx.system.System.local()
    print(system.driver_version)

def momi(signal, fs, aolist, ailist, rangeval, savedirname):
    """Perform multiple output transmission and multiple input reception. The     recorded data is a 2-D array which each row consists of a channel         
    recording. The implementation is referred to test_read_write.py from           nidaqmx.
    
    :signal: transmitted signals
    :fs: sampling rate 
    :aolist: list of analog output channel name
    :ailist: list of analog input channel name
    :rangeval: list contains minimum and maximum amplitude values of the transmitted and received signals
    :savedirname: directory and filename to save the recording
    """
    system = nidaqmx.system.System.local()
    devicename = system.devices[0].name
    outputchannel = [devicename+'/'+ao for ao in aolist]
    inputchannel = [devicename+'/'+ai for ai in ailist]
    numoutputchannel = len(outputchannel)
    numinputchannel = len(inputchannel)
    
    numsignal, numsample = signal.shape
    if numsignal != numoutputchannel:
        raise ValueError('dimension mismatch')
    print('Device name: {}'.format(devicename))
    print('Output channel: {}'.format(outputchannel))
    print('Input channel: {}'.format(inputchannel))
    
    minval, maxval = rangeval
    with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task, \
                nidaqmx.Task() as sample_clk_task:
        sample_clk_task.co_channels.add_co_pulse_chan_freq(
                '{0}/ctr0'.format(devicename), freq=fs)
        sample_clk_task.timing.cfg_implicit_timing(
                samps_per_chan=numsample)
    
        samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(devicename)
    
        write_task.ao_channels.add_ao_voltage_chan(
                flatten_channel_string(outputchannel), max_val=maxval,                         min_val=minval)
        write_task.timing.cfg_samp_clk_timing(
                fs, source=samp_clk_terminal, active_edge=Edge.RISING,
                samps_per_chan=numsample)
    
        read_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(inputchannel), max_val=maxval,   
                min_val=minval)
        read_task.timing.cfg_samp_clk_timing(
                fs, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=numsample)
    
        writer = AnalogMultiChannelWriter(write_task.out_stream)
        reader = AnalogMultiChannelReader(read_task.in_stream)
    
        writer.write_many_sample(signal)
    
        read_task.start()
        write_task.start()
        sample_clk_task.start()
    
        data = _np.zeros((numinputchannel, numsample), dtype=_np.float64)
        reader.read_many_sample(
                data, number_of_samples_per_channel=numsample,
                timeout=2)
        
        if _os.path.isfile(savedirname):
            print('The file exists. The current data has not been saved.' )
        else:
            _np.save(savedirname, data)
        return data