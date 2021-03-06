# -*- coding: utf-8 -*-
"""
Created 2020

@author: trosmeisl, hagonzalezd
"""
#import multiprocessing as mp
import sys
import traceback

import numpy as np
import adc as dl

"""
adc_params: (chirps, recievers, transmitters, samples, bytes per sample, inphase-quadrature)
sensor_params: (radar bandwidth, carrier frequency, chirp time active, chirp time idle)
dsp_params: (range window, doppler window, other window, hadamard)
plot_params: (plots, norm rc, norm rd, norm md)
general_params: ()
"""
class Parameters:
    def __init__(self, adc_params=(128, 4, 2, 256, 2, 2),
                 sensor_params=(1.5351e9, 77e9, 60e-6, 48e-6),
                 dsp_params=("HANNING", "HANNING", None),
                 plot_params=("rd", 50000, 50000, 30000),
                 general_params=()):

        self.FTL = 3e8
        self.params_dict = {"chirps": adc_params[0],
                            "rx": adc_params[1],
                            "tx": adc_params[2],
                            "samples": adc_params[3],
                            "IQ": adc_params[5],
                            "b": adc_params[4],
                            "rangewin": dsp_params[0],
                            "dopplerwin": dsp_params[1]}
        self.params_dict.update({"channels": self.params_dict["rx"] * self.params_dict["tx"],
                                 "doppler": self.params_dict["chirps"] // self.params_dict["tx"],
                                 "range": self.params_dict["samples"] // 2})

    def get(self, key):
        return self.params_dict[key]

"""
1. Reads from one frame from ethernet (buffer).
2. Splits it into real and imagenary part.
3. Separates the channels and flattens the arrays.
4. Returns the flattened arrays and the correct shape to restore the whole frame.

dca    - instance of the radar device
params - adc (at least) parameters for correct operation

return - flattened real and im arrays, original shape
"""
def recording(dca, params):
    #print("trying to read from sensor")
    try:
        real_frame, im_frame, shape = dca.organize_ros(dca.read(2), params.get("chirps"), params.get("rx"), params.get("samples"))
    except Exception as e:
        print("The socket has likely timed out and end of transmission is reached OR an other error occured. \nUncomment next line to get info.")
        #print(e)
        print("EXITING")
        exit()
    real_frame = real_frame.reshape(shape)
    im_frame = real_frame.reshape(shape)
    real_frame = dca.separate_tx_old(real_frame, params.get("tx")).flatten()
    im_frame = dca.separate_tx_old(im_frame, params.get("tx")).flatten()
    print("frame captured, lost packets total:", dca.total_packets_lost, "\n arrived packets total:", dca.total_packets)
    shape = params.get("doppler"), params.get("channels"), params.get("samples")
    return real_frame, im_frame, shape

"""
Sets up the radar device and returns default parameter setup for correct operation.

return - (default) parameters, device instance
"""
def setup_radar():
    # TODO: kwarg or CLI
    #plots = ("range_chirp", "range_doppler") # or None
    plots = None

    dca = dl.DCA1000()
    dca.configure()

    # maybe implement the basic parameters as kwarg
    adc_params = Parameters()

    return adc_params, dca
    
