# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:58:59 2019

@author: trosmeisl, hagonzalezd
"""
import socket
import time
import multiprocessing as mp
import threading
import struct
import sys
import traceback

import numpy as np
import adc as dl
import dsp
#import visualization
import plot
import timing
#7a12

class Parameters:
    def __init__(self, adc_params=(128, 4, 2, 256, 2, 2, 128),
                 sensor_params=(1.5351e9, 77e9, 60e-6, 48e-6),
                 dsp_params=("HANNING", "HANNING", None),
                 plot_params=("rd", 5000, 5000, 3000),
                 general_params=()):
        """
        Holds dictionary with all needed constants and parameters.

        adc_params: (chirps, recievers, transmitters, samples, bytes per sample, inphase-quadrature)
        sensor_params: (radar bandwidth, carrier frequency, chirp time active, chirp time idle)
        dsp_params: (range window, doppler window, other window, hadamard)
        plot_params: (plots, norm rc, norm rd, norm md)
        general_params: ()
        """

        self.FTL = 3e8
        self.params = {"chirps": adc_params[0],
                       "rx": adc_params[1],
                       "tx": adc_params[2],
                       "samples": adc_params[3],
                       "IQ": adc_params[5],
                       "b": adc_params[4],
                       "micro_doppler": adc_params[6],
                       "rangewin": dsp_params[0],
                       "dopplerwin": dsp_params[1],
                       "bandwidth": sensor_params[0],
                       "carrier": sensor_params[1],
                       "c_active": sensor_params[2],
                       "c_idle": sensor_params[3],
                       "plots": plot_params[0],
                       "plot_norm": plot_params[1]}
        self.params.update({"channels": self.params["rx"] * self.params["tx"],
                            "doppler": self.params["chirps"] // self.params["tx"],
                            "range": self.params["samples"] // 2,
                            "chirp_time": self.params["c_active"] + self.params["c_idle"]})
        self.params.update({"frame_time": self.params["chirps"] * self.params["chirp_time"],
                            "range_res": self.FTL / (2 * self.params["bandwidth"])})
        self.params.update({"velocity_res": (self.FTL / self.params["carrier"]) / (2 * self.params["frame_time"])})

    def get(self, key):
        return self.params[key]

    def __call__(self, key):
        return self.params[key]

    def __str__(self):
        return str(self.params)

def recording(data_pipe, info_queue, dca, params):
    #frames = np.zeros((128, params.get("chirps"), params.get("rx"), params.get("samples")))
    counter = 0
    while(True):
        try:
            frame = dca.organize(dca.read(2), params.get("chirps"), params.get("rx"), params.get("samples"))
        except:
            #dca._stop_stream()
            dca.close()
            print("socket timed out")
            print(traceback.format_exc())
            break
        data_pipe.put(frame)
        print("frame", counter,)# "captured and send, lost packets:", dca.lost_packets)
        counter += 1


def radar_capture():
    # TODO: kwarg or CLI
    #plots = ("range_chirp", "range_doppler") # or None
    plots = "rd"

    dca = dl.DCA1000()
    dca.configure()

    # maybe implement the basic parameters as kwarg
    adc_params = Parameters()
    
    dsp_pipe_rec = dsp_pipe_srv = mp.Queue(0) #double assignment just in case of pipe implementation, less work
    plot_pipe_rec = plot_pipe_srv = mp.Queue(0)
    

    info_queue = mp.Queue(0)
    
    print("CPU cores: ", mp.cpu_count())
    processes = []
    try:
        processes.append(mp.Process(target = recording, name="proc_recording", args=(dsp_pipe_srv, info_queue, dca, adc_params)))
        processes.append(mp.Process(target = dsp.dsprocessing, name="proc_dsprocessing", args=(dsp_pipe_rec, plot_pipe_srv, adc_params)))
        if plots:
            processes.append(mp.Process(target = plot.plotting, name="proc_plotting", args=(plot_pipe_rec, adc_params)))

        for process in processes:
            process.deamon = True
            process.start()
            
        print("all processes running, PIDs are:")
        for process in processes:
            print(process.pid, process.name)
    except:
        print (traceback.format_exc())

    try:
        processes[-1].join()
        print("plot process has ended")
        processes.pop(-1)
        #process.close()
        for process in processes:
            process.join()
            #process.close()
    except:
        print(traceback.format_exc())
    print("join completed")


if __name__ == "__main__":
    radar_capture()
