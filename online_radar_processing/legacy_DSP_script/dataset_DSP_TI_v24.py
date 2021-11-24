# Creation date: April 11 2019
# Created by: Hector Gonzalez, Chen Liu


# Contained in this version:
# - Baseline removal
# - Two CFAR 1D for Range and doppler are done
# - Coherent integration across channels is included
# - Range FFT
# - Doppler FFT
# - Angle FFT
#	+ RX2, RX3 and RX4 for elevation
#	+ RX2, RX1 for azimuth
# - FFT length non power of two


# Aug 12 2019. This version seems to be working fine at least in the azimuth direction.
# Check why the yellow plot does not correspond with the csv file. The yellow plot seems correct but not the coordinates.
# HG Note: Equivalent to V25

# Sept 30 2019: We have 1000 samples, which means 1000 Frames.
# Sept 30 2019: Details for Daniel Recordings:
# ar1.ProfileConfig(0, 77, 10, 6, 60, 0, 0, 0, 0, 0, 0, 29.982, 0, 256, 5000, 0, 0, 30)
#
#                          idle_time  = 10 microseconds
#                          chirp_time = 60 microseconds
#                          samples    = 256
#                          Fs         = 5 MHz
#                          Slope      = 29.982 MHz/us
#
# ar1.FrameConfig(0, 0, 1000, 128, 10, 0, 1)
#
#                          Frame_time = 10 miliseconds
#                          No_chirps  = 128 (Still maintained)
#                          No_frames  = 1000 (To achieve 1000 samples)
#
# Nov 25 2019: We have 111 samples, which means 111 Frames.
# Nov 25 2019: Details for IPCEI Recordings:
# ar2.ProfileConfig(0, 77, 10, 6, 60, 0, 0, 0, 0, 0, 0, 29.982, 0, 256, 5000, 0, 0, 30)
#
#                          idle_time  = 10 microseconds
#                          chirp_time = 60 microseconds
#                          samples    = 256
#                          Fs         = 5 MHz
#                          Slope      = 29.982 MHz/us
#
# ar2.FrameConfig(0, 0, 111, 128, 40, 0, 1)
#
#                          Frame_time = 40 miliseconds
#                          No_chirps  = 128 (Still maintained)
#                          No_frames  = 111 (To achieve 111 samples)
#
# The implications of reducing the Frame time is not seen, unless we reduce the transmission time:
# vel_resolution = lambda_mmwave/(2*Time_Frame) # If Frame time is unchanged because we still have the same number of chirps then nothing will happen, we continue with the same velocity resolution
# vel_resolution = 0.2176   m/s For Frame_time = 128*(Tchirp+Tidle) # This should be enough.
# vel_resolution = 0.5 m/s For IPCEI where we have 30 microseconds chirps

# Dec 10 2019 Chen: this script is a simplified version of DSP_clean_Last_NewTI_v3,
# many redundent lines are removed, only retains the main processing steps and only
# supports TDM/BPM recordings since this script targets the data set processing.

# ################################################################################################################
# Jan 30 2020 Hector: Things to do:
# 0. Extract the microdoppler signature using this script for all the 1000 samples case now that it looks cleaner
# 1. Identify our input data to the ML Model.
#    - Check all pool of functions and which ones are participating:
#              * rd_fft
#              * phase_compensation
#              * hadamard2_decode
#              * detect_peaks
#              * cfar_2d
#              * cfar_2d2 // vanilla 2d ?
#              * peak_grouping
#              * plot_rc_map // Range chirps map
#              * plot_rd_map // Range doppler map
#              * plot_angle_map
#              * plot_polar_az_rg
#              * plot_polar_el_rg
#              * plot_dbf_az // Spectrum Density. For comparison purposes
#              * plot_dbf_el // Spectrum Density. For comparison purposes
#    - Check if we can have frames to be passed to the ML model
# 2. Build a standalone model of CNN
# 3. Build a YOLO standalone
#
# ################################################################################################################
# Questions:
#
# print("raw.shape: ", raw.shape) # shape (111,64,256,8) # DELETE
# HG Note: Why do we have 64 chirps ? Because The virtual channels have been extracted already # DELETE
# HG Note: Is X_fft2 and X_fft1 the output of the range doppler FFT
# HG Note: Shape of each of the two virtual channel contributions
# HG Note: Why do we have the full shape in all virtual channels
# HG Note: Do we have all the contributions here ?
# X_fft2 has channel order Tx1,Tx2,Tx1,Tx2,Tx1,Tx2,Tx1,Tx2. It doesn't look like this is the correct order. Why the Rxs are not shown
# HG Note: Information from all targets, not only the person walking is included. We have to isolate it.

import matplotlib.pyplot as plt
import pylab as P
import gc # Garbage collector
import numpy as np
from math import ceil
import csv
import tkinter as tk
from tkinter.filedialog import askopenfilename
import sys
import spectrum as sp
import time
import os
from os.path import exists
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import itertools # To combine lists


######## Parameters #########
# We keep the following parameters consistent with the radar raw data we collected.
# Users don't need to change anything here.
numADCBits = 16 # number of ADC bits per sample
numADCSamples = 256 # number of ADC samples per chirp
numTx = 2
numRx = 4
chirpSize = numADCSamples*numRx
numLanes = 2 # do not change. number of lanes is always 2
isReal = 0 # 1 for real only data, 0 for complex data

#################################
# MANUAL CHANGES REQUIRED BELOW
#################################
# Manually Change each of the following parameters depending on the Experiment type:
#experiment_type = 'Non_Uniform'
#experiment_type = 'Sub_Frames'

# experiment_type = 'Non_Uniform' # When set to Non_Uniform, a different compensation is used, and numChirps is reduced below automatically to numChirps-2
experiment_type = 'Normal' # This is the rest of the cases. For instance, the Heterogeneous frames, which is at the moment only processed individually (ONly subframe C or A or B at a time) is also Normal
# HG Important Note: If you select 'Sub_Frames' here, make sure you change the CFAR2D parameters to consider the
# sides, as the frames are much shorter in the Sub_Frames case


numFrames = 1000 # Sub_Frames: 34 # Normal/Non_Unif: 111 # Daniel: 1000
numChirps = 128 # Non_Unif_ZI: 128*2 as zeros are inserted # Sub_Frames (Not concurrently, One by one): {C: 44*2 # B: 35*2 # A: 2*27}  # Normal/Daniel: 128 # Non_Unif: 126 (numChirps-2=128-2)
Tidle = 48e-6 #48e-6  # Sub_Frames (Not concurrently, One by one): {C: 232us, B: 307us, A: 416us} # Normal/Non_Unif: 232us # Daniel: 48us
# These two values apply only for TDM and BPM
num_Redundant_Vch_Az = 3
num_Redundant_Vch_Elv = 4

SingleTX_TDM_BPM = 0; # SingleTX = 0, TDM = 1, BPM = 2 // Single TX is not supported in this script

fc = 77e9 # 77GHz carrier
c = 3e8 # speed of light
lambda_mmwave = c/fc # = 0.0039
fs=5e6
f_IF_max = 0.5*fs # we discard half range bins
Bw = 1.5351e9   # According to TI Bw should be 1.5351 GHz
Tchirp = 60e-6
Tadc_start_time = 6e-6
Time_Frame  = numChirps*(Tchirp+Tidle) # 128 chirps of 60 microseconds
Kf=29.9817e12 # Slope = Bw/Tc
Kf2=Bw/Tchirp # Slope = 25.585e12 CHEN these 2 values should match

vel_resolution = lambda_mmwave/(2*Time_Frame)
Range_resolution = c/(2*Bw)
max_range = (c*f_IF_max)/(2*Kf) # 11.26
min_range = Range_resolution    # 0.098
max_velocity = lambda_mmwave/(4*numTx*(Tchirp+Tidle)) # 6.957
min_velocity = vel_resolution # 0.217

#################################
# Change the name of the raw.npz:
name_raw = "raw.npz" # Normal/Non_Uniform: raw.npz # Non_Uniform: raw_ZI.npz # Not used in Sub_Frames type

name_raw_A = "raw_A.npz" # 2*27 chirps
name_raw_B = "raw_B.npz" # 2*35 chirps
name_raw_C = "raw_C.npz" # 2*44 chirps

numChirps_A = 2*27
numChirps_B = 2*35
numChirps_C = 2*44

Tidle_A = 416e-6
Tidle_B = 307e-6
Tidle_C = 232e-6

Time_Frame_A = numChirps_A*(Tchirp+Tidle_A)
Time_Frame_B = numChirps_B*(Tchirp+Tidle_B)
Time_Frame_C = numChirps_C*(Tchirp+Tidle_C)

# The resolution should be the same:
vel_resolution_A = lambda_mmwave/(2*Time_Frame_A)
vel_resolution_B = lambda_mmwave/(2*Time_Frame_B)
vel_resolution_C = lambda_mmwave/(2*Time_Frame_C)

max_velocity_A = lambda_mmwave/(4*numTx*(Tchirp+Tidle_A))
max_velocity_B = lambda_mmwave/(4*numTx*(Tchirp+Tidle_B))
max_velocity_C = lambda_mmwave/(4*numTx*(Tchirp+Tidle_C))

# Sub_Frames (Only one supported at a time):
# raw_A.npz -> 2*27 chirps
# raw_B.npz -> 2*35 chirps
# raw_C.npz -> 2*44 chirps
#################################
# END OF MANUAL CHANGES REQUIRED
#################################

#if (experiment_type == 'Non_Uniform'):
#   numChirps = numChirps-2 # Because we remove the first and the last chirp content to make easier the phase compensation





####### function pool ########

nextpow2 = lambda x : pow(2,ceil(np.log2(x))) #function next power of 2

def rd_fft(data):
    """
    Computes a range-doppler map for a given number of chirps and samples per chirp.
    param data: FMCW radar data one frame consisting of <chirps>x<samples>
    return: Returns a 2D dimensional range-doppler map representing the reflected power over all range-doppler bins.
    """
    fft_rangesamples   =  nextpow2(numADCSamples)
    #fft_dopplersamples = nextpow2(Chirps)
    fft_dopplersamples =  Chirps # Do not approximate to power of two. Numpy should take care of this length mismatch with prime factorization

    # data shape before is (64,256,8)
    data = np.moveaxis(data,0,1)
    Ny, Nx, Nz = data.shape
    # data shape after is (256,64,8),Ny=256,Nx=64,Nz=8

    X_range_fft   = np.zeros((fft_rangesamples//2,Nx,Nz),dtype=np.complex64)
    X_doppler_fft = np.zeros((fft_rangesamples//2,fft_dopplersamples,Nz),dtype=np.complex64)

    for ch in range(Nz):
        data_ch = data[:,:,ch]
        # Windowing
        window = np.hanning(Ny).reshape((-1,1))
        scaled = np.sum(window)/Ny
        window2d = np.repeat(window,Nx,axis=1)
        data_ch = data_ch * window2d

        # Range FFT
        x = np.zeros([fft_rangesamples, Nx], dtype = np.complex64)
        start_index = (fft_rangesamples - Ny) // 2 # central zero-padding

        x[start_index:start_index + Ny, :] = data_ch

        X = np.fft.fft(x, fft_rangesamples, 0) / scaled * (2.0 /fft_rangesamples)
        X = X[0:fft_rangesamples // 2, :] # Floor division, export half frequency component
        Ny2, Nx = X.shape# (128,64)

        # Static clutter removal, optional
        static_clutter = np.mean(X,axis=1).reshape(-1,1).repeat(Nx,axis=1)
        X = X - static_clutter

        X_range_fft[:,:,ch] = X# HG NOte: IT needs to return X_range_fft only for plotting outside
        #Doppler FFT
        window = np.hanning(Nx).reshape((1,-1))
        scaled = np.sum(window)/Nx
        window2d = np.repeat(window,Ny2,axis=0)
        X = X * window2d

        x = np.zeros((Ny2, fft_dopplersamples), dtype=np.complex64)
        start_index = (fft_dopplersamples - Nx) // 2 # central zero-padding: [zeros][samples][zeros]
        #start_index = 0 # zero-padding at the end: [samples][zeros]

        #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #print('start_index: ', start_index)
        #print('start_index+Nx: ', start_index+Nx)
        #print('Nx: ', Nx)
        #print('fft_dopplersamples: ', fft_dopplersamples)
        #print('Are we centering the doppler samples???????')
        #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #time.sleep(1000)

        x[:, start_index:start_index + Nx] = X

        X = np.fft.fft(x, fft_dopplersamples, 1) / scaled * (2/fft_dopplersamples)
        X_doppler_fft[:,:,ch] = np.fft.fftshift(X, axes=1)

    del x, X, window, scaled, window2d, data, data_ch
    gc.collect() # All the previous delete get collected here

    return X_doppler_fft, X_range_fft # Returns the X_doppler_fft and the X_range_fft, which the last will be used only for plotting


def rd_fft_non_uniform(data):
    """
    Computes a range-doppler map for a given number of chirps and samples per chirp.
    param data: FMCW radar data one frame consisting of <chirps>x<samples>
    return: Returns a 2D dimensional range-doppler map representing the reflected power over all range-doppler bins.
    """
    fft_rangesamples   =  nextpow2(numADCSamples)
    fft_dopplersamples =  nextpow2(numChirps)
    # fft_dopplersamples =  Chirps # Do not approximate to power of two. Numpy should take care of this length mismatch with prime factorization

    # data shape before is (64,256,8)
    data = np.moveaxis(data,0,1)
    Ny, Nx, Nz = data.shape
    # data shape after is (256,64,8),Ny=256,Nx=64,Nz=8

    X_range_fft   = np.zeros((fft_rangesamples//2,Nx,Nz),dtype=np.complex64)
    X_doppler_fft = np.zeros((fft_rangesamples//2,fft_dopplersamples,Nz),dtype=np.complex64)

    for ch in range(Nz):
        data_ch = data[:,:,ch]
        # Windowing
        window = np.hanning(Ny).reshape((-1,1))
        scaled = np.sum(window)/Ny
        window2d = np.repeat(window,Nx,axis=1)
        data_ch = data_ch * window2d

        # Range FFT
        x = np.zeros([fft_rangesamples, Nx], dtype = np.complex64)
        start_index = (fft_rangesamples - Ny) // 2 # central zero-padding

        x[start_index:start_index + Ny, :] = data_ch

        X = np.fft.fft(x, fft_rangesamples, 0) / scaled * (2.0 /fft_rangesamples)
        X = X[0:fft_rangesamples // 2, :] # Floor division, export half frequency component
        Ny2, Nx = X.shape# (128,64)

        # Static clutter removal, optional
        static_clutter = np.mean(X,axis=1).reshape(-1,1).repeat(Nx,axis=1)
        X = X - static_clutter

        X_range_fft[:,:,ch] = X# HG NOte: IT needs to return X_range_fft only for plotting outside
        #Doppler FFT
        #CHEN padding 0 for X with ABBAABBA pattern
        X_expand = np.zeros([Ny2,Nx*numTx])
        Ny2, Nx2 = X_expand.shape# (128,128)
        j=0#X index
        if ch%2==0:#code A
            for i in range(Nx*numTx):#X_expand index
                if(i%4==0 or i%4==3):
                    X_expand[:,i] = X[:,j]
                    j = j + 1
        else:#code B
            for i in range(Nx*numTx):#X_expand index
                if(i%4==1 or i%4==2):
                    X_expand[:,i] = X[:,j]
                    j = j + 1
        #linear interpolation to replace 0 padding
        if ch%2==0:#code A
            for i in np.arange(2,126,4):
                X_expand[:,i] = X_expand[:,i+1]*2-X_expand[:,i+2]
            for i in np.arange(5,126,4):
                X_expand[:,i] = X_expand[:,i-1]*2-X_expand[:,i-2]
            X_expand[:,1]   = (X_expand[:,0]+X_expand[:,2])/2
            X_expand[:,126] = (X_expand[:,125]+X_expand[:,127])/2
        else:#code B
            for i in np.arange(0,128,4):
                X_expand[:,i] = X_expand[:,i+1]*2-X_expand[:,i+2]
            for i in np.arange(3,128,4):
                X_expand[:,i] = X_expand[:,i-1]*2-X_expand[:,i-2]


        window = np.hanning(Nx2).reshape((1,-1))
        scaled = np.sum(window)/Nx2
        window2d = np.repeat(window,Ny2,axis=0)
        X_expand = X_expand * window2d

        x = np.zeros((Ny2, fft_dopplersamples), dtype=np.complex64)
        start_index = (fft_dopplersamples - Nx2) // 2 # central zero-padding: [zeros][samples][zeros]
        #start_index = 0 # zero-padding at the end: [samples][zeros]

        #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #print('start_index: ', start_index)
        #print('start_index+Nx: ', start_index+Nx)
        #print('Nx: ', Nx)
        #print('fft_dopplersamples: ', fft_dopplersamples)
        #print('Are we centering the doppler samples???????')
        #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #time.sleep(1000)
        x[:, start_index:start_index + Nx2] = X_expand

        X = np.fft.fft(x, fft_dopplersamples, 1) / scaled * (2/fft_dopplersamples)
        X_doppler_fft[:,:,ch] = np.fft.fftshift(X, axes=1)

    del x, X, window, scaled, window2d, data, data_ch
    gc.collect() # All the previous delete get collected here

    return X_doppler_fft, X_range_fft # Returns the X_doppler_fft and the X_range_fft, which the last will be used only for plotting

def phase_compensation(X_fft2):
    rg_bin, dp_bin, vc_bin = X_fft2.shape
    phase_vec     =  (np.arange(dp_bin)-dp_bin/2)*np.pi/(dp_bin/2)
    compen_vec    =  np.exp(1j*phase_vec/2)
    compen_matrix =  np.tile(compen_vec,(rg_bin,1))
    for i in range(1,8,2):# phase compensation for channel 1,3,5,7 (Tx2)
        X_fft2[:,:,i] = X_fft2[:,:,i]*compen_matrix
    return X_fft2

def phase_compensation_nonuniform(X_fft2): # AB BA AB BA AB BA
    rg_bin, dp_bin, vc_bin = X_fft2.shape

    print('rg_bin ', rg_bin) # 128
    print('dp_bin ', dp_bin) # 44
    print('vc_bin ', vc_bin) # 8

    phase_vec     =  (np.arange(dp_bin)-dp_bin/2)*np.pi/(dp_bin/2)
    # -pi to 0 to pi # Unambiguous from -pi to pi, which is half circle possitive or negative
    compen_vec_1    =  - np.exp(1j*phase_vec) # Note: Why did Chen not include the minus in the compensation vector? Is the sign included in the phase grid?
    compen_matrix_1 =  np.tile(compen_vec_1,(rg_bin,1))

    compen_vec_2    =  - np.exp(1j*phase_vec/3) # Note: Why did Chen not include the minus in the compensation vector? TODO: Check this to see if we need to remove it as well
    compen_matrix_2 =  np.tile(compen_vec_2,(rg_bin,1))

    # X_fft2 ->    0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7
    # X_fft2 ->   Tx1,Tx2,Tx1,Tx2,Tx1,Tx2,Tx1,Tx2
    #            R1T1,R1T2,R2T1,R2T2,R3T1,R3T2,R4T1,R4T2
    #            CH11,CH12,CH21,CH22,CH31,CH32,CH41,CH42
    #              0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7
    # Compen  :         X         X         X         X  # Normal Compensation
    # Compen_1:    X         X         X         X       # Nonuniform Compensation occurs in both Channels
    # Compen_2:         X         X         X         X  # Nonuniform Compensation occurs in both Channels

    # Normal sequence:
    # (Tx1+Tx2): 0,2,4,6
    # (Tx1-Tx2): 1,3,5,7

    # Nonuniform sequence:
    #                                Ch1 Ch2 Ch3 Ch4
    # (Tx1+Tx2) received in Channels: 0 , 2 , 4 , 6   # It is still the same, it only changes in the organization that takes place in the npz script
    # (Tx1-Tx2) received in Channels: 1 , 3 , 5 , 7

    for i in range(8):
        if (i % 2 != 0): # Odd groups  # phase compensation for groups: 1,3,5,7 (Tx1-Tx2)
           X_fft2[:,:,i] = X_fft2[:,:,i]*compen_matrix_2
        else:            # Even groups # phase compensation for groups: 0,2,4,6 (Tx1+Tx2)
           X_fft2[:,:,i] = X_fft2[:,:,i]*compen_matrix_1

    return X_fft2


def hadamard2_decode(data, SingleTX_TDM_BPM ):
    # We put TDM also in this decode function beacause we adjust the channel order to group the TX1 data in the first half, TX2 data in the second half.
    if (SingleTX_TDM_BPM == 1): # TDM
        decode1 = data[:,:,0::2]
        decode2 = data[:,:,1::2]
    elif (SingleTX_TDM_BPM == 2): # BPM
        decode1 = (data[:,:,0::2] + data[:,:,1::2]) /2
        decode2 = (data[:,:,0::2] - data[:,:,1::2]) /2
    decode = np.concatenate((decode1,decode2),axis=2)
    #decode channel order is RX1TX1,RX2TX1,RX3TX1,RX4TX1,RX1TX2,RX2TX2,RX3TX2,RX4TX2
    return decode

def detect_peaks(x, num_train, num_guard, rate_fa):
    """
    Detect peaks with CFAR algorithm.

    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate.
    """
    num_cells = x.size
    peak_mask = np.zeros(num_cells,dtype='bool')
    peak_idx = []

    num_train_half = num_train // 2
    num_guard_half = num_guard // 2
    num_side = num_train_half + num_guard_half

    alpha = num_train*(rate_fa**(-1/num_train) - 1)
    alpha = 5# we set alpha as a constant is a little bit tricky to match the data we use

    #boundary expansion for margin detection
    y = np.zeros(2*num_side+num_cells)
    y[:num_side] = x[num_side+1:num_side+1+num_side]
    y[num_side:num_side+num_cells] = x
    y[-num_side:] = x[num_cells-num_side-1:num_cells-1]

    threshold = 0


    for i in range(num_side, num_cells + num_side): # Explore in an interval not starting from zero

        if i == i-num_side+np.argmax(y[i-num_side:i+num_side+1]): # only compare threshold with the local maximum value
            sum1 = np.sum(y[i-num_side:i+num_side+1])
            sum2 = np.sum(y[i-num_guard_half:i+num_guard_half+1])
            p_noise = (sum1 - sum2) / num_train # The guard cells are discarded

            threshold = alpha * p_noise
            if y[i] > threshold: # Clearly the problem is that the threshold is too high so the max value is not detected
                peak_idx.append(i-num_side)
                peak_mask[i-num_side] = True

    # del x, y, threshold, num_cells, num_train_half, num_guard_half, num_side
    # gc.collect() # All the previous delete get collected here
    return peak_mask,peak_idx

def cfar_2d(rd_map,first_axis=0):
    '''
    in this function you can choose the 1d cfar order, first range or doppler
    first_axis = 0: first range CFAR
    first_axis = 1: first doppler CFAR (TI)
    boundary expansion for margin detection is integrated.
    '''
    rg_size, dp_size = rd_map.shape
    rd_map_bottom = np.min(rd_map)
    target_list = []
    # These parameters are not helpful to get activation on the sides, the preferred detected targets are always on the center:
    far = 0.00001  # false alarm rate
    num_train = 10 # both sides total
    num_guard = 2  # both sides total

    # HG Note: Important -  Parameters used during the Heterogeneous frames to avoid cutting the short frames on the sides, especially for
    # targets falling on the edges of the range doppler map
    # This is the preferred set of parameters to detect activation on the sides, however we might need to discard many targets detected here
    # This is why we use a more strict false alarm rate:
    #far = 0.0000001  # false alarm rate
    #num_train = 3 # both sides total
    #num_guard = 2  # both sides total


    if first_axis==0:# Apply 1D CFAR first along range direction
        cfar_mask = np.zeros(rg_size,dtype='bool')
        first_ite_size = dp_size
    else:            # Apply 1D CFAR first along doppler direction
        cfar_mask = np.zeros(dp_size,dtype='bool')
        first_ite_size = rg_size

    ####### Apply 1D CFAR
    for i in range(first_ite_size):
        if first_axis==0: # NUmber of iterations 64
            data = rd_map[:,i]
            ite_mask,_ = detect_peaks(data, num_train, num_guard, far)
            cfar_mask |= ite_mask # boolean vector (rg_size or dp_size) with zeros everywhere except on the peaks
            rd_map[:,i] = (~ite_mask*rd_map_bottom) + (ite_mask*data)# non-detected bins are filled with the bottom value
        else:
            data = rd_map[i,:]
            ite_mask,_ = detect_peaks(data, num_train, num_guard, far)
            cfar_mask |= ite_mask
            rd_map[i,:] = (~ite_mask*rd_map_bottom) + (ite_mask*data)# non-detected bins are filled with the bottom value

    # plot_rd_map(rd_map,'1dcfar')# plot result of the first round CFAR
    ####### Apply 2D CFAR
    for i in range(cfar_mask.size):
        if cfar_mask[i]:
            if first_axis==0: # Out of the 128 iterations, it enters here only in the Range bins where there was a peak
                data = rd_map[i,:]
                ite_mask,idx_list = detect_peaks(data, num_train, num_guard, far)
                rd_map[i,:] = (~ite_mask*rd_map_bottom) + (ite_mask*data)# non-detected bins are filled with the bottom value
                for j in idx_list:
                    target_list.append([i,j])
            else:
                data = rd_map[:,i]
                ite_mask,idx_list = detect_peaks(data, num_train, num_guard, far)
                rd_map[:,i] = (~ite_mask*rd_map_bottom) + (ite_mask*data)# non-detected bins are filled with the bottom value
                for j in idx_list:
                    target_list.append([j,i])

    target_list = np.array(target_list)
    if target_list.size > 0:
        target_list = np.tile(target_list,(1,2))# we reserve 2 columns for the further az,el index. so the target_list has column order: rg,dp,az,el
    return rd_map, target_list

def cfar_2d2(rd_map):
    '''
    this is the vanilla 2d cfar, no boundary expansion and no cfar dimention selection.
    '''
    rg_size, dp_size = rd_map.shape
    cfar_mask = np.zeros(rg_size,dtype='bool')
    rd_map_bottom = np.min(rd_map)
    target_list = []
    far = 0.00001  # false alarm rate
    num_train = 10 # both sides total
    num_guard = 2  # both sides total

    ####### Apply 1D CFAR along range direction
    for dp_ite in range(dp_size):
        data = rd_map[:,dp_ite]
        ite_mask,_ = detect_peaks(data, num_train, num_guard, far)
        cfar_mask |= ite_mask
        rd_map[:,dp_ite] = (~ite_mask*rd_map_bottom) + (ite_mask*data)# non-detected bins are filled with the bottom value
    plot_rd_map(rd_map,'1dcfar')
    ####### Apply 2D CFAR along doppler direction
    for rg_ite in range(rg_size):
        if cfar_mask[rg_ite]:
            data = rd_map[rg_ite,:]
            ite_mask,dp_idx_list = detect_peaks(data, num_train, num_guard, far)
            rd_map[rg_ite,:] = (~ite_mask*rd_map_bottom) + (ite_mask*data)# non-detected bins are filled with the bottom value
            for dp_idx in dp_idx_list:
                target_list.append([rg_ite,dp_idx])

    target_list = np.array(target_list)
    target_list = np.tile(target_list,(1,2))# we reserve 2 columns for the further az,el index. so the target_list has column order: rg,dp,az,el
    return rd_map, target_list

def peak_grouping(rd_map, target_list,rg_radius=3,dp_radius=3):
    '''
    mode=0: peak grouping in range direction
    mode=1: peak grouping in both range and doppler directions
    mode=2: peak grouping in doppler direction
    '''
    if len(target_list) < 2:
        return rd_map, target_list
    coor_list = target_list.tolist()
    rd_map_bottom = np.min(rd_map)

    i = 0 # we use while not for loop since some elements may deleted and loop length is dynamically varied.
    while i < len(coor_list):
        i_rg = coor_list[i][0]
        i_dp = coor_list[i][1]
        j = i+1
        while j < len(coor_list):
            j_rg = coor_list[j][0]
            j_dp = coor_list[j][1]
            if j_rg <= i_rg+rg_radius and j_rg >= i_rg-rg_radius and \
               j_dp <= i_dp+dp_radius and j_dp >= i_dp-dp_radius:
                if rd_map[i_rg,i_dp] > rd_map[j_rg,j_dp]:
                    rd_map[j_rg,j_dp] = rd_map_bottom
                    del coor_list[j]
                    j -= 1
                else:
                    rd_map[i_rg,i_dp] = rd_map_bottom
                    del coor_list[i]
                    i -= 1
                    break
            j += 1
        i += 1

    target_list = np.asarray(coor_list)
    return rd_map, target_list

def thresholding_strongest_magnitude(rd_map, target_list):
    if len(target_list) < 2:
        return target_list
    #
    #
    aux_list = target_list.tolist()

    list_of_magnitudes = []

    for indx0 in range(len(aux_list)):
              i_rg_0 = aux_list[indx0][0]
              i_dp_0 = aux_list[indx0][1]

              list_of_magnitudes.append(rd_map[i_rg_0,i_dp_0])

    avg_dop = np.mean(list_of_magnitudes)
    std_dev = np.std(list_of_magnitudes)

    threshold_av_1stdev = avg_dop + 2*std_dev # 2*std_dev # We control which region of targets we take through the standard deviation

    # Discard the targets below the threshold
    new_aux_list = []
    for indx in range(len(aux_list)):
        if (list_of_magnitudes[indx] >= threshold_av_1stdev):
              new_aux_list.append(aux_list[indx])

    target_list = np.asarray(new_aux_list)

    return target_list

def plot_rc_map(rc_map):
    rg_bin, dp_bin =  rc_map.shape
    axis_doppler   =  np.arange(dp_bin)+1
    fig1           =  plt.figure();
    fig1colormesh  =  plt.pcolormesh(axis_doppler, axis_range, rc_map, edgecolors='None')
    fig1.colorbar(fig1colormesh)
    fig1title      =  plt.title('Range vs Chirps - Sum ALl Channels');
    fig1xlabel     =  plt.xlabel('Chirps ');
    fig1ylabel     =  plt.ylabel('Range (meters)');
    name_string    =  'fig/RangeVersus'+'chirps_Frame_'+'%04d'%(Frame_selected)+'.png'
    fig1.savefig(name_string, bbox_inches='tight')
    plt.close()
    # del fig1, fig1colormesh, fig1title, fig1xlabel, fig1ylabel

def plot_rd_map(rd_map,suffix=''):
    rg_bin, dp_bin =  rd_map.shape
    rd_map = np.where(rd_map>50,50,rd_map)# set saturation value to highlight target.
    fig2           =  plt.figure();
    fig2colormesh  =  plt.pcolormesh(axis_doppler, axis_range, rd_map, edgecolors='None')
    fig2.colorbar(fig2colormesh)
    fig2title      =  plt.title('Range Doppler Map');
    fig2xlabel     =  plt.xlabel('Velocities (m/s)');
    fig2ylabel     =  plt.ylabel('Range (meters)');
    name_string    =  'fig/RangeDopplerMap_Frame_'+'%04d'%(Frame_idx)+suffix+'.png'
    fig2.savefig(name_string, bbox_inches='tight')
    plt.close()

    # Export csv for each range doppler map // Used to plot the RD map in the Journal Paper
    #path_rang_dop = "range_vs_doppler_" + str(Frame_idx) + ".csv"
    #dim1, dim2 = rd_map.shape
    #data_csv_input_rang_dop = []
    #for idx1 in range(dim1):
    #      data_row_input_rang_dop = ''
    #      for idx2 in range(dim2):
    #           data_row_input_rang_dop = data_row_input_rang_dop + "," + str(abs(rd_map[idx1][idx2]))
    #      data_csv_input_rang_dop.append(data_row_input_rang_dop.split(","))
    #csv_writer(data_csv_input_rang_dop, path_rang_dop)


    # del fig2, fig2colormesh, fig2title, fig2xlabel, fig2ylabel

def plot_angle_map():
    fig99 = plt.figure()
    fig99colormesh = plt.pcolormesh(az_deg_vec,el_deg_vec, abs(f_fft3_el), edgecolors='None')
    fig99.colorbar(fig99colormesh)
    fig99title  = plt.title('Elevation vs Azimuth - 64X64 Grid')
    fig99xlabel = plt.xlabel('Azimuth ')
    fig99ylabel = plt.ylabel('Elevation ')
    name_string = 'fig/ElevationVersus'+'Azimuth_Frame_'+str(Frame_selected)+'_TargetNumber_'+str(aux)+'.png'
    fig99.savefig(name_string, bbox_inches='tight')
    print('+++++++++++++++++++++++++++++++++++++++++++++++')
    print('Inside plot_angle_map function')
    print('+++++++++++++++++++++++++++++++++++++++++++++++')

    #### Export elevation and azimuth
    path_az_el = "elevation_vs_azimuth_" + str(Frame_selected) + ".csv"
    dim1, dim2 = f_fft3_el.shape
    data_csv_input_az_el = []
    for idx1 in range(dim1):
          data_row_input_az_el = ''
          for idx2 in range(dim2):
               data_row_input_az_el = data_row_input_az_el + "," + str(abs(f_fft3_el[idx1][idx2]))
          data_csv_input_az_el.append(data_row_input_az_el.split(","))

    csv_writer(data_csv_input_az_el, path_az_el)

    plt.close()

def csv_writer(data, path): # Standard one
    """
    Write data to a CSV file path
    """
    #with open(path, "wb") as csv_file:
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


def plot_polar_az_rg(target_list, target_angle_val, output_3d_fft_Az,az_vec_deg):
    for index_targets in range(target_list.shape[0]):
        output_3d_fft_Az[target_list[index_targets,2],target_list[index_targets,0]] =  target_angle_val[index_targets]

    JdB = sp.mag2db(output_3d_fft_Az)
    JMax = np.max(JdB)
    JNorm = JdB - JMax

    fig30 = plt.figure(30, figsize=(9, 9))
    az_vec_rad = az_vec_deg*(np.pi/180.0)
    mU, mRange = np.meshgrid(az_vec_rad, axis_range)

    ax = fig30.add_subplot(111, projection='polar')
    ax.pcolormesh(mU, mRange, np.transpose(JNorm))
    name_string = 'fig/AzimuthPolarPlot_Frame_'+str(Frame_selected)+'_Total' + '.png'
    fig30.savefig(name_string, bbox_inches='tight')
    plt.close()

def plot_polar_el_rg(target_list, target_angle_val, output_3d_fft_Elv,el_vec_deg):
    for index_targets in range(target_list.shape[0]):
        output_3d_fft_Elv[target_list[index_targets,3],target_list[index_targets,0]] =  target_angle_val[index_targets]

    JdB_Elv = sp.mag2db(output_3d_fft_Elv)
    JMax_Elv = np.max(JdB_Elv)
    JNorm_Elv = JdB_Elv - JMax_Elv

    fig40 = plt.figure(40, figsize=(9, 9))
    el_vec_rad = el_vec_deg* (np.pi / 180.0)

    mU_Elv, mRange = np.meshgrid(el_vec_rad, axis_range) # axis_range is the range grid
    ax_Elv = fig40.add_subplot(111, projection='polar')
    ax_Elv.pcolormesh(mU_Elv, mRange, np.transpose(JNorm_Elv)) # shape of transpose(JNorm_Elv) = (128, 64)
    name_string = 'fig/ElevationPolarPlot_Frame_'+str(Frame_selected)+'_Total.png'
    fig40.savefig(name_string, bbox_inches='tight')
    plt.close()

def plot_dbf_az(az_vec_deg,output_3d_fft_Az):
    fig50 = plt.figure(50, figsize=(9, 9))
    az_db = sp.mag2db(np.sum(output_3d_fft_Az, 1))
    plt.plot(az_vec_deg, (az_db-np.min(az_db))/np.ptp(az_db))#DBF plot
    name_string = 'fig/DBF_Azimuth_Spectrum_Frame_'+str(Frame_selected)+'_Total.png'
    fig50.savefig(name_string, bbox_inches='tight')
    plt.close()

def plot_dbf_el(el_vec_deg,output_3d_fft_Elv):
    fig60 = plt.figure(60, figsize=(9, 9))
    el_db = sp.mag2db(np.sum(output_3d_fft_Elv, 1))
    plt.plot(el_vec_deg, (el_db-np.min(el_db))/np.ptp(el_db))#DBF plot
    name_string_Elv = 'fig/DBF_Elevation_Spectrum_Frame_'+str(Frame_selected)+'_Total.png'
    fig60.savefig(name_string_Elv, bbox_inches='tight')
    plt.close()

#####################   START PROCESSING      ##################################
#####################   USER CONFIGURE HERE   ##################################
# root = tk.Tk();fileName = askopenfilename();root.withdraw()  # assign file via GUI
fileName = name_raw #
# Normal/Non_Uniform:
#    'raw.npz'
# Sub_Frames (Only one supported at a time):
#    'raw_A.npz' -> 2*27 chirps
#    'raw_B.npz' -> 2*35 chirps
#    'raw_C.npz' -> 2*44 chirps
if (experiment_type == "Sub_Frames"): # Open these raw files:
   raw_A = np.load(name_raw_A)['arr_0']
   raw_B = np.load(name_raw_B)['arr_0']
   raw_C = np.load(name_raw_C)['arr_0']
else: # For normal or non uniform experiments open this one:
   raw = np.load(fileName)['arr_0']


# raw has the shape (frame, chirp, sample, channel), e.g. (1000,64,256,8)
# channel order is CH11,CH12,CH21,CH22,CH31,CH32,CH41,CH42 to present CH(RX,TX)
if (experiment_type == "Sub_Frames"):
         process_frames = numFrames*3 # Three times to account for the subframes in each frame
else: # Normal or Non_uniform cases
         process_frames = numFrames # 111 # Frames Loop Numbers
# Here we configure if different kinds of figure are plotted/stored.
plot_rc_map_en     =  0
plot_rd_map_raw_en =  0 # Set to 1 to visualize the Range Doppler map figures individually
plot_rd_map_en     =  0 # With CFAR included
plot_angle_map_en  =  0
plot_az_rg_en      =  0
plot_el_rg_en      =  0
plot_dbf_az_en     =  0
plot_dbf_el_en     =  0
# DBSCAN Batches plus ML application
#dbscan_frames = process_frames # process_frames here means we do DBSCAN only at the end
# Leaving the DBSCAN For the end can cause a dynamic scene to be blurred by the high number of reflections
# A scene with a single isolated object draws correctly
# A solution to this problem of dynamic scenes can be kalman filtering
# check a simple example
dbscan_frames = 10 # Specify here if they are different. The dbscan data can't be now among consecutive frames, since they could be from different types

contador_dbscan = 0

contador_dbscan_A = 0
contador_dbscan_B = 0
contador_dbscan_C = 0

csv_header_full    =  "Frame_i, Target_Number, Range, Speed, Magnitude, x, y, z, angle_theta, angle_phi"
csv_header_uDs     =  "Frame_i, Target_Number, Range, Speed, Magnitude"

#####################   USER CONFIGURE END   ##################################
csv_full = []
csv_uDs  = []
dbscan_input_xyz = []
dbscan_input_xy = []

rd_map_cube= []# export rd_map in each frame for uDs plotting

# For the Sub_Frames case
targets_in_A = []
targets_in_B = []
targets_in_C = []

target_ids_in_A = []
target_ids_in_B = []
target_ids_in_C = []

csv_full_A_original = []
csv_full_B_original = []
csv_full_C_original = []

csv_full_A = []
csv_full_B = []
csv_full_C = []


csv_uDs_A  = []
csv_uDs_B  = []
csv_uDs_C  = []

dbscan_input_xyz_A = []
dbscan_input_xyz_B = []
dbscan_input_xyz_C = []

dbscan_input_xy_A = []
dbscan_input_xy_B = []
dbscan_input_xy_C = []

rd_map_cube_A= []# export rd_map in each frame for uDs plotting
rd_map_cube_B= []# export rd_map in each frame for uDs plotting
rd_map_cube_C= []# export rd_map in each frame for uDs plotting

## prepare figure folder if any plotting function is enabled.
if bool(plot_rc_map_en) or bool(plot_rd_map_raw_en) or bool(plot_rd_map_en) \
  or bool(plot_angle_map_en) or bool(plot_az_rg_en) or bool(plot_el_rg_en)  \
  or bool(plot_dbf_az_en) or bool(plot_dbf_el_en):
    if not exists('fig'):
        os.mkdir('./fig')

#set_of_frames = [87,88,89] # This is to operate over the 3 subframes in Frame 29

# This is only to process frames 29 and 30:
#set_of_frames = [87,88,89,90,91,92] # This is to operate over the 3 subframes in Frame 29

# This is to process from frame 29 to 33
#set_of_frames = [87,88,89,90,91,92,93,94,95,96,97,98,99,100,101] # This is to operate over the 3 subframes in Frame 29
#                29 29 29 30 30 30 31 31 31 32 32 32 33  33  33

#set_of_frames = [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,88,89,90,91,92] # This is to operate over the 3 subframes for FRames 24 25 26 27 28 29 30
#                24  24  24  25  25  25  26  26  26  27  27  27  28  28  28  29 29 29 30 30 30

#set_of_frames = [72, 73, 74, 75, 76, 77] # This is to operate over the 3 subframes for FRames 24 25
#                24  24  24  25  25  25


#for Frame_idx in set_of_frames: # Frame_selected to
for Frame_idx in range(process_frames): # Frame_selected to
        print("Frame_idx: ", Frame_idx)
        if (experiment_type == 'Sub_Frames'): # 0: 27 -> 1:35 -> 2:44
                # Assign raw
                if (Frame_idx % 3 == 0):
                       Chirps = int(numChirps_A/numTx)
                       raw = raw_A
                       vel_resolution = vel_resolution_A
                elif (Frame_idx % 3 == 1):
                       Chirps = int(numChirps_B/numTx)
                       raw = raw_B
                       vel_resolution = vel_resolution_B
                elif (Frame_idx % 3 == 2):
                       Chirps = int(numChirps_C/numTx)
                       raw = raw_C
                       vel_resolution = vel_resolution_C
                Frame_selected = int(Frame_idx/3) # Mapped to groups of 3

        else: # Normal/Non_uniform cases

                Frame_selected = Frame_idx # Direct mapping
                Chirps   = int(numChirps/numTx)
                # Raw is not assigned at every iteration, it is always the same

        if (SingleTX_TDM_BPM == 0): # Single TX
            print("this script doesn't support single TX mode. Quit processing...")
            sys.exit()

            # TODO: Include the support for SIngle TX as well already developed in script DSP_clean_Last_NewTI_Order_vf2.py


        else: # TDM or BPM
            # Non_Uniform exploration:
            #X_fft2, X_fft1 = rd_fft_non_uniform(raw[Frame_selected, :, :, :]) # X_fft2 <- X_doppler_fft, X_fft1 <- X_range_fft
            # Normal rd_fft case:
            #X_fft2, X_fft1 = rd_fft(raw[Frame_selected-1, :, :, :]) # X_fft2 <- X_doppler_fft, X_fft1 <- X_range_fft
            X_fft2, X_fft1 = rd_fft(raw[Frame_selected, :, :, :]) # X_fft2 <- X_doppler_fft, X_fft1 <- X_range_fft

            #if (Frame_selected == 30):
            #   X_fft2, X_fft1 = rd_fft(raw[Frame_selected-1, :, :, :]) # X_fft2 <- X_doppler_fft, X_fft1 <- X_range_fft
            #   # This is only to test the Frame 29 disambiguation
            #else:
            #   X_fft2, X_fft1 = rd_fft(raw[Frame_selected, :, :, :]) # X_fft2 <- X_doppler_fft, X_fft1 <- X_range_fft

        # print("X_fft2.shape: ",X_fft2.shape) # DELETE # (128, 64, 8)
        # print("X_fft1.shape: ",X_fft1.shape) # DELETE # (128, 64, 8)

        # we do phase compensation here for all doppler bins of Tx2.
        # Phase compensation should be before hadamard decoding
        # Phase compensation has no effect on magnitude of rd bins,
        # which means no effect on peak detection.
        # X_fft2 has channel order Tx1,Tx2,Tx1,Tx2,Tx1,Tx2,Tx1,Tx2
        # X_fft2 is (128 range bins, 44 chirps, 8 Channels)

        if (experiment_type == 'Non_Uniform'):
            X_fft2 = phase_compensation_nonuniform(X_fft2) # Only doppler is compensated
        else: # The rest of the cases include the Heterogeneous frames and the Normal BPM
            X_fft2 = phase_compensation(X_fft2) # Only doppler is compensated

        # HG Note: The hadamard decoding function was not modified for the Non_Uniform case
        if (experiment_type != 'Non_Uniform'):
            decode_fft2 = hadamard2_decode(X_fft2, SingleTX_TDM_BPM ) # Doppler fft is the input
            decode_fft1 = hadamard2_decode(X_fft1, SingleTX_TDM_BPM ) # Range   fft is the input

            rd_map = np.sum(abs(decode_fft2),axis=2) # rd: range-doppler DEBUG only check code a of RX1
            rc_map = np.sum(abs(decode_fft1),axis=2)# rc: range-chirp
            
        else: 
            decode_fft2 = X_fft2 #DEBUG
            decode_fft1 = X_fft1 #DEBUG

            rd_map = abs(decode_fft2[:,:,0])# rd: range-doppler DEBUG only check code a of RX1
            rc_map = np.sum(abs(decode_fft1),axis=2)# rc: range-chirp
            
        # decode_fft has channel order Tx1,Tx1,Tx1,Tx1,Tx2,Tx2,Tx2,Tx2


        rg_bin,dp_bin = rd_map.shape
        axis_range = np.arange(rg_bin)*Range_resolution
        axis_doppler = (np.arange(dp_bin)-dp_bin/2)*vel_resolution
        # so far the range-Doppler map is created successfully!
        # Attention! speed=0 bin locates in the middle of axis_doppler

        if (experiment_type == "Sub_Frames"):
                if (Frame_idx % 3 == 0):
                       rd_map_cube_A.append(list(np.copy(rd_map)))# collect raw rd_map for potting
                elif (Frame_idx % 3 == 1):
                       rd_map_cube_B.append(list(np.copy(rd_map)))# collect raw rd_map for potting
                elif (Frame_idx % 3 == 2):
                       rd_map_cube_C.append(list(np.copy(rd_map)))# collect raw rd_map for potting
        else: # Normal and Non_uniform cases
            rd_map_cube.append(list(np.copy(rd_map)))# collect raw rd_map for potting


        if plot_rc_map_en:
            plot_rc_map(rc_map)# PLot for the evolution of range and time
        if plot_rd_map_raw_en:
            plot_rd_map(rd_map,'_raw')# Plot for the range-doppler map before cfar filtering
        ## CFAR for peak detection and export target list with range and doppler index
        rd_map, target_list = cfar_2d(rd_map)
        rd_map, target_list = peak_grouping(rd_map, target_list,rg_radius=3,dp_radius=3)
        #target_list = thresholding_strongest_magnitude(rd_map, target_list) # Used to reduce points in Sub_frame case with alternative cfar params
        target_num = len(target_list)
        # HG Note: UP to this point, the CFAR is made in the two dimensions, we have the list of targets, and we are going to apply the new Angle FFT

        if plot_rd_map_en:
            plot_rd_map(rd_map,'_cfar')# Plot for the range-doppler map after cfar filtering

        if (target_num==0):
            print("In frame ",Frame_selected,", no targets are detected.")
            continue
        else:
            print("In frame ",Frame_selected,", ", str(target_num)," targets are detected.")

        ########## START ANGLE OF ARRIVAL
        # Antenna Organization from behind the radar:
        #
        # | Elevation
        # -------------> Azimuth
        #
        # TDM or BPM:
        #     Tx1-Rx1  Tx2-Rx1  Tx1-Rx2  Tx2-Rx2
        #                       Tx1-Rx3  Tx2-Rx3
        #                       Tx1-Rx4  Tx2-Rx4
        #
        # Single Tx:
        #     Tx1-Rx1           Tx1-Rx2
        #                       Tx1-Rx3
        #                       Tx1-Rx4
        #
        # The contribution per channel to each dimension:
        #		+ RX2, RX3 and RX4 for elevation:
        #		+ RX2, RX1 for azimuth:
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        expand_az_bin = 64
        expand_el_bin = 64

        output_3d_fft_Az = 0.00001*np.ones([expand_az_bin,rg_bin]) # (num_angles, range_bins) # We will fill it up with only the output of the FFt
        output_3d_fft_Elv = 0.00001*np.ones([expand_el_bin,rg_bin]) # (num_angles, range_bins) # We will fill it up with only the output of the FFt
        input_3d_fft = np.zeros([expand_el_bin, expand_az_bin],dtype=complex)
        target_angle_val  = np.zeros(target_num) # for plottintg range-angle polar chart

        # Restart the targets_in_X array before starting every Subframe type
        if (experiment_type == "Sub_Frames"):
                if (Frame_idx % 3 == 0):
                       targets_in_A = []
                       single_target_in_A = []
                       target_ids_in_A = []
                if (Frame_idx % 3 == 1):
                       targets_in_B = []
                       target_ids_in_B = []
                if (Frame_idx % 3 == 2):
                       targets_in_C = []
                       single_target_in_C = []
                       target_ids_in_C = []


        # each target loop
        for aux in range(target_num): # ITerate across each target
            tar_rg = target_list[aux,0]
            tar_dp = target_list[aux,1]

            tmp_var = 0

            # Redundant 1
            input_3d_fft[0,0] = decode_fft2[tar_rg,tar_dp,0] # Tx1-Rx1
            input_3d_fft[0,1] = decode_fft2[tar_rg,tar_dp,4] # Tx2-Rx1
            input_3d_fft[0,2] = decode_fft2[tar_rg,tar_dp,1] # Tx1-Rx2
            input_3d_fft[0,3] = decode_fft2[tar_rg,tar_dp,5] # Tx2-Rx2
            ## Redundant 2
            input_3d_fft[1,2] = decode_fft2[tar_rg,tar_dp,2] # Tx1-Rx3
            input_3d_fft[1,3] = decode_fft2[tar_rg,tar_dp,6] # Tx2-Rx3
            ## Redundant 3
            input_3d_fft[2,2] = decode_fft2[tar_rg,tar_dp,3] # Tx1-Rx4
            input_3d_fft[2,3] = decode_fft2[tar_rg,tar_dp,7] # Tx2-Rx4

            # angle azimuth FFT
            #f_fft3_az = np.fft.fft(input_3d_fft,expand_az_bin, 1) # Original mirrored FFT

            f_fft3_az = np.fft.fft(np.conjugate(input_3d_fft),expand_az_bin, 1) # Fix introduced by Chen
            f_fft3_az= np.fft.fftshift(f_fft3_az, axes=1)

            # angle elevation FFT
            f_fft3_el= np.fft.fft(f_fft3_az,expand_el_bin, 0)
            f_fft3_el= np.fft.fftshift(f_fft3_el, axes=0)

            # Calculate the Maximum value coordinate of the 64X64 angle grid
            index_el, index_az = np.unravel_index(np.argmax(abs(f_fft3_el[:,:])), f_fft3_el.shape)
            val = abs(f_fft3_el[index_el, index_az])
            max_val_num = (abs(f_fft3_el[:])== val).sum()
            # az and el index are integrated into target list here
            target_list[aux,2] = index_az  # az index
            target_list[aux,3] = index_el # el index
            target_angle_val[aux]  = val

            # prepare az and el degree vector
            az_deg_vec = np.degrees(np.arcsin((np.arange(expand_az_bin)-expand_az_bin/2)/(expand_az_bin/2)))# (-90,90)
            el_deg_vec = np.degrees(np.arcsin((np.arange(expand_el_bin)-expand_el_bin/2)/(expand_el_bin/2)))# (-90,90)

            if plot_angle_map_en:
                plot_angle_map()# plot of 64X64 grid angle map

            #calculate azimuth angle (theta) and elevation angle (phi) and convert the polar coordinate to cartesian coordinate
            phi       = el_deg_vec[index_el]    # degree
            theta     = az_deg_vec[index_az]    # degree
            phi_rad   = np.deg2rad(phi)         # radian
            theta_rad = np.deg2rad(theta)       # radian
            # convert angle/range to cardesian coordinate
            pos_x = (axis_range[tar_rg])*(np.cos(phi_rad))*(np.sin(theta_rad))# left/right direction
            pos_y = (axis_range[tar_rg])*(np.cos(phi_rad))*(np.cos(theta_rad))# front/behind direction
            pos_z = (axis_range[tar_rg])*(np.sin(phi_rad))# up/down direction

            print('target ',str(aux+1),'/',str(target_num),':')
            print('Range index:\t',str(tar_rg),'\t\tRange:\t%.2f m'%axis_range[tar_rg])
            print('Doppler index:\t',str(tar_dp),'\t\tSpeed:\t%.2f m'%axis_doppler[tar_dp])
            print('Azimuth index:\t',str(index_az),'\t\tTheta:\t%.2f degree'%theta)
            print('Elevation index:',str(index_el),'\t\tPhi:\t%.2f degree'%phi)
            print('pos_x: %.2f m'%pos_x)
            print('pos_y: %.2f m'%pos_y)
            print('pos_z: %.2f m'%pos_z)
            print('relative power: %.2f'%rd_map[tar_rg,tar_dp])
            print("KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")


            # record full information of each target to the csv file.
            csv_current_line  = [Frame_selected,aux,axis_range[tar_rg],axis_doppler[tar_dp],rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi]
            targets_in_X_line = [Frame_selected,aux,axis_range[tar_rg],axis_doppler[tar_dp],rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi]
            target_ids_in_X_line  = aux

            if (experiment_type == "Sub_Frames"):
                if (Frame_idx % 3 == 0):

                       # Predicting the true velocity here means we are taking data from: B29 C29 A30,
                       # The groups A29 B29 C29 is discarded since A29 is grouped with B28 C28 A29
                       # We should do both, for this we have to do this in C29 as well
                       # But before this we have to export the CSV true vel data correctly

                       #targets_in_A = [] # We can store it here # Don't restart it here
                       #target_ids_in_A = [] # Don't restart it here

                       ### Save normally all the targets in this frame:
                       targets_in_A.append(targets_in_X_line)
                       target_ids_in_A.append(target_ids_in_X_line) # Only one element here always for each target

                       #csv_full_A.append(csv_current_line) # Moved below after the true velocity is determined
                       csv_full_A_original.append(csv_current_line)
                       dbscan_input_xyz_line_A = [pos_x,pos_y,pos_z] # The mapping with velocities will be made one to one with csv_current_line
                       dbscan_input_xyz_A.append(dbscan_input_xyz_line_A) # xyz coordinates considered in the clustering process

                       dbscan_input_xy_line_A = [pos_x,pos_y]
                       dbscan_input_xy_A.append(dbscan_input_xy_line_A) # Only XY coordinates for the clustering


                       # Also perform disambiguation:
                       single_target_in_A = []
                       single_target_id_in_A = []

                       single_target_in_A.append(targets_in_X_line)
                       single_target_id_in_A.append(target_ids_in_X_line)

                       ###############################################################################################################
                       # DBSCAN Velocity Disambiguation Functionality
                       if (Frame_idx > 1): # To ensure the Frame 0 is discarded since there is no information available from earlier

                           # Find the corresponding match for each target in the other frames
                           number_of_matches_in_B = 0
                           index_of_matching_target_in_B = []
                           for tgt_id in range(len(targets_in_B)): # Iterate along all targets in B Frame
                                  if ((abs(abs(axis_range[tar_rg]) - abs(targets_in_B[tgt_id][2])) < 0.4) and abs(axis_range[tar_rg])):
                                      number_of_matches_in_B = number_of_matches_in_B + 1
                                      index_of_matching_target_in_B.append(targets_in_B[tgt_id][1]) # Get the aux index

                           number_of_matches_in_C = 0
                           index_of_matching_target_in_C = []
                           for tgt_id in range(len(targets_in_C)): # Iterate along all targets in C Frame
                                  if ((abs(abs(axis_range[tar_rg]) - abs(targets_in_C[tgt_id][2])) < 0.4) and abs(axis_range[tar_rg])):
                                      number_of_matches_in_C = number_of_matches_in_C + 1
                                      index_of_matching_target_in_C.append(targets_in_C[tgt_id][1]) # Get the aux index

                           print('')
                           print('axis_range[tar_rg] ', axis_range[tar_rg])
                           print('')
                           print('targets_in_B: ',targets_in_B)
                           print('targets_in_C: ',targets_in_C)
                           print('')

                           print('number_of_matches_in_B: ',number_of_matches_in_B)
                           print('number_of_matches_in_C: ',number_of_matches_in_C)

                           if ((number_of_matches_in_B > 0) and (number_of_matches_in_C > 0)):
                               # Check if we have only one or more than one
                               input_data = []
                               input_data.append(single_target_id_in_A) # Only one element here for each target
                               input_data.append(index_of_matching_target_in_B)
                               input_data.append(index_of_matching_target_in_C) # TODO: Do this only with the index_of_matching_target
                               #input_data.append(target_ids_in_B)
                               #input_data.append(target_ids_in_C) # TODO: Do this only with the index_of_matching_target

                               result_input_data = list(itertools.product(*input_data))

                               # result_input_data has the tuples

                               # Generate the Conjecture velocities in an interleaved way to ensure the first cluster is always the smallest
                               number_of_valid_tuples = 0
                               for tuple_id in range(len(result_input_data)):
                                   # Do conjecture expansion here for each tuple
                                   print('')
                                   print('Current_tuple: ', result_input_data[tuple_id])
                                   # TgtID
                                   #   0  - targets_in_X_line =
                                   # [Frame_selected,aux,axis_range[tar_rg],axis_doppler[tar_dp],rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi]
                                   #   1  - targets_in_X_line =
                                   # [Frame_selected,aux,axis_range[tar_rg],axis_doppler[tar_dp],rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi]
                                   #   2  - targets_in_X_line =
                                   # [Frame_selected,aux,axis_range[tar_rg],axis_doppler[tar_dp],rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi]
                                   #      0           1      2                    3                   4                  5     6     7      8   9

                                   # Check how many clean clusters do we have?
                                   E = vel_resolution_A*2 #% Twice the threshold to be more flexible
                                   minPts = 3

                                   BinFrame1 = int(axis_doppler[tar_dp]/vel_resolution_A) # 27 chirps - Same as A1 in CRT
                                   BinFrame2 = int(targets_in_B[result_input_data[tuple_id][1]][3]/vel_resolution_B)
                                   # 35 chirps # Dimension [1] has targets from B # [3] refers to axis_doppler value
                                   BinFrame3 = int(targets_in_C[result_input_data[tuple_id][2]][3]/vel_resolution_C)
                                   # 44 chirps # Dimension [2] has targets from C # [3] refers to axis_doppler value

                                   print('')
                                   print('BinFrame1: ', BinFrame1)
                                   print('BinFrame2: ', BinFrame2)
                                   print('BinFrame3: ', BinFrame3)

                                   Range_MeasFrame1 = axis_range[tar_rg]
                                   Range_MeasFrame2 = targets_in_B[result_input_data[tuple_id][1]][2]
                                   # Dimension [1] has targets from B # [2] refers to axis_range value
                                   Range_MeasFrame3 = targets_in_C[result_input_data[tuple_id][2]][2]
                                   # Dimension [2] has targets from C # [2] refers to axis_range value

                                   print('')
                                   print('Range_MeasFrame1: ', Range_MeasFrame1)
                                   print('Range_MeasFrame2: ', Range_MeasFrame2)
                                   print('Range_MeasFrame3: ', Range_MeasFrame3)


                                   # Max velocity to constraint the algorithm to realistic values
                                   MaxVelScale = 200 # 200 m/s

                                   ## Frame 1 - Measurement 1
                                   Ndblocks_1 = int(numChirps_A/numTx)
                                   max_Number_bins_200mPers_Frame1_pos = abs((MaxVelScale/vel_resolution_A - BinFrame1)/Ndblocks_1)
                                   interleaved_scale_Frame1 = np.zeros([2*ceil(max_Number_bins_200mPers_Frame1_pos+1),1]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)
                                   interleaved_scale_Frame1_2d = np.zeros([2*ceil(max_Number_bins_200mPers_Frame1_pos+1),2]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)

                                   ## Frame 2 - Measurement 2
                                   Ndblocks_2 = int(numChirps_B/numTx)
                                   max_Number_bins_200mPers_Frame2_pos = abs((MaxVelScale/vel_resolution_B - BinFrame2)/Ndblocks_2);
                                   interleaved_scale_Frame2 = np.zeros([2*ceil(max_Number_bins_200mPers_Frame2_pos+1),1]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)
                                   interleaved_scale_Frame2_2d = np.zeros([2*ceil(max_Number_bins_200mPers_Frame2_pos+1),2]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)

                                   ## Frame 3 - Measurement 3
                                   Ndblocks_3 = int(numChirps_C/numTx)
                                   max_Number_bins_200mPers_Frame3_pos = abs((MaxVelScale/vel_resolution_C - BinFrame3)/Ndblocks_3);
                                   interleaved_scale_Frame3 = np.zeros([2*ceil(max_Number_bins_200mPers_Frame3_pos+1),1]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)
                                   interleaved_scale_Frame3_2d = np.zeros([2*ceil(max_Number_bins_200mPers_Frame3_pos+1),2]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)

                                   # Eacn value in the interleaved scale contains: [ConjF1,Range]
                                   #interleaved_scale = np.zeros([2*ceil(max_Number_bins_200mPers_Frame1_pos+1),2]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)


                                   # Extrapolation of measurements:
                                   NHeterogeneousFrames = 3
                                   #################################################################
                                   # Generating Conjecture velocities
                                   for frame_idx2 in range(NHeterogeneousFrames):
                                            if(frame_idx2 == 0):
                                                for indices in range(ceil(max_Number_bins_200mPers_Frame1_pos+1)):
                                                    #            0 1 2 3 4 5
                                                    # Pos scale: 0   2   4
                                                    # Neg scale:   1   3   5
                                                    interleaved_scale_Frame1[indices*2][0] = (BinFrame1 + Ndblocks_1*indices)*vel_resolution_A
                                                    interleaved_scale_Frame1[indices*2+1][0] = (BinFrame1 - Ndblocks_1*(indices+1))*vel_resolution_A

                                                    interleaved_scale_Frame1_2d[indices*2][0] = (BinFrame1 + Ndblocks_1*indices)*vel_resolution_A
                                                    interleaved_scale_Frame1_2d[indices*2+1][0] = (BinFrame1 - Ndblocks_1*(indices+1))*vel_resolution_A
                                                    # Fill in also the range information
                                                    interleaved_scale_Frame1_2d[indices*2][1] = Range_MeasFrame1
                                                    interleaved_scale_Frame1_2d[indices*2+1][1] = Range_MeasFrame1

                                            if(frame_idx2 == 1):
                                                for indices in range(ceil(max_Number_bins_200mPers_Frame2_pos+1)):
                                                    interleaved_scale_Frame2[indices*2][0] = (BinFrame2 + Ndblocks_2*indices)*vel_resolution_B
                                                    interleaved_scale_Frame2[indices*2+1][0] = (BinFrame2 - Ndblocks_2*(indices+1))*vel_resolution_B

                                                    interleaved_scale_Frame2_2d[indices*2][0] = (BinFrame2 + Ndblocks_2*indices)*vel_resolution_B
                                                    interleaved_scale_Frame2_2d[indices*2+1][0] = (BinFrame2 - Ndblocks_2*(indices+1))*vel_resolution_B
                                                    interleaved_scale_Frame2_2d[indices*2][1] = Range_MeasFrame2
                                                    interleaved_scale_Frame2_2d[indices*2+1][1] = Range_MeasFrame2

                                            if(frame_idx2 == 2):
                                                for indices in range(ceil(max_Number_bins_200mPers_Frame3_pos+1)):
                                                    interleaved_scale_Frame3[indices*2][0] = (BinFrame3 + Ndblocks_3*indices)*vel_resolution_C
                                                    interleaved_scale_Frame3[indices*2+1][0] = (BinFrame3 - Ndblocks_3*(indices+1))*vel_resolution_C

                                                    interleaved_scale_Frame3_2d[indices*2][0] = (BinFrame3 + Ndblocks_3*indices)*vel_resolution_C
                                                    interleaved_scale_Frame3_2d[indices*2+1][0] = (BinFrame3 - Ndblocks_3*(indices+1))*vel_resolution_C
                                                    interleaved_scale_Frame3_2d[indices*2][1] = Range_MeasFrame3
                                                    interleaved_scale_Frame3_2d[indices*2+1][1] = Range_MeasFrame3



                                   interleaved_scale_Frame1_np = np.array(interleaved_scale_Frame1, dtype=np.float32)
                                   interleaved_scale_Frame2_np = np.array(interleaved_scale_Frame2, dtype=np.float32)
                                   interleaved_scale_Frame3_np = np.array(interleaved_scale_Frame3, dtype=np.float32)
                                   interleaved_scale_Frame1_np_2d = np.array(interleaved_scale_Frame1_2d, dtype=np.float32)
                                   interleaved_scale_Frame2_np_2d = np.array(interleaved_scale_Frame2_2d, dtype=np.float32)
                                   interleaved_scale_Frame3_np_2d = np.array(interleaved_scale_Frame3_2d, dtype=np.float32)

                                   interleaved_scale_AllFrames_with_range_np = np.vstack((interleaved_scale_Frame1_np,interleaved_scale_Frame2_np,interleaved_scale_Frame3_np))# [ConjF1,RangeF1],[ConjF2,RangeF2],[ConjF3,RangeF3]...
                                   interleaved_scale_AllFrames_with_range_np_2d = np.vstack((interleaved_scale_Frame1_np_2d,interleaved_scale_Frame2_np_2d,interleaved_scale_Frame3_np_2d))# [ConjF1,RangeF1],[ConjF2,RangeF2],[ConjF3,RangeF3]...

                                   print("interleaved_scale_AllFrames_with_range_np.shape ", interleaved_scale_AllFrames_with_range_np.shape)
                                   db = DBSCAN(E, min_samples=minPts).fit(interleaved_scale_AllFrames_with_range_np) # epsilon default = 0.3. epsilon is set to be 1.5meters to consider the extent of a human being

                                   core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                                   core_samples_mask[db.core_sample_indices_] = True
                                   labels = db.labels_

                                   # Number of clusters in labels, ignoring noise if present.
                                   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                                   n_noise_ = list(labels).count(-1)

                                   print('Estimated number of clusters: %d' % n_clusters_)
                                   print('Estimated number of noise points: %d' % n_noise_)

                                   print('')
                                   print('labels: ', labels)
                                   print('interleaved_scale_AllFrames_with_range_np: ', interleaved_scale_AllFrames_with_range_np)

                                   if (n_clusters_ > 0):
                                      true_vel_cluster_loc = list(labels).index(0)

                                      ########### Verify if this is the smallest cluster #########3
                                      smallest_value_so_far = interleaved_scale_AllFrames_with_range_np[true_vel_cluster_loc]  # is this the smallest?

                                      for aux_idx2 in range(n_clusters_): # Iterate among the clusters
                                                   # The indexing excludes the -1, which is good because that is the noise cluster
                                                   if (aux_idx2 > 0): # Since we captured by default outside
                                                       current_cluster_loc = list(labels).index(aux_idx2)
                                                       current_evaluated_value = interleaved_scale_AllFrames_with_range_np[current_cluster_loc]
                                                       #print('')
                                                       #print('current_evaluated_value: ', current_evaluated_value)
                                                       if (abs(smallest_value_so_far) > abs(current_evaluated_value)):
                                                                       true_vel_cluster_loc = aux_idx2
                                                                       smallest_value_so_far = current_evaluated_value

                                      ########### End of verification                    #########3

                                      print('')
                                      print('true_vel_cluster_loc: ',true_vel_cluster_loc)
                                      print('interleaved_scale_AllFrames_with_range_np[true_vel_cluster_loc]: ',interleaved_scale_AllFrames_with_range_np[true_vel_cluster_loc])
                                      print('')
                                      # Then we have to replace the true velocity in the target list
                                      true_vel = interleaved_scale_AllFrames_with_range_np[true_vel_cluster_loc][0]
                                      tmp_var = aux + 1000
                                      # Run all experiments to see if we still have unrealistic velocities
                                      if (abs(true_vel)<15): # Unrealistic velocities are discarded -> This shouldn't be done but the target detection is not perfect
                                          csv_current_line  = [Frame_selected,tmp_var,axis_range[tar_rg],true_vel,rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi] # Update the CSV line
                                          csv_full_A.append(csv_current_line) # Append each cluster from each tuple, which had convergence

                                          print('targets_in_B[result_input_data[tuple_id][1]][1]) ', targets_in_B[result_input_data[tuple_id][1]][1])
                                          print('targets_in_C[result_input_data[tuple_id][2]][1]) ', targets_in_C[result_input_data[tuple_id][2]][1])

                                          aux_csv_current_line_B = targets_in_B[result_input_data[tuple_id][1]]
                                          aux_csv_current_line_C = targets_in_C[result_input_data[tuple_id][2]]

                                          aux_csv_current_line_B_cp = aux_csv_current_line_B.copy()
                                          aux_csv_current_line_C_cp = aux_csv_current_line_C.copy()

                                          aux_csv_current_line_B_cp[1] = tmp_var # 0: A, 1: B, 2: C | [1] => tmp_var
                                          aux_csv_current_line_C_cp[1] = tmp_var # 0: A, 1: B, 2: C | [1] => tmp_var

                                          csv_full_B.append(aux_csv_current_line_B_cp)
                                          # Duplicating targets to see which ones took part in the vel disambiguation process
                                          csv_full_C.append(aux_csv_current_line_C_cp)

                                          #print('targets_in_B: ', targets_in_B)
                                          #print('targets_in_C: ', targets_in_C)

                                          ######


                                   # Black removed and is used for noise instead.
                                   unique_labels = set(labels)
                                   colors = [plt.cm.Spectral(each)
                                             for each in np.linspace(0, 1, len(unique_labels))]
                                   for k, col in zip(unique_labels, colors):
                                       if k == -1:
                                           # Black used for noise.
                                           col = [0, 0, 0, 1]

                                       class_member_mask = (labels == k)

                                       xy = interleaved_scale_AllFrames_with_range_np_2d[class_member_mask & core_samples_mask]
                                       plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                                                markeredgecolor='k', markersize=14)

                                       xy = interleaved_scale_AllFrames_with_range_np_2d[class_member_mask & ~core_samples_mask]
                                       plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                                                markeredgecolor='k', markersize=6)

                                   plt.title('Estimated number of clusters: %d' % n_clusters_)
                                   #plt.show()
                                   ######################## End of DBSCAN Application

                                   ##### End of DBSCAN Numpy


                                   #################################################################

                           else: # No matching targets were found. else of if ((number_of_matches_in_B > 0) and (number_of_matches_in_A > 0)):
                               csv_full_A.append(csv_current_line) # If no matching targets were found, store the detected targets by this frame anyway

                       ###############################################################################################################

                elif (Frame_idx % 3 == 1):
                       target_ids_in_B.append(target_ids_in_X_line)
                       targets_in_B.append(targets_in_X_line)
                       csv_full_B.append(csv_current_line)
                       csv_full_B_original.append(csv_current_line)
                       dbscan_input_xyz_line_B = [pos_x,pos_y,pos_z] # The mapping with velocities will be made one to one with csv_current_line
                       dbscan_input_xyz_B.append(dbscan_input_xyz_line_B) # xyz coordinates considered in the clustering process

                       dbscan_input_xy_line_B = [pos_x,pos_y]
                       dbscan_input_xy_B.append(dbscan_input_xy_line_B) # Only XY coordinates for the clustering

                elif (Frame_idx % 3 == 2):
                       # Capture your targets in the normal list
                       target_ids_in_C.append(target_ids_in_X_line)
                       targets_in_C.append(targets_in_X_line)

                       #csv_full_C.append(csv_current_line)
                       csv_full_C_original.append(csv_current_line)
                       dbscan_input_xyz_line_C = [pos_x,pos_y,pos_z] # The mapping with velocities will be made one to one with csv_current_line
                       dbscan_input_xyz_C.append(dbscan_input_xyz_line_C) # xyz coordinates considered in the clustering process

                       dbscan_input_xy_line_C = [pos_x,pos_y]
                       dbscan_input_xy_C.append(dbscan_input_xy_line_C) # Only XY coordinates for the clustering

                       # But achieve disambiguation as well here with A29 B29
                       # perform disambiguation:
                       single_target_in_C = []
                       single_target_id_in_C = []

                       single_target_in_C.append(targets_in_X_line)
                       single_target_id_in_C.append(target_ids_in_X_line)

                       ###############################################################################################################
                       # DBSCAN Velocity Disambiguation Functionality

                       #if (Frame_idx > 1): # This is not really necessary for Frame C, since we already went through A and B

                       # Find the corresponding match for each target in the other frames
                       number_of_matches_in_B = 0
                       index_of_matching_target_in_B = []
                       for tgt_id in range(len(targets_in_B)): # Iterate along all targets in B Frame
                                  if ((abs(abs(axis_range[tar_rg]) - abs(targets_in_B[tgt_id][2])) < 0.4)  and abs(axis_range[tar_rg])):
                                      number_of_matches_in_B = number_of_matches_in_B + 1
                                      index_of_matching_target_in_B.append(targets_in_B[tgt_id][1]) # Get the aux index

                       number_of_matches_in_A = 0
                       index_of_matching_target_in_A = []
                       for tgt_id in range(len(targets_in_A)): # Iterate along all targets in C Frame
                                  if ((abs(abs(axis_range[tar_rg]) - abs(targets_in_A[tgt_id][2])) < 0.4) and abs(axis_range[tar_rg])):
                                      number_of_matches_in_A = number_of_matches_in_A + 1
                                      index_of_matching_target_in_A.append(targets_in_A[tgt_id][1]) # Get the aux index

                       print('')
                       print('targets_in_A: ',targets_in_A)
                       print('targets_in_B: ',targets_in_B)
                       print('axis_range[tar_rg] in C ', axis_range[tar_rg])
                       print('')
                       print('')
                       print('number_of_matches_in_A: ',number_of_matches_in_A)
                       print('number_of_matches_in_B: ',number_of_matches_in_B)

                       if ((number_of_matches_in_A > 0) and (number_of_matches_in_B > 0)):
                               # Check if we have only one or more than one
                               input_data = []
                               input_data.append(index_of_matching_target_in_A) # Only one element here for each target
                               input_data.append(index_of_matching_target_in_B)
                               input_data.append(single_target_id_in_C) # TODO: Do this only with the index_of_matching_target

                               result_input_data = list(itertools.product(*input_data))

                               # result_input_data has the tuples

                               # Generate the Conjecture velocities in an interleaved way to ensure the first cluster is always the smallest
                               number_of_valid_tuples = 0
                               for tuple_id in range(len(result_input_data)):
                                   # Do conjecture expansion here for each tuple
                                   print('')
                                   print('Current_tuple: ', result_input_data[tuple_id])
                                   # TgtID
                                   #   0  - targets_in_X_line =
                                   # [Frame_selected,aux,axis_range[tar_rg],axis_doppler[tar_dp],rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi]
                                   #   1  - targets_in_X_line =
                                   # [Frame_selected,aux,axis_range[tar_rg],axis_doppler[tar_dp],rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi]
                                   #   2  - targets_in_X_line =
                                   # [Frame_selected,aux,axis_range[tar_rg],axis_doppler[tar_dp],rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi]
                                   #      0           1      2                    3                   4                  5     6     7      8   9

                                   # Check how many clean clusters do we have?
                                   E = vel_resolution_C*2 #% Twice the threshold to be more flexible
                                   minPts = 3

                                   BinFrame1 = int(targets_in_A[result_input_data[tuple_id][0]][3]/vel_resolution_A)
                                   # 27 chirps # Dimension [0] has targets from B # [3] refers to axis_doppler value
                                   BinFrame2 = int(targets_in_B[result_input_data[tuple_id][1]][3]/vel_resolution_B)
                                   # 35 chirps # Dimension [1] has targets from B # [3] refers to axis_doppler value
                                   BinFrame3 = int(axis_doppler[tar_dp]/vel_resolution_C)
                                   # 44 chirps # Dimension [2] has targets from C # [3] refers to axis_doppler value

                                   print('')
                                   print('BinFrame1: ', BinFrame1)
                                   print('BinFrame2: ', BinFrame2)
                                   print('BinFrame3: ', BinFrame3)

                                   Range_MeasFrame1 = targets_in_A[result_input_data[tuple_id][0]][2]
                                   # Dimension [0] has targets from C # [2] refers to axis_range value
                                   Range_MeasFrame2 = targets_in_B[result_input_data[tuple_id][1]][2]
                                   # Dimension [1] has targets from B # [2] refers to axis_range value
                                   Range_MeasFrame3 = axis_range[tar_rg] # Dimension [2] has targets from C # [2] refers to axis_range value

                                   print('')
                                   print('Range_MeasFrame1: ', Range_MeasFrame1)
                                   print('Range_MeasFrame2: ', Range_MeasFrame2)
                                   print('Range_MeasFrame3: ', Range_MeasFrame3)


                                   # Max velocity to constraint the algorithm to realistic values
                                   MaxVelScale = 200 # 200 m/s

                                   ## Frame 1 - Measurement 1
                                   Ndblocks_1 = int(numChirps_A/numTx)
                                   max_Number_bins_200mPers_Frame1_pos = abs((MaxVelScale/vel_resolution_A - BinFrame1)/Ndblocks_1)
                                   interleaved_scale_Frame1 = np.zeros([2*ceil(max_Number_bins_200mPers_Frame1_pos+1),1]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)
                                   interleaved_scale_Frame1_2d = np.zeros([2*ceil(max_Number_bins_200mPers_Frame1_pos+1),2]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)
                                   ## Frame 2 - Measurement 2
                                   Ndblocks_2 = int(numChirps_B/numTx)
                                   max_Number_bins_200mPers_Frame2_pos = abs((MaxVelScale/vel_resolution_B - BinFrame2)/Ndblocks_2);
                                   interleaved_scale_Frame2 = np.zeros([2*ceil(max_Number_bins_200mPers_Frame2_pos+1),1]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)
                                   interleaved_scale_Frame2_2d = np.zeros([2*ceil(max_Number_bins_200mPers_Frame2_pos+1),2]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)

                                   ## Frame 3 - Measurement 3
                                   Ndblocks_3 = int(numChirps_C/numTx)
                                   max_Number_bins_200mPers_Frame3_pos = abs((MaxVelScale/vel_resolution_C - BinFrame3)/Ndblocks_3);
                                   interleaved_scale_Frame3 = np.zeros([2*ceil(max_Number_bins_200mPers_Frame3_pos+1),1]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)
                                   interleaved_scale_Frame3_2d = np.zeros([2*ceil(max_Number_bins_200mPers_Frame3_pos+1),2]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)

                                   # Eacn value in the interleaved scale contains: [ConjF1,Range]
                                   #interleaved_scale = np.zeros([2*ceil(max_Number_bins_200mPers_Frame1_pos+1),2]);
                                   # Interleaved data with the Smallest Values First (Positive and Negative)


                                   # Extrapolation of measurements:
                                   NHeterogeneousFrames = 3
                                   #################################################################
                                   # Generating Conjecture velocities
                                   for frame_idx2 in range(NHeterogeneousFrames):
                                            if(frame_idx2 == 0):
                                                for indices in range(ceil(max_Number_bins_200mPers_Frame1_pos+1)):
                                                    #            0 1 2 3 4 5
                                                    # Pos scale: 0   2   4
                                                    # Neg scale:   1   3   5
                                                    interleaved_scale_Frame1[indices*2][0] = (BinFrame1 + Ndblocks_1*indices)*vel_resolution_A
                                                    interleaved_scale_Frame1[indices*2+1][0] = (BinFrame1 - Ndblocks_1*(indices+1))*vel_resolution_A

                                                    # Fill in also the range information
                                                    interleaved_scale_Frame1_2d[indices*2][0] = (BinFrame1 + Ndblocks_1*indices)*vel_resolution_A
                                                    interleaved_scale_Frame1_2d[indices*2+1][0] = (BinFrame1 - Ndblocks_1*(indices+1))*vel_resolution_A
                                                    interleaved_scale_Frame1_2d[indices*2][1] = Range_MeasFrame1
                                                    interleaved_scale_Frame1_2d[indices*2+1][1] = Range_MeasFrame1

                                            if(frame_idx2 == 1):
                                                for indices in range(ceil(max_Number_bins_200mPers_Frame2_pos+1)):
                                                    interleaved_scale_Frame2[indices*2][0] = (BinFrame2 + Ndblocks_2*indices)*vel_resolution_B
                                                    interleaved_scale_Frame2[indices*2+1][0] = (BinFrame2 - Ndblocks_2*(indices+1))*vel_resolution_B

                                                    interleaved_scale_Frame2_2d[indices*2][0] = (BinFrame2 + Ndblocks_2*indices)*vel_resolution_B
                                                    interleaved_scale_Frame2_2d[indices*2+1][0] = (BinFrame2 - Ndblocks_2*(indices+1))*vel_resolution_B
                                                    interleaved_scale_Frame2_2d[indices*2][1] = Range_MeasFrame2
                                                    interleaved_scale_Frame2_2d[indices*2+1][1] = Range_MeasFrame2

                                            if(frame_idx2 == 2):
                                                for indices in range(ceil(max_Number_bins_200mPers_Frame3_pos+1)):
                                                    interleaved_scale_Frame3[indices*2][0] = (BinFrame3 + Ndblocks_3*indices)*vel_resolution_C
                                                    interleaved_scale_Frame3[indices*2+1][0] = (BinFrame3 - Ndblocks_3*(indices+1))*vel_resolution_C

                                                    interleaved_scale_Frame3_2d[indices*2][0] = (BinFrame3 + Ndblocks_3*indices)*vel_resolution_C
                                                    interleaved_scale_Frame3_2d[indices*2+1][0] = (BinFrame3 - Ndblocks_3*(indices+1))*vel_resolution_C
                                                    interleaved_scale_Frame3_2d[indices*2][1] = Range_MeasFrame3
                                                    interleaved_scale_Frame3_2d[indices*2+1][1] = Range_MeasFrame3


                                   #% Define the range array for each extrapolated measurement:
                                   #range_Meas_Array_Frame1 = Range_MeasFrame1*np.ones(2*ceil(max_Number_bins_200mPers_Frame1_pos+1));
                                   #range_Meas_Array_Frame2 = Range_MeasFrame2*np.ones(2*ceil(max_Number_bins_200mPers_Frame2_pos+1));
                                   #range_Meas_Array_Frame3 = Range_MeasFrame3*np.ones(2*ceil(max_Number_bins_200mPers_Frame3_pos+1));

                                   interleaved_scale_Frame1_np = np.array(interleaved_scale_Frame1, dtype=np.float32)
                                   interleaved_scale_Frame2_np = np.array(interleaved_scale_Frame2, dtype=np.float32)
                                   interleaved_scale_Frame3_np = np.array(interleaved_scale_Frame3, dtype=np.float32)

                                   interleaved_scale_Frame1_np_2d = np.array(interleaved_scale_Frame1_2d, dtype=np.float32)
                                   interleaved_scale_Frame2_np_2d = np.array(interleaved_scale_Frame2_2d, dtype=np.float32)
                                   interleaved_scale_Frame3_np_2d = np.array(interleaved_scale_Frame3_2d, dtype=np.float32)


                                   interleaved_scale_AllFrames_with_range_np = np.vstack((interleaved_scale_Frame1_np,interleaved_scale_Frame2_np,interleaved_scale_Frame3_np))# [ConjF1,RangeF1],[ConjF2,RangeF2],[ConjF3,RangeF3]...
                                   interleaved_scale_AllFrames_with_range_np_2d = np.vstack((interleaved_scale_Frame1_np_2d,interleaved_scale_Frame2_np_2d,interleaved_scale_Frame3_np_2d))# [ConjF1,RangeF1],[ConjF2,RangeF2],[ConjF3,RangeF3]...


                                   print("interleaved_scale_AllFrames_with_range_np.shape ", interleaved_scale_AllFrames_with_range_np.shape)
                                   db = DBSCAN(E, min_samples=minPts).fit(interleaved_scale_AllFrames_with_range_np) # epsilon default = 0.3. epsilon is set to be 1.5meters to consider the extent of a human being

                                   core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                                   core_samples_mask[db.core_sample_indices_] = True
                                   labels = db.labels_

                                   # Number of clusters in labels, ignoring noise if present.
                                   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                                   n_noise_ = list(labels).count(-1)

                                   print('Estimated number of clusters: %d' % n_clusters_)
                                   print('Estimated number of noise points: %d' % n_noise_)

                                   print('')
                                   print('labels: ', labels)
                                   print('interleaved_scale_AllFrames_with_range_np: ', interleaved_scale_AllFrames_with_range_np)

                                   if (n_clusters_ > 0):
                                      true_vel_cluster_loc = list(labels).index(0)

                                      ########### Verify if this is the smallest cluster #########3
                                      smallest_value_so_far = interleaved_scale_AllFrames_with_range_np[true_vel_cluster_loc]  # is this the smallest?

                                      for aux_idx2 in range(n_clusters_): # Iterate among the clusters
                                                   # The indexing excludes the -1, which is good because that is the noise cluster
                                                   if (aux_idx2 > 0): # Since we captured by default outside
                                                       current_cluster_loc = list(labels).index(aux_idx2)
                                                       current_evaluated_value = interleaved_scale_AllFrames_with_range_np[current_cluster_loc]
                                                       #print('')
                                                       #print('current_evaluated_value: ', current_evaluated_value)
                                                       if (abs(smallest_value_so_far) > abs(current_evaluated_value)):
                                                                       true_vel_cluster_loc = aux_idx2
                                                                       smallest_value_so_far = current_evaluated_value

                                      ########### End of verification                    #########3

                                      print('')
                                      print('true_vel_cluster_loc: ',true_vel_cluster_loc)
                                      print('interleaved_scale_AllFrames_with_range_np[true_vel_cluster_loc]: ',interleaved_scale_AllFrames_with_range_np[true_vel_cluster_loc])
                                      print('')
                                      # Then we have to replace the true velocity in the target list
                                      true_vel = interleaved_scale_AllFrames_with_range_np[true_vel_cluster_loc][0]
                                      tmp_var = aux + 2000 # 2000 when the prediction occurs in SubFrame C. 1000 when the prediction occurs in SubFrame A
                                      if (abs(true_vel)<15): # Unrealistic velocities are discarded -> This shouldn't be done but the target detection is not perfect
                                          csv_current_line  = [Frame_selected,tmp_var,axis_range[tar_rg],true_vel,rd_map[tar_rg,tar_dp],pos_x,pos_y,pos_z,theta,phi] # Update the CSV line
                                          # If we want to store all the velocities for all tuples, then we have to include the append here
                                          csv_full_C.append(csv_current_line) # csv line should have the true velocity here

                                          # Edit the target ID on the used Frames to see which one was used on the prediction
                                          #print('targets_in_A[result_input_data[tuple_id][0]][1]) ', targets_in_A[result_input_data[tuple_id][0]][1])
                                          #print('targets_in_B[result_input_data[tuple_id][1]][1]) ', targets_in_B[result_input_data[tuple_id][1]][1])

                                          aux_csv_current_line_A = targets_in_A[result_input_data[tuple_id][0]]
                                          aux_csv_current_line_B = targets_in_B[result_input_data[tuple_id][1]]

                                          aux_csv_current_line_A_cp = aux_csv_current_line_A.copy()
                                          aux_csv_current_line_B_cp = aux_csv_current_line_B.copy()

                                          aux_csv_current_line_A_cp[1] = tmp_var # 0: A, 1: B, 2: C | [1] => tmp_var
                                          aux_csv_current_line_B_cp[1] = tmp_var # 0: A, 1: B, 2: C | [1] => tmp_var

                                          csv_full_A.append(aux_csv_current_line_A_cp)
                                          csv_full_B.append(aux_csv_current_line_B_cp)



                                   # Black removed and is used for noise instead.
                                   unique_labels = set(labels)
                                   colors = [plt.cm.Spectral(each)
                                             for each in np.linspace(0, 1, len(unique_labels))]
                                   for k, col in zip(unique_labels, colors):
                                       if k == -1:
                                           # Black used for noise.
                                           col = [0, 0, 0, 1]

                                       class_member_mask = (labels == k)

                                       xy = interleaved_scale_AllFrames_with_range_np_2d[class_member_mask & core_samples_mask]
                                       plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                                                markeredgecolor='k', markersize=14)

                                       xy = interleaved_scale_AllFrames_with_range_np_2d[class_member_mask & ~core_samples_mask]
                                       plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                                                markeredgecolor='k', markersize=6)

                                   plt.title('Estimated number of clusters: %d' % n_clusters_)
                                   #plt.show()
                                   ######################## End of DBSCAN Application

                                   ##### End of DBSCAN Numpy


                                   #################################################################

                       else: # No matching targets were found. else of if ((number_of_matches_in_B > 0) and (number_of_matches_in_A > 0)):
                               csv_full_C.append(csv_current_line) # If there are no matching targets in the other frames, store the targets detected by this frame anyway
                               # TODO: Another option is to discard all targets that were not detected in the three frames, we need only to comment this else statement

                       ########################################################################

            else: # Normal or Non_Uniform case
                csv_full.append(csv_current_line)

                dbscan_input_xyz_line = [pos_x,pos_y,pos_z] # The mapping with velocities will be made one to one with csv_current_line
                dbscan_input_xyz.append(dbscan_input_xyz_line) # xyz coordinates considered in the clustering process

                dbscan_input_xy_line = [pos_x,pos_y]
                dbscan_input_xy.append(dbscan_input_xy_line) # Only XY coordinates for the clustering

            # Extract Micro Doppler Signature to a csv file
            # The static targets (ground cluster) are filtered out for uD signature
            #if abs(axis_doppler[tar_dp]) > 0.3:            
            if abs(axis_doppler[tar_dp]) > 0.1: # After fixing vel_resol with a correct tidle, 0.1 compensates instead of 0.3 to get same old results
                csv_uDs.append(csv_current_line[:5])

        ########### target loop end

        print("") # DELETE
        print("rd_map.shape ", rd_map.shape)  # DELETE # (128,64)
        print("len(csv_current_line) ", len(csv_current_line)) # DELETE # Number of targets detected in a frame, with the coordinates
        print("len(csv_uDs) ", len(csv_uDs)) # DELETE # Number of targets that are part of the microdoppler signature



        # plot polar chart and DBF chart
        if plot_az_rg_en:
            plot_polar_az_rg(target_list, target_angle_val, output_3d_fft_Az,az_deg_vec)
        if plot_el_rg_en:
            plot_polar_el_rg(target_list, target_angle_val, output_3d_fft_Elv,el_deg_vec)
        if plot_dbf_az_en:
            plot_dbf_az(az_deg_vec,output_3d_fft_Az)
        if plot_dbf_el_en:
            plot_dbf_el(el_deg_vec,output_3d_fft_Elv)

        if (experiment_type == "Sub_Frames"):
                if (Frame_idx % 3 == 0):
                     contador_dbscan = contador_dbscan_A
                if (Frame_idx % 3 == 1):
                     contador_dbscan = contador_dbscan_B
                if (Frame_idx % 3 == 2):
                     contador_dbscan = contador_dbscan_C
        # No need to code for the Normal or Non_uniform since contador_dbscan is the only one considered

        ########### DBSCAN applied to targets from consecutie Frames
        # Apply DBSCAN once dbscan_frames are reached
        if contador_dbscan==dbscan_frames-1: # or any of the other dbscan counters
              ######################## Start DBSCAN Application
              # X = StandardScaler().fit_transform(X) # Removes baseline

              if (experiment_type == "Sub_Frames"):
                if (Frame_idx % 3 == 0):
                   dbscan_input_xyz_np = np.array(dbscan_input_xyz_A, dtype=np.float32) # XYZ - Uncomment to consider the XYZ
                   #dbscan_input_xyz_np = np.array(dbscan_input_xy_A, dtype=np.float32)   # XY - Uncomment to consider the XY
                if (Frame_idx % 3 == 1):
                   dbscan_input_xyz_np = np.array(dbscan_input_xyz_B, dtype=np.float32) # XYZ - Uncomment to consider the XYZ
                   #dbscan_input_xyz_np = np.array(dbscan_input_xy_B, dtype=np.float32)   # XY - Uncomment to consider the XY
                if (Frame_idx % 3 == 2):
                   dbscan_input_xyz_np = np.array(dbscan_input_xyz_C, dtype=np.float32) # XYZ - Uncomment to consider the XYZ
                   #dbscan_input_xyz_np = np.array(dbscan_input_xy_C, dtype=np.float32)   # XY - Uncomment to consider the XY
              else: # Normal or Non_uniform cases
                dbscan_input_xyz_np = np.array(dbscan_input_xyz, dtype=np.float32) # XYZ - Uncomment to consider the XYZ
                #dbscan_input_xyz_np = np.array(dbscan_input_xy, dtype=np.float32)   # XY - Uncomment to consider the XY

              print("dbscan_input_xyz.shape ", dbscan_input_xyz_np.shape)

              db = DBSCAN(1.5, min_samples=4).fit(dbscan_input_xyz_np) # epsilon default = 0.3. epsilon is set to be 1.5meters to consider the extent of a human being

              core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
              core_samples_mask[db.core_sample_indices_] = True
              labels = db.labels_

              # Number of clusters in labels, ignoring noise if present.
              n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
              n_noise_ = list(labels).count(-1)

              print('Estimated number of clusters: %d' % n_clusters_)
              print('Estimated number of noise points: %d' % n_noise_)

              # Black removed and is used for noise instead.
              unique_labels = set(labels)
              colors = [plt.cm.Spectral(each)
                        for each in np.linspace(0, 1, len(unique_labels))]
              for k, col in zip(unique_labels, colors):
                  if k == -1:
                      # Black used for noise.
                      col = [0, 0, 0, 1]

                  class_member_mask = (labels == k)

                  xy = dbscan_input_xyz_np[class_member_mask & core_samples_mask]
                  plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                           markeredgecolor='k', markersize=14)

                  xy = dbscan_input_xyz_np[class_member_mask & ~core_samples_mask]
                  plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                           markeredgecolor='k', markersize=6)

              plt.title('Estimated number of clusters: %d' % n_clusters_)
              #plt.show()
              ######################## End of DBSCAN Application

              if (experiment_type == "Sub_Frames"):
                if (Frame_idx % 3 == 0):
                     dbscan_input_xyz_A = []
                     dbscan_input_xy_A = []
                     contador_dbscan_A = 0
                if (Frame_idx % 3 == 1):
                     dbscan_input_xyz_B = []
                     dbscan_input_xy_B = []
                     contador_dbscan_B = 0
                if (Frame_idx % 3 == 2):
                     dbscan_input_xyz_C = []
                     dbscan_input_xy_C = []
                     contador_dbscan_C = 0
              else: # Normal or Non_uniform cases
                dbscan_input_xyz = []
                dbscan_input_xy = []

                contador_dbscan = 0


              #time.sleep(10)
        else:
              if (experiment_type == "Sub_Frames"):
                if (Frame_idx % 3 == 0):
                     contador_dbscan_A = contador_dbscan_A + 1
                if (Frame_idx % 3 == 1):
                     contador_dbscan_B = contador_dbscan_B + 1
                if (Frame_idx % 3 == 2):
                     contador_dbscan_C = contador_dbscan_C + 1
              else: # Normal or Non_uniform cases
                contador_dbscan = contador_dbscan + 1
        ########### End of DBSCAN applied to targets from consecutie Frames

        gc.collect()
        ### End frame loop

# HG Note: Continuar aca
# HG Note: Important to keep in mind in the ML problem
# - Not only data from the walking object is captured in the microdoppler signature
# - We need isolation for the target in mention
# - There will be frames where no target is detected
# - How does the movement of the platform affects? If we move at a constant speed, we consider the speed of the platform to remove the baseline. The constant approximation
#    should be good enough due to the small time intervals
# - We need to take the csv data and probably the rd data for training while for testing we have to figure out how to consider every ten or 100 frames a valid input data
# - Which are the variables with the relevant information in terms of range doppler map, range chirps (Micro range), and microdoppler
# HG Note continuar aca definiendo alternativas para el algoritmo de ML Feb 1 2020


if (experiment_type == "Sub_Frames"):
   np.save('rd_map_cube_A.npy',np.asarray(rd_map_cube_A)) # save this for plotting animation and uDs with external tool.
   np.save('rd_map_cube_B.npy',np.asarray(rd_map_cube_B)) # save this for plotting animation and uDs with external tool.
   np.save('rd_map_cube_C.npy',np.asarray(rd_map_cube_C)) # save this for plotting animation and uDs with external tool.
   csv_full_A = np.asarray(csv_full_A)
   csv_full_B = np.asarray(csv_full_B)
   csv_full_C = np.asarray(csv_full_C)

   np.savetxt('csv_full_A.csv',csv_full_A,fmt='%.2f',delimiter=',',header=csv_header_full)
   np.savetxt('csv_full_B.csv',csv_full_B,fmt='%.2f',delimiter=',',header=csv_header_full)
   np.savetxt('csv_full_C.csv',csv_full_C,fmt='%.2f',delimiter=',',header=csv_header_full)

   csv_full_A_original = np.asarray(csv_full_A_original)
   csv_full_B_original = np.asarray(csv_full_B_original)
   csv_full_C_original = np.asarray(csv_full_C_original)

   np.savetxt('csv_full_A_original.csv',csv_full_A_original,fmt='%.2f',delimiter=',',header=csv_header_full)
   np.savetxt('csv_full_B_original.csv',csv_full_B_original,fmt='%.2f',delimiter=',',header=csv_header_full)
   np.savetxt('csv_full_C_original.csv',csv_full_C_original,fmt='%.2f',delimiter=',',header=csv_header_full)


   csv_uDs_A = np.asarray(csv_uDs_A)
   csv_uDs_B = np.asarray(csv_uDs_B)
   csv_uDs_C = np.asarray(csv_uDs_C)

   np.savetxt('csv_uDs_A.csv',csv_uDs_A,fmt='%.2f',delimiter=',',header=csv_header_uDs)
   np.savetxt('csv_uDs_B.csv',csv_uDs_B,fmt='%.2f',delimiter=',',header=csv_header_uDs)
   np.savetxt('csv_uDs_C.csv',csv_uDs_C,fmt='%.2f',delimiter=',',header=csv_header_uDs)

else: # Normal and Non_Uniform cases
   np.save('rd_map_cube.npy',np.asarray(rd_map_cube)) # save this for plotting animation and uDs with external tool.

   csv_full = np.asarray(csv_full)
   np.savetxt('csv_full.csv',csv_full,fmt='%.2f',delimiter=',',header=csv_header_full)

   csv_uDs = np.asarray(csv_uDs)
   np.savetxt('csv_uDs.csv',csv_uDs,fmt='%.2f',delimiter=',',header=csv_header_uDs)


plt.close('all')
print("Finished.")
