#**************************************************************************
#                        IMPORTANT NOTICE
#**************************************************************************
# THIS SOFTWARE IS PROVIDED "AS IS". NO WARRANTIES, WHETHER EXPRESS,
# IMPLIED OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS
# SOFTWARE. INFINEON SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, FOR ANY REASON WHATSOEVER.
#**************************************************************************

import sys
sys.path.insert(0, '..')

import strata

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

# get config structs
import strata_RangeDoppler_cfg
mmicConfig, sequenceConfig, processingConfig = strata_RangeDoppler_cfg.getConfigs()

# connect to board
print("Strata Software Version: ", strata.getVersion())
board = strata.connection.withAutoAddress()
print("Board Image Version", board.getVersion())

moduleRadar = board.getIModuleRadar()

# configure all settings
moduleRadar.setConfiguration(mmicConfig)
moduleRadar.setSequence(sequenceConfig)
moduleRadar.setProcessingStages(processingConfig)
moduleRadar.configure()

# Get values for visualization
dataProperties = strata.IDataProperties_t()
moduleRadar.getDataProperties(dataProperties)
NoSamples = dataProperties.samples
NoRX = dataProperties.rxChannels

radarInfo = strata.IProcessingRadarInput_t()
moduleRadar.getRadarInfo(radarInfo)
NoRampsPerTx = radarInfo.rampsPerTx
NoTX = radarInfo.txChannels

NoRangeBins = strata.utils.nextpow2(NoSamples)
if processingConfig.fftSettings[0].flags & strata.IfxRsp.FFT_FLAGS.DISCARD_HALF:
    NoRangeBins = NoRangeBins // 2
    
if processingConfig.fftSettings[1].size == 0:
    NoVelocityBins = strata.utils.nextpow2(NoRampsPerTx)
else:
    NoVelocityBins = strata.utils.nextpow2(stages.fftsettings[1].size)

print("NoRX:", NoRX, ", NoTX:", NoTX, ", NoVelocityBins:", NoVelocityBins, ", NoRangeBins:", NoRangeBins)

nciEn = (processingConfig.nciFormat != strata.IfxRsp.DataFormat.Disabled)

if processingConfig.format == strata.IfxRsp.DataFormat.Complex32:
    FftBinSz = 8
    nciBinSz = 4
else:
    print("Error: FFT format not supported!")
    exit()
Fft2Sz = NoRangeBins * NoVelocityBins * FftBinSz

if processingConfig.fftSettings[1].flags & strata.IfxRsp.FFT_FLAGS.INPLACE:
    RD_offset = 0
else:
    RD_offset = Fft2Sz * NoRX * NoTX

if ~nciEn:
    RD_size = Fft2Sz
else:
    RD_offset = RD_offset + Fft2Sz * NoRX * NoTX
    RD_size = NoRangeBins * NoVelocityBins * nciBinSz

# scaling for plot
rmax = radarInfo.maxRange
vmax = radarInfo.maxVelocity
r_vect = np.linspace(0, 0.5, NoRangeBins, False) * 2 * rmax
v_vect = np.linspace(-0.5, 0.5, NoVelocityBins, False) * 2 * vmax
v_vect = np.reshape(v_vect, (NoVelocityBins, 1))

# prepare for data plotting
fig = plt.figure()
sp = fig.add_subplot(1, 1, 1, projection='3d')
sp.set_xlabel('Velocity (m/s)')
sp.set_ylabel('Range (m)')
sp.set_zlabel('Magnitude (dB)')

stopClicked = False


def stopFunction(event):
    global stopClicked
    stopClicked = True


#add stop button to the figure
axstop = fig.add_axes([0.01, 0.05, 0.05, 0.05])
stopButton = Button(axstop, 'stop', color='0.75', hovercolor='r')
stopButton.on_clicked(stopFunction)

# start measurements and do one
measurementTime = 50e-3
moduleRadar.startMeasurements(measurementTime)

# show rangeDoppler data
while not stopClicked:
    measurement = board.getFrame()
      
    if len(measurement.data) == 0:
        print("Frame dropped due to packet loss! Continuing with next frame...")
        continue

    measurement = np.frombuffer(measurement.data, np.dtype(np.int32).newbyteorder('<'), int(RD_size/4), int(RD_offset/4))
    RD_data = measurement.astype(np.float32).view(np.complex64)

    zPower = np.absolute(RD_data)
    zPower = 10 * np.log10(zPower)
    zPower = np.reshape(zPower, (NoRangeBins, NoVelocityBins), order='F')
    zPower = zPower.transpose()
    zPower = np.fft.fftshift(zPower, axes=(0,))

    sp.cla()
    surf = sp.plot_surface(v_vect, r_vect, zPower, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.pause(1e-6)  # pause to allow plot to draw

moduleRadar.stopMeasurements()
