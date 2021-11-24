#! /usr/bin/python3
import sys
print(sys.version)
sys.path.insert(0, '/media/tud-jxavier/SSD/infineon/Strata_release_x64_2.0.0/Python/')
import strata
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# get config structs
import strata_RangeDoppler_cfg
from sensor_msgs.msg import Image,Imu
import cv2
from std_msgs.msg import String


import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import message_filters
import datetime,time
from datetime import timedelta
from zed_interfaces.msg import Objects
import numpy as np
import glob
import os
import pickle


 







def radarCallback(dummytime):
    time = rospy.get_rostime()
    measurement = board.getFrame()



    timestamp = time.to_time()	
    date = datetime.datetime.fromtimestamp(timestamp)
    print(time,timestamp,date)

    

    # this has to be string in order to use npz format!!
    npz_idx_tsp = str(int(int(str(time))/1000))#str(int(str(time))/1000)#str(int(int(str(time))/1000))



      
    if len(measurement.data) == 0:
        print("Frame dropped due to packet loss! Continuing with next frame...")
    else:
        f.write(str(int(int(str(time))/1000))+' '+ str(date)+"\n")
        measurement = np.frombuffer(measurement.data, np.dtype(np.int32).newbyteorder('<'), int(RD_size/4), int(RD_offset/4))
        RD_data = measurement.astype(np.float32).view(np.complex64)
        print("test",RD_data.shape)
        zPower = np.absolute(RD_data)
        zPower = 10 * np.log10(zPower)
        zPower = np.reshape(zPower, (NoRangeBins, NoVelocityBins), order='F')
        print("test",zPower.shape)
        zPower = zPower.transpose()
        print("test",zPower.shape)
        zPower_fft = np.fft.fftshift(zPower, axes=(0,))
        print("test",zPower_fft.shape)




    print("I am done with one measurement!")


    dict_measurement[npz_idx_tsp] = zPower
    dict_measurement_fft[npz_idx_tsp] = zPower_fft
    
    
    

 





    
 
	
    rospy.sleep( 0.00005)#max grabing is 30 or so 


def main():
    rospy.init_node('infeneon_radar_range_dop')
    rospy.Subscriber("/dummy4radar", String, radarCallback)
  
 
    rospy.spin()

 

if __name__ == '__main__':
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
    print(rmax,r_vect.shape)
    v_vect = np.linspace(-0.5, 0.5, NoVelocityBins, False) * 2 * vmax
    print(vmax,v_vect.shape)
    v_vect = np.reshape(v_vect, (NoVelocityBins, 1))

    # prepare for data plotting
    fig = plt.figure()
    sp = fig.add_subplot(1, 1, 1, projection='3d')
    sp.set_xlabel('Velocity (m/s)')
    sp.set_ylabel('Range (m)')
    sp.set_zlabel('Magnitude (dB)')

    # start measurements and do one
    measurementTime = 50e-3
    moduleRadar.startMeasurements(measurementTime)

    firstFrame = True

    f = open("/media/tud-jxavier/SSD/data/radar/timestamp_radar_rangedop.txt",'a')
    f.truncate(0)

    dict_measurement = {}
    dict_measurement_fft = {}
    dict_measurement_path ="/media/tud-jxavier/SSD/data/radar/measurement_rangedop.npz"
    dict_measurement_fft_path ="/media/tud-jxavier/SSD/data/radar/measurement_fft_rangedop.npz"


 
 
    main()

    print("total number of data ",len(dict_measurement))

    np.savez(dict_measurement_path,**dict_measurement)
    np.savez(dict_measurement_fft_path,**dict_measurement_fft)


    f.close()
 
