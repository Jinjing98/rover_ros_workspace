#! /usr/bin/python3
import sys
print(sys.version)
sys.path.insert(0, '/home/tud-jxavier/infineon/Strata_release_x64_2.0.0/Python/')
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

# get config structs
import strata_Timedata_cfg


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
    moduleRadar.doMeasurement()
    measurement = board.getFrame()



    timestamp = time.to_time()	
    date = datetime.datetime.fromtimestamp(timestamp)
    print(time,timestamp,date)
    

    # this has to be string in order to use npz format!!
    npz_idx_tsp = str(int(str(time))/1000)


  
    if len(measurement.data) == 0:
        print("Frame dropped due to packet loss! Re-doing measurement...")
    else:
        #jinjing   comment out below break and add more
        #break


	#jinjing     direct get each data
        f.write(str(int(str(time))/1000)+' '+ str(date)+"\n")

        measurement = np.frombuffer(measurement.data, np.dtype(np.int16).newbyteorder('<'))
        #(16384,)        
        measurement = measurement[0::2]/2**15
	 #  [256,1,8,4]  NoSamples, NoTX, NoRampsPerTx, NoRX 
        measurement = np.reshape(measurement, (NoSamples, NoTX, NoRampsPerTx, NoRX), order='F')
        print(measurement.shape,measurement)
        print("get one timedata!")
        dict_measurement[npz_idx_tsp]= measurement

 
	
    rospy.sleep( 0.00005)#max grabing is 30 or so 


def main():
    rospy.init_node('infeneon_radar_timedata')
    rospy.Subscriber("/dummy4radar", String, radarCallback)
  
 
    rospy.spin()

 

if __name__ == '__main__':
    mmicConfig, sequence, stages = strata_Timedata_cfg.getConfigs()

    # connect to board
    print("Strata Software Version: ", strata.getVersion())
    board = strata.connection.withAutoAddress()
    print("Board Image Version", board.getVersion())

    moduleRadar = board.getIModuleRadar()

    # configure all settings
    moduleRadar.setConfiguration(mmicConfig)
    moduleRadar.setSequence(sequence)
    moduleRadar.setProcessingStages(stages)
    moduleRadar.configure()

    # Extra configuration
    #radarRxs = board.getIRadarRxs(0)
    #registers = radarRxs.getIRegisters()
    #radarRxs.enableDividerOutput(True)            # enable divider output for test
    #registers.setBits(0x042C, 0x0001)     # DMUX1 as output
    #registers.write(0x0434, 0x0020)       # DMUX1 map DMUX_A

    # Get values for visualization
    dataProperties = strata.IDataProperties_t()
    moduleRadar.getDataProperties(dataProperties)
    NoSamples = dataProperties.samples
    NoRX = dataProperties.rxChannels

    radarInfo = strata.IProcessingRadarInput_t()
    moduleRadar.getRadarInfo(radarInfo)
    NoRampsPerTx = radarInfo.rampsPerTx
    NoTX = radarInfo.txChannels

    # start measurements and do one
    moduleRadar.startMeasurements(0)




















    f = open("/media/tud-jxavier/SSD/data/radar/timestamp_radar_timedata.txt",'a')
    f.truncate(0)

    dict_measurement = {}
    # dict_measurement_fft = {}
    dict_measurement_path ="/media/tud-jxavier/SSD/data/radar/measurement_timedata.npz"
    # dict_measurement_fft_path ="/media/tud-jxavier/SSD/data/radar/measurement_fft_rangedop.npz"


 
 
    main()

    print("total number of data ",len(dict_measurement))

    np.savez(dict_measurement_path,**dict_measurement)
    # np.savez(dict_measurement_fft_path,**dict_measurement_fft)


    f.close()



 
