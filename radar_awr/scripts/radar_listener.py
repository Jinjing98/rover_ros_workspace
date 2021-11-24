#!/usr/bin/env python
PKG = 'radar_awr'
import roslib; roslib.load_manifest(PKG)

import rospy
from radar_awr.msg import RadarFrame
from rospy.numpy_msg import numpy_msg
import numpy as np

# Callback funtion for ROS subscriber node of radar AWR1642
# RadarFrame message contains three arrays:
# real: flattened real values for one frame of the adc data
# im: flattened imagenary values for one frame of the adc data
# shape: shape of the original numpy array of the transmitter-seperated adc data
# As ROS does not allow the use of complex numbers in messages and is not well
# to use with numpy the data gets split into flattened real and imagenary values.
# Add both flattened arrays to restore the data. Use the shape array to reshape the
# the complex valued array into its original dimensions.
# Otherwise the here shown restore method can be used as well.
def callback(msg):
    print("shape should be:", msg.shape)
    shape = msg.shape
    frame = np.ndarray(msg.shape, dtype=np.complex64)
    print(msg.real, msg.im)
    print("shape of msg arrays:", msg.real.shape, msg.im.shape)
    print("dtype of msg arrays:", msg.real.dtype, msg.im.dtype)
    frame.real = msg.real.reshape(shape)
    frame.imag = msg.im.reshape(shape)
    print("shape of final frame:", frame.shape, frame.dtype)
    #print rospy.get_name(), "I heard %s"%str(data.shape)

def listener():
    rospy.init_node('radar_sub')
    rospy.Subscriber("radar_frame", numpy_msg(RadarFrame), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
