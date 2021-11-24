#!/usr/bin/env python
#PKG = 'radar_awr'
#import roslib; roslib.load_manifest(PKG)

import rospy
from radar_awr.msg import RadarFrame
import capture_helper as helper

# actual publisher node
def talker(params, dca):
    pub = rospy.Publisher('radar_frame', RadarFrame, queue_size=10)
    rospy.init_node('radar_pub', anonymous=True) #without anonymous?
    # 100Hz, keep this high, otherwise ethernet packages could get lost
    r = rospy.Rate(1000)
    while not rospy.is_shutdown():
        real, im, shape = helper.recording(dca, params)
        pub.publish(shape, real, im)
        #rospy.loginfo("pub %d", real[0,0,0])
        r.sleep()

if __name__ == '__main__':
    params, dca = helper.setup_radar()
    talker(params, dca)
