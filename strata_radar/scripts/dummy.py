#! /usr/bin/python
 
import sys
# license removed for brevity
import rospy
from std_msgs.msg import String

 
 
 

def main(pub_rate):
    pub = rospy.Publisher('/dummy4radar',String,queue_size = 10)
 
    rospy.init_node('dummy_node', anonymous=True)
    rate = rospy.Rate(pub_rate)
 
    while not rospy.is_shutdown():
        dummy_str = "current time %s" % rospy.get_time()
        rospy.loginfo(dummy_str)
        pub.publish(dummy_str)
        rate.sleep()

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("usage: dummy.py pub_rate(int)  recommend: 30 ")
    else:
	pub_rate = int(sys.argv[1])

 
    try:
        main(pub_rate)
    except rospy.ROSInterruptException:
        pass 

 
 


        





















 
