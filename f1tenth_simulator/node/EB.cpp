#include <ros/ros.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Bool.h>
#include <iostream>
#include <cstdlib>

class AEB {
private:
    ros::NodeHandle n;
    ros::Subscriber odom_sub;
    ros::Subscriber scan_sub;

    ros::Publisher brake_drive_pub;
    ros::Publisher brake_bool_pub;
    

    float speed_online;
 


public:
    AEB() {
        // Initialize the node handle
        n = ros::NodeHandle();//"~"  will effect the topic_name in sub and pub!! but why?
        speed_online = 0;

        // // get topic names
        // std::string brake_drive_topic, odom_topic, brake_bool_topic,scan_topic;
        // n.getParam("brake_drive_topic", brake_drive_topic);
        // n.getParam("odom_topic", odom_topic);

        // n.getParam("brake_bool_topic",brake_bool_topic);
        // n.getParam("scan_topic",scan_topic);

 
        // Make a publisher for drive messages
        brake_drive_pub = n.advertise<ackermann_msgs::AckermannDriveStamped>("/brake", 10);
        brake_bool_pub = n.advertise<std_msgs::Bool>("/brake_bool",10);
        // Start a subscriber to listen to odom messages
        odom_sub = n.subscribe("/odom", 1, &AEB::odom_callback, this);
    	scan_sub = n.subscribe("/scan",1,&AEB::scan_callback,this);//  the scan_topic var is wrong?
        

    }


    void scan_callback(const sensor_msgs::LaserScan& msg){

       std_msgs::Bool bool_msg;
       bool_msg.data = 1;  // true

       // accroding to the scan msg, ang min and ang max are -pi and pi
       //range min and range max are  0 and 100
       // they can be got via   "rostopic echo /scan |grep "range_max"


        double TTC;
        int data_size = msg.ranges.size();
        double delta_angle = msg.angle_increment;
        for(int i = 0; i < data_size; i++){
            //get the range and proj vel

            double proj_vel = -speed_online*std::cos(i*delta_angle);
            double distance = msg.ranges[i];
            if(distance < 0 || distance >100) continue;

            //std::cout<<"the distance is"<<distance<<" the proj vel is "<<proj_vel<<std::endl;

            if(proj_vel>0){
                TTC = distance/proj_vel;
            } else TTC = 1000;

            if (TTC<1){  //  the param 3 is a magic number need to be tunned!
                brake_bool_pub.publish(bool_msg);
                ROS_INFO("Emergency Brake Engaged!");
                break;
            }





        }








 
	







    }





    double compute_Length(const geometry_msgs::Vector3 &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }







    void odom_callback( const nav_msgs::Odometry & msg) {
 
        //update the speed for TTC computation
        geometry_msgs::Vector3 vel = msg.twist.twist.linear;
        speed_online = AEB::compute_Length(vel);




        ackermann_msgs::AckermannDriveStamped drive_st_msg;
        ackermann_msgs::AckermannDrive drive_msg;

 

        drive_msg.steering_angle = 0;
        drive_msg.steering_angle_velocity = 0;
        drive_msg.speed = 0;
        drive_msg.acceleration = 0;
        drive_msg.jerk = 0;

 

        // set drive message in drive stamped message
        drive_st_msg.drive = drive_msg;

        // publish AckermannDriveStamped message to drive topic
        brake_drive_pub.publish(drive_st_msg);


    }

}; // end of class definition


int main(int argc, char ** argv) {
    ros::init(argc, argv, "AEB_node");
    AEB aeb;
    ros::spin();
    return 0;
}
