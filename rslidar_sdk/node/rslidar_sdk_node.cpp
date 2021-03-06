/*********************************************************************************************************************
Copyright (c) 2020 RoboSense
All rights reserved

By downloading, copying, installing or using the software you agree to this license. If you do not agree to this
license, do not download, install, copy or use the software.

License Agreement
For RoboSense LiDAR SDK Library
(3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the names of the RoboSense, nor Suteng Innovation Technology, nor the names of other contributors may be used
to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*********************************************************************************************************************/

#include <iostream>
#include <assert.h>
// boost.format ???????????????
#include <boost/format.hpp>

#include <signal.h>
#include "manager/adapter_manager.h"
using namespace robosense::lidar;
std::mutex g_mtx;
std::condition_variable g_cv;
static void sigHandler(int sig)
{
  RS_MSG << "RoboSense-LiDAR-Driver is stopping....." << RS_REND;
#ifdef ROS_FOUND
  ros::shutdown();
#endif
  g_cv.notify_all();
}

int main(int argc, char** argv)
{
  signal(SIGINT, sigHandler);  ///< bind ctrl+c signal with the sigHandler function
  RS_TITLE << "********************************************************" << RS_REND;
  RS_TITLE << "**********                                    **********" << RS_REND;
  RS_TITLE << "**********    RSLidar_SDK Version: v" << RSLIDAR_VERSION_MAJOR << "." << RSLIDAR_VERSION_MINOR << "."
           << RSLIDAR_VERSION_PATCH << "     **********" << RS_REND;
  RS_TITLE << "**********                                    **********" << RS_REND;
  RS_TITLE << "********************************************************" << RS_REND;

  std::shared_ptr<AdapterManager> demo_ptr = std::make_shared<AdapterManager>();
  YAML::Node config;
  try
  {
    
    if (argc==2){
	string pcd_path = argv[1];
	config = YAML::LoadFile(pcd_path);}//  for data collection, we use the old version 
    else{ // for general use and calibration, we use the Veloydne and /raw points version.
    config = YAML::LoadFile((std::string)PROJECT_PATH +"/config/config_ori_data_collection.yaml");//  be default will use this one!
}//config_Velodyne_rawpoints.yaml

///home/tud-jxavier/catkin_ws/src/rslidar_sdk/config/config_ori_data_collection.yaml
    
    //config = YAML::LoadFile((std::string)PROJECT_PATH +"/config/"argv[1]);//"/config/config.yaml");
  }
  catch (...)
  {
    RS_ERROR << "Config file format wrong! Please check the format(e.g. indentation) " << RS_REND;
    return -1;
  }

#ifdef ROS_FOUND  ///< if ROS is found, call the ros::init function
  ros::init(argc, argv, "rslidar_sdk_node", ros::init_options::NoSigintHandler);
#endif

#ifdef ROS2_FOUND  ///< if ROS2 is found, call the rclcpp::init function
  rclcpp::init(argc, argv);
#endif

  demo_ptr->init(config);
  demo_ptr->start();
  RS_MSG << "RoboSense-LiDAR-Driver is running....." << RS_REND;

#ifdef ROS_FOUND
  ros::spin();
#else
  std::unique_lock<std::mutex> lck(g_mtx);
  g_cv.wait(lck);
#endif
  return 0;
}
