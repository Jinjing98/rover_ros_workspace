/*
 *  Copyright (C) 2009, 2010 Austin Robot Technology, Jack O'Quin
 *  Copyright (C) 2011 Jesse Vera
 *  Copyright (C) 2012 Austin Robot Technology, Jack O'Quin
 *  Copyright (C) 2017 Robosense, Tony Zhang
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** @file

    This class converts raw RSLIDAR 3D LIDAR packets to PointCloud2.

*/
#include "convert.h"
#include <pcl_conversions/pcl_conversions.h>

namespace rslidar_pointcloud {
/** @brief Constructor. */
Convert::Convert(ros::NodeHandle node, ros::NodeHandle private_nh)
    : data_(new rslidar_rawdata::RawData()), out_points(new pcl::PointCloud<pcl::PointXYZI>)
{
  // load lidar parameters
  data_->loadConfigFile(node, private_nh); // load lidar parameters

  std::string model;
  private_nh.param("model", model, std::string("MEMS"));

  // advertise output point cloud (before subscribing to input data)
  point_output_ = node.advertise<sensor_msgs::PointCloud2>("rslidar_points", 10);

  srv_ = boost::make_shared<
      dynamic_reconfigure::Server<rslidar_pointcloud::CloudNodeConfig>>(
      private_nh);
  dynamic_reconfigure::Server<rslidar_pointcloud::CloudNodeConfig>::CallbackType
      f;
  f = boost::bind(&Convert::callback, this, _1, _2);
  srv_->setCallback(f);

  // subscribe to rslidarScan packets
  rslidar_scan_ = node.subscribe("rslidar_packets", 10, &Convert::processScan,
                     (Convert *)this, ros::TransportHints().tcpNoDelay(true));
}

void Convert::callback(rslidar_pointcloud::CloudNodeConfig &config,
                       uint32_t level) {
  ROS_INFO("Reconfigure Request");
  // config_.time_offset = config.time_offset;
}

/** @brief Callback for raw scan messages. */
void Convert::processScan(const rslidar_msgs::rslidarScan::ConstPtr &scanMsg) {

  out_points->header.stamp = pcl_conversions::toPCL(scanMsg->header).stamp;
  out_points->header.frame_id = scanMsg->header.frame_id;

  // out_points init
  out_points->height = 5;
  out_points->width = 15750; // 630 packets * 25 blocks
  out_points->is_dense = false;
  out_points->resize(out_points->height * out_points->width);

  // unpack packet repeatly
  for (size_t i = 0; i < scanMsg->packets.size(); ++i) {
    data_->unpack_MEMS(scanMsg->packets[i], out_points);
  }

  // publish point cloud data 
  sensor_msgs::PointCloud2 outMsg;
  pcl::toROSMsg(*out_points, outMsg);
  point_output_.publish(outMsg);

  // reprocesss the last packet to get start points of next frame
  size_t last_pkt_index = scanMsg->packets.size() - 1;
  data_->unpack_MEMS(scanMsg->packets[last_pkt_index], out_points);
}
} // namespace rslidar_pointcloud
