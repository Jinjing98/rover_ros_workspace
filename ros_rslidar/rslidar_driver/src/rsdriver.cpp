/*
 *  Copyright (C) 2007 Austin Robot Technology, Patrick Beeson
 *  Copyright (C) 2009-2012 Austin Robot Technology, Jack O'Quin
 *	Copyright (C) 2017 Robosense, Tony Zhang
 *
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** \file
 *
 *  ROS driver implementation for the RILIDAR 3D LIDARs
 */
#include "rsdriver.h"
#include <rslidar_msgs/rslidarScan.h>

namespace rs_driver {

rslidarDriver::rslidarDriver(ros::NodeHandle node, ros::NodeHandle private_nh) {
  // use private node handle to get parameters
  private_nh.param("frame_id", config_.frame_id, std::string("rslidar"));

  std::string tf_prefix = tf::getPrefixParam(private_nh);
  ROS_DEBUG_STREAM("tf_prefix: " << tf_prefix);
  config_.frame_id = tf::resolve(tf_prefix, config_.frame_id);
  config_.time_offset = 0.0;
  
  // get model name, validate string, determine packet rate
  private_nh.param("model", config_.model, std::string("MEMS"));
  private_nh.param("start_from_edge", config_.start_from_edge, true);
  double packet_rate;  // packet frequency (Hz)
  std::string model_full_name;

  // product model
  if (config_.model == "MEMS") 
  {
    packet_rate = 9450;
    model_full_name = "RS-LiDAR-MEMS";
  } 
  else 
  {
    ROS_ERROR_STREAM("unknown LIDAR model: " << config_.model);
    packet_rate = 4000;
  }
  std::string deviceName(std::string("Robosense ") + model_full_name);

  double frequency = 15; // expected Hz rate

  // default number of packets for each scan is a single revolution
  // (fractions rounded up)

  int npackets = (int)ceil(packet_rate / frequency);
  private_nh.param("npackets", config_.npackets, npackets);
  ROS_INFO_STREAM("publishing " << config_.npackets << " packets per scan");

  std::string dump_file;
  private_nh.param("pcap", dump_file, std::string(""));

  int msop_udp_port;
  int difop_udp_port;
  private_nh.param("msop_port", msop_udp_port, (int)MSOP_DATA_PORT_NUMBER);
  private_nh.param("difop_port", difop_udp_port, (int)DIFOP_DATA_PORT_NUMBER);

  // Initialize dynamic reconfigure
  srv_ = boost::make_shared<
      dynamic_reconfigure::Server<rslidar_driver::rslidarNodeConfig>>(
      private_nh);
  dynamic_reconfigure::Server<rslidar_driver::rslidarNodeConfig>::CallbackType
      f;
  f = boost::bind(&rslidarDriver::callback, this, _1, _2);
  srv_->setCallback(f); // Set callback function und call initially

  // initialize diagnostics
  diagnostics_.setHardwareID(deviceName);
  const double diag_freq = packet_rate / config_.npackets;
  diag_max_freq_ = diag_freq;
  diag_min_freq_ = diag_freq;
  // ROS_INFO("expected frequency: %.3f (Hz)", diag_freq);

  using namespace diagnostic_updater;
  diag_topic_.reset(new TopicDiagnostic(
      "rslidar_packets", diagnostics_,
      FrequencyStatusParam(&diag_min_freq_, &diag_max_freq_, 0.1, 10),
      TimeStampStatusParam()));

  // open rslidar input device or file
  if (dump_file != "") // have PCAP file?
  {
     printf("dump_file != null\n");
    // read data from packet capture file
    msop_input_.reset(new rs_driver::InputPCAP(private_nh, msop_udp_port, packet_rate, dump_file));
    difop_input_.reset(new rs_driver::InputPCAP(private_nh, difop_udp_port, packet_rate, dump_file));
  } 
  else 
  {
    printf("dump_file == null\n");
    // read data from live socket
    msop_input_.reset(new rs_driver::InputSocket(private_nh, msop_udp_port));
    difop_input_.reset(new rs_driver::InputSocket(private_nh, difop_udp_port));
  }

  // raw packet output topic
  msop_output_ = node.advertise<rslidar_msgs::rslidarScan>("rslidar_packets", 10);
  difop_output_ = node.advertise<rslidar_msgs::rslidarPacket>("rslidar_difop_packets", 10);

  // receive difop packet thread
  difop_thread_ = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&rslidarDriver::difopPoll, this)));
}

/** poll the device
 *
 *  @returns true unless end of file reached
 */
bool rslidarDriver::poll(void) {
  // Allocate a new shared pointer for zero-copy sharing with other nodelets.
  rslidar_msgs::rslidarScanPtr scan(new rslidar_msgs::rslidarScan);
  if (config_.start_from_edge) {
    scan->packets.reserve(config_.npackets);
    rslidar_msgs::rslidarPacket tmp_packet;
    while (true) {
      bool isNewFrame = false;
      while (true) {
        int rc = msop_input_->getPacket(&tmp_packet, config_.time_offset);
        if (rc == 0)
          break; // got a full packet?
        if (rc < 0)
          return false; // end of file reached?
      }
      scan->packets.push_back(tmp_packet);
      // is first pkt of new frame? one packet include 25 blocks, one block offset: 52bytes, first block offset: 20
      for (int i = 0; i < 25; i++)
      {
        int index_temp = 256 * tmp_packet.data[20 + 52 * i] + tmp_packet.data[21 + 52 * i];
        static int index_temp_zero = index_temp;

        if ( index_temp < index_temp_zero)
        {
          index_temp_zero = index_temp;
          isNewFrame = true;
          break;
        }
        index_temp_zero = index_temp;
      }

      if (isNewFrame)
        break;
    }
  } else {
    scan->packets.resize(config_.npackets);
    // Since the rslidar delivers data at a very high rate, keep
    // reading and publishing scans as fast as possible.
    for (int i = 0; i < config_.npackets; i++) {
      while (true) {
        // keep reading until full packet received
        int rc = msop_input_->getPacket(&scan->packets[i], config_.time_offset);
        if (rc == 0)
          break; // got a full packet?
        if (rc < 0)
          return false; // end of file reached?
      }
    }
  }

  // publish message using time of last packet read
  // ROS_DEBUG("Publishing a full rslidar scan.");
  scan->header.stamp = scan->packets.back().stamp;
  scan->header.frame_id = config_.frame_id;
  msop_output_.publish(scan);

  // notify diagnostics that a message has been published, updating its status
  diag_topic_->tick(scan->header.stamp);
  diagnostics_.update();

  return true;
}

void rslidarDriver::difopPoll(void)
{
  // reading and publishing scans as fast as possible.
  rslidar_msgs::rslidarPacketPtr difop_packet_ptr(new rslidar_msgs::rslidarPacket);
  while (ros::ok())
  {
    // keep reading
    rslidar_msgs::rslidarPacket difop_packet_msg;
    int rc = difop_input_->getPacket(&difop_packet_msg, config_.time_offset);
    if (rc == 0)
    {
      // std::cout << "Publishing a difop data." << std::endl;
      ROS_DEBUG("Publishing a difop data.");
      *difop_packet_ptr = difop_packet_msg;
      difop_output_.publish(difop_packet_ptr);
    }
    if (rc < 0)
      return;  // end of file reached?
    ros::spinOnce();
  }
}

void rslidarDriver::callback(rslidar_driver::rslidarNodeConfig &config,
                             uint32_t level) {
  ROS_INFO("Reconfigure Request");
  config_.time_offset = config.time_offset;
}

}
