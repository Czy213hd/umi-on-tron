// Copyright information
//
// © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.

#include "robot_controllers/ControllerBase.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace robot_controller {

bool ControllerBase::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& nh) {
  nh_ = nh;

  const char* robotTypeEnv = std::getenv("ROBOT_TYPE");
  if (robotTypeEnv != nullptr && std::strlen(robotTypeEnv) > 0) {
    robotType_ = std::string(robotTypeEnv);
  } else {
    nh_.getParam("/robot_type", robotType_);
  }

  bool isSim = false;
  nh_.param<bool>("/is_sim", isSim, false);
  isSim_ = isSim ? 1 : 0;

  bool gotJointNames = false;
  if (robotType_.find("WF") != std::string::npos) {
    gotJointNames = nh_.getParam("/LeggedRobotCfg/joint_names", jointNames_);
  } else {
    gotJointNames = nh_.getParam("/PointfootCfg/init_state/joint_names", jointNames_);
  }

  if (!gotJointNames || jointNames_.empty()) {
    ROS_ERROR("Failed to retrieve joint names from the parameter server.");
    return false;
  }

  auto* hybridJointInterface = robot_hw->get<robot_common::HybridJointInterface>();
  auto* imuInterface = robot_hw->get<hardware_interface::ImuSensorInterface>();
  auto* contactInterface = robot_hw->get<robot_common::ContactSensorInterface>();

  if (hybridJointInterface == nullptr) {
    ROS_ERROR("HybridJointInterface not found from RobotHW.");
    return false;
  }
  if (imuInterface == nullptr) {
    ROS_ERROR("ImuSensorInterface not found from RobotHW.");
    return false;
  }
  (void)contactInterface;

  hybridJointHandles_.clear();
  hybridJointHandles_.reserve(jointNames_.size());
  try {
    for (const auto& jointName : jointNames_) {
      hybridJointHandles_.push_back(hybridJointInterface->getHandle(jointName));
    }
  } catch (const std::exception& e) {
    ROS_ERROR("Failed to get joint handles: %s", e.what());
    return false;
  }

  try {
    imuSensorHandles_ = imuInterface->getHandle("limx_imu");
  } catch (const std::exception& e) {
    ROS_ERROR("Failed to get IMU handle: %s", e.what());
    return false;
  }

  defaultJointAngles_.resize(hybridJointHandles_.size());
  initJointAngles_.resize(hybridJointHandles_.size());
  defaultJointAngles_.setZero();
  initJointAngles_.setZero();

  loopCount_ = 0;
  loopCountKeep_ = 0;
  commands_.setZero();
  scaledCommands_.setZero();

  cmdVelSub_ = nh_.subscribe("/cmd_vel", 10, &ControllerBase::cmdVelCallback, this);
  return true;
}

}  // namespace robot_controller
