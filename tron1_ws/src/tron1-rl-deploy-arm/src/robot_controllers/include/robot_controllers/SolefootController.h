// Copyright information
//
// © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.

#ifndef _LIMX_SOLEFOOT_CONTROLLER_H_
#define _LIMX_SOLEFOOT_CONTROLLER_H_

#include <iostream>
#include <thread>
#include <fstream>
#include <csignal>
#include <array>
#include <atomic>
#include <random>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <limits>

#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32MultiArray.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include "robot_controllers/ControllerBase.h"
#include "limxsdk/pointfoot.h"
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

namespace robot_controller {

// Class for controlling a biped robot with point foot
class SolefootController : public ControllerBase {
  using tensor_element_t = float; // Type alias for tensor elements
  using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>; // Type alias for matrices

public:
  SolefootController() = default; // Default constructor

  ~SolefootController() override = default; // Destructor

  // Enumeration for controller modes
  enum class Mode : uint8_t {
    SOLE_STAND, 
    SOLE_WALK, 
    SOLE_STOP,
  };

  // Initialize the controller
  bool init(hardware_interface::RobotHW *robot_hw, ros::NodeHandle &nh) override;

  // Perform actions when the controller starts
  void starting(const ros::Time &time) override;

  // Update the controller
  void update(const ros::Time &time, const ros::Duration &period) override;

protected:
  // Load the model for the controller
  bool loadModel() override;

  // Load RL configuration settings
  bool loadRLCfg() override;

  // Compute actions for the controller
  void computeActions() override;

  // Compute observations for the controller
  void computeObservation() override;

  // Compute encoder for the controller
  void computeEncoder() override;

  void handleSoleStandMode() override;

  void handleSoleStopMode();

  void handleRLSoleWalkMode() override;

  void cmdVelCallback(const geometry_msgs::TwistConstPtr &msg) override;
  void EEPoseCmdRCCallback(const std_msgs::Float32MultiArrayConstPtr &msg);
  void EEPoseTargetWorldCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void groundTruthCallback(const nav_msgs::OdometryConstPtr& msg);

  void handleExtraCommands();

  // compute gait
  vector_t handleGaitCommand();

  // compute gait phase
  vector_t handleGaitPhase(vector_t &gait);

  Mode mode_{Mode::SOLE_STAND}; // Controller mode
  Mode last_mode_{Mode::SOLE_STAND};

  int work_mode_flag_{10};

private:
  double sliding_window(std::vector<double>& data, int window_size);

  void clearData();

  struct RCEECmd{
      Eigen::Vector3d ee_position;
      Eigen::Vector3d ee_rpy;
      bool gripper_cmd{false};
      void zero() {
          ee_position << 0, 0, 0;
          ee_rpy << 0, 0, 0;
      }
  };

  vector3_t ee_init_pos_;
  vector3_t ee_init_rpy_;

  ros::Publisher gripper_cmd_pub_;
  std_msgs::Bool gripper_cmd_msg_;

  ros::Publisher obs_debug_pub_;
  std_msgs::Float32MultiArray obs_debug_msg_;

  geometry_msgs::Pose ee_pos_cmd_debug_msg_;

  ros::Subscriber ee_pos_cmd_rc_delta_;
  std_msgs::Float32MultiArray ee_pos_cmd_rc_delta_msg_;
  RCEECmd rc_ee_cmd;
  // File path for policy model
  std::string policyFilePath_;

  std::shared_ptr<Ort::Env> onnxEnvPtr_; // Shared pointer to ONNX environment

  // ONNX session pointers
  std::unique_ptr<Ort::Session> encoderSessionPtr_;
  std::vector<const char *> encoderInputNames_;
  std::vector<const char *> encoderOutputNames_;

  // Names and shapes of inputs and outputs for ONNX sessions
  std::vector<std::vector<int64_t>> policyInputShapes_;
  std::vector<std::vector<int64_t>> policyOutputShapes_;

  std::unique_ptr<Ort::Session> policySessionPtr_;
  std::vector<const char *> policyInputNames_;
  std::vector<const char *> policyOutputNames_;
  std::vector<std::vector<int64_t>> encoderInputShapes_;
  std::vector<std::vector<int64_t>> encoderOutputShapes_;

  std::unique_ptr<Ort::Session> gruSessionPtr_;
  std::vector<const char *> gruInputNames_;
  std::vector<const char *> gruOutputNames_;
  std::vector<std::vector<int64_t>> gruInputShapes_;
  std::vector<std::vector<int64_t>> gruOutputShapes_;

  std::unique_ptr<Ort::Session> gaitGeneratorSessionPtr_;
  std::vector<const char *> gaitGeneratorInputNames_;
  std::vector<const char *> gaitGeneratorOutputNames_;
  std::vector<std::vector<int64_t>> gaitGeneratorInputShapes_;
  std::vector<std::vector<int64_t>> gaitGeneratorOutputShapes_;

  std::vector<tensor_element_t> proprioHistoryVector_;
  Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer_;
  Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBufferForEstimation_;

  bool isfirstRecObs_{true};
  int encoderInputSize_{0}, encoderOutputSize_{0};
  int oneHotEncoderInputSize_{0}, oneHotEncoderOutputSize_{0};
  int contactNetObsSize_{0};
  int contactNetOutputSize_{0};
  int gruLatentSize_{0};
  int nextObsLatentSize_{0};

  std::vector<tensor_element_t> cnOutput_;
  std::vector<tensor_element_t> gruHiddenState_;
  std::array<tensor_element_t, 3> baseLinVelObs_{0.0f, 0.0f, 0.0f};
  std::mt19937 rng_{std::random_device{}()};
  std::normal_distribution<tensor_element_t> standardNormal_{0.0f, 1.0f};
  bool sampleNextObsLatent_{true};

  // --- Simulation-only: use gazebo ground truth velocity for policy obs (to match IsaacLab) ---
  ros::Subscriber ground_truth_sub_;
  bool useGroundTruthBaseLinVel_{false};
  bool groundTruthTwistInWorldFrame_{false};
  std::atomic<tensor_element_t> gtBaseLinVelX_{0.0f};
  std::atomic<tensor_element_t> gtBaseLinVelY_{0.0f};
  std::atomic<tensor_element_t> gtBaseLinVelZ_{0.0f};
  std::atomic<bool> gtBaseLinVelValid_{false};

  // --- World-frame EE target support (UMI-style): target in world/odom, error expressed in EE frame ---
  bool useWorldFrameEeTarget_{false};
  bool anchorEeTargetToFirstBasePose_{true};
  ros::Subscriber ee_target_world_sub_;
  std::atomic<tensor_element_t> eeTargetWorldPosX_{0.0f};
  std::atomic<tensor_element_t> eeTargetWorldPosY_{0.0f};
  std::atomic<tensor_element_t> eeTargetWorldPosZ_{0.0f};
  std::atomic<tensor_element_t> eeTargetWorldQuatW_{1.0f};
  std::atomic<tensor_element_t> eeTargetWorldQuatX_{0.0f};
  std::atomic<tensor_element_t> eeTargetWorldQuatY_{0.0f};
  std::atomic<tensor_element_t> eeTargetWorldQuatZ_{0.0f};
  std::atomic<bool> eeTargetWorldValid_{false};

  // Manual world target (used when no world target message is received).
  bool manualEeTargetWorldEnabled_{false};
  vector3_t manualEeTargetWorldPos_{0.0, 0.0, 0.0};
  vector3_t manualEeTargetWorldRpy_{0.0, 0.0, 0.0};  // roll, pitch, yaw [rad]

  // Ground-truth base pose (world frame), required for world-frame EE tracking error.
  std::atomic<tensor_element_t> gtBasePosX_{0.0f};
  std::atomic<tensor_element_t> gtBasePosY_{0.0f};
  std::atomic<tensor_element_t> gtBasePosZ_{0.0f};
  std::atomic<tensor_element_t> gtBaseQuatW_{1.0f};
  std::atomic<tensor_element_t> gtBaseQuatX_{0.0f};
  std::atomic<tensor_element_t> gtBaseQuatY_{0.0f};
  std::atomic<tensor_element_t> gtBaseQuatZ_{0.0f};
  std::atomic<bool> gtBasePoseValid_{false};

  // Anchor pose (world <- base) used to convert base-frame targets into world targets without following base drift.
  bool eeTargetAnchorValid_{false};
  vector3_t eeTargetAnchorPosW_{0.0, 0.0, 0.0};
  Eigen::Matrix<scalar_t, 3, 3> eeTargetAnchorRotWb_{Eigen::Matrix<scalar_t, 3, 3>::Identity()};

  // --- Optional: lock J6 command to a fixed angle (useful if policy didn't train/use J6) ---
  bool lockArmJ6_{false};
  tensor_element_t lockArmJ6Angle_{0.0f};

  bool eeFkReady_{false};
  KDL::Chain eeChain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> eeFkSolver_;
  std::vector<std::string> eeChainJointNames_;
  std::unordered_map<std::string, size_t> jointNameToIndex_;

  vector3_t baseLinVel_;
  vector3_t basePosition_;
  vector_t lastActions_;

  double baseHeightCmd_ = 0.7; 

  int actionsSize_{0};
  int observationSize_{0};
  int obsHistoryLength_{0};
  int gaitGeneratorOutputSize_{0};
  float imuOrientationOffset_[3]{0.0f, 0.0f, 0.0f};
  std::vector<tensor_element_t> actions_;
  std::vector<tensor_element_t> filtered_actions_;
  std::vector<tensor_element_t> observations_;
  std::vector<tensor_element_t> encoderOut_;
  std::vector<tensor_element_t> oneHotEncoderOut_;
  std::vector<tensor_element_t> gaitGeneratorOut_;

  double gaitIndex_{0.0};

  vector3_t extraCommands_;
  vector5_t scaledCommandsSole_;

  int locomotionFlag_ = 0;
  int standStillFlag_ = 0;
  int sitDownFlag_ = 0;

  std::vector<double> commandX_;
  std::vector<double> commandY_;
  std::vector<double> commandYaw_;
  const int windowLen_ = 100;

  // stand pos
  vector_t standCenterPos_;
  vector_t standJointPos_;
  vector_t stopSitPos_;
  scalar_t standCenterPercent_{0.0};
  scalar_t standCenterDuration_{1.0};
  scalar_t stopCenterPercent_{0.0};
  scalar_t stopCenterDuration_{1.0};
  scalar_t initStandPercent_{0.0};
  scalar_t initStandDuration_{1.0};
  vector3_t lastEePos_, lastEeRpy_;

  bool armHoldStill_{false};
  bool stopJointAnglesUpdated_{false};

  // SE(3) distance reference for policy obs: initialized to 2*pos_err+rot_err, decays to 0
  tensor_element_t se3DistanceRef_{5.0f};
  bool needDamping_{false};
  bool isNoCommand_{false};

};

} // namespace robot_controller
#endif //_LIMX_SOLEFOOT_CONTROLLER_H_