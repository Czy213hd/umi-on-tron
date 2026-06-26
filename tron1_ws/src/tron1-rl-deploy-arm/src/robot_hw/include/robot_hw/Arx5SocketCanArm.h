// Copyright information
//
// © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.
//
// ARX5 SocketCAN minimal arm driver.
// - Designed for TRON1 ROS1 (Noetic) docker deployment
// - Depends ONLY on legacy arx5-sdk "libhardware.so" (SocketCAN) and headers under include/hardware/
// - Does NOT depend on soem, ROS2 humble runtime, or solver (KDL/urdf) libraries.
#ifndef ROBOT_HW_ARX5_SOCKETCAN_ARM_H_
#define ROBOT_HW_ARX5_SOCKETCAN_ARM_H_

#ifdef USE_ARX5_SDK

#include <array>
#include <cstdint>
#include <string>

// Legacy arx5-sdk (SocketCAN) header (must come from ARX5_SDK_DIR/include).
#include "hardware/arx_can.h"

namespace hw {

class Arx5SocketCanArm {
public:
  enum class JointMotorType : uint8_t {
    EC_A4310 = 0,
    DM_J4310 = 1,
    DM_J4340 = 2,
  };

  struct Config {
    std::string model{"L5"};         // "X5" or "L5"
    std::string can_interface{"can0"}; // e.g. "can0"

    // 6-DoF joint motors (CAN IDs).
    std::array<uint16_t, 6> joint_motor_id{{1, 2, 4, 5, 6, 7}};
    std::array<JointMotorType, 6> joint_motor_type{
        {JointMotorType::DM_J4340, JointMotorType::DM_J4340, JointMotorType::DM_J4340,
         JointMotorType::DM_J4310, JointMotorType::DM_J4310, JointMotorType::DM_J4310}};

    // Gripper motor (CAN ID) (DM motor).
    uint16_t gripper_motor_id{8};
    JointMotorType gripper_motor_type{JointMotorType::DM_J4310};

    // Torque constants (Nm/A). The legacy SDK expects CURRENT command, not torque.
    double torque_constant_ec_a4310{1.4};
    double torque_constant_dm_j4310{0.424};
    double torque_constant_dm_j4340{1.0};

    // Gripper mapping (meters -> motor rad readout).
    double gripper_width{0.088};          // fully opened width (m)
    double gripper_open_readout{5.03};    // motor angle (rad) at fully open
    double gripper_kp{30.0};
    double gripper_kd{0.2};

    bool enable_motors_on_init{true};
  };

  struct State {
    std::array<double, 6> q{};
    std::array<double, 6> dq{};
    std::array<double, 6> tau{};
    double gripper_pos{0.0}; // meters
    bool valid{false};
  };

  explicit Arx5SocketCanArm(Config cfg);
  ~Arx5SocketCanArm() = default;

  // Non-copyable.
  Arx5SocketCanArm(const Arx5SocketCanArm&) = delete;
  Arx5SocketCanArm& operator=(const Arx5SocketCanArm&) = delete;

  bool ok() const { return ok_; }

  // Read latest motor state snapshot (from SDK's internal receiver).
  bool read(State& out_state);

  // Send one set of MIT commands to motors.
  // NOTE: tau_ff is in Nm. It will be converted to CURRENT command internally.
  bool write(const std::array<double, 6>& q_des,
             const std::array<double, 6>& dq_des,
             const std::array<double, 6>& kp,
             const std::array<double, 6>& kd,
             const std::array<double, 6>& tau_ff,
             double gripper_pos_m);

  const Config& config() const { return cfg_; }

private:
  static JointMotorType motorTypeFromModelAndIndex(const std::string& model, int joint_index);
  static double torqueConstant(const Config& cfg, JointMotorType type);
  static double clamp(double x, double lo, double hi);

  // Find motor message by motor_id (robust to any internal indexing scheme).
  static const OD_Motor_Msg* findMotorMsg(const std::array<OD_Motor_Msg, 10>& msgs, uint16_t motor_id);

  Config cfg_;
  ArxCan can_;
  bool ok_{false};
};

} // namespace hw

#endif // USE_ARX5_SDK

#endif // ROBOT_HW_ARX5_SOCKETCAN_ARM_H_


