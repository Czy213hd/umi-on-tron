// Copyright information
//
// © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.

#ifdef USE_ARX5_SDK

#include "robot_hw/Arx5SocketCanArm.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hw {

static bool startsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

Arx5SocketCanArm::JointMotorType Arx5SocketCanArm::motorTypeFromModelAndIndex(const std::string& model, int joint_index) {
  // X5: J1-3 are EC, J4-6 are DM_J4310.
  // L5: J1-3 are DM_J4340, J4-6 are DM_J4310.
  if (startsWith(model, "X5")) {
    return (joint_index < 3) ? JointMotorType::EC_A4310 : JointMotorType::DM_J4310;
  }
  if (startsWith(model, "L5")) {
    return (joint_index < 3) ? JointMotorType::DM_J4340 : JointMotorType::DM_J4310;
  }
  // Default to safe option: DM (most common).
  return JointMotorType::DM_J4310;
}

double Arx5SocketCanArm::torqueConstant(const Config& cfg, JointMotorType type) {
  switch (type) {
    case JointMotorType::EC_A4310:
      return cfg.torque_constant_ec_a4310;
    case JointMotorType::DM_J4310:
      return cfg.torque_constant_dm_j4310;
    case JointMotorType::DM_J4340:
      return cfg.torque_constant_dm_j4340;
    default:
      return cfg.torque_constant_dm_j4310;
  }
}

double Arx5SocketCanArm::clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

const OD_Motor_Msg* Arx5SocketCanArm::findMotorMsg(const std::array<OD_Motor_Msg, 10>& msgs, uint16_t motor_id) {
  for (const auto& m : msgs) {
    if (m.motor_id == motor_id) {
      return &m;
    }
  }
  // Some implementations store at index=(motor_id-1) without filling motor_id consistently.
  if (motor_id >= 1 && motor_id <= msgs.size()) {
    return &msgs[static_cast<size_t>(motor_id - 1)];
  }
  return nullptr;
}

Arx5SocketCanArm::Arx5SocketCanArm(Config cfg)
    : cfg_(std::move(cfg)), can_(cfg_.can_interface) {
  // Fill motor type mapping from model if caller keeps defaults.
  for (int i = 0; i < 6; ++i) {
    cfg_.joint_motor_type[static_cast<size_t>(i)] = motorTypeFromModelAndIndex(cfg_.model, i);
  }
  cfg_.gripper_motor_type = JointMotorType::DM_J4310;

  // Enable DM motors (EC motors typically don't need explicit enable).
  if (cfg_.enable_motors_on_init) {
    for (int i = 0; i < 6; ++i) {
      const auto type = cfg_.joint_motor_type[static_cast<size_t>(i)];
      const uint16_t id = cfg_.joint_motor_id[static_cast<size_t>(i)];
      if (type == JointMotorType::DM_J4310 || type == JointMotorType::DM_J4340) {
        (void)can_.enable_DM_motor(id);
      }
    }
    (void)can_.enable_DM_motor(cfg_.gripper_motor_id);
  }

  ok_ = true;
}

bool Arx5SocketCanArm::read(State& out_state) {
  if (!ok_) {
    out_state.valid = false;
    return false;
  }

  const auto msgs = can_.get_motor_msg();

  for (int i = 0; i < 6; ++i) {
    const uint16_t id = cfg_.joint_motor_id[static_cast<size_t>(i)];
    const auto* m = findMotorMsg(msgs, id);
    if (m == nullptr) {
      out_state.valid = false;
      return false;
    }
    out_state.q[static_cast<size_t>(i)] = static_cast<double>(m->angle_actual_rad);
    out_state.dq[static_cast<size_t>(i)] = static_cast<double>(m->speed_actual_rad);

    const double tq_const = torqueConstant(cfg_, cfg_.joint_motor_type[static_cast<size_t>(i)]);
    const double cur = static_cast<double>(m->current_actual_float);
    out_state.tau[static_cast<size_t>(i)] = cur * tq_const;
  }

  // Gripper (optional)
  {
    const auto* gm = findMotorMsg(msgs, cfg_.gripper_motor_id);
    if (gm != nullptr && cfg_.gripper_open_readout > 1e-6 && cfg_.gripper_width > 1e-6) {
      const double motor_pos = static_cast<double>(gm->angle_actual_rad);
      out_state.gripper_pos = motor_pos / cfg_.gripper_open_readout * cfg_.gripper_width;
    }
  }

  out_state.valid = true;
  return true;
}

bool Arx5SocketCanArm::write(const std::array<double, 6>& q_des,
                             const std::array<double, 6>& dq_des,
                             const std::array<double, 6>& kp,
                             const std::array<double, 6>& kd,
                             const std::array<double, 6>& tau_ff,
                             double gripper_pos_m) {
  if (!ok_) {
    return false;
  }

  bool ok_all = true;
  for (int i = 0; i < 6; ++i) {
    const uint16_t id = cfg_.joint_motor_id[static_cast<size_t>(i)];
    const auto type = cfg_.joint_motor_type[static_cast<size_t>(i)];
    const double tq_const = torqueConstant(cfg_, type);
    const double current_cmd = (std::abs(tq_const) > 1e-9) ? (tau_ff[static_cast<size_t>(i)] / tq_const) : 0.0;

    const float f_kp = static_cast<float>(kp[static_cast<size_t>(i)]);
    const float f_kd = static_cast<float>(kd[static_cast<size_t>(i)]);
    const float f_pos = static_cast<float>(q_des[static_cast<size_t>(i)]);
    const float f_vel = static_cast<float>(dq_des[static_cast<size_t>(i)]);
    const float f_cur = static_cast<float>(current_cmd);

    bool ok_one = true;
    if (type == JointMotorType::EC_A4310) {
      ok_one = can_.send_EC_motor_cmd(id, f_kp, f_kd, f_pos, f_vel, f_cur);
    } else {
      ok_one = can_.send_DM_motor_cmd(id, f_kp, f_kd, f_pos, f_vel, f_cur);
    }
    ok_all = ok_all && ok_one;
  }

  // Gripper command (position control in motor readout).
  if (cfg_.gripper_width > 1e-6 && cfg_.gripper_open_readout > 1e-6) {
    const double gp = clamp(gripper_pos_m, 0.0, cfg_.gripper_width);
    const double motor_pos = gp / cfg_.gripper_width * cfg_.gripper_open_readout;
    const bool ok_g = can_.send_DM_motor_cmd(cfg_.gripper_motor_id,
                                            static_cast<float>(cfg_.gripper_kp),
                                            static_cast<float>(cfg_.gripper_kd),
                                            static_cast<float>(motor_pos),
                                            0.0f,
                                            0.0f);
    ok_all = ok_all && ok_g;
  }

  return ok_all;
}

} // namespace hw

#endif // USE_ARX5_SDK


