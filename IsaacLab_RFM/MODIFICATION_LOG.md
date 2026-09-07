# IsaacLab_RFM 修改记录

这个文件用来记录项目里的重要修改。后面每次改训练配置、URDF、reward、sim2sim、PPO 逻辑，都继续追加到这里。

## 2026-07-14：SF_TRON1A + L5_umi 机械臂/夹爪与训练配置更新

### 1. 机器人 URDF / 模型

- 修改文件：
  - `source/ext_loco/ext_loco/assets/SF_TRON1A_ARXR5ARM/assembly.urdf`
  - `source/ext_loco/ext_loco/assets/SF_TRON1A_ARXR5ARM/models/L5_umi.urdf`
  - `source/ext_loco/ext_loco/assets/SF_TRON1A_ARXR5ARM/models/meshes/...`

- 主要变化：
  - 将 `assembly.urdf` 中原来的机械臂和旧夹爪替换为当前 `models/L5_umi.urdf` 里的 L5 机械臂 + UMI 夹爪。
  - 删除 `L5_umi.urdf` 自带的 `world -> base_link` 根节点连接，避免整机出现多个 root。
  - 将新机械臂接到原来的安装位置：

    ```xml
    <joint name="tron1computer-assembly" type="fixed">
      <parent link="tron1computer" />
      <child link="base_link" />
      <origin xyz="0 0 0.096" rpy="0 0 0" />
    </joint>
    ```

  - 将 `joint1` ~ `joint6` 改名为 `J1` ~ `J6`，保持训练配置、action、reward、sim2sim 中已有的关节命名不变。
  - 将 mesh 路径修正为从 `assembly.urdf` 可以加载的相对路径：

    ```text
    models/meshes/...
    ```

- 验证：
  - XML 解析通过。
  - mesh 文件路径检查通过。
  - URDF 树只有一个 root：`base_Link`。
  - `J1` ~ `J6` 均存在。

### 2. EEF 从 `link6` 改到夹爪 base frame

- 修改文件：
  - `source/ext_loco/ext_loco/tasks/loco_manipulation/EE_pose/config/sf_tron1_arm/sf_tron1_arm_env_cfg.py`
  - `source/ext_loco/ext_loco/tasks/loco_manipulation/EE_pose/mdp/events.py`
  - `source/ext_loco/ext_loco/assets/limx.py`

- 主要变化：
  - 将训练 command 的 EE body 从：

    ```python
    body_name="link6"
    ```

    改为：

    ```python
    body_name="eef_link"
    ```

  - 将 policy/contactNet/next_obs/critic 中的 EE pose 和 EE velocity 也改为读取 `eef_link`。
  - 将 `prepare_quantity_for_tron1_piper()` 中的 `_ee_link_idx` 从 `link6` 改为 `eef_link`。
  - 在 `limx.py` 中打开：

    ```python
    merge_fixed_joints=False
    ```

    目的是保留 fixed joint 下面的 `eef_link`，否则 URDF importer 可能把它合并掉，导致训练找不到 `eef_link`。

- 注意：
  - 当前 UMI 夹爪本体 mesh 仍然挂在 `link6` 里面。
  - `eef_link` 是一个 fixed frame，没有自己的 mesh/collision。
  - `ee_contact` termination 仍然保留检测 `link6`，因为夹爪 collision mesh 在 `link6` 上。

### 3. Pickle trajectory offset 清零

- 修改位置：
  - `CommandsCfgPlay.EE_pose`

- 变化：

  原来使用 `link6 -> UMI tip` offset：

  ```python
  tip_offset_pos=(0.08657, -0.0249, -0.00024366)
  tip_offset_rpy=(-math.pi * 0.5, 0.0, -math.pi * 0.5)
  ```

  现在因为 EE body 改成 `eef_link`，不再从 `link6` 额外偏移：

  ```python
  tip_offset_pos=(0.0, 0.0, 0.0)
  tip_offset_rpy=(0.0, 0.0, 0.0)
  ```

### 4. 新增手臂 reset 随机化

- 修改文件：
  - `source/ext_loco/ext_loco/tasks/loco_manipulation/EE_pose/config/sf_tron1_arm/sf_tron1_arm_env_cfg.py`

- 新增 event：

  ```python
  reset_arm_joints = EventTerm(
      func=mdp.reset_selected_joints_by_offset,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg(
              "robot",
              joint_names=["J1", "J2", "J3", "J4", "J5", "J6"],
          ),
          "position_range": (-0.1, 0.1),
          "velocity_range": (-0.2, 0.2),
      },
  )
  ```

- 作用：
  - 每次训练 reset 时，机械臂 6 个关节在默认角度附近随机扰动。
  - 位置扰动：`±0.1 rad`
  - 速度扰动：`±0.2 rad/s`

- PLAY/eval 中关闭该随机化：

  ```python
  self.events.reset_arm_joints.params["position_range"] = (0.0, 0.0)
  self.events.reset_arm_joints.params["velocity_range"] = (0.0, 0.0)
  ```

- 风险判断：
  - 当前没有专门针对手臂关节角/速度的 termination。
  - 该随机化不会直接触发手臂 termination。
  - 可能间接触发的只有：
    - `ee_contact`：如果随机后 `link6`/夹爪碰撞过大。
    - `bad_orientation` / `bad_height`：如果手臂姿态导致整机摔倒。
  - `±0.1 rad` 属于比较温和的随机范围。

### 5. 鲁棒性相关配置说明

- 当前 push 扰动位置：

  ```python
  push_robot = EventTerm(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(10.0, 15.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
  )
  ```

- 说明：
  - 这里用的是 velocity push，不是 force push。
  - 它直接给 root 设置随机速度，常用于训练抗扰动。
  - 如果要增强鲁棒性，可以考虑：

    ```python
    interval_range_s=(8.0, 12.0)
    params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "z": (-0.05, 0.05)}}
    ```

  - `z` 方向可以加，但建议很小，否则容易造成弹飞、落地冲击和训练不稳定。

### 6. Reward / penalty 相关说明

- `action_rate_l2`
  - 惩罚 action 的变化速度。
  - 关注的是：

    ```text
    action_t - action_{t-1}
    ```

  - 用来让 policy 输出更平滑。

- `dof_vel_non_ankle_l2`
  - 惩罚实际关节速度。
  - 当前配置：

    ```python
    params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!ankle_).*")}
    ```

  - 这会包含：

    ```text
    abad_L/R, hip_L/R, knee_L/R, J1~J6
    ```

- 注意：
  - `dof_vel_non_ankle_l2` 已经包含手臂 `J1~J6`。
  - 如果同时启用 `dof_vel_arm_l2`，机械臂速度会被惩罚两次。

### 7. 新增训练启动脚本

- 新增文件：
  - `train_sf_tron1_arm.sh`

- 运行方式：

  ```bash
  cd /home/phi5090ii/UMI-ON-TRON/umi-on-tron-lab-main/IsaacLab_RFM
  ./train_sf_tron1_arm.sh
  ```

- 脚本内容等价于：

  ```bash
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$PWD/rsl_rl:$PWD/source/ext_loco:$PYTHONPATH" \
  /home/phi5090ii/UMI-ON-TRON/conda_envs/isaaclab_tron/bin/python \
  scripts/rsl_rl/ios_train.py \
    --task Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0 \
    --num_envs 8192 \
    --headless \
    --logger wandb
  ```

### 8. Git / GitHub 记录

- 当前远端：

  ```text
  origin = https://github.com/Czy213hd/umi-on-tron.git
  ```

- 如果要推到 main：

  ```bash
  cd /home/phi5090ii/UMI-ON-TRON/umi-on-tron-lab-main
  git status
  git add -A
  git commit -m "update robot URDF and EE frame"
  git checkout main
  git pull origin main
  git merge feature/bidirectional-heading-alignment
  git push origin main
  ```

- 如果 merge 有冲突，先不要 push，先解决冲突。

## 后续记录模板

## 2026-07-15：同步 MuJoCo sim2sim 到当前 eef_link 训练配置

### 修改目的

- 当前训练已经把 EE frame 从 `link6` 改成固定在 UMI 夹爪 base 的 `eef_link`。
- MuJoCo sim2sim 脚本仍使用旧的 `link6` body 和旧的 link6->tip offset，导致策略观测和训练不一致，进入 MuJoCo 后容易直接倒机。

### 修改文件

- `run_sf_tron1_arm_mujoco.py`
- `IsaacLab_RFM/scripts/sim2sim/run_sf_tron1_arm_mujoco.py`
- `IsaacLab_RFM/scripts/sim2sim_quest/run_sf_tron1_arm_mujoco.py`
- `IsaacLab_RFM/scripts/sim2sim_quest/run_sf_tron1_arm_mujoco_quest_delta.py`

### 具体改动

- 默认路径修正：
  - 根目录脚本现在能从 `umi-on-tron-lab-main/IsaacLab_RFM` 找到 MJCF，不再错误解析到 `/home/phi5090ii/source/...`。

- EE frame 同步：
  - 在运行时给 MuJoCo `link6` 动态添加 collision-free site：

    ```text
    site name = eef_link
    pos       = 0.0999414 0.0000388 0.0767217
    quat      = 0.99144482142 0 0.130526495702 0
    ```

  - sim2sim 的 EE observation 从 `link6` body 改成读取 `eef_link` site：

    ```python
    data.site_xpos[ee_site_id]
    data.site_xmat[ee_site_id]
    ```

- trajectory offset 同步：
  - 清零旧的 `TIP_OFFSET_POS` / `TIP_OFFSET_RPY`，与 IsaacLab play 配置保持一致：

    ```python
    TIP_OFFSET_POS = np.zeros(3)
    TIP_OFFSET_RPY = (0.0, 0.0, 0.0)
    ```

- joint 顺序同步：
  - MuJoCo deploy 脚本的 `JOINT_NAMES` 改为当前 URDF/MJCF/articulation 顺序：

    ```text
    J1, J2, J3, J4, J5, J6,
    abad_L_Joint, hip_L_Joint, knee_L_Joint, ankle_L_Joint,
    abad_R_Joint, hip_R_Joint, knee_R_Joint, ankle_R_Joint
    ```

  - 同步调整 `DEFAULT_JOINT_POS`、`KP`、`KD`、`TORQUE_LIMIT` 的顺序。

### 验证方式

- 语法检查通过：

  ```bash
  python3 -m py_compile \
    run_sf_tron1_arm_mujoco.py \
    IsaacLab_RFM/scripts/sim2sim/run_sf_tron1_arm_mujoco.py \
    IsaacLab_RFM/scripts/sim2sim_quest/run_sf_tron1_arm_mujoco.py \
    IsaacLab_RFM/scripts/sim2sim_quest/run_sf_tron1_arm_mujoco_quest_delta.py
  ```

- 短时间 headless 测试能正常加载 MJCF 和 ONNX，不再出现路径或缺少 `eef_link` 的错误。

### 注意事项 / 风险

- 当前 MuJoCo 中仍观察到机器人会倒：
  - 使用 `2026-07-14_20-40-56/exported` 会倒。
  - 使用旧的 `2026-06-30_22-18-19/exported` 也会倒。
  - 甚至只用 PD 保持默认关节、不加载策略，MuJoCo 模型也会逐渐倒下。

- 因此目前剩余问题不只是 EE frame：
  - 需要继续排查当前 `assembly.xml` 的动力学/contact/默认站姿是否与 IsaacLab/旧 sim2sim 版本一致。
  - 也需要确认当前 checkpoint 是否已经具备 MuJoCo sim2sim 稳定站立能力。

## 2026-07-15：恢复旧 MuJoCo XML 结构，只持久化 eef_link

### 修改目的

- 保留原来的 MuJoCo 底盘、腿、电脑盒子和 body 层级，不让完整 URDF 导出改变下半身模型外观。
- 在原 XML 里持久化训练/控制使用的 `eef_link` frame。

### 修改文件

- `source/ext_loco/ext_loco/assets/SF_TRON1A_ARXR5ARM/assembly.xml`

### 具体改动

- 撤回完整 URDF 导出的 `assembly.xml`，因为 MuJoCo 会合并 fixed body，导致：
  - `tron1computer` / `base_link_arm` 层级消失；
  - 腿部 visual mesh 被简化；
  - 下半身外观变成和原来不一致。
- 恢复原来的 XML 结构，只在 `link6` 下增加：

  ```xml
  <site
    name="eef_link"
    pos="0.0999414 0.0000388 0.0767217"
    quat="0.99144482142 0 0.130526495702 0"
  />
  ```

- 这个 `eef_link` 的位置和姿态来自 URDF 里的 `gripper_fixed_joint`。

### 验证方式

- 真实 XML 直接加载通过：

  ```text
  nbody 27, njnt 15, nu 14, nsensor 31, nsite 2
  ```

- 使用 `run_sf_tron1_arm_mujoco.py` 的 `load_sim_model()` 加载通过：

  ```text
  nbody 27, njnt 15, nu 14, nsensor 31, nsite 3
  ```

- 确认这些对象存在：
  - `base_Link`
  - `tron1computer`
  - `base_link_arm`
  - `eef_link`
  - `command_target`
  - `imu_gyro`

### 注意事项 / 风险

- 不再使用完整 URDF 导出的简化 XML，避免下半身模型变化。
- 如果后面确实要同步新的机械臂/夹爪 mesh，应该只替换 `base_link_arm -> link6` 这一段 subtree，不要重新导出整机。

## YYYY-MM-DD：修改标题

### 修改目的

- 

### 修改文件

- 

### 具体改动

- 

### 验证方式

- 

### 注意事项 / 风险

- 
