# UMI on Tron1 Lab

在 Tron1 足式机器人上部署 UMI（Universal Manipulation Interface）操作技能的实验仓库。

本项目基于 Stanford 的 [UMI on Legs](https://umi-on-legs.github.io/) 研究，将机械臂操作技能部署到逐际动力（LimX Dynamics）的 Tron1 双足机器人上。

## 项目概述

本项目实现了以下功能：

- **仿真环境**：在 Gazebo 中仿真 Tron1 机器人 + ARX5 机械臂，自动回放 UMI 采集的 pkl 轨迹
- **真实部署**：在真实 Tron1 机器人上通过 SocketCAN 驱动 ARX5 机械臂，部署强化学习策略
- **操作技能迁移**：使用 UMI 收集的操作数据训练末端位姿控制策略（IsaacLab RFM）

## 仓库结构

```
umi-on-tron-lab/
├── tron1_ws/                              # Tron1 机器人 ROS Noetic 工作空间
│   ├── build_and_run_real_arx5_socketcan.sh  # 容器内编译 + 运行脚本
│   └── src/tron1-rl-deploy-arm/src/
│       ├── robot_hw/                      # 硬件接口层
│       ├── robot_controllers/             # RL 控制器（含 pkl 轨迹发布器）
│       ├── robot-description/             # 机器人 URDF 模型
│       ├── robot_common/                  # 公共库
│       ├── arx5-sdk/                      # ARX5 机械臂 SocketCAN SDK
│       ├── airbot-sdk-2.9/                # Airbot 机械臂 SDK
│       ├── onnxruntime_sdk/               # ONNX 推理库
│       ├── pointfoot-gazebo-ros/          # Gazebo 仿真插件
│       └── pointfoot-sdk-lowlevel/        # 底层通信 SDK
├── IsaacLab_RFM/                          # IsaacLab 强化学习训练框架
│   ├── scripts/rsl_rl/
│   │   ├── ios_train.py                   # 训练脚本
│   │   └── ios_play.py                    # 推理 + ONNX 导出脚本
│   ├── source/ext_loco/                   # 自定义任务环境
│   └── data/                              # UMI pkl 轨迹数据
├── host_run_tron1_real_arx5_socketcan.sh  # 宿主机侧真实机器人部署脚本
└── start_tron1_sim.sh                     # 仿真一键启动脚本
```

## 快速开始（仿真）

### 前置要求

- Ubuntu 20.04
- Docker 已安装
- 宿主机支持 X11 显示（用于 Gazebo GUI）

### 0. 构建 Docker 镜像（首次使用）

项目根目录已包含 `Dockerfile`，一键构建完整环境：

```bash
docker build -t tron1-rl-deploy:noetic .
```

镜像基于 `osrf/ros:noetic-desktop-full`，包含：
- ROS Noetic Desktop Full + Gazebo
- ros_control 全套控制器包
- Eigen / PCL / OpenCV / Boost / spdlog 等 C++ 库
- Python 3 + pip

> 构建约需 5~10 分钟（取决于网速），只需执行一次。之后重新克隆仓库也无需重新构建，除非 `Dockerfile` 有改动。

### 1. 准备 pkl 轨迹数据

将 UMI 采集的 pkl 文件放入 `IsaacLab_RFM/data/`，例如 `tossing.pkl`。

在 `start_tron1_sim.sh` 顶部修改以下参数：

```bash
PKL_FILE="tossing.pkl"    # pkl 文件名
PKL_TRAJ_IDX=0            # 轨迹索引
PKL_START_DELAY=15.0      # 等待 Gazebo + 控制器完全启动后再开始跟踪（秒）
PKL_LOOP=true             # 是否循环回放
```

### 2. 启动仿真

```bash
./start_tron1_sim.sh
```

脚本会自动完成：
1. 挂载 `tron1_ws` 和 `IsaacLab_RFM/data` 到 Docker 容器
2. 执行 `catkin_make` 编译工作空间
3. 后台启动 Gazebo 仿真（`roslaunch robot_hw pointfoot_hw_sim.launch`）
4. 延迟 `PKL_START_DELAY` 秒后启动 pkl 轨迹发布器

<!-- ## 真实机器人部署

### 硬件要求

- Tron1 双足机器人（`SF_TRON1A_ARX5ARM`）
- ARX5 机械臂（SocketCAN 接口，**非** soem 版本）
- 宿主机上已配置好 `can0` 接口

### 1. 配置 CAN 接口（宿主机）

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set up can0
```

### 2. 准备 Docker 容器

容器需要满足以下条件：

```bash
docker run -d \
  --name tron1 \
  --network host \
  --cap-add=NET_RAW \
  --cap-add=NET_ADMIN \
  -v /path/to/tron1_ws:/root/tron1_ws \
  -v /path/to/umi-on-legs:/root/umi-on-legs \
  tron1-rl-deploy:noetic \
  sleep infinity
```

### 3. 运行部署脚本

```bash
# 编译并运行（默认）
./host_run_tron1_real_arx5_socketcan.sh all

# 仅编译
./host_run_tron1_real_arx5_socketcan.sh build

# 仅运行（已编译过）
./host_run_tron1_real_arx5_socketcan.sh run

# 检查环境配置
./host_run_tron1_real_arx5_socketcan.sh check
```

#### 可选环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONTAINER_NAME` | `tron1` | Docker 容器名 |
| `ROBOT_TYPE` | `SF_TRON1A_ARX5ARM` | 机器人类型 |
| `CAN_IF` | `can0` | CAN 接口名 |
| `ARX5_SDK_DIR_IN_CONTAINER` | 自动检测 | 容器内 arx5-sdk 路径 |
| `NO_TTY` | `0` | 设为 `1` 禁用 `-it` 标志 | -->

## 支持的机器人类型

| 类型 | 描述 |
|------|------|
| `SF_TRON1A` | 标准 Tron1 双足（无机械臂） |
| `WF_TRON1A` | 轮足 Tron1（无机械臂） |
| `SF_TRON1A_ARX5ARM` | 标准 Tron1 双足 + ARX5 机械臂 |

## IsaacLab RFM 训练

### 环境配置

1. 安装 IsaacSim 4.5.0 和 IsaacLab 2.0.1

2. 安装项目依赖：

```bash
cd IsaacLab_RFM
python -m pip install -e source/ext_loco
pip install -e rsl_rl
```

<!-- 3. 拉取 LFS 文件（USD 资产等）：

```bash
git lfs install
git lfs pull
``` -->

### 训练

```bash
cd IsaacLab_RFM

# 足式 Tron1 + ARX5 机械臂
python scripts/rsl_rl/ios_train.py \
    --task Template-Isaac-EEPose-Rough-Limx-SF-Tron1A-v0 \
    --headless \
    --logger wandb

# 轮足 Tron1 + ARX5 机械臂
python scripts/rsl_rl/ios_train.py \
    --task Template-Isaac-EEPose-Rough-Limx-WF-Tron1A-v0 \
    --headless \
    --logger wandb
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--num_envs <N>` | 并行环境数量 |
| `--max_iterations <N>` | 最大训练迭代次数 |
| `--seed <N>` | 随机种子 |
| `--video` | 录制训练视频 |

### 推理 & ONNX 导出

```bash
python scripts/rsl_rl/ios_play.py \
    --task Template-Isaac-EEPose-Rough-Limx-SF-Tron1A-Play-v0 \
    --num_envs 1
```

Play 脚本自动导出 ONNX 模型到 `logs/rsl_rl/ImplicitOneStageARXR5Arm/<timestamp>/exported/`：

| 文件 | 说明 |
|------|------|
| `actor.onnx` | Actor 网络 |
| `gru.onnx` | GRU 时序记忆网络（latent dim=128） |
| `contactNet.onnx` | Transformer 接触力估计网络 |

导出的模型可直接复制到 `tron1_ws/src/tron1-rl-deploy-arm/src/robot_controllers/config/pointfoot/SF_TRON1A_ARX5ARM/policy/` 用于实机部署。

### 训练配置

主要超参数位于：
`source/ext_loco/ext_loco/tasks/loco_manipulation/EE_pose/config/sf_tron1_arm/agents/implicit_one_stage_cfg.py`

- **算法**：PPO（gamma=0.99, lam=0.95）
- **训练迭代**：20000 次，每 200 次保存检查点
- **日志**：`IsaacLab_RFM/logs/rsl_rl/ImplicitOneStageARXR5Arm/`

## 参考资源

- [UMI on Legs 论文](https://arxiv.org/abs/2407.10353)
- [UMI on Legs 项目主页](https://umi-on-legs.github.io/)
- [LimX Dynamics GitHub](https://github.com/limxdynamics)
- [ARX5 SDK](https://github.com/yihuai-gao/arx5-sdk)

## 致谢

- [UMI on Legs](https://github.com/real-stanford/umi-on-legs) — Stanford 的移动操作框架
- [Tron1 RL Deploy](https://github.com/limxdynamics/tron1-rl-deploy-arm) — 逐际动力的强化学习部署代码
- [ARX5 SDK](https://github.com/yihuai-gao/arx5-sdk) — ARX5 机械臂 SocketCAN 驱动
- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — NVIDIA 机器人学习框架

## 许可证

本项目仅供研究和学习使用。各子模块遵循其原始许可证。
