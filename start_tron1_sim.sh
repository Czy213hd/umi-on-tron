#!/usr/bin/env bash
set -e

IMAGE_NAME="tron1-rl-deploy:noetic"
CONTAINER_NAME="tron1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_HOST="$REPO_DIR/tron1_ws"
DATA_HOST="$REPO_DIR/IsaacLab_RFM/data"

# pkl 轨迹参数（pushing.pkl，复现 LimxEEposeRoughEnvCfg_PLAY）
PKL_FILE="tossing.pkl"
PKL_TRAJ_IDX=0
PKL_START_DELAY=15.0   # 等待 Gazebo + 控制器完全启动后再开始跟踪（秒）
PKL_LOOP=true
PKL_HOST_PATH="$DATA_HOST/$PKL_FILE"

# 1. 检查 workspace 是否存在
if [ ! -d "$WS_HOST" ]; then
  echo "找不到工作空间目录: $WS_HOST"
  echo "请确认你的 tron1_ws 在这个路径下，然后再运行本脚本。"
  exit 1
fi

# 2. 允许容器访问宿主机显示（X11）
xhost +local:root >/dev/null 2>&1 || true

# 3. 启动容器 + 挂载工作空间和pkl数据 + 编译 + 仿真 + 可选 pkl 轨迹发布
docker run --rm -it \
  --name "$CONTAINER_NAME" \
  --net=host \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e ROBOT_TYPE=SF_TRON1A_ARX5ARM \
  -e PKL_FILE="$PKL_FILE" \
  -e PKL_TRAJ_IDX="$PKL_TRAJ_IDX" \
  -e PKL_START_DELAY="$PKL_START_DELAY" \
  -e PKL_LOOP="$PKL_LOOP" \
  -v "$WS_HOST":/root/tron1_ws \
  -v "$DATA_HOST":/root/umi_data:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "$IMAGE_NAME" \
  bash -lc "
    set -e
    echo '--- 载入 ROS 环境 ---'
    source /opt/ros/noetic/setup.bash

    echo '--- 切换到 /root/tron1_ws ---'
    cd /root/tron1_ws

    echo '--- 运行 catkin_make ---'
    catkin_make

    echo '--- 编译完成，source devel ---'
    source devel/setup.bash

    echo '--- 后台启动 Gazebo 仿真 ---'
    roslaunch robot_hw pointfoot_hw_sim.launch &
    ROSLAUNCH_PID=\$!

    PKL_PID=''
    if [ -f \"/root/umi_data/\$PKL_FILE\" ]; then
      echo '--- 等待 '\$PKL_START_DELAY's 让 Gazebo + 控制器完全初始化 ---'
      sleep \$PKL_START_DELAY

      echo '--- 启动 pkl 轨迹发布器: '\$PKL_FILE' ---'
      python3 /root/tron1_ws/src/tron1-rl-deploy-arm/src/robot_controllers/scripts/publish_eepose_target_world_from_pkl.py \
        _pickle_path:=/root/umi_data/\$PKL_FILE \
        _traj_idx:=\$PKL_TRAJ_IDX \
        _start_delay_s:=3.0 \
        _loop:=\$PKL_LOOP &
      PKL_PID=\$!
    else
      echo '--- 未找到 /root/umi_data/'\"\$PKL_FILE\"'，跳过 pkl 发布器 ---'
      echo '--- 将使用 params.yaml 中 ee_target.manual_enable 的 fallback 目标 ---'
    fi

    echo '--- 运行中（Ctrl+C 退出）---'
    wait \$ROSLAUNCH_PID || true
    if [ -n \"\$PKL_PID\" ]; then
      kill \$PKL_PID 2>/dev/null || true
    fi
  "
