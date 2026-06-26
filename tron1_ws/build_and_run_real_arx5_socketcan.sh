#!/usr/bin/env bash
set -euo pipefail

# Build + run TRON1 real deployment with ARX5 arm using legacy SocketCAN arx5-sdk (NO soem, NO solver/ROS2 runtime).
#
# Usage:
#   ./build_and_run_real_arx5_socketcan.sh [all|build|run|check]
#
# Common overrides (environment variables):
#   ROS_DISTRO=noetic
#   ROBOT_TYPE=SF_TRON1A_ARX5ARM
#   CAN_IF=can0
#   ARX5_SDK_DIR=/abs/path/to/legacy/arx5-sdk   # must contain include/hardware/arx_can.h and lib/<arch>/libhardware.so
#   CMAKE_BUILD_TYPE=RelWithDebInfo
#
# Notes:
# - This script is intended to be executed INSIDE your ROS1 (Noetic) docker container.
# - CAN interface (can0) should be configured on the HOST, and the container should run with host network
#   and sufficient caps (NET_RAW, optionally NET_ADMIN) so it can open SocketCAN.

cmd="${1:-all}"

log()  { echo "[deploy] $*"; }
warn() { echo "[deploy][WARN] $*" >&2; }
die()  { echo "[deploy][ERROR] $*" >&2; exit 1; }

ROS_DISTRO="${ROS_DISTRO:-noetic}"
ROBOT_TYPE="${ROBOT_TYPE:-SF_TRON1A_ARX5ARM}"
CAN_IF="${CAN_IF:-can0}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

guess_arx5_sdk_dir() {
  # Common repo layout: <repo_root>/tron1_ws and <repo_root>/umi-on-legs/arx5-sdk
  local cand
  cand="${WS_DIR}/../umi-on-legs/arx5-sdk"
  if [[ -d "${cand}" ]]; then
    echo "${cand}"
    return
  fi
  # Fallback: allow co-located copy (user-provided).
  cand="${WS_DIR}/src/tron1-rl-deploy-arm/src/arx5-sdk"
  if [[ -d "${cand}" ]]; then
    echo "${cand}"
    return
  fi
  echo ""
}

ARX5_SDK_DIR="${ARX5_SDK_DIR:-$(guess_arx5_sdk_dir)}"
[[ -n "${ARX5_SDK_DIR}" ]] || die "ARX5_SDK_DIR not set and default not found. Please export ARX5_SDK_DIR=/path/to/legacy/arx5-sdk"

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64|amd64) ARX5_LIB_DIR="${ARX5_SDK_DIR}/lib/x86_64" ;;
  aarch64|arm64) ARX5_LIB_DIR="${ARX5_SDK_DIR}/lib/aarch64" ;;
  *) die "Unsupported arch: ${ARCH}. Expected x86_64 or aarch64." ;;
esac

check_runtime_deps() {
  if ! command -v ldd >/dev/null 2>&1; then
    warn "ldd not found; skip dependency sanity check."
    return 0
  fi
  local so="${ARX5_LIB_DIR}/libhardware.so"
  if ldd "${so}" | grep -qE "libsoem\\.so"; then
    warn "Detected libsoem dependency in ${so}. This script expects legacy SocketCAN arx5-sdk WITHOUT soem."
    warn "Please point ARX5_SDK_DIR to the legacy SDK (e.g. ../umi-on-legs/arx5-sdk)."
  fi
  if ldd "${so}" | grep -qE "rcutils|ament_index|class_loader"; then
    warn "Detected ROS2 runtime dependency in ${so}. This deployment expects ROS1-only runtime."
  fi
}

check_env() {
  [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]] || die "ROS not found: /opt/ros/${ROS_DISTRO}/setup.bash"
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"

  command -v catkin_make >/dev/null 2>&1 || die "catkin_make not found. Run inside ROS1 Noetic docker (or install catkin-tools)."

  [[ -f "${ARX5_SDK_DIR}/include/hardware/arx_can.h" ]] || die "Missing ${ARX5_SDK_DIR}/include/hardware/arx_can.h (wrong ARX5_SDK_DIR?)"
  [[ -f "${ARX5_LIB_DIR}/libhardware.so" ]] || die "Missing ${ARX5_LIB_DIR}/libhardware.so (wrong ARX5_SDK_DIR or arch?)"

  if command -v ip >/dev/null 2>&1; then
    if ip link show "${CAN_IF}" >/dev/null 2>&1; then
      log "CAN interface found: ${CAN_IF}"
    else
      warn "CAN interface not found: ${CAN_IF}"
      warn "If you're in docker, make sure container uses --network host and has --cap-add=NET_RAW."
      warn "Also make sure the HOST has brought up can0 (ip link set up can0 type can bitrate 1000000)."
    fi
  else
    warn "ip command not found; cannot check CAN interface."
  fi

  check_runtime_deps
}

do_build() {
  log "Building workspace: ${WS_DIR}"
  cd "${WS_DIR}"

  catkin_make -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DUSE_ARX5_SDK=ON \
    -DARX5_SDK_DIR="${ARX5_SDK_DIR}"

  log "Build done."
}

do_run() {
  cd "${WS_DIR}"
  [[ -f "${WS_DIR}/devel/setup.bash" ]] || die "Missing devel/setup.bash. Run build first."

  # shellcheck disable=SC1090
  source "${WS_DIR}/devel/setup.bash"

  export ROBOT_TYPE="${ROBOT_TYPE}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ARX5_LIB_DIR}"

  log "ROBOT_TYPE=${ROBOT_TYPE}"
  log "ARX5_SDK_DIR=${ARX5_SDK_DIR}"
  log "ARX5_LIB_DIR=${ARX5_LIB_DIR}"
  log "Launching real robot stack (robot_hw/pointfoot_hw.launch)..."
  exec roslaunch robot_hw pointfoot_hw.launch
}

case "${cmd}" in
  all)
    check_env
    do_build
    do_run
    ;;
  build)
    check_env
    do_build
    ;;
  run)
    check_env
    do_run
    ;;
  check)
    check_env
    log "Check OK."
    ;;
  -h|--help|help)
    sed -n '1,80p' "${BASH_SOURCE[0]}"
    ;;
  *)
    die "Unknown command: ${cmd}. Use: all|build|run|check"
    ;;
esac


