#!/usr/bin/env bash
set -euo pipefail

# Host-side helper:
# - Start existing container (default name: tron1) if needed
# - Find workspace & legacy SocketCAN arx5-sdk paths inside container
# - Exec into container and run tron1_ws/build_and_run_real_arx5_socketcan.sh
#
# Usage (on HOST):
#   ./host_run_tron1_real_arx5_socketcan.sh [all|build|run|check]
#
# Optional env overrides:
#   CONTAINER_NAME=tron1
#   ROBOT_TYPE=SF_TRON1A_ARX5ARM
#   CAN_IF=can0
#   CONTAINER_WS=/root/tron1_ws
#   ARX5_SDK_DIR_IN_CONTAINER=/root/umi-on-tron-lab/umi-on-legs/arx5-sdk
#   NO_TTY=1   # disable -it
#   DOCKER_BIN=docker

cmd="${1:-all}"

CONTAINER_NAME="${CONTAINER_NAME:-tron1}"
ROBOT_TYPE="${ROBOT_TYPE:-SF_TRON1A_ARX5ARM}"
CAN_IF="${CAN_IF:-can0}"
CONTAINER_WS="${CONTAINER_WS:-}"
ARX5_SDK_DIR_IN_CONTAINER="${ARX5_SDK_DIR_IN_CONTAINER:-}"
NO_TTY="${NO_TTY:-0}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

log()  { echo "[host-deploy] $*"; }
warn() { echo "[host-deploy][WARN] $*" >&2; }
die()  { echo "[host-deploy][ERROR] $*" >&2; exit 1; }

# Pick docker command (docker vs sudo -n docker).
DOCKER_CMD=("${DOCKER_BIN}")
if ! "${DOCKER_CMD[@]}" ps >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n "${DOCKER_BIN}" ps >/dev/null 2>&1; then
    DOCKER_CMD=(sudo -n "${DOCKER_BIN}")
    log "Using docker via sudo -n (non-interactive)."
  else
    die "Cannot access docker daemon. Try: sudo ${DOCKER_BIN} ps  OR  add your user to docker group."
  fi
fi

# Host-side CAN hint (optional)
if command -v ip >/dev/null 2>&1; then
  if ip link show "${CAN_IF}" >/dev/null 2>&1; then
    log "Host CAN interface exists: ${CAN_IF}"
  else
    warn "Host CAN interface not found: ${CAN_IF}. Make sure you have configured can0 on the host."
  fi
fi

# Ensure container exists.
if ! "${DOCKER_CMD[@]}" inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  cat >&2 <<EOF
[host-deploy][ERROR] Container not found: ${CONTAINER_NAME}

This script is designed to work with an existing container.
If you haven't created it yet, create your tron1 container with:
  - --network host
  - --cap-add=NET_RAW (and optionally --cap-add=NET_ADMIN)
  - volume mounts that expose your repo inside the container

Then re-run this script.
EOF
  exit 1
fi

# Start container if needed.
running="$("${DOCKER_CMD[@]}" inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")"
if [[ "${running}" != "true" ]]; then
  log "Starting container: ${CONTAINER_NAME}"
  "${DOCKER_CMD[@]}" start "${CONTAINER_NAME}" >/dev/null
else
  log "Container already running: ${CONTAINER_NAME}"
fi

TTY_FLAGS=()
if [[ "${NO_TTY}" == "0" ]]; then
  TTY_FLAGS=(-it)
fi

container_has_file() {
  local path="$1"
  "${DOCKER_CMD[@]}" exec "${CONTAINER_NAME}" bash -lc "[[ -f \"${path}\" ]]" >/dev/null 2>&1
}

# Detect tron1_ws path inside container if not provided.
if [[ -z "${CONTAINER_WS}" ]]; then
  for d in \
    /root/tron1_ws \
    /root/umi-on-tron-lab/tron1_ws \
    /workspace/umi-on-tron-lab/tron1_ws \
    /src/umi-on-tron-lab/tron1_ws \
    /src/tron1_ws \
    /workspace/tron1_ws
  do
    if container_has_file "${d}/build_and_run_real_arx5_socketcan.sh"; then
      CONTAINER_WS="${d}"
      break
    fi
  done
fi
[[ -n "${CONTAINER_WS}" ]] || die "Cannot find tron1_ws inside container. Set CONTAINER_WS=/path/to/tron1_ws"

log "Container workspace: ${CONTAINER_WS}"

# Detect legacy arx5-sdk dir inside container if not provided.
if [[ -z "${ARX5_SDK_DIR_IN_CONTAINER}" ]]; then
  candidates=(
    "${CONTAINER_WS}/../umi-on-legs/arx5-sdk"
    "/root/umi-on-tron-lab/umi-on-legs/arx5-sdk"
    "/workspace/umi-on-tron-lab/umi-on-legs/arx5-sdk"
    "/src/umi-on-tron-lab/umi-on-legs/arx5-sdk"
    "/root/umi-on-legs/arx5-sdk"
    "/workspace/umi-on-legs/arx5-sdk"
    "/src/umi-on-legs/arx5-sdk"
  )
  for d in "${candidates[@]}"; do
    if "${DOCKER_CMD[@]}" exec "${CONTAINER_NAME}" bash -lc "[[ -f \"${d}/include/hardware/arx_can.h\" ]]" >/dev/null 2>&1; then
      ARX5_SDK_DIR_IN_CONTAINER="${d}"
      break
    fi
  done
fi

if [[ -n "${ARX5_SDK_DIR_IN_CONTAINER}" ]]; then
  log "Container ARX5_SDK_DIR: ${ARX5_SDK_DIR_IN_CONTAINER}"
else
  warn "Could not auto-detect legacy arx5-sdk in container. build script may fall back to a wrong SDK."
  warn "Strongly recommended to set: ARX5_SDK_DIR_IN_CONTAINER=/path/to/legacy/arx5-sdk"
fi

# Run inside container.
inner_env="export ROBOT_TYPE=\"${ROBOT_TYPE}\"; export CAN_IF=\"${CAN_IF}\";"
if [[ -n "${ARX5_SDK_DIR_IN_CONTAINER}" ]]; then
  inner_env="${inner_env} export ARX5_SDK_DIR=\"${ARX5_SDK_DIR_IN_CONTAINER}\";"
fi

inner_cmd="${inner_env} cd \"${CONTAINER_WS}\" && chmod +x build_and_run_real_arx5_socketcan.sh && ./build_and_run_real_arx5_socketcan.sh \"${cmd}\""

log "Exec into container and run: ${cmd}"
exec "${DOCKER_CMD[@]}" exec "${TTY_FLAGS[@]}" "${CONTAINER_NAME}" bash -lc "${inner_cmd}"


