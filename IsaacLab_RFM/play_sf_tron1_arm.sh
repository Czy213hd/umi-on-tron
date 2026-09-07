#!/usr/bin/env bash
set -e

cd /home/phi5090ii/UMI-ON-TRON/umi-on-tron-lab-main/IsaacLab_RFM

LOAD_RUN="2026-07-19_17-08-18"
CHECKPOINT="model_2000.pt"
NUM_ENVS="50"

PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD/rsl_rl:$PWD/source/ext_loco:$PYTHONPATH" \
/home/phi5090ii/UMI-ON-TRON/conda_envs/isaaclab_tron/bin/python \
scripts/rsl_rl/ios_play.py \
  --task Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-Command-Play-v0 \
  --num_envs "$NUM_ENVS" \
  --load_run "$LOAD_RUN" \
  --checkpoint "$CHECKPOINT"