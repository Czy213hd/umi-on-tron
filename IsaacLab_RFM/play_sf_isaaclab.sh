cd /home/phi5090ii/UMI-ON-TRON/umi-on-tron-lab-main/IsaacLab_RFM

LOAD_RUN="2026-07-18_21-30-27"
CHECKPOINT="model_12600.pt"
NUM_ENVS="50"

PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD/rsl_rl:$PWD/source/ext_loco:$PYTHONPATH" \
/home/phi5090ii/UMI-ON-TRON/conda_envs/isaaclab_tron/bin/python \
scripts/rsl_rl/ios_play.py \
  --task Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-Command-Play-v0 \
  --num_envs "$NUM_ENVS" \
  --load_run "$LOAD_RUN" \
  --checkpoint "$CHECKPOINT" \
  --target_x 0.5 \
  --target_y 0.0 \
  --target_z 1.3 \
  --target_roll 0.0 \
  --target_pitch 0.0 \
  --target_yaw 0.0