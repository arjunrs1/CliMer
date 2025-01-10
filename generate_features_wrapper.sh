#!/bin/bash
# Be very wary of this explicit setting of CUDA_VISIBLE_DEVICES. Say you are
# running one task and asked for --gpus-per-node=1 then setting this variable will mean
# all your processes will want to run GPU 0 - disaster!! Setting this variable
# only makes sense in specific cases that I have described above where you are
# using --gpus-per-node=8 and I have spawned 8 tasks. So I need to divvy up the GPUs
# between the tasks. Think THRICE before you set this!!

# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

# Your CUDA enabled program here
python tools/run_net.py \
  --cfg /path/to/config/EPIC-KITCHENS/OMNIVORE_feature.yaml \
  NUM_GPUS <num_gpus> \
  OUTPUT_DIR /path/to/output/dataset_split \
  EPICKITCHENS.VISUAL_DATA_DIR /path/to/epic_frames \
  EPICKITCHENS.TEST_LIST /path/to/EPIC_100_feature_interval_times \
  TEST.BATCH_SIZE <batch_size> \
  TEST.ENABLE True \
  TEST.NUM_FEATURES <num_features>