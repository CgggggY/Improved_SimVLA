#!/bin/bash
cd ~/simvla_projects/SimVLA/evaluation/libero

# 1. spatial
CUDA_VISIBLE_DEVICES=5 python -u libero_client_debug.py \
    --host 127.0.0.1 --port 8102 --client_type websocket \
    --task_suite libero_spatial --num_trials 1 --video_out ./eval_results_3tasks/spatial

# 2. object
CUDA_VISIBLE_DEVICES=5 python -u libero_client_debug.py \
    --host 127.0.0.1 --port 8102 --client_type websocket \
    --task_suite libero_object --num_trials 1 --video_out ./eval_results_3tasks/object

# 3. goal
CUDA_VISIBLE_DEVICES=5 python -u libero_client_debug.py \
    --host 127.0.0.1 --port 8102 --client_type websocket \
    --task_suite libero_goal --num_trials 1 --video_out ./eval_results_3tasks/goal