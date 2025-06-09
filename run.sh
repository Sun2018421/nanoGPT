#!/bin/bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train_non_clip.py config/train_gpt2.py
