#!/bin/bash
# 根据ASCENDXX来确定使用的NPU
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
