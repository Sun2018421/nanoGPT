#!/bin/bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
