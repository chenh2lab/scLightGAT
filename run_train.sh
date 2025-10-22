#!/bin/bash


source venv/bin/activate


export PYTHONPATH=.


python scLightGAT/pipeline.py \
  --mode train \
  --train data/train/train_GSE153935.h5ad \
  --test "data/final results/GSE153935_final_complete.h5ad" \
  --output results/ \
  --model_dir scLightGAT/models \
  --train_dvae \
  --use_gat \
  --dvae_epochs 3 \
  --gat_epochs 300
