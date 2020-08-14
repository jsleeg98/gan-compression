#!/usr/bin/env bash
python export.py \
  --input_path logs/pix2pix/map2sat_lite/finetune/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/map2sat_lite/compressed/latest_net_G.pth \
  --ngf 64 --config_str $1
