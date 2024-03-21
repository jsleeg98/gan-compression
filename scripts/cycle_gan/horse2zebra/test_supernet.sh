#!/usr/bin/env bash
python test.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --results_dir results-pretrained/cycle_gan/horse2zebra/supernet \
  --ngf 32 --netG super_mobile_resnet_9blocks \
  --config_str 32_32_32_32_32_32_32_32 \
  --restore_G_path logs/cycle_gan/horse2zebra/supernet_baseline/checkpoints/latest_net_G.pth \
  --need_profile \
  --real_stat_path real_stat/horse2zebra_B.npz
