#!/usr/bin/env bash
python test_SPM.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --results_dir results-pretrained/cycle_gan/horse2zebra/supernet_SPM \
  --ngf 32 --netG super_mobile_resnet_9blocks_SPM_bi \
  --config_str 32_32_32_32_32_32_32_32 \
  --restore_G_path logs/cycle_gan/horse2zebra/cyclegan_batch1/supernet_FDRU_YYYY_tr_0.78_mac_0.5_nuc_0.001_biStart_1_biInterval_2_Rmax_100/checkpoints/latest_net_G.pth \
  --need_profile \
  --real_stat_path real_stat/horse2zebra_B.npz

