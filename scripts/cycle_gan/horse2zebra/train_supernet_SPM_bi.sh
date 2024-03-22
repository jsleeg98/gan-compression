#!/usr/bin/env bash
python train_supernet.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnet \
  --student_netG super_mobile_resnet_9blocks_SPM \
  --log_dir logs/cycle_gan/horse2zebra/supernet_SPM_test_loss \
  --gan_mode lsgan \
  --student_ngf 32 --ndf 64 \
  --restore_teacher_G_path ./pretrained/cycle_gan/horse2zebra/mobile/latest_net_G.pth \
  --restore_D_path ./pretrained/cycle_gan/horse2zebra/mobile/latest_net_D.pth \
  --restore_student_G_path logs/cycle_gan/horse2zebra/supernet_baseline/checkpoints/latest_net_G.pth \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 6 --nepochs_decay 4 \
  --save_epoch_freq 5 \
  --metaA_path datasets/metas/horse2zebra/train1A.meta \
  --metaB_path datasets/metas/horse2zebra/train1B.meta \
  --config_str 32_32_32_32_32_32_32_32 \
  --alpha_mac 5.0 \
  --alpha_nuc 0.001 \
  --target_ratio 0.5 \
  --R_max 6 \
  --proj_name cyclegan_bi_level \
  --bi_level_train \
  --bi_level_interval 2 \
  --bi_level_start_epoch 1 \
  #  --config_set channels-32 \
#  --restore_student_G_path logs/cycle_gan/horse2zebra/distill/checkpoints/latest_net_G.pth \