#!/usr/bin/env bash
python train_supernet.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnet \
  --student_netG super_mobile_resnet_9blocks_SPM_bi \
  --log_dir logs/cycle_gan/horse2zebra/ \
  --gan_mode lsgan \
  --student_ngf 32 --ndf 64 \
  --restore_teacher_G_path ./pretrained/cycle_gan/horse2zebra/mobile/latest_net_G.pth \
  --restore_student_G_path logs/cycle_gan/horse2zebra/cyclegan_bi_level/supernet_SPM_2mac_tr_0.49_mac_10.0_nuc_0.001_biStart_1_biInterval_10_Rmax_400/checkpoints/400_net_G.pth \
  --restore_D_path logs/cycle_gan/horse2zebra/cyclegan_bi_level/supernet_SPM_2mac_tr_0.49_mac_10.0_nuc_0.001_biStart_1_biInterval_10_Rmax_400/checkpoints/400_net_D.pth \
  --restore_A_path logs/cycle_gan/horse2zebra/cyclegan_bi_level/supernet_SPM_2mac_tr_0.49_mac_10.0_nuc_0.001_biStart_1_biInterval_10_Rmax_400/checkpoints/400_net_A \
  --restore_O_path logs/cycle_gan/horse2zebra/cyclegan_bi_level/supernet_SPM_2mac_tr_0.49_mac_10.0_nuc_0.001_biStart_1_biInterval_10_Rmax_400/checkpoints/400_optim \
  --epoch_base 400 \
  --iter_base 485600 \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 400 --nepochs_decay 200 \
  --save_epoch_freq 5 \
  --metaA_path datasets/metas/horse2zebra/train1A.meta \
  --metaB_path datasets/metas/horse2zebra/train1B.meta \
  --config_str 32_32_32_32_32_32_32_32 \
  --alpha_mac 10.0 \
  --alpha_nuc 0.001 \
  --target_ratio 0.49 \
  --R_max 400 \
  --proj_name cyclegan_bi_level \
  --log_name supernet_SPM_2mac_resume \
  --bi_level_train \
  --bi_level_interval 10 \
  --bi_level_start_epoch 1
  #  --config_set channels-32 \
#  --restore_student_G_path logs/cycle_gan/horse2zebra/distill/checkpoints/latest_net_G.pth \
