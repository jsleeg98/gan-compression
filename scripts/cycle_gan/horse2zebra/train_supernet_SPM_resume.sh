#!/usr/bin/env bash
python train_supernet.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnet \
  --student_netG super_mobile_resnet_9blocks_SPM \
  --log_dir logs/cycle_gan/horse2zebra/supernet_SPM_test_resume \
  --gan_mode lsgan \
  --student_ngf 32 --ndf 64 \
  --restore_teacher_G_path ./pretrained/cycle_gan/horse2zebra/mobile/latest_net_G.pth \
  --restore_student_G_path logs/cycle_gan/horse2zebra/supernet_SPM_test/checkpoints/iter120000_net_G.pth \
  --restore_D_path logs/cycle_gan/horse2zebra/supernet_SPM_test/checkpoints/iter120000_net_D.pth \
  --restore_A_path logs/cycle_gan/horse2zebra/supernet_SPM_test/checkpoints/iter120000_net_A \
  --restore_O_path logs/cycle_gan/horse2zebra/supernet_SPM_test/checkpoints/iter40000_optim \
  --epoch_base 100 \
  --iter_base 120000 \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 400 --nepochs_decay 200 \
  --save_epoch_freq 20 \
  --metaA_path datasets/metas/horse2zebra/train1A.meta \
  --metaB_path datasets/metas/horse2zebra/train1B.meta \
  --config_str 32_32_32_32_32_32_32_32 \
  --no_mac_loss \
  --no_nuc_loss
  #  --config_set channels-32 \
#  --restore_student_G_path logs/cycle_gan/horse2zebra/distill/checkpoints/latest_net_G.pth \
