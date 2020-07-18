

source ~/v3env/bin/activate
python patch_cnn.py \
  --train_dir /ssd_scratch/cvit/piyush/KIRC/train/ \
  --val_dir /ssd_scratch/cvit/piyush/KIRC/val/ \
  --model_checkpoint checkpoints/KIRC_model_epoch_11.pth \
  --save_prefix KIRC \
  --num_epochs 25  | tee kirc_train_log_full.txt
