

source /home/delta_one/v3env/bin/activate

python inference.py \
  --model_checkpoint checkpoints/KIRC_Dense_model_epoch_11.pth \
  --test_dir /ssd_scratch/cvit/piyush/KIRC/test \
  --log_dir inf_dense_logs/ | tee kirc_inference_dense_log.txt
