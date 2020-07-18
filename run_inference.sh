
source /home/delta_one/v3env/bin/activate

python inference.py \
  --model_checkpoint checkpoints/KIRC_model_epoch_24.pth \
  --test_dir /ssd_scratch/cvit/piyush/KIRC/test \
  --log_dir inf_logs/ > kirc_inference_log.txt
