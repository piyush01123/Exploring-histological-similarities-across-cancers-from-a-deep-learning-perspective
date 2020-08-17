#!/bin/bash

for mode in "IntGrads" "GradCAM" "GuidedGradCAM" "Occlusion" ;
  do
    cmd="python visualize.py --model_checkpoint $HOME/.torch/models/sairam_model.pth \
                             --train_dir /ssd_scratch/cvit/TCGA/KIRC/train/ \
                             --test_dir /ssd_scratch/cvit/TCGA/KIRC/test \
                             --log_dir KIRC_log_viz \
                             --record_csv KIRC_viz_record.csv \
                             --batch_size 8 \
                             --num_images 16
                             --visu_mode $mode"
    echo $cmd
    $cmd | tee KIRC_viz_log.txt
  done