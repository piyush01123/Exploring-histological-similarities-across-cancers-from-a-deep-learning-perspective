#!/bin/bash
for subtype in "KIRC" "KICH" "KIRP" ;
  do
    echo "Subtype: $subtype"
    echo "Running Visualizations... " | tee "$subtype"_viz_log.txt
    for mode in "IntGrads" "GradCAM" "GuidedGradCAM" "Occlusion" ;
      do
        cmd="python visualize.py --model_checkpoint checkpoints/"$subtype"_model_epoch_24.pth \
                                --train_dir /ssd_scratch/cvit/TCGA/"$subtype"/train/ \
                                --test_dir /ssd_scratch/cvit/TCGA/"$subtype"/test \
                                --log_dir "$subtype"_log_viz \
                                --record_csv "$subtype"_viz_record.csv \
                                --batch_size 8 \
                                --num_images 32
                                --visu_mode $mode"
        echo $cmd | tee -a "$subtype"_viz_log.txt
        $cmd | tee -a "$subtype"_viz_log.txt
      done
  done