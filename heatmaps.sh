
source /home/delta_one/v3env/bin/activate

python create_heatmaps.py \
  --root_dir_cancer /ssd_scratch/cvit/piyush/KIRC/test/cancer \
  --dest_dir /ssd_scratch/cvit/piyush/KIRC/heatmaps \
  --thumbnail_dir /ssd_scratch/cvit/piyush/KIRC/heatmap_thumbnails \
  --patch_extraction_csv KIRC_stats.csv \
  --record_csv record.csv \
  --alpha 0.75  | tee heatmap_logs.txt
