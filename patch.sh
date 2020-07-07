
source ~/v3env/bin/activate
python extract_patches.py \
  --root_dir /ssd_scratch/cvit/piyush/KIRC/SLIDES/ \
  --dest_dir /ssd_scratch/cvit/piyush/KIRC/PATCHES/ \
  --extras_dir /ssd_scratch/cvit/piyush/KIRC/EXTRAS/  | tee log_patch.txt
