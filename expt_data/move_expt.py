

import argparse
import os
import shutil
import glob
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--expt_train_dir", required=True)
    parser.add_argument("--expt_val_dir", required=True)
    parser.add_argument("--expt_test_dir", required=True)
    parser.add_argument("--expt_json", required=True)
    args = parser.parse_args()

    with open(args.expt_json, 'r') as fh:
        expt_dict = json.load(fh)

    for  split, root_original, root_expt in zip(["train", "val", "test"], \
                [args.train_dir, args.val_dir,  args.test_dir], \
                [ args.expt_train_dir, args.expt_val_dir,  args.expt_test_dir]):
        for type in ["cancer", "normal"]:
            for slide_id, patches in expt_dict[split][type].items():
                os.makedirs(os.path.join(root_expt, type, slide_id),exist_ok=True)
                for patch in patches:
                    src = os.path.join(root_original, type, slide_id, patch)
                    dest = os.path.join(root_expt, type, slide_id, patch)
                    shutil.move(src,dest) if os.path.isfile(src) else None


if __name__=="__main__":
    main()

# Usage:
# python3 move_expt.py \
#         --train_dir /ssd_scratch/cvit/piyush/LIHC/train \
#         --val_dir /ssd_scratch/cvit/piyush/LIHC/val \
#         --test_dir /ssd_scratch/cvit/piyush/LIHC/test \
#         --expt_train_dir /ssd_scratch/cvit/piyush/LIHC/train_expt \
#         --expt_val_dir /ssd_scratch/cvit/piyush/LIHC/val_expt \
#         --expt_test_dir /ssd_scratch/cvit/piyush/LIHC/test_expt \
#         --expt_json LIHC_expt.json \
