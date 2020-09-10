

import argparse
import os
import glob
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_train_dir", required=True)
    parser.add_argument("--expt_val_dir", required=True)
    parser.add_argument("--expt_test_dir", required=True)
    parser.add_argument("--subtype", required=True)
    args = parser.parse_args()

    expt_dict = {split: {type: {} for type in ["cancer", "normal"]} for split in ["train", "val", "test"]}
    for  split, root in zip(["train", "val", "test"], [args.expt_train_dir, args.expt_val_dir,  args.expt_test_dir]):
        for type in ["cancer", "normal"]:
            slide_ids = os.listdir(os.path.join(root,type))
            for slide_id in slide_ids:
                patches = os.listdir(os.path.join(root, type, slide_id))
                expt_dict[split][type][slide_id] = sorted(patches)

    with open("{}_expt.json".format(args.subtype), 'w') as fh:
        json.dump(expt_dict, fh)

if __name__=="__main__":
    main()

# Usage:
# python3 list_expt.py \
#         --expt_train_dir /ssd_scratch/cvit/piyush/LIHC/train_data_for_expt \
#         --expt_val_dir /ssd_scratch/cvit/piyush/LIHC/val_data_for_expt \
#         --expt_test_dir /ssd_scratch/cvit/piyush/LIHC/test_data_for_expt \
#         --subtype LIHC
