
import argparse
import os
import glob
import json
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtype")
    parser.add_argument("--root_dir")
    parser.add_argument("--train_dir")
    parser.add_argument("--val_dir")
    parser.add_argument("--test_dir")
    parser.add_argument("--ratio", nargs='+', type=float)
    args = parser.parse_args()
    print(args, flush=True)

    dictionary = {partition: {"cancer": [], "normal": []} for partition in ["train","val","test"]}
    for cls in ["cancer", "normal"]:
        slide_ids = os.listdir(os.path.join(args.root_dir, cls))
        N = len(slide_ids)
        train_slide_ids = slide_ids[:int(.7*N)]
        val_slide_ids = slide_ids[int(.7*N):int(.9*N)]
        test_slide_ids = slide_ids[int(.9*N):]

        dictionary['train'][cls] = train_slide_ids
        dictionary['val'][cls] = val_slide_ids
        dictionary['test'][cls] = test_slide_ids

        os.mkdir(os.path.join(args.train_dir, cls))
        os.mkdir(os.path.join(args.val_dir, cls))
        os.mkdir(os.path.join(args.test_dir, cls))

        for slide_id in train_slide_ids:
            src = os.path.join(args.root_dir, cls, slide_id)
            dest = os.path.join(args.train_dir, cls)
            shutil.move(src, dest)

        for slide_id in val_slide_ids:
            src = os.path.join(args.root_dir, cls, slide_id)
            dest = os.path.join(args.val_dir, cls)
            shutil.move(src, dest)

        for slide_id in test_slide_ids:
            src = os.path.join(args.root_dir, cls, slide_id)
            dest = os.path.join(args.test_dir, cls)
            shutil.move(src, dest)

    with open('{}_train_val_test.json'.format(args.subtype), 'w') as f:
        json.dump(dictionary, f)



if __name__=="__main__":
    main()
