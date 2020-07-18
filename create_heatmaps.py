
import numpy as np
import glob, os
import argparse
from PIL import Image
import cv2
import time
from multiprocessing import Pool
import multiprocessing
import itertools
import pandas as pd


def create_patches(slide_id, root_dir, patch_size, dest_dir, thumbnail_dir, patch_df, record_df, slide_class, alpha):
    _,_,W,H,_= patch_df[patch_df.slide_id==slide_id].values.squeeze(0)
    im_slide = np.full((W//4,H//4,3),240,dtype=np.uint8)

    record_subset = record_df[record_df.slide_ids==slide_id]
    probs = record_subset.probs
    record_subset.probs = (probs-probs.min())/(probs.max()-probs.min())

    for x in range(0,W//4,patch_size):
        for y in range(0,H//4,patch_size):
            fp = os.path.join(root_dir_cancer, slide_id, "{}_X_{}_Y_{}.png".format(slide_id, x, y))
            if not os.path.isfile(fp):
                continue
            patch = Image.open(fp)
            patch_arr = np.swapaxes(np.array(patch),1,0)
            prob = record_subset[record_subset.paths==fp].probs.values[0]
            r, g, b = int(prob*255), int((1-prob)*255), 0
            color_tile = np.empty((patch_size,patch_size,3), dtype=np.uint8)
            color_tile[:,:,0] = r
            color_tile[:,:,1] = g
            color_tile[:,:,2] = b
            overlap_tile = (alpha*patch_arr + (1-alpha)*color_tile).astype(np.uint8)
            im_slide[x:x+patch_size,y:y+patch_size,:] = overlap_tile

    im_slide_full = Image.fromarray(np.swapaxes(im_slide,1,0))
    fp = os.path.join(dest_dir, slide_class, "{}.png".format(slide_id))
    im_slide_full.save(fp)

    im_slide_thumbnail = im_slide_full.resize((1024,int(H/W*1024)))
    fp = os.path.join(thumbnail_dir, slide_class, "{}.png".format(slide_id))
    im_slide_thumbnail.save(fp)

    print("[INFO: {}] Slide {} Done.".format(time.strftime("%d-%b-%Y %H:%M:%S"),slide_id), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Process args for creating heatmap from record file')
    parser.add_argument("--root_dir_cancer", type=str, help="Root directory of test set cancer tiles")
    parser.add_argument("--root_dir_normal", type=str, help="Root directory of test set normal tiles")
    parser.add_argument("--dest_dir", type=str, help="Destination directory for heatmaps")
    parser.add_argument("--thumbnail_dir", type=str, help="Destination directory for heatmap thumbnails")
    parser.add_argument("--level", type=int, help="0 means 40x, 1 means 20x", default=1)
    parser.add_argument("--patch_size", type=int, default=512, help="Patch Size")
    parser.add_argument("--patch_extraction_csv", type=str,, help="Patch extraction logger CSV file")
    parser.add_argument("--record_csv", type=str, help="Predictions of Patch CNN logger with slide ids")
    parser.add_argument("--alpha", type=float, default=0.75, help="Heatmap = alpha*img + (1-alpha)*color")


    args = parser.parse_args()
    pool = Pool(multiprocessing.cpu_count())
    patch_df = pd.read_csv(args.patch_extraction_csv)
    record_df = pd.read_csv(args.record_csv)

    ## change this if your storage format is different
    if args.root_dir_cancer is not None:
        os.makedirs(os.path.join(args.dest_dir,, "cancer"), exist_ok=True)
        os.makedirs(os.path.join(args.thumbnail_dir, "cancer"), exist_ok=True)
        cancer_slide_ids = os.listdir(args.root_dir_cancer)
        P = itertools.product(cancer_slide_ids, [args.root_dir_cancer],[args.patch_size],\
            [args.dest_dir], [args.thumbnail_dir], [patch_df], [record_df], ["cancer"])
        pool.starmap(create_heatmaps, P)

    if args.root_dir_normal is not None:
        os.makedirs(os.path.join(args.dest_dir,, "normal"), exist_ok=True)
        os.makedirs(os.path.join(args.thumbnail_dir, "normal"), exist_ok=True)
        normal_slide_ids = os.listdir(args.root_dir_normal)
        pool = Pool(multiprocessing.cpu_count())
        P = itertools.product(normal_slide_ids, [args.root_dir_normal],[args.patch_size],\
            [args.dest_dir], [args.thumbnail_dir], [patch_df], [record_df], ["normal"], [args.alpha])
        pool.starmap(create_heatmaps, P)



if __name__=="__main__":
    main()
