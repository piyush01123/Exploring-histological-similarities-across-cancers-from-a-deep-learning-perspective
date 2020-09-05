
import openslide
import numpy as np
import glob, os
import argparse
from PIL import Image
import cv2
import time
from multiprocessing import Pool
import multiprocessing
import itertools
from matplotlib import pyplot as plt


def get_connected_components(img):
    img = cv2.cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    num_labels, labels_im = cv2.connectedComponents(img)
    return num_labels,labels_im

def go_up(path, n):
    for i in range(n):
        path = os.path.dirname(path)
    return path

def get_expt_data(path,patch_size, whiteness_limit, blackness_limit, max_faulty_pixels, min_conn_comp):
    mode = go_up(path,3).split('/')[-1]
    img_save_path = path.replace(mode, mode+'_data_for_expt')
    patch_im = Image.open(path)
    patch_arr = np.swapaxes(np.array(patch_im),1,0)
    is_white = np.all([patch_arr[:,:,i]>whiteness_limit for i in range(3)], axis=0)
    is_black = np.all([patch_arr[:,:,i]<blackness_limit for i in range(3)], axis=0)
    num_labels, labels_im = get_connected_components(patch_arr)
    count = 0
    if np.sum(is_white+is_black)<=patch_size*patch_size*max_faulty_pixels and num_labels>=min_conn_comp:
        os.makedirs(go_up(img_save_path,1), exist_ok=True)
        patch_im.save(img_save_path)
    else:
        count+=1


def main():
    parser = argparse.ArgumentParser(description='Process args for patch extraction')
    parser.add_argument("--data_dir", type=str, help="Root directory", default="/ssd_scratch/cvit/MED/KIRC/SLIDES")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch Size")
    parser.add_argument("--whiteness_limit", type=int, default=230, help="Whiteness Limit")
    parser.add_argument("--blackness_limit", type=int, default=10, help="Blackness Limit")
    parser.add_argument("--max_faulty_pixels", type=float, default=0.7, help="Max allowed fraction of only B/W pixels")
    parser.add_argument("--min_conn_comp", type=int, default=10, help="Min allowed number of connected components")

    args = parser.parse_args()
    # os.makedirs(os.path.join(args.extras_dir, "thumbnails"), exist_ok=True)

    ## change this if your storage format is different
    # file_pattern = os.path.join(args.root_dir, "*.svs")
    # file_pattern = os.path.join('/ssd_scratch/cvit/ashishmenon/LUAD/train/normal/*/', "*.png")
    # file_pattern = os.path.join(args.data_dir,"/*/*.png")

    file_pattern = os.path.join(args.data_dir,"*/*/*.png")    
    file_paths = glob.glob(file_pattern)

    pool = Pool(multiprocessing.cpu_count())

    P = itertools.product(file_paths,[args.patch_size],[args.whiteness_limit],\
    [args.blackness_limit], [args.max_faulty_pixels], [args.min_conn_comp])

    pool.starmap(get_expt_data, P)



if __name__=="__main__":
    main()
