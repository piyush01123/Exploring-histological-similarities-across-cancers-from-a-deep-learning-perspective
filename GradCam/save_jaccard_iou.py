import json 
import cv2
import glob
import numpy as np
import shutil
import os
import itertools
from matplotlib import pyplot as plt
from PIL import Image
import json 
from sklearn.metrics import jaccard_score
import multiprocessing
from multiprocessing import Pool
import pandas as pd
import argparse

def bb_intersection_over_union(truth_coords_og, pred_coords_og):
    truth_coords = list(truth_coords_og).copy()
    pred_coords = list(pred_coords_og).copy()
    pred_area = pred_coords[2]*pred_coords[3]
    truth_area = truth_coords[2]*truth_coords[3]
    
    truth_coords[2]+=truth_coords[0]
    truth_coords[3]+=truth_coords[1]
    
    pred_coords[2]+=pred_coords[0]
    pred_coords[3]+=pred_coords[1]
    
    # coords of intersection rectangle
    x1 = max(truth_coords[0], pred_coords[0])
    y1 = max(truth_coords[1], pred_coords[1])
    x2 = min(truth_coords[2], pred_coords[2])
    y2 = min(truth_coords[3], pred_coords[3])
    # area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # area of prediction and truth rectangles
    boxTruthArea = (truth_coords[2] - truth_coords[0] + 1) * (truth_coords[3] - truth_coords[1] + 1)
    boxPredArea = (pred_coords[2] - pred_coords[0] + 1) * (pred_coords[3] - pred_coords[1] + 1)
    # intersection over union 
    dr  = boxTruthArea+boxPredArea-interArea
    if dr==0:
        iou=1
    else:
        iou = interArea / float(boxTruthArea + boxPredArea - interArea)
    return iou



def check_list_of_lists(list_gt):
    if type(list_gt[0]) != type(list_gt):
        list_gt_up = []
        list_gt_up.append(list_gt)
    else:
        list_gt_up = list_gt
    return list_gt_up

def get_iou_from_lists(list_gt,list_pr):
    iou_list = []
    list_gt_up = check_list_of_lists(list_gt)
    list_pr_up = check_list_of_lists(list_pr)
    for i,j in itertools.product(list_pr_up,list_gt_up):
        iou_list.append(bb_intersection_over_union(i,j))
    return np.max(iou_list)
    
def get_unique_items(x):
    list_x = x.values()
    list_x_u = list(list_x for list_x,_ in itertools.groupby(list_x))
    return list_x_u

def get_area(x):
    return abs(x[2])*abs(x[3])

def remove_redundant_bb(bb_list):
    to_remove_coord = []
    for k in itertools.combinations(bb_list,2):
        iou_req = get_iou_from_lists(k[0],k[1])
        if iou_req>0:
            area1=get_area(k[0])
            area2=get_area(k[1])
            to_remove_coord.append(k[np.argmin([area1,area2])])
    for i in to_remove_coord:
        try:
            bb_list.pop(bb_list.index(i))
        except:
            continue
    return bb_list


def get_iou_from_fp(gt_fp,pr_fp):
    with open(gt_fp,'r') as f:
        s1 = json.load(f)
    gt_list = get_unique_items(s1)
    gt_list = remove_redundant_bb(gt_list)
    with open(pr_fp,'r') as f:
        s2 = json.load(f)
    pr_list = get_unique_items(s2)
    pr_list = remove_redundant_bb(pr_list)
    iou = get_iou_from_lists(gt_list,pr_list)
    return iou,gt_fp   


def get_jaccard(fp_gt,fp_pr):
    overall_jcard = []
    orig_img1 = cv2.imread(fp_gt)
    orig_img1 = cv2.resize(orig_img1, (224,224))
    orig_img2 = cv2.imread(fp_pr)
    orig_img2 = cv2.resize(orig_img2, (224,224))
    x1 = np.uint8(orig_img1/255)
    x2 = np.uint8(orig_img2/255)
    overall_jcard = jaccard_score(x2.reshape(-1), x1.reshape(-1))
    return overall_jcard,fp_gt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process args for Classifer')
    parser.add_argument("--vis_output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Class Names")
    parser.add_argument("--infer_on", type=str,default=None, help="Class Names")
    parser.add_argument("--save_dir", type=str, default='./')
    args = parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    
    gt_coordinates_fp = glob.glob(args.vis_output_dir+'/{}_{}/*.txt'.format(args.infer_on,args.infer_on))
    gt_coordinates_fp.sort()

    pr_coordinates_fp = glob.glob(args.vis_output_dir+'/{}_{}/*.txt'.format(args.model,args.infer_on))
    pr_coordinates_fp.sort()


    gt_th_img_fp = glob.glob(args.vis_output_dir+'/{}_{}/*Cam_thresholded.png'.format(args.infer_on,args.infer_on))
    gt_th_img_fp.sort()

    pr_th_img_fp = glob.glob(args.vis_output_dir+'/{}_{}/*Cam_thresholded.png'.format(args.model,args.infer_on))
    pr_th_img_fp.sort()

    pool = Pool(multiprocessing.cpu_count())
    res_iou = pool.starmap(get_iou_from_fp,zip(gt_coordinates_fp,pr_coordinates_fp))

    pool = Pool(multiprocessing.cpu_count())
    res_jacc = pool.starmap(get_jaccard,zip(gt_th_img_fp,pr_th_img_fp) )

    final_dict_jacc = {'filepath':[i[1].split('/')[-1].split('_C')[0] for i in res_jacc],'jacc':[i[0] for i in res_jacc]}
    df_jacc = pd.DataFrame(final_dict_jacc)

    final_dict_iou = {'filepath':[i[1].split('/')[-1].split('_b')[0] for i in res_iou],'iou':[i[0] for i in res_iou]}
    df_iou = pd.DataFrame(final_dict_iou)

    df_total = pd.merge(df_jacc, df_iou, on='filepath')
    df_total.to_csv(os.path.join(args.save_dir,'{}_{}.csv').format(args.model,args.infer_on))