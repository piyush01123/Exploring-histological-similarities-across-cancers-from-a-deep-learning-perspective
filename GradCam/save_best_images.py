import json 
import cv2
import glob
import numpy as np
import shutil
import os
import argparse
overall_iou_dict  = {}
import itertools
import pandas as pd
import ast
parser = argparse.ArgumentParser(description='Process args for Classifer')
parser.add_argument("--vis_output_dir", type=str, required=True)
parser.add_argument("--organs", type=str, nargs='+', default=None, help="Class Names")
parser.add_argument("--infer_on", type=str, nargs='+', default=None, help="Class Names")
parser.add_argument("--save_dir", type=str, default='./')

args = parser.parse_args()


def get_IoU(truth_coords_dict, pred_coords_dict):
    truth_coords = truth_coords_dict['box0']
    pred_coords = pred_coords_dict['box0']
    pred_area = pred_coords[2]*pred_coords[3]
    truth_area = truth_coords[2]*truth_coords[3]
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


def cal_iou(dict1,dict2):
    iou_list = []
    iou_dict = {}
    for i in dict1.keys():
        iou_list.append(get_IoU(dict1[i],dict2[i]))
        iou_dict[i.split('/')[-1].split('.')[0]] = get_IoU(dict1[i],dict2[i])
    return [np.mean(iou_list),len(iou_list)],iou_dict



def save_best_IOU_pairwise_images(IOU,IOU_listwise,vis_output_dir):
    for k in IOU.keys():
        images = np.array(list(IOU_listwise[k].keys()))
        ious = np.array(list(IOU_listwise[k].values()))
        reqd_imgs = []
        reqd_imgs = ["_".join(i.split('/')[-1].split('_')[:5]) for i in images[ious>0.7]]
        total_img_list1 = glob.glob('{}/{}/*.png'.format(vis_output_dir,k.split('_')[0]+'_'+k.split('_')[2]))
        total_img_list2 = glob.glob('{}/{}/*.png'.format(vis_output_dir,k.split('_')[1]+'_'+k.split('_')[2]))
        reqd_img_path1 = [i for i in total_img_list1 if "_".join(i.split('/')[-1].split('_')[:5]) in reqd_imgs]
        reqd_img_path2 = [i for i in total_img_list2 if "_".join(i.split('/')[-1].split('_')[:5]) in reqd_imgs]
        patients1 = ['_'.join(i.split('/')[-1].split('_')[:5]) for i in reqd_img_path1]
        patients2 = ['_'.join(i.split('/')[-1].split('_')[:5]) for i in reqd_img_path2]
        reqd_img_path1_sorted = list(np.array(reqd_img_path1)[sorted(range(len(patients1)), key=lambda k: patients1[k])])
        reqd_img_path2_sorted = list(np.array(reqd_img_path2)[sorted(range(len(patients2)), key=lambda k: patients2[k])])
        with open('{}/pairwise_best{}.txt'.format(args.save_dir,k), 'w') as filehandle:
            json.dump(reqd_img_path1_sorted, filehandle)
        with open('{}/pairwise_best{}.txt'.format(args.save_dir,k.split('_')[1]+'_'+k.split('_')[0]+'_'+k.split('_')[2]), 'w') as f:
            json.dump(reqd_img_path2_sorted, f)


def save_best_IOU_common_images(IOU,IOU_listwise,vis_output_dir):
    images_iou= {}
    images_common = []
    
    for i,k in enumerate(IOU.keys()):
        images = np.array(list(IOU_listwise[k].keys()))
        ious = np.array(list(IOU_listwise[k].values()))
        reqd_imgs = ["_".join(i.split('/')[-1].split('_')[:5]) for i in images[ious>0.7]]
        images_iou[k] = ["_".join(i.split('/')[-1].split('_')[:5]) for i in images[ious>0.7]] 
        images_common.extend(reqd_imgs)
        images_common = list(set(images_common)&set(reqd_imgs))
    
    overall_common = []

    for k in IOU.keys():
        total_img_list1 = glob.glob('{}/{}/*.png'.format(vis_output_dir,k.split('_')[0]+'_'+k.split('_')[2]))
        total_img_list2 = glob.glob('{}/{}/*.png'.format(vis_output_dir,k.split('_')[1]+'_'+k.split('_')[2]))
        reqd_img_path1 = [i for i in total_img_list1 if "_".join(i.split('/')[-1].split('_')[:5]) in images_common]
        reqd_img_path2 = [i for i in total_img_list2 if "_".join(i.split('/')[-1].split('_')[:5]) in images_common]
        patients1 = ['_'.join(i.split('/')[-1].split('_')[:5]) for i in reqd_img_path1]
        patients2 = ['_'.join(i.split('/')[-1].split('_')[:5]) for i in reqd_img_path2]
        reqd_img_path1_sorted = list(np.array(reqd_img_path1)[sorted(range(len(patients1)), key=lambda k: patients1[k])])
        reqd_img_path2_sorted = list(np.array(reqd_img_path2)[sorted(range(len(patients2)), key=lambda k: patients2[k])])
        overall_common.extend(reqd_img_path1)
        overall_common.extend(reqd_img_path2)


    with open('{}/common_imgs_{}.txt'.format(args.save_dir,args.infer_on[0]), 'w') as f:
        json.dump(overall_common, f)


if __name__ == '__main__':
    os.makedirs(args.save_dir,exist_ok=True)
    z = itertools.combinations(args.organs, 2)
    IOU = {}
    IOU_listwise = {}
    cnt=0
    for o1 in z:
        for o2 in args.infer_on:
            dict1_exception_list = []
            dict2_exception_list = []
    
            dict1 = {}
            dict2 = {}
            for i in glob.glob("{}/{}_{}/*.txt".format(args.vis_output_dir,o1[0],o2)):
                with open(i) as json_file: 
                    try:
                        dict1[i.split('/')[-1].split('.')[0]] = json.load(json_file)
                    except Exception as e:
                        dict1_exception_list.append(i)

            for i in glob.glob("{}/{}_{}/*.txt".format(args.vis_output_dir,o1[1],o2)):
                with open(i) as json_file: 
                    try:
                        dict2[i.split('/')[-1].split('.')[0]] = json.load(json_file)
                    except Exception as e:
                        dict2_exception_list.append(i)
            for i in dict1_exception_list:
                with open(i) as json_file:
                    dict1[i.split('/')[-1].split('.')[0]] = ast.literal_eval(json_file.read().split('}')[0]+'}')

            for i in dict2_exception_list:
                with open(i) as json_file:
                    dict2[i.split('/')[-1].split('.')[0]] = ast.literal_eval(json_file.read().split('}')[0]+'}')

            try:
                IOU[o1[0] + '_' + o1[1] + '_'+ o2],IOU_listwise[o1[0] + '_' + o1[1] + '_'+ o2] = cal_iou(dict1,dict2)
                IOU[o1[1] + '_' + o1[0] + '_'+ o2],IOU_listwise[o1[1] + '_' + o1[0] + '_'+ o2] = cal_iou(dict2,dict1)
            except Exception as e:
                print(e)
                continue
    df = pd.DataFrame(IOU).T
    df.columns = ['Mean IOU','Number of Images']
    print(df)
    df.to_csv('{}/IOU_stats_{}.csv'.format(args.save_dir,args.infer_on[0]))
    save_best_IOU_pairwise_images(IOU,IOU_listwise,args.vis_output_dir)
    save_best_IOU_common_images(IOU,IOU_listwise,args.vis_output_dir) 