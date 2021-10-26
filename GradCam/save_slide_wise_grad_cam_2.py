import openslide
import numpy as np
import glob, os
import argparse
from PIL import Image
import subprocess
import shutil
import pandas as pd
import itertools
from multiprocessing import Pool
import multiprocessing
def command_reqd(infer_on,folder):
    c = ["bash","selective_rsync.sh","{}".format(infer_on),"{}".format(folder)]
    subprocess.call(c)

def save_slidewise_gradcam(patient,visualization_fp,model,infer_on):
    print(patient,visualization_fp,model,infer_on)
    command_reqd(infer_on,patient)
    slide_fp = glob.glob('/ssd_scratch/cvit/ashishmenon/{}/*{}*'.format(infer_on,patient))
    patch_fp = glob.glob('/ssd_scratch/cvit/ashishmenon/test_patches/{}/{}'.format(infer_on,patient))
    slide = openslide.OpenSlide(slide_fp[0])
    os.makedirs('/ssd_scratch/cvit/ashishmenon/slide_thumbnails/{}/'.format(infer_on),exist_ok=True)
    slide.get_thumbnail((1024,1024)).save('/ssd_scratch/cvit/ashishmenon/slide_thumbnails/{}/{}.png'.format(infer_on,patient))
    W,H = slide.level_dimensions[0]
    patch_size = 512
    patch_size_upscaled = 4*patch_size
    im_slide = np.ones((W//4,H//4,3),dtype=np.uint8)*240
    for i in glob.glob('{}/*.png'.format(patch_fp[0])):
        patch_pil = Image.open(i).convert('RGB')
        patch_pil_224 = patch_pil.resize((224,224))
        x = int(i.split('_X_')[-1].split('_Y')[0])
        y = int(i.split('_X_')[-1].split('_Y_')[-1].split('.')[0])
        path_pil_resized = patch_pil_224.resize((patch_size,patch_size))
        patch_arr = np.swapaxes(np.array(path_pil_resized),1,0)
        im_slide[x:x+patch_size,y:y+patch_size,:] = patch_arr
    
    im_slide_gt= im_slide.copy()
    
    for k in glob.glob('{}/{}_{}/*{}*'.format(visualization_fp,model,infer_on,patient)):
        if '_Cam_On_Image.png' in k:
            cam_patch = Image.open(k).convert('RGB')
            cam_patch = cam_patch.resize((patch_size,patch_size))
            patch_arr = np.swapaxes(np.array(cam_patch),1,0)
            x = int(k.split('_X_')[-1].split('_Y')[0])
            y = int(k.split('_X_')[-1].split('_Y_')[-1].split('_')[0])
            im_slide[x:x+patch_size,y:y+patch_size,:] = patch_arr
    for l in glob.glob('{}/{}_{}/*{}*'.format(visualization_fp,infer_on,infer_on,patient)):
        if '_Cam_On_Image.png' in l:
            cam_patch = Image.open(l).convert('RGB')
            cam_patch = cam_patch.resize((patch_size,patch_size))
            patch_arr = np.swapaxes(np.array(cam_patch),1,0)
            x = int(l.split('_X_')[-1].split('_Y')[0])
            y = int(l.split('_X_')[-1].split('_Y_')[-1].split('_')[0])
            im_slide_gt[x:x+patch_size,y:y+patch_size,:] = patch_arr
    shutil.rmtree(patch_fp[0])
    os.remove(slide_fp[0])
    os.makedirs("/ssd_scratch/cvit/ashishmenon/saved_slides_gradcam_heatmap/{}_{}".format(model,infer_on),exist_ok=True)
    img_pil_pr = Image.fromarray(np.swapaxes(im_slide[::8,::8],1,0))
    img_pil_gt = Image.fromarray(np.swapaxes(im_slide_gt[::8,::8],1,0))
    img_pil_pr.save("/ssd_scratch/cvit/ashishmenon/saved_slides_gradcam_heatmap/{}_{}/{}_pr.png".format(model,infer_on,patient))
    img_pil_gt.save("/ssd_scratch/cvit/ashishmenon/saved_slides_gradcam_heatmap/{}_{}/{}_gt.png".format(model,infer_on,patient))


def get_top_k_iou(df,k=1000,asc=False):
    df = df.iloc[:,1:]
    df_sorted = df.sort_values('iou',ascending=asc)
    iou = list(df_sorted['iou'])
    return df_sorted.iloc[:k,:],np.mean(iou[:k])

def get_top_k_jacc(df,k=1000,asc=False):
    df = df.iloc[:,1:]
    df_sorted = df.sort_values('jacc',ascending=asc)
    jacc = list(df_sorted['jacc'])   
    return df_sorted.iloc[:k,:],np.mean(jacc[:k])





if __name__=='__main__':
    organs = ['BRCA','COAD', 'KICH','KIRP','KIRC','LIHC','PRAD','READ', 'STAD', 'LUAD','LUSC']
    best_list = [["COAD","LIHC", "READ"],
    ["BRCA","LIHC","READ"],
    ["COAD","KIRP" , "LIHC" , "READ"],
    ["COAD" ,"KICH"  ,"LIHC"  , "READ"],

    ["BRCA" , "KICH" , "KIRP" , "LIHC"],
    ["BRCA" , "COAD" ,"READ"],
    ["BRCA","READ","LIHC"],
    ["COAD"  , "LIHC"],
    ["BRCA" , "COAD" , "LIHC" , "READ"],

    ["BRCA" , "COAD" , "LIHC" , "LUSC" , "READ"],
    ["BRCA" , "LIHC" , "LUAD" , "READ"]]


    worst_list = [["KICH","KIRC"],
    ["KICH","KIRC"],
    ["KIRC","STAD"],
    ["PRAD","STAD"],
    ["PRAD","STAD"],
    ["KICH","KIRC"],
    ["KICH","KIRC"],
    ["KICH","KIRC"],
    ["KICH","KIRC"],
    ["KICH","KIRC"],
    ["KICH","KIRC"]]

    best_list_dict = {}
    for i in range(len(organs)):
        best_list_dict[organs[i]] = [organs[i]+'_'+j for j in best_list[i]]
    worst_list_dict = {}
    for i in range(len(organs)):
        worst_list_dict[organs[i]] = [organs[i]+'_'+j for j in worst_list[i]]
    worst_list_dict



    all_csv = glob.glob('/ssd_scratch/cvit/ashishmenon/IOU_inferences/*.csv')
    organ_wise_best_k_iou = {}
    organ_wise_best_k_iou_fp = {}
    organ_wise_best_k_jacc = {}
    organ_wise_best_k_jacc_fp = {}
    for i in organs:
        for k in all_csv:
            if k.split('/')[-1].split('.')[0].split('_')[0] ==i and k.split('/')[-1].split('.')[0] in best_list_dict[i]:
                df = pd.read_csv(k)
                df_sorted_iou,iou_val = get_top_k_iou(df,k=max(1000,int(0.5*len(df))))
                df_sorted_jacc,jacc_val = get_top_k_jacc(df,k=max(1000,int(0.5*len(df))))
                organ_wise_best_k_iou[k.split('/')[-1].split('.')[0]] = [iou_val]
                organ_wise_best_k_iou_fp[k.split('/')[-1].split('.')[0]] = list(df_sorted_iou['filepath'])
                organ_wise_best_k_jacc[k.split('/')[-1].split('.')[0]] = [jacc_val]
                organ_wise_best_k_jacc_fp[k.split('/')[-1].split('.')[0]] = list(df_sorted_jacc['filepath'])



    all_csv = glob.glob('/ssd_scratch/cvit/ashishmenon/IOU_inferences/*.csv')
    organ_wise_worst_k_iou = {}
    organ_wise_worst_k_iou_fp = {}
    organ_wise_worst_k_jacc = {}
    organ_wise_worst_k_jacc_fp = {}
    for i in organs:
        for k in all_csv:
            if k.split('/')[-1].split('.')[0].split('_')[0] ==i and k.split('/')[-1].split('.')[0] in worst_list_dict[i]:
                df = pd.read_csv(k)
                df_sorted_iou,iou_val = get_top_k_iou(df.copy(),k=int(0.8*len(df)),asc=True)
                df_sorted_jacc,jacc_val = get_top_k_jacc(df.copy(),k=int(0.8*len(df)),asc=True)
                organ_wise_worst_k_iou[k.split('/')[-1].split('.')[0]] = [iou_val]
                organ_wise_worst_k_iou_fp[k.split('/')[-1].split('.')[0]] = list(df_sorted_iou['filepath'])
                organ_wise_worst_k_jacc[k.split('/')[-1].split('.')[0]] = [jacc_val]
                organ_wise_worst_k_jacc_fp[k.split('/')[-1].split('.')[0]] = list(df_sorted_jacc['filepath'])

    
    for i in organ_wise_worst_k_iou_fp.keys():
        patients = list(set([i.split('_X')[0] for i in organ_wise_worst_k_iou_fp[i]]))
        visual_fp = '/ssd_scratch/cvit/ashishmenon/visualization_output_4'
        model = i.split('_')[0]
        infer_on = i.split('_')[1]
        # pool = Pool(multiprocessing.cpu_count())
        # P = itertools.product(patients,[visual_fp],[model],[infer_on])
        # pool.starmap(save_slidewise_gradcam, P)
        for j in patients:
            save_slidewise_gradcam(j,visual_fp,model,infer_on)

    # for i in organ_wise_best_k_iou_fp.keys():
    #     patients = list(set([i.split('_X')[0] for i in organ_wise_best_k_iou_fp[i]]))
    #     visual_fp = '/ssd_scratch/cvit/ashishmenon/visualization_output_4'
    #     model = i.split('_')[0]
    #     infer_on = i.split('_')[1]
    #     pool = Pool(multiprocessing.cpu_count())
    #     P = itertools.product(patients,[visual_fp],[model],[infer_on])
    #     pool.starmap(save_slidewise_gradcam, P)
        # for i in patients:
        #     save_slidewise_gradcam(i,visual_fp,model,infer_on)




