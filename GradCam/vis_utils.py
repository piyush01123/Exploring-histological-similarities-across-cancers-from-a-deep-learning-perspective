import os
import copy
import numpy as np
from PIL import Image, ImageFilter,ImageOps
import matplotlib.cm as mpl_color_map
import cv2
import torch
from torch.autograd import Variable
from torchvision import models
import json 
import random as rng
from save_jaccard_iou import remove_redundant_bb
def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, result_dir, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join(result_dir, file_name + '.jpg')
    save_image(gradient, path_to_file)


# def save_class_activation_images(org_img, activation_map, results_dir,file_name,sample_type):
#     """
#         Saves cam activation map and activation map on the original image
#     Args:
#         org_img (PIL img): Original image
#         activation_map (numpy arr): Activation map (grayscale) 0-255
#         file_name (str): File name of the exported image
#     """
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
#     os.makedirs(os.path.join(results_dir),exist_ok=True)
#     # os.makedirs(os.path.join(results_dir,sample_type,"Cam_HeatMap"),exist_ok=True)
#     # os.makedirs(os.path.join(results_dir,sample_type,"Cam_On_Image"),exist_ok=True)
#     # os.makedirs(os.path.join(results_dir,sample_type,"Cam_Grayscale"),exist_ok=True)
#     # os.makedirs(os.path.join(results_dir,sample_type,"bounding_box"),exist_ok=True)

#     # Grayscale activation map
#     heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'brg')
#     # Save colored heatmap
#     # path_to_file = os.path.join(results_dir,sample_type, "Cam_HeatMap" , file_name+'.png')
#     path_to_file = os.path.join(results_dir,file_name+"_Cam_HeatMap"+'.png')
#     save_image(heatmap, path_to_file)
#     # # Save heatmap on iamge
#     path_to_file = os.path.join(results_dir,file_name+"_Cam_On_Image"+'.png')
#     save_image(heatmap_on_image, path_to_file)
#     # # SAve grayscale heatmap
#     # path_to_file = os.path.join(results_dir,sample_type,"Cam_Grayscale" ,file_name+'.png')
#     # path_to_file = os.path.join(results_dir,file_name+"_Cam_Grayscale"+'.png')
#     # save_image(activation_map, path_to_file)
#     #
#     heatmap = heatmap.convert('RGB')
#     heatmap = np.array(heatmap)
#     heatmap = heatmap[:, :, ::-1].copy() 
    
#     orig_img_cv2 = np.array(org_img.resize((224,224)).convert('RGB'))[:,:,::-1].copy()

#     # grey_img = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
#     # thresh = cv2.threshold(grey_img,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    

#     hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)

#     ## mask of green (36,25,25) ~ (86, 255,255)
#     # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
#     mask = cv2.inRange(hsv, (20, 25, 25), (70, 255,255))

#     ## slice the green
#     imask = mask>0
#     green = np.zeros_like(heatmap, np.uint8)
#     green[imask] = heatmap[imask]
#     grey_img = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(grey_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#     path_to_file = os.path.join(results_dir,file_name+'_Cam_thresholded.png')
#     cv2.imwrite(path_to_file,thresh)

#     bb_img,bb_coordinates = get_bb(orig_img_cv2,thresh)

#     path_to_json = os.path.join(results_dir,file_name+"_bb_box_coordinates"+'.txt')
    
#     with open(path_to_json, 'w') as outfile:
#         json.dump(bb_coordinates, outfile)

#     path_to_bb_img = os.path.join(results_dir,file_name+"_bb_box_image"+'.png')
#     cv2.imwrite(path_to_bb_img,bb_img)



def save_class_activation_images(org_img, activation_map, results_dir,inferred_on,file_name):
    cam_on_img_dir = os.path.join(results_dir,"gc_simp",inferred_on)
    os.makedirs(cam_on_img_dir,exist_ok=True)
    cam_dir = os.path.join(results_dir,"gradcam",inferred_on)
    os.makedirs(cam_dir,exist_ok=True)
    cam_th_dir = os.path.join(results_dir,"gc_th",inferred_on)
    os.makedirs(cam_th_dir,exist_ok=True)
    cam_bb_box_dir = os.path.join(results_dir,"gc_bb_box",inferred_on)
    os.makedirs(cam_bb_box_dir,exist_ok=True)
    
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    # os.makedirs(os.path.join(results_dir,sample_type,"Cam_HeatMap"),exist_ok=True)
    # os.makedirs(os.path.join(results_dir,sample_type,"Cam_On_Image"),exist_ok=True)
    # os.makedirs(os.path.join(results_dir,sample_type,"Cam_Grayscale"),exist_ok=True)
    # os.makedirs(os.path.join(results_dir,sample_type,"bounding_box"),exist_ok=True)

    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'brg')
    # Save colored heatmap
    # path_to_file = os.path.join(results_dir,sample_type, "Cam_HeatMap" , file_name+'.png')
    path_to_file = os.path.join(cam_dir,file_name+'.png')
    save_image(heatmap, path_to_file)
    # # Save heatmap on iamge
    path_to_file = os.path.join(cam_on_img_dir,file_name+'.png')
    save_image(heatmap_on_image, path_to_file)
    # # SAve grayscale heatmap
    # path_to_file = os.path.join(results_dir,sample_type,"Cam_Grayscale" ,file_name+'.png')
    # path_to_file = os.path.join(results_dir,file_name+"_Cam_Grayscale"+'.png')
    # save_image(activation_map, path_to_file)
    #
    
    heatmap = heatmap.convert('RGB')
    heatmap = np.array(heatmap)
    heatmap = heatmap[:, :, ::-1].copy() 
    
    orig_img_cv2 = np.array(org_img.resize((224,224)).convert('RGB'))[:,:,::-1].copy()

    # grey_img = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(grey_img,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    

    hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv2.inRange(hsv, (20, 25, 25), (70, 255,255))

    ## slice the green
    imask = mask>0
    green = np.zeros_like(heatmap, np.uint8)
    green[imask] = heatmap[imask]
    grey_img = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    path_to_file = os.path.join(cam_th_dir,file_name+'.png')
    cv2.imwrite(path_to_file,thresh)

    bb_img,bb_coordinates = get_bb(orig_img_cv2,thresh)

    path_to_json = os.path.join(cam_bb_box_dir,file_name+'.txt')
    
    with open(path_to_json, 'w') as outfile:
        json.dump(bb_coordinates, outfile)

    path_to_bb_img = os.path.join(cam_bb_box_dir,file_name+'.png')
    cv2.imwrite(path_to_bb_img,bb_img)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap

    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, mean,std,resize_im=True,centre_crop=True):

    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]


    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((256, 256), Image.ANTIALIAS)
    if centre_crop:
        left = (256 - 224)/2
        top = (256 - 224)/2
        right = (256 + 224)/2
        bottom = (256 + 224)/2
        pil_im = pil_im.crop((left, top, right, bottom))

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var,mean,var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-mean[0], -mean[1], -mean[2]]
    reverse_std = [1/std[0], 1/std[1], 1/std[2]]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im



def get_bb(orig_img,thresh):
    threshold = 150
    canny_output = cv2.Canny(thresh, threshold, threshold * 2)

    try:
        _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    except:
        contours,_ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    boundRect = [tuple(x) for x in set(tuple(x) for x in boundRect)]
    boundRect = remove_redundant_bb(boundRect)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    boxes = {}
    for i in range(len(boundRect)):
        color = (0, 255, 0)
        x = int(boundRect[i][0])
        y = int(boundRect[i][1])
        w = int(boundRect[i][2])
        h = int(boundRect[i][3])
        boxes['box{}'.format(i)] = [x,y,w,h]
        cv2.rectangle(orig_img, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    
    return orig_img,boxes
