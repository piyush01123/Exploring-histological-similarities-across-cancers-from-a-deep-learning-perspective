import os
import numpy as np
import glob, os
import argparse
from PIL import Image
import cv2
import time
from multiprocessing import Pool
import multiprocessing
import itertools
from torchvision import datasets, models, transforms
import torch
from torchvision import models
from vis_utils import preprocess_image,save_class_activation_images
import torch.nn as nn
import json
from torch.utils.data import Subset
parser = argparse.ArgumentParser(description='Process args for Classifer')
parser.add_argument("--test_dir", type=str, required=True)
parser.add_argument("--model_checkpoint", type=str, required=True)
parser.add_argument("--hparam_json", type=str, required=True)
parser.add_argument("--save_dir", type=str, default='/ssd_scratch/cvit/ashishmenon/unknown/results_test/')
parser.add_argument("--model_chosen", type=str, default='BRCA')
parser.add_argument("--inferred_on", type=str, default='BRCA')

args = parser.parse_args()



def get_hyperpara(hparams):
    dropouts = []
    hidden_layer_units = []
    for i in range(hparams['n_layers']):
        dropouts.append(hparams['dropout_l{}'.format(i)])
        hidden_layer_units.append(hparams['n_units_l{}'.format(i)])
    optimizer = hparams['optimizer']
    lr = hparams['lr']
    return dropouts,hidden_layer_units,optimizer,lr


def define_model(dropouts,hidden_layer_units,num_classes=2):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    layers = []
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    for i in range(len(dropouts)):
        out_features = hidden_layer_units[i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = dropouts[i]
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*layers)
    return model

class ModImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(datasets.ImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path



class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model._modules.items():
            try:
                x = module(x)  # Forward
            except:
                x = x.view(-1)
                x = module(x)
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x
    

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        return conv_output, x
    
class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model,target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, dataset_obj):

        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        input_image = dataset_obj[0].view(1,3,224,224)
        target_class = dataset_obj[1]
        conv_output, model_output = self.extractor.forward_pass(input_image)
        model_output = model_output.view(1,-1)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam,target_class    


def main():
    print(args, flush=True)

    os.makedirs(args.save_dir,exist_ok=True)

    
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ])

    test_dataset = ModImageFolder(root=args.test_dir, transform=data_transform)

    samples = [i[1] for i in test_dataset.samples]
    cancer_indices = np.where(np.array(samples)==0)[0]
    # cancer_indices = np.random.choice(cancer_indices,min(len(cancer_indices),1000),replace=False)
    cancer_indices = cancer_indices[:min(len(cancer_indices),1000)]
    normal_indices = np.where(np.array(samples)==1)[0]
    # normal_indices = np.random.choice(normal_indices,min(len(normal_indices),1000),replace=False)
    normal_indices = normal_indices[:min(len(normal_indices),1000)]
    # cancer_dataset = Subset(test_dataset, cancer_indices)
    # normal_dataset = Subset(test_dataset, normal_indices)
    combined_dataset = Subset(test_dataset, np.array(list(cancer_indices)+list(normal_indices)))

    hparams = json.load(open(args.hparam_json, 'r'))[args.model_chosen]
    dropouts,hidden_layer_units,optimizer_name,lr = get_hyperpara(hparams)
    model = define_model(dropouts,hidden_layer_units,num_classes=2)

    ckpt = torch.load(args.model_checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)

    print(model)

    grad_cam = GradCam(model, target_layer="layer4")
    # with Pool(processes = multiprocessing.cpu_count()) as pool:
    #     pool.map(save_grad_cam_results, test_dataset)

    for i in combined_dataset:
        cam_out,pred = grad_cam.generate_cam(i)
        original_image = Image.open(i[2]).convert('RGB')
        original_image = original_image.resize((224,224,))
        save_class_activation_images(original_image, cam_out, args.save_dir , i[2].split('/')[-1].split('.')[0],test_dataset.classes[i[1]])

if __name__ == '__main__':

    # Get params
    main()