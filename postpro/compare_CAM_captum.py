


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torchvision
from PIL import Image
import argparse
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image, ImageFile
import json
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution, GuidedGradCam
from captum.attr import visualization as viz
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.ioff()

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


class UnNormalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--model1_checkpoint", required=True)
    parser.add_argument("--model2_checkpoint", required=True)
    parser.add_argument("--model1_organ", required=True)
    parser.add_argument("--model2_organ", required=True)
    parser.add_argument("--hparam_json", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    print(args,flush=True)
    assert os.path.isfile(args.img_path), "Image Does not exist"
    assert os.path.isfile(args.model1_checkpoint), "CheckPoint does not exist"
    assert os.path.isfile(args.model2_checkpoint), "CheckPoint does not exist"

    data_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ])

    data_transform_inv = transforms.Compose([
        UnNormalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186]),
        transforms.ToPILImage(),
        ])

    img = Image.open(args.img_path)
    X = data_transform(img).unsqueeze(0)

    hparams = json.load(open(args.hparam_json, 'r'))[args.model1_organ]
    dropouts,hidden_layer_units,optimizer_name,lr = get_hyperpara(hparams)
    model1 = define_model(dropouts,hidden_layer_units)
    ckpt = torch.load(args.model1_checkpoint, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model1.load_state_dict(ckpt)
    print("[MSG] Model 1 loaded from {}".format(args.model1_checkpoint), flush=True)

    model1.eval()
    output = model1(X)
    probs = F.softmax(output, dim=1)
    preds = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

    if args.method=="GradCam":
        layer_gc = LayerGradCam(model1, model1.layer4)
        attr = layer_gc.attribute(X, preds)
        attr = LayerAttribution.interpolate(attr, (X.size()[2], X.size()[3]))
        attrib = attr.detach().numpy()
        original_image = UnNormalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])(X).detach().numpy()
        fig, ax = viz.visualize_image_attr(np.swapaxes(attrib[0],0,2), np.swapaxes(original_image[0],0,2), "blended_heat_map")
        plt.close('all')
        fig.savefig("captum_gcam.jpg")

    elif args.method=="IntGrad":
        ig = IntegratedGradients(model1)
        attr, delta = ig.attribute(X,target=preds[:,0], baselines = torch.ones_like(X), return_convergence_delta=True)
        attrib = attr.detach().numpy()
        original_image = UnNormalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])(X).detach().numpy()
        # fig, ax = viz.visualize_image_attr(np.swapaxes(attrib[0],0,2), np.swapaxes(original_image[0],0,2), "blended_heat_map")
        # fig, ax = viz.visualize_image_attr(attrib[0], original_image[0],"blended_heat_map")
        fig, ax = viz.visualize_image_attr(np.swapaxes(np.swapaxes(attrib[0],0,2),0,1), np.swapaxes(np.swapaxes(original_image[0],0,2),0,1), "blended_heat_map")
        plt.close('all')
        fig.savefig("captum_ig.jpg")

    elif args.method=="Occ":
        occlusion = Occlusion(model1)
        print("HERE?")
        attr = occlusion.attribute(X, strides = (3, 4, 4), target=preds[:,0], sliding_window_shapes=(3, 15, 15), baselines=0)
        print("OR HERE?")
        attrib = attr.detach().numpy()
        original_image = X.detach().numpy()
        fig, ax = viz.visualize_image_attr(np.swapaxes(attrib[0],0,2), np.swapaxes(original_image[0],0,2), "blended_heat_map")
        plt.close('all')
        fig.savefig("captum_occ.jpg")

    elif args.method=="GuGradCam":
        layer_gc = GuidedGradCam(model1, model1.layer4)
        attr = layer_gc.attribute(X, preds[:,0])
        attrib = attr.detach().numpy()
        original_image = X.detach().numpy()
        fig, ax = viz.visualize_image_attr(np.swapaxes(attrib[0],0,2), np.swapaxes(original_image[0],0,2), "blended_heat_map")
        plt.close('all')
        fig.savefig("captum_guided_gcam.jpg")

    else:
        raise ValueError("Invalid Method.")


    # hparams = json.load(open(args.hparam_json, 'r'))[args.model2_organ]
    # dropouts,hidden_layer_units,optimizer_name,lr = get_hyperpara(hparams)
    # model2 = define_model(dropouts,hidden_layer_units)
    # ckpt = torch.load(args.model2_checkpoint, map_location='cpu')
    # ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    # model2.load_state_dict(ckpt)
    # print("[MSG] Model 2 loaded from {}".format(args.model2_checkpoint), flush=True)
    #
    # model2.eval()
    # output = model2(X)
    # probs = F.softmax(output, dim=1)
    # preds = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    #
    # layer_gc = LayerGradCam(model2, model2.layer4)
    # attr = layer_gc.attribute(X, preds)
    # attr = LayerAttribution.interpolate(attr, (X.size()[2], X.size()[3]))
    #
    # attrib = attr.detach().numpy()
    # original_image = X.detach().numpy()
    #
    # fig, ax = viz.visualize_image_attr(np.swapaxes(attrib[0],0,2), np.swapaxes(original_image[0],0,2), "blended_heat_map")
    # fig.savefig("captum_test2.jpg")





if __name__=="__main__":
    main()

"""
python3 compare_CAM_captum.py \
--img_path ~/Music/TCGA-A1-A0SB-01A-01-BS1/TCGA-A1-A0SB-01A-01-BS1_X_19968_Y_1536.png \
--model1_checkpoint ~/Music/BRCA_best_model.pth \
--model1_organ BRCA \
--model2_checkpoint ~/Music/COAD_best_model.pth \
--model2_organ COAD \
--hparam_json best_hparams.json \
--method=GradCam
"""
