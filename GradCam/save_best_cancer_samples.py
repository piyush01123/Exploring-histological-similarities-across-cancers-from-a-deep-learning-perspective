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
import pandas as pd
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Process args for Classifer')
parser.add_argument("--test_dir", type=str, required=True)
parser.add_argument("--model_checkpoint", type=str, required=True)
parser.add_argument("--hparam_json", type=str, required=True)
parser.add_argument("--save_dir", type=str, default='/ssd_scratch/cvit/ashishmenon/unknown/results_test/')
parser.add_argument("--model_chosen", type=str, default='BRCA')

args = parser.parse_args()


class ModImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(datasets.ImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path

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

def save_best_cancerous_samples(model,test_dataset,organ):
    correct = 0
    y_pred = []
    y_true = []
    score_all =[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024,num_workers=4)
    df = pd.DataFrame(columns=['paths', 'slide_ids', 'targets', 'preds', 'probs'])
    with torch.no_grad():
        for batch_id, (data, targets, paths) in enumerate(dataloader):
                data, targets = data.to(device), targets.to(device)
                output = model(data)
                score_all.append(output)
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

                pred_class_probs = probs[torch.arange(len(preds))[:,None],preds].squeeze(1)
                df_batch = pd.DataFrame({'paths': list(paths),
                                         'slide_ids': [p.split('/')[-2] for p in paths],
                                         'targets': targets.tolist(),
                                         'preds': preds.squeeze(1).tolist(),
                                         'probs': pred_class_probs.tolist(),
                                        })
                df = df.append(df_batch)
    df_sorted = df.iloc[np.argsort(np.array(list(df['probs'])))]
    df_filtered = df_sorted[df_sorted['probs']>0.98]
    df_filtered_cancer = df_filtered[df_filtered['slide_ids'].apply(lambda x:x.split('-')[3][:2]=='01')]
    img_list_high_prob_cancer = df_filtered_cancer[df_filtered_cancer['targets'] == df_filtered_cancer['preds']]
    img_list_high_prob_cancer.to_csv('{}/{}_best_cancer_samples_train.csv'.format(args.save_dir,organ))


if __name__=='__main__':
    data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
    ])

    test_dataset = ModImageFolder(root=args.test_dir, transform=data_transform)

    hparams = json.load(open(args.hparam_json, 'r'))[args.model_chosen]
    dropouts,hidden_layer_units,optimizer_name,lr = get_hyperpara(hparams)
    model = define_model(dropouts,hidden_layer_units,num_classes=2)

    class_ids = [int(i[1]) for i in test_dataset.samples]
    reqd_indices = np.where(np.array(class_ids)==0)[0] #to get cancer samples only
    test_dataset_subset  = Subset(test_dataset,reqd_indices)
    
    ckpt = torch.load(args.model_checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    save_best_cancerous_samples(model,test_dataset_subset,args.model_chosen)