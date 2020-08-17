
import numpy as np
import os
from os.path import join as opj
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torchvision

import argparse
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy import stats
import pickle
import skimage.io as skio
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution, GuidedGradCam
from captum.attr import visualization as viz


class ModImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(datasets.ImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path

class VisualizeMaps:
    def __init__(self, model, mode, out_dir, baseline=None):
        self.model = model
        self.mode = mode
        self.out_dir = out_dir
        if mode == "IntGrads":
            #self.baseline = baseline.cuda()
            self.baseline = torch.ones_like(baseline).cuda()
            self.title = "Integrated Gradients"
            self.tag = "IG"
        elif self.mode == "Occlusion":
            self.title = "Occlusion"
            self.tag = "Occ"
        elif self.mode == "GradCAM":
            self.title = "Grad CAM"
            self.tag = "GCAM"
        elif self.mode == "GuidedGradCAM":
            self.title = "Guided Grad CAM"
            self.tag = "GuGCAM"

    def visualize(self, in_tensor, out_tensor, target, paths):
        in_tensor.requires_grad_()
        if self.mode == "IntGrads":
            #print("Getting IG attributions")
            ig = IntegratedGradients(self.model)
            attr, delta = ig.attribute(in_tensor,target=0, baselines = torch.ones_like(in_tensor).cuda(), return_convergence_delta=True)
        elif self.mode == "Occlusion":
            occlusion = Occlusion(self.model)
            attr = occlusion.attribute(in_tensor, strides = (3, 4, 4), target=0, sliding_window_shapes=(3, 15, 15), baselines=0)
        elif self.mode == "GradCAM":
            layer_gc = LayerGradCam(self.model, self.model.module.layer4)
            attr = layer_gc.attribute(in_tensor, 0)
            attr = LayerAttribution.interpolate(attr, (in_tensor.size()[2], in_tensor.size()[3]))
        elif self.mode == "GuidedGradCAM":
            layer_gc = GuidedGradCam(self.model, self.model.module.layer4)
            attr = layer_gc.attribute(in_tensor, 0)

        attr = attr.detach().cpu().numpy()
        #print("Attribution shapes: ", attr.shape)
        original_image = in_tensor.cpu().detach().numpy()
        for idx in range(in_tensor.size()[0]):
            #print("Images: ", attr[idx,...].shape, original_image[idx,...].shape)
            y_pred = out_tensor[idx,0]
            y_true = target[idx]
            #print("Shapes: ", out_tensor.size(), target.size(), y_pred.shape, y_true.shape)
            suffix_true = "Cancer" if y_true == 0 else "Normal"
            suffix_pred = "Cancer" if y_pred == 0 else "Normal"
            fig, ax = viz.visualize_image_attr_multiple(np.swapaxes(np.swapaxes( attr[idx,...],0,2),0,1), np.swapaxes(np.swapaxes(original_image[idx,...],0,2),0,1), methods=["original_image", "blended_heat_map"],signs=["all", "positive"], show_colorbar=True, titles=["Original: "+suffix_true, self.title+":"+suffix_pred], use_pyplot=False)
            #fig, ax = viz.visualize_image_attr_multiple(np.swapaxes(np.swapaxes( attr[idx,...],0,2),0,1), np.swapaxes(np.swapaxes(original_image[idx,...],0,2),0,1), method="blended_heat_map",sign="positive", show_colorbar=True, title=self.title, use_pyplot=False)
            path = paths[idx]
            can_norm = path.split('/')[-3]
            folder_name = path.split('/')[-2]
            file_name = path.split('/')[-1]
            file_name = file_name.split('.')[0]
            #print("Dump: ", can_norm, folder_name, file_name)
            if not os.path.exists( opj(self.out_dir, "../viz", can_norm, folder_name) ):
                os.makedirs(opj(self.out_dir, "../viz", can_norm, folder_name))
            fig.savefig( opj(self.out_dir, "../viz", can_norm, folder_name, file_name+'_'+self.tag+'.png'), bbox_inches='tight')
            """
            # Save the original image as well.
            img = np.swapaxes(np.swapaxes(original_image[idx,...],0,2),0,1)
            img = 255*(img - img.min())/(img.max() - img.min())
            img = img.astype(np.uint8)
            skio.imsave( opj(self.out_dir, "../viz", can_norm, folder_name, file_name+'_Orig.png'), img)
            """

def get_stratified_sampler(img_dataset):
    imgs, targets, num_classes = img_dataset.imgs, np.array(img_dataset.targets), len(img_dataset.classes)
    class_counts = np.empty(num_classes,)
    for i in range(num_classes):
        class_counts[i] = sum(targets==i)
    class_weights = sum(class_counts)/class_counts # N/count not count/N
    sample_weights = torch.empty((len(imgs),), dtype=torch.double)
    for i in range(num_classes):
        sample_weights[targets==i] = class_weights[i]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler, class_weights

def test(model, test_dataloader, device, writer, args, baseline=None):
    record_csv = args.record_csv
    test_dir = args.test_dir
    visu_mode = args.visu_mode
    NUM_IMGS = args.num_images
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    print("Y=0-->CANCER, Y=1-->NORMAL", flush=True)
    df = pd.DataFrame(columns=['paths', 'slide_ids', 'targets', 'preds', 'probs'])
    visMaps = VisualizeMaps(model, visu_mode, test_dir, baseline)
    with torch.no_grad():
        img_idx = 0
        for batch_id, (data, targets, paths) in enumerate(test_dataloader):

            data, targets = data.to(device), targets.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            #print("Test data shapes:",data.size(), targets.size() )
            # Call the visualization code and visualize the activation maps
            visMaps.visualize(data, preds, targets, paths)
            pred_class_probs = probs[torch.arange(len(preds))[:,None],preds].squeeze(1)
            df_batch = pd.DataFrame({'paths': list(paths),
                                     'slide_ids': [p.split('/')[-2] for p in paths],
                                     'targets': targets.tolist(),
                                     'preds': preds.squeeze(1).tolist(),
                                     'probs': pred_class_probs.tolist(),
                                    })
            df = df.append(df_batch)
            y_pred.append(output.argmax(dim=1))
            y_true.append(targets)
            correct += preds.eq(targets.view_as(preds)).sum().item()
            #print("Act. and Pred. : ", y_true, y_pred, list(paths), targets.tolist())
            print('[{}/{}] Done'.format((batch_id+1)*len(data),len(test_dataloader.dataset)), flush=True)
            img_idx = img_idx + len(data)
            if img_idx >= NUM_IMGS:
                break
    test_acc = 100.*correct/NUM_IMGS
    print("Test set: Accuracy: {}/{} ({:.2f}%)".format(correct, NUM_IMGS, test_acc), flush=True)

    y_true, y_pred = torch.cat(y_true).cpu().numpy(), torch.cat(y_pred).cpu().numpy()
    classes = test_dataloader.dataset.classes
    report = classification_report(y_true, y_pred, labels=[0,1], target_names=classes, output_dict=True, digits=4)
    print(report, flush=True)
    df.to_csv(record_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description='Process args for Classifer')
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--record_csv", type=str, required=True)
    parser.add_argument("--visu_mode", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=16)


    args = parser.parse_args()
    print(args, flush=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    baseline = None

    if args.visu_mode == "IntGrads":
        if os.path.isfile(os.path.join(args.test_dir, "int_grads_baseline.pkl") ):
            baseline = pickle.load( open(os.path.join(args.test_dir, "int_grads_baseline.pkl"), "rb") )
            print("Loaded saved int grads baseline.")
        else:
            train_data_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
                ])
            train_dataset = datasets.ImageFolder(root=args.train_dir, transform=train_data_transform)
            sampler, class_weights = get_stratified_sampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

            # Generate gradients baseline
            baseline = torch.Tensor(1,3,args.image_size,args.image_size)
            for batch_id, (data, target) in enumerate(train_dataloader):
                #print("Data shapes: ",data.size(), torch.sum(data, dim=0).size(), baseline.size(), target.size())
                baseline = baseline + torch.sum(data, dim=0)

            pickle.dump(baseline, open(os.path.join(args.test_dir, "int_grads_baseline.pkl"), "wb"))

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ])
    test_dataset = ModImageFolder(root=args.test_dir, transform=data_transform)
    nw = 4 if torch.cuda.is_available() else 0
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=nw, shuffle=False)
    #print("Number of classes: ", len(test_dataset.classes), test_dataloader.dataset.classes)
    model = models.resnet18()
    model.fc = nn.Sequential(
                nn.Dropout(p=0.2), # p is prob. of a neuron not being dropped out
                nn.Linear(model.fc.in_features, len(test_dataset.classes))
                )

    ckpt = torch.load(args.model_checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    #model = nn.DataParallel(model)
    model.load_state_dict(ckpt, strict=False)
    print("[MSG] Model loaded from {}".format(args.model_checkpoint), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE {}".format(device), flush=True)
    model = nn.DataParallel(model).to(device)
    #print(model.module.layer4)
    test(model, test_dataloader, device, writer, args, baseline)


if __name__=="__main__":
    main()





