
# Code to find optimum batch size for feature extraction
"""
Results:
Feature Extraction Optimal Batch Size Test for N = 12800 images
---------------------------------------------------------------
200.0 batches of batch_size=64 processing time = 84.56591486930847 seconds
100.0 batches of batch_size=128 processing time = 85.99241518974304 seconds
50.0 batches of batch_size=256 processing time = 95.34101510047913 seconds
25.0 batches of batch_size=512 processing time = 102.53731894493103 seconds
So for 600,000 images total time = 84/12800*600000/3600 hours = 1.03 hours
"""


from torchvision import models, datasets, transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import os, glob

# Test on 12800 images
N = 12800


def test(model, dataset, device, batch_size):
    # TEST on 12800 examples
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    N = dataloader.dataset.size
    model.eval()
    output = np.empty((N, 512))
    tit = time.time()
    for i, (batch, _) in enumerate(dataloader):
        batch = batch.to(device)
        out = model(batch)
        out = out.reshape((-1, 512))
        output[i*batch_size : (i+1)*batch_size] = out.cpu().numpy()
        np.save("/ssd_scratch/cvit/piyush/features_batch_{}.npy".format(i), out.cpu().numpy())
    np.save("/ssd_scratch/cvit/piyush/features.npy", output)
    tat = time.time()
    print("{} batches of batch_size={} processing time = {} seconds".format(N/batch_size,\
          batch_size, tat-tit), flush=True)


def main():
    print("Feature Extraction Optimal Batch Size Test for N = 12800 images", flush=True)
    model = models.resnet18(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, \
            model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool)
    for param in model.parameters():
      param.requires_grad = False
    model = model.to(device)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = datasets.FakeData(size=N, image_size=(3,224,224), num_classes=2, transform=transform)
    for batch_size in [64, 128, 256, 512]:
        test(model, dataset, device, batch_size)
        for f in glob.glob("*.npy"):
          os.remove(f)



if __name__=="__main__":
    main()
