from torchvision import transforms, datasets
import numpy as np, torch
tdir = '/ssd_scratch/cvit/piyush/KIRC/train'
image_size=224
t = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
    ])
img_dataset = datasets.ImageFolder(root=tdir, transform=t)
imgs, targets, num_classes = img_dataset.imgs, np.array(img_dataset.targets), len(img_dataset.classes)
class_weights = np.empty(num_classes,)
for i in range(num_classes):
    class_weights[i] = sum(targets==i)
class_weights = class_weights/sum(class_weights)
sample_weights = torch.empty((len(imgs),), dtype=torch.double)
for i in range(num_classes):
    sample_weights[targets==i] = class_weights[i]
sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
