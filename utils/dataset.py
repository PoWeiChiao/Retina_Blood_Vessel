import glob
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RetinaDataset(Dataset):
    def __init__(self, image_dir, label_dir, mask_dir, crop=True, image_transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.crop = crop
        self.image_transforms = image_transforms

        self.image_list = glob.glob(os.path.join(image_dir, '*.tif'))
        self.label_list = glob.glob(os.path.join(label_dir, '*.gif'))
        self.mask_list = glob.glob(os.path.join(mask_dir, '*.gif'))

        self.image_list.sort()
        self.label_list.sort()
        self.mask_list.sort()  

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = Image.open(self.label_list[idx])

        if self.crop:
            ltx, lty, rbx, rby = getMaskBox(self.mask_list[idx])
            image = image.crop((ltx, lty, rbx + 1, rby + 1))
            label = label.crop((ltx, lty, rbx + 1, rby + 1))

        if self.image_transforms is not None:
            image = self.image_transforms(image)
            label = self.image_transforms(label)

        return image, label

def getMaskBox(mask_path):
    mask = Image.open(mask_path)
    mask = np.array(mask)
    y, x = np.where(mask == 255)
    ltx = np.min(x)
    lty = np.min(y)
    rbx = np.max(x)        
    rby = np.max(y)
    return ltx, lty, rbx, rby

def main():
    image_dir = os.path.join('D:/pytorch/Segmentation/Retina_Blood_Vessel/data/train', 'image')
    label_dir = os.path.join('D:/pytorch/Segmentation/Retina_Blood_Vessel/data/train', 'label')
    mask_dir = os.path.join('D:/pytorch/Segmentation/Retina_Blood_Vessel/data/train', 'mask')
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    dataset = RetinaDataset(image_dir, label_dir, mask_dir, image_transforms)
    print(dataset.__len__())

if __name__ == '__main__':
    main()