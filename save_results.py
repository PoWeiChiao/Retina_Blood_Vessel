import cv2 as cv
import glob
from model.DeepLab_ResNet import DeepLabv3_plus
from model.ResUNet import BasicBlock, BottleNeck, ResUNet
from model.UNet import UNet
import numpy as np
import os
from PIL import Image
from predict import predict
import torch
from torchvision import transforms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = DeepLabv3_plus(in_channels=1, n_classes=1, os=16)
    # net = UNet(n_channels=1, n_classes=1)
    # net = ResUNet(in_channel=1, out_channel=1, block=BottleNeck, num_block=[3, 4, 6 ,3])
    net.to(device=device)
    net.load_state_dict(torch.load('model.pth', map_location=device))

    data_dir = 'data/test'
    image_dir = os.path.join(data_dir, 'image', '*.tif')
    mask_dir = os.path.join(data_dir, 'mask', '*.gif')

    image_list = glob.glob(image_dir)
    mask_list = glob.glob(mask_dir)

    image_list.sort()
    mask_list.sort()

    save_dir = 'predict'
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    for i in range(len(image_list)):
        image, pred = predict(net=net, device=device, image_path=image_list[i], mask_path=mask_list[i], image_transforms=image_transforms, crop=False, threshold=0.5)
        image = image[:, :, ::-1] # rgb to bgr
        result = np.concatenate((image, (np.stack((pred,)*3, axis=-1))), axis=1)
        cv.imwrite(os.path.join(save_dir, os.path.basename(image_list[i])[:-4] + '.jpg'), result)
        print(os.path.join(save_dir, os.path.basename(image_list[i])[:-4] + '.jpg'))

if __name__ == "__main__":
    main()
