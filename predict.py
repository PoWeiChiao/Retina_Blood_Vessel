import os
from model.ResUNet import BasicBlock, BottleNeck, ResUNet
from model.UNet import UNet
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from utils.dataset import getMaskBox

def predict(net, device, image_path, mask_path, image_transforms, crop=True, threshold=0.5):
    ltx, lty, rbx, rby = getMaskBox(mask_path)

    image = Image.open(image_path)
    if crop:
        image = image.crop((ltx, lty, rbx + 1, rby + 1))
    output_image = np.array(image, dtype=np.uint8)
    image = image_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        pred = net(image)
        pred = np.array(pred.data.cpu()[0])[0]
        pred = np.where(pred >= threshold, 255, 0)
        pred = np.array(pred, dtype=np.uint8)
        return output_image, pred

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=1, n_classes=1)
    # net = ResUNet(in_channel=1, out_channel=1, block=BottleNeck, num_block=[3, 4, 6, 3])

    net.to(device=device)
    net.load_state_dict(torch.load('saved/20210419_UNet/model.pth', map_location=device))

    data_dir = 'data/test'
    image_path = os.path.join(data_dir, 'image', '02_test.tif')
    mask_path = os.path.join(data_dir, 'mask', '02_test_mask.gif')
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    image, pred = predict(net=net, device=device, image_path=image_path, mask_path=mask_path, image_transforms=image_transforms)

if __name__ == '__main__':
    main()