import glob
import random
import xml.etree.cElementTree as ET
from os import path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from utils.augmentations import horisontal_flip


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


class ListDataset(Dataset):
    def __init__(self, image_set, img_size=416, augment=True, multiscale=True,
                 binary_class=False,
                 dataset_dir='/home/nico/Dataset/Fabric-Final/'):

        with open(path.join(dataset_dir, 'ImageSets', 'All', image_set + '.txt'), "r") as file:
            self.image_ids = file.read().splitlines()

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.binary_class = binary_class
        self.dataset_dir = dataset_dir

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        img_path = path.join(self.dataset_dir, 'Images', self.image_ids[index] + '.jpg')

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        xml_path = path.join(self.dataset_dir, 'Annotations', 'xmls', self.image_ids[index] + '.xml')

        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for xmlbox in root.iter('bbox'):
            if self.binary_class:
                cls_idx = 0  # index start from 0
            else:
                cls_idx = int(root.find('pattern').text) - 1  # index start from 0
            # Extract coordinates for unpadded + unscaled image
            x1 = float(xmlbox.find('xmin').text)
            x2 = float(xmlbox.find('xmax').text)
            y1 = float(xmlbox.find('ymin').text)
            y2 = float(xmlbox.find('ymax').text)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            x = ((x1 + x2) / 2) / padded_w
            y = ((y1 + y2) / 2) / padded_h
            w = (x2 - x1) / padded_w
            h = (y2 - y1) / padded_h
            box = (cls_idx, x, y, w, h)
            boxes.append(box)
        boxes = torch.from_numpy(np.array(boxes, dtype=np.float32).reshape(-1, 5))
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.image_ids)
