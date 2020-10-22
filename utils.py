#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from PIL import Image


class ImageFolder(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(image_paths + "*")
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, 0

    def __len__(self):
        return len(self.image_paths)


class ShapeDataset(object):

    def __init__(self,
                 image_size=32,
                 data_size=50000):

        self.image_size = image_size
        self.seed_is_set = False  # multi threaded loading
        self.channels = 3
        self.N = data_size

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        x = np.zeros((self.image_size,
                      self.image_size,
                      self.channels),
                     dtype=np.float32)

        # color
        color = tuple(np.random.random(size=[3]))

        center_x = np.random.randint(8, high=25)
        center_y = np.random.randint(8, high=25)

        size = np.random.randint(3, high=9)

        # shape
        s = np.random.randint(0, high=2)

        if s == 0:
            # circle
            cv.circle(x, (center_x, center_y), size, color, -1)
        elif s == 1:
            # square
            cv.rectangle(x, (center_x, center_y),
                         (center_x + size, center_y + size), color, -1)

        x = np.transpose(x, [2, 0, 1])
        return x, s
