#!/usr/bin/env python
# encoding: utf-8


import torch.utils.data as data
import torchvision
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tif'])


def image_to_tensor():
    return Compose([
        ToTensor(),
    ])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')

    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_HR, image_dir_x2, image_dir_TextGt, image_dir_x4_input):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames_HR = [join(image_dir_HR, x) for x in listdir(image_dir_HR) if is_image_file(x)]
        self.image_filenames_x2 = [join(image_dir_x2, x) for x in listdir(image_dir_x2) if
                                   is_image_file(x)]
        self.image_dir_TextGt = [join(image_dir_TextGt, x) for x in listdir(image_dir_TextGt) if
                                 is_image_file(x)]
        self.image_filenames_x4_input = [join(image_dir_x4_input, x) for x in listdir(image_dir_x4_input) if
                                         is_image_file(x)]
        self.image_to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        x8 = load_img(self.image_filenames_HR[index])
        x8 = self.image_to_tensor(x8)

        x2 = load_img(self.image_filenames_x2[index])
        x2 = self.image_to_tensor(x2)

        textGt = load_img(self.image_dir_TextGt[index])
        textGt = self.image_to_tensor(textGt)

        x4_input = load_img(self.image_filenames_x4_input[index])
        x4_input = self.image_to_tensor(x4_input)

        return x4_input, x2, x8, textGt

    def __len__(self):
        return len(self.image_filenames_x2)
