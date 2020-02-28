from torch.utils import data
import numpy as np
from glob import glob
import torch as t
from utils.util import normalize
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import zoom
import random
import cv2
from config import opt
import warnings
warnings.filterwarnings("ignore")

def mynormalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def aug_flip(x, mode):
    y = cv2.flip(x, mode)
    return y

def data_cut(img):
    x_start = random.randint(0, 8)
    y_start = random.randint(0, 8)
    z_start = random.randint(0, 8)
    cut_img = img[x_start:x_start + 48, y_start:y_start + 48, z_start:z_start + 48]
    return cut_img

def dataTranspose(img,mode):
    if mode == 1:
        return img.transpose(1, 0, 2)
    elif mode == 2:
        return img.transpose(0, 2, 1)
    elif mode == 3:
        return img.transpose(2, 1, 0)

def path_to_image_mask(paths1,paths2):
    images = []
    for path in paths1:
        image = np.load(path)
        images.append(image)
    masks = []
    for path in paths2:
        mask = np.load(path)
        masks.append(mask)
    return images, masks

def path_to_data(paths):
    dataList = []
    for path in paths:
        label = int(path[-5])
        raw_img = np.load(path)
        dataList.append([raw_img, label])
    return dataList


class clsDataLoader(data.Dataset):
    def __init__(self):
        self.img_p_path= opt.cls_train_path + "/p/"
        self.img_n_path = opt.cls_train_path + "/n/"
        self.img_path_p = glob(self.img_p_path+"*.npy")
        self.img_path_n = glob(self.img_n_path + "*.npy")

        self.p_List = path_to_data(self.img_path_p)
        self.n_List = path_to_data(self.img_path_n)

        self.balanceList = self.p_List + self.n_List
        self.file_len = len(self.balanceList)

    def __getitem__(self, index):
        img = self.balanceList[index][0]
        label = self.balanceList[index][1]

        flag = random.random()
        if flag < 0.50:
            mode = random.choice([-1, 0, 1])
            img = aug_flip(img, mode)

        flag = random.random()
        if flag < 0.50:
            mode = random.choice([1, 2, 3])
            img = dataTranspose(img, mode)

        img = data_cut(img)
        img = mynormalize(img)
        img = img * 2 - 1
        img = np.expand_dims(img, axis=0)
        img_tensor = t.from_numpy(img.astype(np.float32))
        return img_tensor, label

    def __len__(self):
        return self.file_len

class clsValDataLoader(data.Dataset):
    def __init__(self):
        self.img_p_path = opt.cls_test_path + "/p/"
        self.img_n_path = opt.cls_test_path + "/n/"
        self.img_path_p = glob(self.img_p_path+"*.npy")
        self.img_path_n = glob(self.img_n_path + "*.npy")
        self.img_path = self.img_path_p + self.img_path_n
        self.file_len = len(self.img_path)

    def __getitem__(self, index):
        img_name=self.img_path[index]
        label = int(img_name[-5])
        img = np.load(self.img_path[index])
        img = data_cut(img)
        img = mynormalize(img)
        img = img * 2 - 1
        img = np.expand_dims(img, axis=0)
        img_tensor = t.from_numpy(img.astype(np.float32))
        return img_tensor, label, index

    def __len__(self):
        return self.file_len


class SegDataLoader(data.Dataset):
    def __init__(self):
        self.img_file_path = opt.seg_train_path + "/images/"
        self.mask_file_path = opt.seg_train_path + "/masks/"
        self.img_path = glob(self.img_file_path+"*.npy")
        self.mask_path = glob(self.mask_file_path+"*.npy")

        self.image, self.mask = path_to_image_mask(self.img_path,self.mask_path)
        self.file_len = len(self.image)

    def __getitem__(self, index):
        img = self.image[index]
        mask = self.mask[index]

        flag = random.random()
        if flag < 0.50:
            mode = random.choice([-1, 0, 1])
            img = aug_flip(img, mode)
            mask = aug_flip(mask, mode)

        flag = random.random()
        if flag < 0.50:
            mode = random.choice([1, 2, 3])
            img = dataTranspose(img, mode)
            mask = dataTranspose(mask, mode)

        img = mynormalize(img)
        img = img * 2 - 1

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        img_tensor = t.from_numpy(img.astype(np.float32))
        mask_tensor = t.from_numpy(mask.astype(np.float32))
        return img_tensor, mask_tensor

    def __len__(self):
        return self.file_len

class SegvalDataLoader(data.Dataset):
    def __init__(self):
        self.img_file_path = opt.seg_test_path + "/images/"
        self.mask_file_path = opt.seg_test_path + "/masks/"
        self.img_path = glob(self.img_file_path+"*.npy")
        self.mask_path = glob(self.mask_file_path+"*.npy")
        self.image, self.mask = path_to_image_mask(self.img_path,self.mask_path)

        self.file_len = len(self.image)

    def __getitem__(self, index):
        img = self.image[index]
        mask = self.mask[index]

        img = mynormalize(img)
        img = img * 2 - 1

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        img_tensor = t.from_numpy(img.astype(np.float32))
        mask_tensor = t.from_numpy(mask.astype(np.float32))
        return img_tensor, mask_tensor

    def __len__(self):
        return self.file_len



