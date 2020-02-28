from torch.utils import data
import numpy as np
from glob import glob
import torch as t
from utils.util import normalize
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import zoom
import random
import cv2
import warnings
warnings.filterwarnings("ignore")

def mynormalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def resample(x, new_shape):
    scale_factor = new_shape / max(x.shape)
    if scale_factor < 1.0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = zoom(x, scale_factor, mode="nearest", order=1)

    pad = []
    for i in x.shape:
        pad_i = new_shape - i
        pad.append([pad_i // 2, pad_i - pad_i // 2])
    x = np.pad(x, pad, mode="constant", constant_values=-1)
    return x

def sifit_negsample(neg_list,png_list):
    negsample_list = []
    sifit_list = [file[-25:-8] for file in png_list]
    for negfile in neg_list:
        negname = negfile[-25:-8]
        if negname in sifit_list:
            negsample_list.append(negfile)
    return negsample_list

def aug_flip(x, mode):
    y = cv2.flip(x, mode)
    return y

def data_seg_cut(img,mask):
    x_start = random.randint(0, 8)
    y_start = random.randint(0, 8)
    z_start = random.randint(0, 8)
    cut_img = img[x_start:x_start + 48, y_start:y_start + 48, z_start:z_start + 48]
    cut_mask = mask[x_start:x_start + 48, y_start:y_start + 48, z_start:z_start + 48]
    return cut_img,cut_mask

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
        self.img_p_path="./data/cls_3d/train_p_midsize56/"
        self.img_n_path = "./data/cls_3d/train_n_midsize56/"
        self.img_path_p = glob(self.img_p_path+"*.npy")
        self.img_path_n = glob(self.img_n_path + "*.npy")

        self.p_List = path_to_data(self.img_path_p)
        self.n_List = path_to_data(self.img_path_n)
        #数据平衡
        self.p_List = self.p_List * int((len(self.n_List) / len(self.p_List))*0.7)

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
        self.img_p_path="./data/cls_3d/train_p_bigsize56/"
        self.img_n_path = "./data/cls_3d/train_n_bigsize56/"
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
        #img = resample(img, new_shape=48)
        img = np.expand_dims(img, axis=0)
        img_tensor = t.from_numpy(img.astype(np.float32))
        return img_tensor, label, index

    def __len__(self):
        return self.file_len


class SegDataLoader(data.Dataset):
    def __init__(self):
        self.img_file_path="./data/cls_3d/seg_train/images/"
        self.mask_file_path="./data/cls_3d/seg_train/masks/"
        self.img_path = glob(self.img_file_path+"*.npy")
        self.mask_path = glob(self.mask_file_path+"*.npy")

        self.image, self.mask = path_to_image_mask(self.img_path,self.mask_path)

        self.image = self.image * 6
        self.mask = self.mask * 6
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

        img, mask = data_seg_cut(img,mask)
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
        self.img_file_path="./data/cls_3d/seg_validate/images/"
        self.mask_file_path="./data/cls_3d/seg_validate/masks/"
        self.img_path = glob(self.img_file_path+"*.npy")
        self.mask_path = glob(self.mask_file_path+"*.npy")
        self.image, self.mask = path_to_image_mask(self.img_path,self.mask_path)

        self.file_len = len(self.image)

    def __getitem__(self, index):
        img = self.image[index]
        mask = self.mask[index]

        img, mask = data_seg_cut(img, mask)
        img = mynormalize(img)
        img = img * 2 - 1

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        img_tensor = t.from_numpy(img.astype(np.float32))
        mask_tensor = t.from_numpy(mask.astype(np.float32))
        return img_tensor, mask_tensor

    def __len__(self):
        return self.file_len



