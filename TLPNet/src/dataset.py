import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from os import listdir
from os.path import join
import torch
import cv2
import math


def rotate_image(images, angle=90, img_scale=1.0, color=(0, 0, 0)):
    """
    rotate with angle, background filled with color, default black (0, 0, 0)
    label_box = (cls_type, box)
    box = [x0, y0, x1, y1, x2, y2, x3, y3]
    """
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise),
    # then grab the sine and cosine (i.e., the rotation components of the matrix)
    # if angle < 0, counterclockwise rotation; if angle > 0, clockwise rotation
    # 1.0 - scale, to adjust the size scale (image scaling parameter), recommended 0.75
    height_ori, width_ori = images[0].shape[:2]
    x_center_ori, y_center_ori = (width_ori // 2, height_ori // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        (x_center_ori, y_center_ori), angle, img_scale)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # compute the new bounding dimensions of the image
    width_new = int((height_ori * sin) + (width_ori * cos))
    height_new = int((height_ori * cos) + (width_ori * sin))

    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (width_new / 2) - x_center_ori
    rotation_matrix[1, 2] += (height_new / 2) - y_center_ori

    # perform the actual rotation and return the image
    # borderValue - color to fill missing background, default black, customizable
    image_news = []
    for image in images:
        image_new = cv2.warpAffine(
            image, rotation_matrix, (width_new, height_new), borderValue=color)
        image_news.append(image_new)
    # img_info_list = x_center_ori, y_center_ori, width_new, height_new

    # x_center_ori, y_center_ori, width_new, height_new = img_info_list
    # each point coordinates
    # angle = angle / 180 * math.pi
    # box_rot_list = cal_rotate_box(
    #     label_box_list, angle, (x_center_ori, y_center_ori), (width_new//2, height_new//2))
    # # box_new_list = []
    # # for box_rot in box_rot_list:
    # for index in range(len(box_rot_list)//2):
    #     box_rot_list[index*2] = int(box_rot_list[index*2])
    #     box_rot_list[index*2] = max(min(box_rot_list[index*2], width_new), 0)
    #     box_rot_list[index*2+1] = int(box_rot_list[index*2+1])
    #     box_rot_list[index*2 +
    #                  1] = max(min(box_rot_list[index*2+1], height_new), 0)

    return image_news  # , box_rot_list


def cal_rotate_box(box_list, angle, ori_center, new_center):
    # box = [x0, y0, x1, y1, x2, y2, x3, y3]
    # image_shape - [width, height]
    # box_list_new = []
    # for box in box_list:
    box_new = []
    for index in range(len(box_list)//2):
        box_new.extend(cal_rotate_coordinate(
            box_list[index*2], box_list[index*2+1], angle, ori_center, new_center))
    # label_box = box_new
    # box_list_new.append(label_box)
    return box_new  # box_list_new


def cal_rotate_coordinate(x_ori, y_ori, angle, ori_center, new_center):
    # box = [x0, y0, x1, y1, x2, y2, x3, y3]
    # image_shape - [width, height]
    x_0 = x_ori - ori_center[0]
    y_0 = ori_center[1] - y_ori
    x_new = x_0 * math.cos(angle) - y_0 * math.sin(angle) + new_center[0]
    y_new = new_center[1] - (y_0 * math.cos(angle) + x_0 * math.sin(angle))
    return (x_new, y_new)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DatasetIC1519Ens(Dataset):
    def __init__(self,  dataset_dir, image_size=768, is_for_train=True):
        super(DatasetIC1519Ens, self).__init__()
        self.image_size = image_size
        if is_for_train:
            self.TE_img_dir = os.path.join(dataset_dir, "TE_img")
            self.HF_img_dir = os.path.join(dataset_dir, "Heatmap_fin")
        else:
            self.TE_img_dir = os.path.join(
                dataset_dir.replace('1_tra', '1_val'), "TE_img")
            self.HF_img_dir = os.path.join(
                dataset_dir.replace('1_tra', '1_val'), "Heatmap_fin")

        self.HF_img_filenames = [x for x in listdir(
            self.HF_img_dir) if is_image_file(x)]
        # transforms.RandomResizedCrop(self.image_size),
        transform_list = [transforms.ColorJitter(
            brightness=0.6, contrast=0.5, saturation=0.4, hue=0.4), transforms.RandomGrayscale(0.05), transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        transform_list_mask = [transforms.ToTensor()]
        self.transform_mask = transforms.Compose(transform_list_mask)
        self.transform_Flip = transforms.RandomHorizontalFlip()

    def __getitem__(self, index):

        HF_img = cv2.imread(
            join(self.HF_img_dir, self.HF_img_filenames[index]), 0)
        TE_img = cv2.imread(
            join(self.TE_img_dir, self.HF_img_filenames[index].replace('.png', '.jpg')))

        rot = np.random.randint(-20, 20)
        [TE_img, HF_img] = rotate_image([TE_img, HF_img], rot)
        # img_height, img_width = TE_img.shape[0:2]
        H, W = TE_img.shape[0:2]

        if H > W:
            re_H = self.image_size
            re_W = int(((self.image_size/H)*W//16)*16)
        else:
            re_W = self.image_size
            re_H = int(((self.image_size/W)*H//16)*16)
        HF_img = cv2.resize(HF_img, (re_W, re_H))
        TE_img = cv2.resize(TE_img, (re_W, re_H))
        img_height, img_width = re_H, re_W

        TE_img_padding = np.zeros(
            (self.image_size, self.image_size, 3), np.uint8)
        TE_img_padding[:img_height, :img_width, :3] = TE_img
        HF_img_padding = np.zeros(
            (self.image_size, self.image_size, 1), np.uint8)

        HF_img_padding[:img_height, :img_width, 0] = HF_img

        if np.sum(HF_img/255) > 10:
            loc = np.where(HF_img > 0)
            miny, minx = np.min(loc[0]), np.min(loc[1])
            maxy, maxx = np.max(loc[0]), np.max(loc[1])
            t_W, t_H = maxx - minx, maxy - miny
            miny, minx = miny-t_H, minx-t_W
            maxy, maxx = maxy+t_H, maxx+t_W

            minx = 1 if minx <= 0 else minx
            miny = 1 if miny <= 0 else miny
            maxx = self.image_size - 1 if maxx >= self.image_size else maxx
            maxy = self.image_size - 1 if maxy >= self.image_size else maxy

            xa = np.random.randint(0, minx)
            ya = np.random.randint(0, miny)
            xb = np.random.randint(maxx, self.image_size)
            yb = np.random.randint(maxy, self.image_size)
        else:
            xa = 0
            ya = 0
            xb = 768
            yb = 768

        HF_img = HF_img_padding[ya:yb, xa:xb]
        TE_img = TE_img_padding[ya:yb, xa:xb, ...]
        HF_img = cv2.resize(HF_img, (640, 640))
        TE_img = cv2.resize(TE_img, (640, 640))

        # print('1', HF_img.shape)
        HF_img = Image.fromarray(HF_img, mode="L")
        TE_img = Image.fromarray(cv2.cvtColor(
            TE_img, cv2.COLOR_BGR2RGB), mode="RGB")

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        HF_img = self.transform_Flip(HF_img)
        HF_tensor = self.transform_mask(HF_img)
        torch.random.manual_seed(seed)
        TE_img = self.transform_Flip(TE_img)
        TE_tensor = self.transform(TE_img)

        # TE_tensor_pad = TE_tensor.new_full(
        #     (3, self.image_size, self.image_size), -1)
        # TE_tensor_pad[:TE_tensor.shape[0], : TE_tensor.shape[1],
        #               : TE_tensor.shape[2]].copy_(TE_tensor)

        # HF_tensor_pad = HF_tensor.new_full(
        #     (1, self.image_size, self.image_size), 0)
        # HF_tensor_pad[:HF_tensor.shape[0], : HF_tensor.shape[1],
        #               : HF_tensor.shape[2]].copy_(HF_tensor)

        return {'TE': TE_tensor, 'HF_GT': HF_tensor}

    def __len__(self):
        return len(self.HF_img_filenames)


class DatasetIC1519Ens_val(Dataset):
    def __init__(self,  dataset_dir, image_size=768):
        super(DatasetIC1519Ens_val, self).__init__()
        self.image_size = image_size

        self.TE_img_dir = os.path.join(dataset_dir, "TE_img")
        self.HF_img_dir = os.path.join(dataset_dir, "Heatmap_fin")

        self.HF_img_filenames = [x for x in listdir(
            self.HF_img_dir) if is_image_file(x)]

        transform_list = [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        transform_list_mask = [transforms.ToTensor()]
        self.transform_mask = transforms.Compose(transform_list_mask)
        # self.transform_Flip = transforms.RandomHorizontalFlip()

    def __getitem__(self, index):

        HF_img = cv2.imread(
            join(self.HF_img_dir, self.HF_img_filenames[index]), 0)
        TE_img = cv2.imread(
            join(self.TE_img_dir, self.HF_img_filenames[index].replace('.png', '.jpg')))

        # rot = np.random.randint(-15, 15)
        # [TE_img, HF_img] = rotate_image([TE_img, HF_img], rot)
        # img_height, img_width = TE_img.shape[0:2]
        H, W = TE_img.shape[0:2]

        if H > W:
            re_H = self.image_size
            re_W = int(((self.image_size/H)*W//16)*16)
        else:
            re_W = self.image_size
            re_H = int(((self.image_size/W)*H//16)*16)
        HF_img = cv2.resize(HF_img, (re_W, re_H))
        TE_img = cv2.resize(TE_img, (re_W, re_H))
        img_height, img_width = re_H, re_W

        TE_img_padding = np.zeros(
            (self.image_size, self.image_size, 3), np.uint8)
        TE_img_padding[:img_height, :img_width, :3] = TE_img
        HF_img_padding = np.zeros(
            (self.image_size, self.image_size), np.uint8)

        HF_img_padding[:img_height, :img_width] = HF_img
        # print(HF_img_padding.shape)
        # if np.sum(HF_img/255) > 10:
        #     loc = np.where(HF_img > 0)
        #     miny, minx = np.min(loc[0]), np.min(loc[1])
        #     maxy, maxx = np.max(loc[0]), np.max(loc[1])
        #     t_W, t_H = maxx - minx, maxy - miny
        #     miny, minx = miny-t_H, minx-t_W
        #     maxy, maxx = maxy+t_H, maxx+t_W

        #     minx = 1 if minx <= 0 else minx
        #     miny = 1 if miny <= 0 else miny
        #     maxx = self.image_size - 1 if maxx >= self.image_size else maxx
        #     maxy = self.image_size - 1 if maxy >= self.image_size else maxy

        #     xa = np.random.randint(0, minx)
        #     ya = np.random.randint(0, miny)
        #     xb = np.random.randint(maxx, self.image_size)
        #     yb = np.random.randint(maxy, self.image_size)
        # else:
        #     xa = 0
        #     ya = 0
        #     xb = 768
        #     yb = 768

        # HF_img = HF_img_padding[ya:yb, xa:xb]
        # TE_img = TE_img_padding[ya:yb, xa:xb, ...]
        # HF_img = cv2.resize(HF_img, (640, 640))
        # TE_img = cv2.resize(TE_img, (640, 640))

        HF_img = Image.fromarray(HF_img_padding, mode="L")
        TE_img = Image.fromarray(cv2.cvtColor(
            TE_img_padding, cv2.COLOR_BGR2RGB), mode="RGB")

        # seed = torch.random.seed()
        # torch.random.manual_seed(seed)
        # HF_img = self.transform_Flip(HF_img)
        HF_tensor = self.transform_mask(HF_img)
        # torch.random.manual_seed(seed)
        # TE_img = self.transform_Flip(TE_img)
        TE_tensor = self.transform(TE_img)

        return {'TE': TE_tensor, 'HF_GT': HF_tensor}

    def __len__(self):
        return len(self.HF_img_filenames)
