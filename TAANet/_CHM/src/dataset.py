import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import cv2
import cfg
import math


def rotate_image(images, label_box_list, angle=90, img_scale=1.0, color=(0, 0, 0)):
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

    # each point coordinates
    angle = angle / 180 * math.pi
    box_rot_list = cal_rotate_box(
        label_box_list, angle, (x_center_ori, y_center_ori), (width_new//2, height_new//2))
    for index in range(len(box_rot_list)//2):
        box_rot_list[index*2] = int(box_rot_list[index*2])
        box_rot_list[index*2] = max(min(box_rot_list[index*2], width_new), 0)
        box_rot_list[index*2+1] = int(box_rot_list[index*2+1])
        box_rot_list[index*2 +
                     1] = max(min(box_rot_list[index*2+1], height_new), 0)

    return image_news, box_rot_list


def cal_rotate_box(box_list, angle, ori_center, new_center):
    box_new = []
    for index in range(len(box_list)//2):
        box_new.extend(cal_rotate_coordinate(
            box_list[index*2], box_list[index*2+1], angle, ori_center, new_center))
    return box_new


def cal_rotate_coordinate(x_ori, y_ori, angle, ori_center, new_center):
    # box = [x0, y0, x1, y1, x2, y2, x3, y3]
    # image_shape - [width, height]
    x_0 = x_ori - ori_center[0]
    y_0 = ori_center[1] - y_ori
    x_new = x_0 * math.cos(angle) - y_0 * math.sin(angle) + new_center[0]
    y_new = new_center[1] - (y_0 * math.cos(angle) + x_0 * math.sin(angle))
    return (x_new, y_new)


def croppaste(canvas, tar_img, quad):
    quad = np.array(quad, np.int32).reshape((4, 2))
    img_height, img_width = canvas.shape[0:2]
    mask = np.zeros((img_height, img_width, 1), np.uint8)
    cv2.fillConvexPoly(mask, quad, (255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel)
    mask = mask.reshape((img_height, img_width, 1))
    dst = np.where(mask > 0, tar_img, canvas)
    return dst


def Kmeans(img, k):
    data = img.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    compactness, labels, centers = cv2.kmeans(
        data, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    # 图像转换回uint8二维类型
    centers = np.uint8(centers)
    labels = labels.flatten()
    res = centers[labels.flatten()]
    dst = res.reshape((img.shape))
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst


def get_datasetlist(src_txt, text_loc):
    gt_lines_txt = open(os.path.os.path.join(src_txt),
                        mode='r', encoding="UTF-8-sig")
    gt_lines = gt_lines_txt.readlines()
    pixel_data_list = []
    gterase_data_list = []
    gterase_dataset_dict = {}
    img_name = 'img_?'
    for i, line in enumerate(gt_lines[:]):  # 0:30
        if len(line) == 14 or len(line) == 13 or len(line) == 12:
            for line in pixel_data_list:  # 过滤无用img
                if line['pixel'] == 1:
                    gterase_dataset_dict[img_name] = gterase_data_list
                    break
            gterase_data_list = []
        else:
            line_parts = line.strip().split(",")
            quad = list(map(int, list(map(int, line_parts[0:text_loc]))))
            img_name = str(line_parts[text_loc])
            # index = int(line_parts[text_loc+1])
            text = str(line_parts[text_loc+2])
            pixel = int(line_parts[text_loc+3])
            erase = int(line_parts[text_loc+4])
            oneline_dict = {'img_name': img_name, 'quad': quad,
                            'text': text, 'pixel': pixel, 'erase': erase}  # 'index': index,
            if erase == 1 and pixel == 1:  # 过滤出useful的line
                pixel_data_list.append(oneline_dict)
            # elif erase == 1 and pixel != 1:
            if erase == 1:
                # img_name = oneline_dict.pop('img_name')
                gterase_data_list.append(oneline_dict)
    return pixel_data_list, gterase_dataset_dict


class DecompST4CHM(Dataset):
    def __init__(self, dataset_dir, image_size=512):
        super(DecompST4CHM, self).__init__()
        self.image_size = image_size
        self.src_img_dir = os.path.join(dataset_dir, "src")
        self.pixel_img_dir = os.path.join(dataset_dir, "text_pixel")
        self.mask_img_dir = os.path.join(dataset_dir, "stroke_mask")
        self.erase_img_dir = os.path.join(dataset_dir, "text_erased")
        self.src_txt = os.path.join(dataset_dir, "annotation.txt")

        self.pixel_dataset_list, self.gterase_dataset_dict = get_datasetlist(
            self.src_txt, 8)

        transform_list = [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.8  0.1
        self.transform = transforms.Compose(transform_list)

        transform_list_aug = [transforms.ColorJitter(
            brightness=cfg.brightness, contrast=cfg.contrast, saturation=cfg.saturation, hue=cfg.hue), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.8  0.1  0.8  0.2
        self.transform_aug = transforms.Compose(transform_list_aug)
        transform_list_mask = [transforms.ToTensor()]
        self.transform_mask = transforms.Compose(transform_list_mask)

    def __getitem__(self, index):
        img_name = self.pixel_dataset_list[index]['img_name']
        four_pts = self.pixel_dataset_list[index]['quad']

        aug_len = len(
            self.gterase_dataset_dict[self.pixel_dataset_list[index]['img_name']])

        src_img = cv2.imread(os.path.join(
            self.src_img_dir, img_name+'.jpg'))
        img_height, img_width = src_img.shape[0:2]
        pixel_img = cv2.imread(os.path.join(
            self.pixel_img_dir, img_name+'.png'))
        mask_img = cv2.imread(os.path.join(
            self.mask_img_dir, img_name+'.png'), 0)
        erase_img = cv2.imread(os.path.join(
            self.erase_img_dir, img_name+'.jpg'))

        ########## prepare bg and gt #############
        bg_img = croppaste(src_img, erase_img, four_pts)
        gt_img = src_img.copy()
        if aug_len > 0:
            aug_num = random.randint(1, aug_len)
            aug_lines = random.sample(
                self.gterase_dataset_dict[self.pixel_dataset_list[index]['img_name']], aug_num)
            for aug_line in aug_lines:
                aug_pts = aug_line['quad']
                aug_quad = np.array(aug_pts, np.float32).reshape((4, 2))
                bg_img = croppaste(bg_img, erase_img, aug_quad)
                gt_img = croppaste(gt_img, erase_img, aug_quad)
        gt_img = croppaste(gt_img, src_img, four_pts)

############# rotation and crop#############################################
        rot = np.random.randint(-20, 20)  # (-15, 15)
        [src_img, pixel_img, mask_img, bg_img, gt_img], four_pts = rotate_image(
            [src_img, pixel_img, mask_img, bg_img, gt_img], four_pts, rot)

        img_height, img_width = src_img.shape[0:2]
        # print(four_pts)
        pts_x = four_pts[0:8:2]
        pts_y = four_pts[1:8:2]
        t_W, t_H = max(pts_x)-min(pts_x), max(pts_y)-min(pts_y)
        interval = max(t_W, t_H)
        x1 = np.random.randint(0, max(min(pts_x)-interval, 1))
        x2 = np.random.randint(
            min(max(pts_x)+interval, img_width), img_width+1)
        y1 = np.random.randint(0, max(min(pts_y)-interval, 1))
        y2 = np.random.randint(
            min(max(pts_y)+interval, img_height), img_height+1)

        src_img = src_img[y1:y2, x1:x2, ...]
        pixel_img = pixel_img[y1:y2, x1:x2, ...]
        mask_img = mask_img[y1:y2, x1:x2, ...]
        bg_img = bg_img[y1:y2, x1:x2, ...]
        gt_img = gt_img[y1:y2, x1:x2, ...]

        four_pts = np.array(four_pts, np.float32).reshape((4, 2))
        four_pts = four_pts-np.array([x1, y1], np.float32).reshape((-1, 2))

########################################################################
        H, W, = src_img.shape[0:2]
        re_H = re_W = 512
        src_img_padding = cv2.resize(src_img, (re_W, re_H))
        pixel_img_padding = cv2.resize(pixel_img, (re_W, re_H))
        mask_img_padding = cv2.resize(mask_img, (re_W, re_H))
        bg_img_padding = cv2.resize(bg_img, (re_W, re_H))
        gt_img_padding = cv2.resize(gt_img, (re_W, re_H))
        ratio_x = re_W/W
        ratio_y = re_H/H
        img_height, img_width = re_H, re_W

        white_img = np.ones(
            (self.image_size, self.image_size, 3), np.uint8)*255
        black_img = np.zeros(
            (self.image_size, self.image_size, 1), np.uint8)

        four_pts[:, 0] = four_pts[:, 0]*ratio_x
        four_pts[:, 1] = four_pts[:, 1]*ratio_y


##################textRGBA+QM###############################################
        # prepare single textRGB
        textRGB_T_cv = croppaste(white_img, pixel_img_padding, four_pts)
        textRGB_cv = Kmeans(textRGB_T_cv, 4)
        textA_T_cv = croppaste(
            black_img, mask_img_padding[..., np.newaxis], four_pts)

##############text_T########################################
        textRGB_T_pil = Image.fromarray(cv2.cvtColor(
            textRGB_T_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        textRGB_T_tensor = self.transform(textRGB_T_pil)
        textA_T_tensor = self.transform_mask(textA_T_cv)
        text_T_tensor = torch.cat(
            (textRGB_T_tensor, textA_T_tensor), 0)

############I_tt ##colaug_text########################################
        colaug_textRGB_pil = Image.fromarray(cv2.cvtColor(
            textRGB_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        colaug_textRGB_tensor = self.transform_aug(colaug_textRGB_pil)
        colaug_textRGB_tensor = colaug_textRGB_tensor*textA_T_tensor + \
            torch.ones_like(colaug_textRGB_tensor)*(-1.) * \
            (1 - textA_T_tensor)
        colaug_text_tensor = torch.cat(
            (colaug_textRGB_tensor, textA_T_tensor), 0)

########## prepare bg and gt #############
        self.transform_list_bg = [transforms.ColorJitter(
            brightness=0.6, contrast=0.5, saturation=0.5, hue=0.2), transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.8  0.1
        self.transform_bg = transforms.Compose(self.transform_list_bg)

        gt_pil = Image.fromarray(cv2.cvtColor(
            gt_img_padding, cv2.COLOR_BGR2RGB), mode="RGB")
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        gt_tensor = self.transform_bg(gt_pil)

        # bg_cv = bg_img
        bg_pil = Image.fromarray(cv2.cvtColor(
            bg_img_padding, cv2.COLOR_BGR2RGB), mode="RGB")
        torch.random.manual_seed(seed)
        bg_tensor = self.transform_bg(bg_pil)

        return {'bg': bg_tensor, 'I_tt': colaug_text_tensor, 'gt_text': text_T_tensor, 'real': gt_tensor, 'four_pts': four_pts}

    def __len__(self):
        return len(self.pixel_dataset_list)


################### DecompST4CHM_val########################################
class DecompST4CHM_val(Dataset):
    def __init__(self, dataset_dir, image_size=768):
        super(DecompST4CHM_val, self).__init__()
        self.image_size = image_size
        self.src_img_dir = os.path.join(dataset_dir, "src")
        self.pixel_img_dir = os.path.join(dataset_dir, "text_pixel")
        self.mask_img_dir = os.path.join(dataset_dir, "stroke_mask")
        self.erase_img_dir = os.path.join(dataset_dir, "text_erased")
        self.src_txt = os.path.join(dataset_dir, "annotation_val.txt")

        self.pixel_dataset_list, self.gterase_dataset_dict = get_datasetlist(
            self.src_txt, 8)

        transform_list = [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.8  0.1
        self.transform = transforms.Compose(transform_list)

        transform_list_aug = [transforms.ColorJitter(
            brightness=cfg.brightness, contrast=cfg.contrast, saturation=cfg.saturation, hue=cfg.hue), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.8  0.1  0.8  0.2
        self.transform_aug = transforms.Compose(transform_list_aug)
        transform_list_mask = [transforms.ToTensor()]
        self.transform_mask = transforms.Compose(transform_list_mask)

    def __getitem__(self, index):
        img_name = self.pixel_dataset_list[index]['img_name']
        four_pts = self.pixel_dataset_list[index]['quad']

        aug_len = len(
            self.gterase_dataset_dict[self.pixel_dataset_list[index]['img_name']])

        src_img = cv2.imread(os.path.join(
            self.src_img_dir, img_name+'.jpg'))
        img_height, img_width = src_img.shape[0:2]
        pixel_img = cv2.imread(os.path.join(
            self.pixel_img_dir, img_name+'.png'))
        mask_img = cv2.imread(os.path.join(
            self.mask_img_dir, img_name+'.png'), 0)
        erase_img = cv2.imread(os.path.join(
            self.erase_img_dir, img_name+'.jpg'))

        ########## prepare bg and gt #############
        bg_img = croppaste(src_img, erase_img, four_pts)
        gt_img = src_img
        if aug_len > 0:
            aug_num = random.randint(1, aug_len)
            aug_lines = random.sample(
                self.gterase_dataset_dict[self.pixel_dataset_list[index]['img_name']], aug_num)
            for aug_line in aug_lines:
                aug_pts = aug_line['quad']
                aug_quad = np.array(aug_pts, np.float32).reshape((4, 2))
                bg_img = croppaste(bg_img, erase_img, aug_quad)
                gt_img = croppaste(gt_img, erase_img, aug_quad)
        gt_img = croppaste(gt_img, src_img, four_pts)

        ########## Ipt #############
        H, W, = img_height, img_width
        if H > W:
            re_H = self.image_size
            re_W = int(((self.image_size/H)*W//16)*16)
        else:
            re_W = self.image_size
            re_H = int(((self.image_size/W)*H//16)*16)
        src_img = cv2.resize(src_img, (re_W, re_H))
        pixel_img = cv2.resize(pixel_img, (re_W, re_H))
        mask_img = cv2.resize(mask_img, (re_W, re_H))
        bg_img = cv2.resize(bg_img, (re_W, re_H))
        gt_img = cv2.resize(gt_img, (re_W, re_H))
        ratio_x = re_W/W
        ratio_y = re_H/H
        img_height, img_width = re_H, re_W

        src_img_padding = np.zeros(
            (self.image_size, self.image_size, 3), np.uint8)
        src_img_padding[:img_height, :img_width, :3] = src_img
        pixel_img_padding = np.ones(
            (self.image_size, self.image_size, 3), np.uint8)*255
        pixel_img_padding[:img_height, :img_width, :3] = pixel_img
        mask_img_padding = np.zeros(
            (self.image_size, self.image_size, 1), np.uint8)

        mask_img_padding[:img_height, :img_width, 0] = mask_img
        bg_img_padding = np.zeros(
            (self.image_size, self.image_size, 3), np.uint8)
        bg_img_padding[:img_height, :img_width, :3] = bg_img
        gt_img_padding = np.zeros(
            (self.image_size, self.image_size, 3), np.uint8)
        gt_img_padding[:img_height, :img_width, :3] = gt_img

        white_img = np.ones(
            (self.image_size, self.image_size, 3), np.uint8)*255
        black_img = np.zeros(
            (self.image_size, self.image_size, 1), np.uint8)

        four_pts = np.array(four_pts, np.float32).reshape((4, 2))
        four_pts[:, 0] = four_pts[:, 0]*ratio_x
        four_pts[:, 1] = four_pts[:, 1]*ratio_y
        # quad = four_pts
##################textRGBA+QM###############################################
        # prepare single textRGB
        textRGB_T_cv = croppaste(white_img, pixel_img_padding, four_pts)
        textRGB_cv = Kmeans(textRGB_T_cv, 4)
        textA_T_cv = croppaste(
            black_img, mask_img_padding, four_pts)

##############text_T########################################
        textRGB_T_pil = Image.fromarray(cv2.cvtColor(
            textRGB_T_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        textRGB_T_tensor = self.transform(textRGB_T_pil)
        # print(textA_T_cv.shape)
        textA_T_tensor = self.transform_mask(textA_T_cv)
        text_T_tensor = torch.cat(
            (textRGB_T_tensor, textA_T_tensor), 0)

##############colaug_text########################################
        colaug_textRGB_pil = Image.fromarray(cv2.cvtColor(
            textRGB_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        colaug_textRGB_tensor = self.transform_aug(colaug_textRGB_pil)
        colaug_textRGB_tensor = colaug_textRGB_tensor*textA_T_tensor + \
            torch.ones_like(colaug_textRGB_tensor)*(-1.) * \
            (1 - textA_T_tensor)
        colaug_text_tensor = torch.cat(
            (colaug_textRGB_tensor, textA_T_tensor), 0)

########## prepare bg and gt #####################
        self.transform_list_bg = [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform_bg = transforms.Compose(self.transform_list_bg)

        gt_pil = Image.fromarray(cv2.cvtColor(
            gt_img_padding, cv2.COLOR_BGR2RGB), mode="RGB")
        gt_tensor = self.transform_bg(gt_pil)

        bg_pil = Image.fromarray(cv2.cvtColor(
            bg_img_padding, cv2.COLOR_BGR2RGB), mode="RGB")
        bg_tensor = self.transform_bg(bg_pil)

        return {'bg': bg_tensor, 'I_tt': colaug_text_tensor, 'gt_text': text_T_tensor, 'real': gt_tensor, 'four_pts': four_pts}

    def __len__(self):
        return len(self.pixel_dataset_list)
