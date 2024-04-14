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


def cal_maxWidth_maxHeight(quad):
    (tl, tr, br, bl) = quad
    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    mean_Width = int((widthA+widthB)/2)
    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    mean_Height = int((heightA+heightB)/2)
    return maxWidth, maxHeight, mean_Width, mean_Height


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


def GetLinePara(line):
    line.a = line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x * line.p2.y - line.p2.x * line.p1.y


def GetCrossPoint(l1, l2):
    GetLinePara(l1)
    GetLinePara(l2)
    d = l1.a * l2.b - l2.a * l1.b
    p = Point()
    p.x = (l1.b * l2.c - l2.b * l1.c)*1.0 / d
    p.y = (l1.c * l2.a - l2.c * l1.a)*1.0 / d
    return p


def cross_point(quad):  # 计算交点函数
    x1, y1 = quad[0]  # 取四点坐标
    x2, y2 = quad[1]
    x3, y3 = quad[2]
    x4, y4 = quad[3]
    # x1, y1, x2, y2, x3, y3, x4, y4 = quad
    p1 = Point(x1, y1)
    p3 = Point(x3, y3)
    line1 = Line(p1, p3)

    p2 = Point(x2, y2,)
    p4 = Point(x4, y4)
    line2 = Line(p2, p4)
    Pc = GetCrossPoint(line1, line2)
    # print("Cross point:", Pc.x, Pc.y)
    return (Pc.x, Pc.y)


def quadr2rect(quadr, img_height, img_width, ratio=1.5):

    # quadr = np.array(quadr, np.float32).reshape((4, 2))
    (tl, tr, br, bl) = quadr

    maxWidth, maxHeight, mean_Width, mean_Height = cal_maxWidth_maxHeight(
        quadr)
    HWratio = mean_Height/mean_Width

    if mean_Width < 14 or mean_Height < 14:
        ratio = 6 + 2*np.random.rand()  # 3~4
        text_len = min(maxWidth, maxHeight) * ratio
    elif mean_Width <= 18 or mean_Height <= 18:
        ratio = 5 + 2*np.random.rand()  # 2.5~3.5
        text_len = min(maxWidth, maxHeight) * ratio
    else:
        if HWratio < 0.2 or HWratio > 2:
            ratio = 1.25 + 0.5*np.random.rand()  # 1.5~1.75
            text_len = max(maxWidth, maxHeight)*ratio
        else:
            ratio = 4 + np.random.rand()
            text_len = min(maxWidth, maxHeight) * ratio
    maxlength = max(max(maxWidth, maxHeight)*1.25, text_len)

    c_pts = cross_point(quadr)
    jitting_x = np.random.rand()*max(maxWidth, maxHeight)*0.2
    jitting_y = np.random.rand()*max(maxWidth, maxHeight)*0.2

    negposx = -1 if np.random.rand() <= 0.5 else 1
    negposy = -1 if np.random.rand() <= 0.5 else 1
    tl_x = int(c_pts[0]-maxlength//2 + negposx * jitting_x)
    tl_y = int(c_pts[1]-maxlength//2 + negposy * jitting_y)
    br_x = int(c_pts[0]+maxlength//2 + negposx * jitting_x)
    br_y = int(c_pts[1]+maxlength//2 + negposy * jitting_y)

    tl_x = 0 if tl_x < 0 else tl_x
    tl_y = 0 if tl_y < 0 else tl_y
    br_x = img_height - 1 if br_x >= img_height else br_x
    br_y = img_width - 1 if br_y >= img_width else br_y

    tl = np.array([tl_x, tl_y])
    br = np.array([br_x, br_y])

    return tl, br


def transfrom_text_roi(textRGB_roi, textA_roi, quad_in_rect_rescale, quad_in_rect, ori_HW):
    '''input:
        textRGB: fullsize  
    '''
    ori_H, ori_W = ori_HW
    text_shape = textRGB_roi.shape
    c_pts = (text_shape[0]//2, text_shape[1]//2)
    maxWidth, maxHeight, mean_Width, mean_Height = cal_maxWidth_maxHeight(
        quad_in_rect_rescale)
    # mean_Width = mean_Width*(0.7+0.5*(np.random.rand()))

    HWratio = mean_Height/mean_Width
    text_width = np.random.randint(190, 220)
    text_height = mean_Height*text_width/mean_Width
    text_height = 160 if text_height >= 160 else text_height
    text_width = text_height/HWratio

    center_rect = np.array([[c_pts[0]-text_width/2, c_pts[1]-text_height/2],
                            [c_pts[0]+text_width/2, c_pts[1]-text_height/2],
                            [c_pts[0]+text_width/2, c_pts[1]+text_height/2],
                            [c_pts[0]-text_width/2, c_pts[1]+text_height/2]],
                           np.float32)

    hori_textBM_roi = np.zeros_like(textA_roi)
    center_quad = np.array(center_rect, np.int32)
    cv2.fillConvexPoly(hori_textBM_roi, center_quad, (255, 255, 255))

    M = cv2.getPerspectiveTransform(quad_in_rect_rescale, center_rect)
    # center_rect[:, 0] = center_rect[:, 0] / text_shape[1]*ori_W
    # center_rect[:, 1] = center_rect[:, 1] / text_shape[0]*ori_H
    homo_cv = cv2.getPerspectiveTransform(center_rect, quad_in_rect_rescale)

    # 进行仿射变换
    # print(text_shape[:2])
    hori_textRGB_roi = cv2.warpPerspective(
        textRGB_roi, M, text_shape[:2], borderValue=(255, 255, 255))
    hori_textA_roi = cv2.warpPerspective(textA_roi, M, text_shape[:2])

    # hori_textRGB_roi_ori = cv2.resize(hori_textRGB_roi, ori_HW[::-1])

    return hori_textRGB_roi, hori_textA_roi, hori_textBM_roi, homo_cv


def transfrom_text(textRGB, textA, quad, center_rect, tl):
    quad = np.array(quad, np.float32).reshape((4, 2))
    text_shape = textRGB.shape
    center_rect = center_rect + tl

    hori_textBM = np.zeros_like(textA)
    center_quad = np.array(center_rect, np.int32)
    cv2.fillConvexPoly(hori_textBM, center_quad, (255, 255, 255))

    # print(quad.dtype, center_rect.dtype)
    M = cv2.getPerspectiveTransform(quad, center_rect.astype(np.float32))

    hori_textRGB = cv2.warpPerspective(
        textRGB, M, text_shape[:2], borderValue=(255, 255, 255))
    hori_textA = cv2.warpPerspective(textA, M, text_shape[:2])

    # hori_textRGB_roi_ori = cv2.resize(hori_textRGB_roi, ori_HW)
    return hori_textRGB, hori_textA, hori_textBM


def cvhomo2torch(homo_cv, size):
    N = np.array([[2/size, 0, -1], [0, 2/size, -1], [0, 0, 1]], np.float32)
    N_inv = np.linalg.inv(N)
    homo_cv_inv = np.linalg.inv(homo_cv)
    homo_gt = N@homo_cv_inv@N_inv
    homo_gt = np.array(homo_gt, np.float32)
    return homo_gt


def croppaste(canvas, tar_img, quad):
    quad = np.array(quad, np.int32).reshape((4, 2))
    img_width, img_height = canvas.shape[0:2]
    mask = np.zeros((img_width, img_height, 1), np.uint8)
    cv2.fillConvexPoly(mask, quad, (255, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel)
    mask = mask.reshape((img_width, img_height, -1))
    dst = np.where(mask > 0, tar_img, canvas)
    return dst


def Kmeans(img, k):
    data = img.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
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


class DecompST4GTM(Dataset):
    def __init__(self, dataset_dir, image_size=768, is_for_train=True):
        super(DecompST4GTM, self).__init__()
        self.image_size = image_size
        self.text_size = 256
        self.src_img_dir = os.path.join(dataset_dir, "src")
        self.pixel_img_dir = os.path.join(dataset_dir, "text_pixel")
        self.mask_img_dir = os.path.join(dataset_dir, "stroke_mask")
        self.erase_img_dir = os.path.join(dataset_dir, "text_erased")

        self.src_txt = os.path.join(dataset_dir, "annotation.txt")  # _tra

        self.pixel_dataset_list, self.gterase_dataset_dict = get_datasetlist(
            self.src_txt, 8)

        transform_list = [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        transform_list_aug = [transforms.ColorJitter(
            brightness=0.8, contrast=0.4, saturation=0.6, hue=0.5), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform_aug = transforms.Compose(transform_list_aug)
        transform_list_mask = [transforms.ToTensor()]
        self.transform_mask = transforms.Compose(transform_list_mask)

        self.transform_list_bg = [transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.3, hue=0.2), transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform_bg = transforms.Compose(self.transform_list_bg)

    def __getitem__(self, index):
        img_name = self.pixel_dataset_list[index]['img_name']
        four_pts = self.pixel_dataset_list[index]['quad']

        aug_len = len(
            self.gterase_dataset_dict[self.pixel_dataset_list[index]['img_name']])

        src_img = cv2.imread(os.path.join(
            self.src_img_dir, img_name+'.jpg'))
        pixel_img = cv2.imread(os.path.join(
            self.pixel_img_dir, img_name+'.png'))
        mask_img = cv2.imread(os.path.join(
            self.mask_img_dir, img_name+'.png'), 0)
        erase_img = cv2.imread(os.path.join(
            self.erase_img_dir, img_name+'.jpg'))

#############random text erase#############################################
        bg_img = croppaste(src_img, erase_img, four_pts)
        if aug_len > 0:
            aug_num = random.randint(1, aug_len)
            aug_lines = random.sample(
                self.gterase_dataset_dict[self.pixel_dataset_list[index]['img_name']], aug_num)
            for aug_line in aug_lines:
                aug_pts = aug_line['quad']
                aug_quad = np.array(aug_pts, np.float32).reshape((4, 2))
                bg_img = croppaste(bg_img, erase_img, aug_quad)

############# rotation and crop#############################################
        rot = np.random.randint(-20, 20)  # (-15, 15)
        [src_img, pixel_img, mask_img, bg_img], four_pts = rotate_image(
            [src_img, pixel_img, mask_img, bg_img], four_pts, rot)

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

        four_pts = np.array(four_pts, np.float32).reshape((4, 2))
        four_pts = four_pts - \
            np.array([x1, y1], np.float32).reshape((-1, 2))

########################################################################
        H, W, = src_img.shape[0:2]
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
        mask_img_padding[:img_height, :img_width,
                         :1] = mask_img.reshape(img_height, img_width, -1)
        bg_img_padding = np.zeros(
            (self.image_size, self.image_size, 3), np.uint8)
        bg_img_padding[:img_height, :img_width, :3] = bg_img

        white_img = np.ones(
            (self.image_size, self.image_size, 3), np.uint8)*255
        black_img = np.zeros((self.image_size, self.image_size, 1), np.uint8)

        four_pts[:, 0] = four_pts[:, 0]*ratio_x
        four_pts[:, 1] = four_pts[:, 1]*ratio_y
        quad = four_pts  # np.array(four_pts, np.float32).reshape((4, 2))

        # expend quadr
        tl, br = quadr2rect(four_pts, self.image_size,
                            self.image_size)
        rect = np.array([tl, br])

##################textRGBA+QM###############################################
        # prepare single textRGB
        textRGB_T_cv = croppaste(white_img, pixel_img_padding, four_pts)
        textRGB_cv = Kmeans(textRGB_T_cv, 2)
        textA_T_cv = croppaste(black_img, mask_img_padding, four_pts)

        textBM_T_cv = np.zeros_like(textA_T_cv)
        quadint = np.array(quad, np.int32)
        cv2.fillConvexPoly(textBM_T_cv, quadint, (255, 255, 255))

        textRGB_roi_cv = textRGB_cv[tl[1]:br[1], tl[0]:br[0]]
        ori_H, ori_W, _ = textRGB_roi_cv.shape
        textRGB_roi_cv256 = cv2.resize(
            textRGB_roi_cv, (self.text_size, self.text_size))

        textA_T_roi_cv = textA_T_cv[tl[1]:br[1], tl[0]:br[0]]
        textA_T_roi_cv = cv2.resize(
            textA_T_roi_cv, (self.text_size, self.text_size))

        quad_in_rect = (quad-tl).astype(np.float32)
        quad_in_rect_rescale = quad_in_rect.copy()
        quad_in_rect_rescale[:, 0] = quad_in_rect_rescale[:, 0] * \
            self.text_size/ori_W
        quad_in_rect_rescale[:, 1] = quad_in_rect_rescale[:, 1] * \
            self.text_size/ori_H

        hori_textRGB_cv, hori_textA_cv, hori_textBM_cv, homo_cv = transfrom_text_roi(
            textRGB_roi_cv256, textA_T_roi_cv, quad_in_rect_rescale, quad_in_rect, (ori_H, ori_W))

        homo_gt = cvhomo2torch(homo_cv, 256)

# ##############text_T_roi########################################
        quad_roi = np.array(four_pts, np.float32).reshape((4, 2))
        quad_roi = (quad_roi-tl).astype(np.float32)
        quad_roi[:, 0] = quad_roi[:, 0]*self.text_size/textRGB_roi_cv.shape[1]
        quad_roi[:, 1] = quad_roi[:, 1]*self.text_size/textRGB_roi_cv.shape[0]

        textRGB_T_roi_cv = textRGB_T_cv[tl[1]:br[1], tl[0]:br[0]]
        textRGB_T_roi_cv = cv2.resize(
            textRGB_T_roi_cv, (self.text_size, self.text_size))
        textBM_T_roi_cv = np.zeros(
            (self.text_size, self.text_size, 1), np.uint8)
        quad_roi = np.array(quad_roi, np.int32)
        cv2.fillConvexPoly(textBM_T_roi_cv, quad_roi, (255, 255, 255))

##############text_T########################################
        textRGB_T_pil = Image.fromarray(cv2.cvtColor(
            textRGB_T_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        textRGB_T_tensor = self.transform(textRGB_T_pil)
        textA_T_tensor = self.transform_mask(textA_T_cv)
        textBM_T_tensor = self.transform_mask(textBM_T_cv)
        text_T_tensor = torch.cat(
            (textRGB_T_tensor, textA_T_tensor, textBM_T_tensor), 0)


##############hori_text########################################
        hori_textRGB_pil = Image.fromarray(cv2.cvtColor(
            hori_textRGB_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        hori_textRGB_tensor = self.transform_aug(hori_textRGB_pil)
        hori_textA_tensor = self.transform_mask(hori_textA_cv)
        hori_textBM_tensor = self.transform_mask(hori_textBM_cv)
        hori_textRGB_tensor = hori_textRGB_tensor*hori_textA_tensor + \
            torch.ones_like(hori_textRGB_tensor)*(-1.) * \
            (1 - hori_textA_tensor)
        hori_text_tensor = torch.cat(
            (hori_textRGB_tensor, hori_textA_tensor, hori_textBM_tensor), 0)

        bg_cv = bg_img_padding  # bg_img
        bg_pil = Image.fromarray(cv2.cvtColor(
            bg_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        bg_tensor = self.transform_bg(bg_pil)

        return {'bg': bg_tensor, 'P_pt': hori_text_tensor, 'gt_text': text_T_tensor, 'rect': rect, 'homo_gt': homo_gt}

    def __len__(self):
        return len(self.pixel_dataset_list)


class DecompST4GTM_val(Dataset):
    def __init__(self, dataset_dir, image_size=768):
        super(DecompST4GTM_val, self).__init__()
        self.image_size = image_size
        self.text_size = 256
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
        transform_list_aug = [transforms.ColorJitter(brightness=0.8, contrast=0.4, saturation=0.6, hue=0.5), transforms.ToTensor(
        ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.8  0.1  0.8  0.2
        self.transform_aug = transforms.Compose(transform_list_aug)
        transform_list_mask = [transforms.ToTensor()]
        self.transform_mask = transforms.Compose(transform_list_mask)

        self.transform_list_bg = [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.8  0.1
        self.transform_bg = transforms.Compose(self.transform_list_bg)

    def __getitem__(self, index):
        img_name = self.pixel_dataset_list[index]['img_name']
        four_pts = self.pixel_dataset_list[index]['quad']

        aug_len = len(
            self.gterase_dataset_dict[self.pixel_dataset_list[index]['img_name']])

        src_img = cv2.imread(os.path.join(
            self.src_img_dir, img_name+'.jpg'))
        pixel_img = cv2.imread(os.path.join(
            self.pixel_img_dir, img_name+'.png'))
        mask_img = cv2.imread(os.path.join(
            self.mask_img_dir, img_name+'.png'), 0)
        erase_img = cv2.imread(os.path.join(
            self.erase_img_dir, img_name+'.jpg'))

        bg_img = croppaste(src_img, erase_img, four_pts)
        # gt_img = src_img_padding
        if aug_len > 0:
            aug_num = random.randint(1, aug_len)
            aug_lines = random.sample(
                self.gterase_dataset_dict[self.pixel_dataset_list[index]['img_name']], aug_num)
            for aug_line in aug_lines:
                aug_pts = aug_line['quad']
                aug_quad = np.array(aug_pts, np.float32).reshape((4, 2))
                bg_img = croppaste(bg_img, erase_img, aug_quad)
                # gt_img = croppaste(gt_img, erase_img_padding, aug_quad)

########################################################################
        # print(img_height, img_width)
        H, W, = src_img.shape[0:2]
        if H > W:
            re_H = self.image_size
            re_W = int(((self.image_size/H)*W//16)*16)
        else:
            re_W = self.image_size
            re_H = int(((self.image_size/W)*H//16)*16)
        src_img = cv2.resize(src_img, (re_W, re_H))
        pixel_img = cv2.resize(pixel_img, (re_W, re_H))
        mask_img = cv2.resize(mask_img, (re_W, re_H))
        # erase_img = cv2.resize(erase_img, (re_W, re_H))
        bg_img = cv2.resize(bg_img, (re_W, re_H))
        ratio_x = re_W/W
        ratio_y = re_H/H
        img_height, img_width = re_H, re_W
        # print(img_height, img_width)
        # print(img_height, img_width)

        src_img_padding = np.zeros(
            (self.image_size, self.image_size, 3), np.uint8)
        src_img_padding[:img_height, :img_width, :3] = src_img
        pixel_img_padding = np.ones(
            (self.image_size, self.image_size, 3), np.uint8)*255
        pixel_img_padding[:img_height, :img_width, :3] = pixel_img
        mask_img_padding = np.zeros(
            (self.image_size, self.image_size, 1), np.uint8)
        mask_img_padding[:img_height, :img_width,
                         :1] = mask_img.reshape(img_height, img_width, -1)
        bg_img_padding = np.zeros(
            (self.image_size, self.image_size, 3), np.uint8)
        bg_img_padding[:img_height, :img_width, :3] = bg_img

        white_img = np.ones(
            (self.image_size, self.image_size, 3), np.uint8)*255
        black_img = np.zeros((self.image_size, self.image_size, 1), np.uint8)

        four_pts = np.array(four_pts, np.float32).reshape((4, 2))
        four_pts[:, 0] = four_pts[:, 0]*ratio_x
        four_pts[:, 1] = four_pts[:, 1]*ratio_y
        quad = four_pts

        # expend quadr
        tl, br = quadr2rect(four_pts, self.image_size,
                            self.image_size)
        rect = np.array([tl, br])

##################textRGBA+QM###############################################
        # prepare single textRGB
        textRGB_T_cv = croppaste(white_img, pixel_img_padding, four_pts)
        textRGB_cv = Kmeans(textRGB_T_cv, 2)
        textA_T_cv = croppaste(black_img, mask_img_padding, four_pts)

        textBM_T_cv = np.zeros_like(textA_T_cv)
        quadint = np.array(quad, np.int32)
        cv2.fillConvexPoly(textBM_T_cv, quadint, (255, 255, 255))

        textRGB_roi_cv = textRGB_cv[tl[1]:br[1], tl[0]:br[0]]
        # textRGB_roi_cv
        ori_H, ori_W, _ = textRGB_roi_cv.shape
        # print(textRGB_roi_cv.shape)
        textRGB_roi_cv256 = cv2.resize(
            textRGB_roi_cv, (self.text_size, self.text_size))

        textA_T_roi_cv = textA_T_cv[tl[1]:br[1], tl[0]:br[0]]
        textA_T_roi_cv = cv2.resize(
            textA_T_roi_cv, (self.text_size, self.text_size))

        quad_in_rect = (quad-tl).astype(np.float32)
        quad_in_rect_rescale = quad_in_rect.copy()
        quad_in_rect_rescale[:, 0] = quad_in_rect_rescale[:, 0] * \
            self.text_size/ori_W
        quad_in_rect_rescale[:, 1] = quad_in_rect_rescale[:, 1] * \
            self.text_size/ori_H

        hori_textRGB_cv, hori_textA_cv, hori_textBM_cv, homo_cv = transfrom_text_roi(
            textRGB_roi_cv256, textA_T_roi_cv, quad_in_rect_rescale, quad_in_rect, (ori_H, ori_W))

        # hori_textRGB_cv, hori_textA_cv, hori_textBM_cv = transfrom_text(
        #     textRGB_cv, textA_T_cv, quad, center_rect, tl)

        homo_gt = cvhomo2torch(homo_cv, 256)

# ##############text_T_roi########################################
        quad_roi = np.array(four_pts, np.float32).reshape((4, 2))
        quad_roi = (quad_roi-tl).astype(np.float32)
        quad_roi[:, 0] = quad_roi[:, 0]*self.text_size/textRGB_roi_cv.shape[1]
        quad_roi[:, 1] = quad_roi[:, 1]*self.text_size/textRGB_roi_cv.shape[0]

        textRGB_T_roi_cv = textRGB_T_cv[tl[1]:br[1], tl[0]:br[0]]
        textRGB_T_roi_cv = cv2.resize(
            textRGB_T_roi_cv, (self.text_size, self.text_size))
        textRGB_T_roi_pil = Image.fromarray(cv2.cvtColor(
            textRGB_T_roi_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        textRGB_T_roi_tensor = self.transform(textRGB_T_roi_pil)
        textBM_T_roi_cv = np.zeros(
            (self.text_size, self.text_size, 1), np.uint8)
        quad_roi = np.array(quad_roi, np.int32)
        cv2.fillConvexPoly(textBM_T_roi_cv, quad_roi, (255, 255, 255))
        # textA_T_roi_tensor = self.transform_mask(textA_T_roi_cv)
        # textBM_T_roi_tensor = self.transform_mask(textBM_T_roi_cv)
        # text_T_roi_tensor = torch.cat(
        #     (textRGB_T_roi_tensor, textA_T_roi_tensor, textBM_T_roi_tensor), 0)

##############text_T########################################
        textRGB_T_pil = Image.fromarray(cv2.cvtColor(
            textRGB_T_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        textRGB_T_tensor = self.transform(textRGB_T_pil)
        textA_T_tensor = self.transform_mask(textA_T_cv)
        textBM_T_tensor = self.transform_mask(textBM_T_cv)
        text_T_tensor = torch.cat(
            (textRGB_T_tensor, textA_T_tensor, textBM_T_tensor), 0)


##############hori_text########################################
        hori_textRGB_pil = Image.fromarray(cv2.cvtColor(
            hori_textRGB_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        hori_textRGB_tensor = self.transform_aug(hori_textRGB_pil)
        hori_textA_tensor = self.transform_mask(hori_textA_cv)
        hori_textBM_tensor = self.transform_mask(hori_textBM_cv)
        hori_textRGB_tensor = hori_textRGB_tensor*hori_textA_tensor + \
            torch.ones_like(hori_textRGB_tensor)*(-1.) * \
            (1 - hori_textA_tensor)
        hori_text_tensor = torch.cat(
            (hori_textRGB_tensor, hori_textA_tensor, hori_textBM_tensor), 0)

        bg_cv = bg_img_padding  # bg_img
        bg_pil = Image.fromarray(cv2.cvtColor(
            bg_cv, cv2.COLOR_BGR2RGB), mode="RGB")
        # torch.random.manual_seed(seed)
        bg_tensor = self.transform_bg(bg_pil)

        return {'bg': bg_tensor, 'P_pt': hori_text_tensor, 'gt_text': text_T_tensor, 'rect': rect, 'homo_gt': homo_gt}

    def __len__(self):
        return len(self.pixel_dataset_list)
