import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import cv2
# from Iptgen.gen import Ipt_gen
# from copy import deepcopy
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None

################################################
# hyper para
# squa len LB =
# plain text W H
# random trial max
# overlap_rate =
################################################


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg"])


def pad_text_256(text, text_shape=(256, 256)):
    text_center = (text_shape[0] // 2, text_shape[1] // 2)
    H, W, = text.shape
    HW_ratio = H / W
    text_W = 220
    text_H = int(H * text_W / W)
    text_H = 160 if text_H >= 160 else text_H
    text_W = int(text_H / HW_ratio)
    x1, x2 = int(text_center[0] - text_W / 2), int(text_center[0] + text_W / 2)
    y1, y2 = int(text_center[1] - text_H / 2), int(text_center[1] + text_H / 2)
    # print(text_c.shape, (H, W), (text_W, text_H))
    text = cv2.resize(text, (text_W, text_H), cv2.INTER_CUBIC)

    text256 = np.zeros((text_shape[0], text_shape[1]), np.uint8)
    text256[y1:y2, x1:x2] = text
    return text256, (text_W,
                     text_H), np.array([x1, y1, x2, y1, x2, y2, x1, y2])


class DataIptIbg(Dataset):
    def __init__(self, data_dict, img_list, Ipt_gen, Ibg_dir, BB_map_dir,
                 seg_map_dir):
        super(DataIptIbg, self).__init__()
        self.img_size = 256
        self.data_dict = data_dict
        self.img_list = img_list
        self.Ibg_dir = Ibg_dir
        self.BB_map_dir = BB_map_dir
        self.seg_map_dir = seg_map_dir
        self.Ipt_gen = Ipt_gen
        transform_bg_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.transform_Ibg = transforms.Compose(transform_bg_list)
        transform_list_mask = [transforms.ToTensor()]
        self.transform_mask = transforms.Compose(transform_list_mask)
        transform_text_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1))
        ]
        self.transform_Ipt = transforms.Compose(transform_text_list)

    def __getitem__(self, index):

        bg_name, region_name = self.img_list[index]
        if random.random() < 0.8:  # 0.8
            seed = self.data_dict[bg_name][region_name]['seed']
            np.random.seed(seed)

        text, I_textA = self.Ipt_gen.get_i_textA()

        aa = np.where(I_textA > 0)
        x1, y1, x2, y2 = aa[0].min(), aa[1].min(), aa[0].max(), aa[1].max()
        I_textA256, (T_W, T_H), quad = pad_text_256(I_textA)

        Ibg_cv = cv2.imread(os.path.join(self.Ibg_dir, bg_name + '.jpg'))

        I_Height, I_Width, _ = Ibg_cv.shape

        Ibg_pad = np.zeros([I_Height * 2, I_Width * 2, 3], np.uint8)
        Ibg_pad[int(I_Height / 2):I_Height + int(I_Height / 2),
                int(I_Width / 2):I_Width + int(I_Width / 2), :] = Ibg_cv
        X, Y, W, H, _ = self.data_dict[bg_name][region_name]['stats']
        label_map = cv2.imread(
            os.path.join(self.seg_map_dir, bg_name + '.png'), 0)
        max_length = max([W, H])
        min_length = min([W, H])

        ng_map = cv2.imread(os.path.join(self.BB_map_dir, bg_name + '.png'),
                            0).astype(np.int32)
        ones_map = np.ones_like(ng_map)
        ones_map[ng_map > 0] = 0
        safe_label_map = label_map * ones_map

        HALF_squa_len = np.random.randint(min_length / 3.5 - 1, max_length / 3)
        while HALF_squa_len * 2 > min(I_Height, I_Width):
            HALF_squa_len = np.random.randint(min_length / 3.5 - 1,
                                              max_length / 3)

        rescale_rate = HALF_squa_len * 2 / self.img_size
        safemask = (safe_label_map == int(region_name))

        overlap_rate = 0
        if safemask.sum() < 2:
            safemask = (label_map == int(region_name))
            ctr_locs = np.transpose(np.nonzero(safemask))
            overlap_rate = 1
            ctr = ctr_locs[np.random.choice(ctr_locs.shape[0]), :]
            ctr = ctr[::-1]
            cx, cy = int(ctr[0]), int(ctr[1])
        else:
            ctr_locs = np.transpose(np.nonzero(safemask))

        for i in range(2000):
            if overlap_rate < 0.7:
                ctr = ctr_locs[np.random.choice(ctr_locs.shape[0]), :]
                ctr = ctr[::-1]
                cx, cy = int(ctr[0]), int(ctr[1])
                squa_map = np.zeros_like(label_map)
                cv2.rectangle(squa_map, (cx - int(T_W / 2 * rescale_rate),
                                         cy - int(T_H / 2 * rescale_rate)),
                              (cx + int(T_W / 2 * rescale_rate),
                               cy + int(T_H / 2 * rescale_rate)), 1.0, -1)
                overlap_map = safemask * (squa_map == 1)
                overlap_rate = np.sum(overlap_map) / \
                    (T_W*T_H*(rescale_rate)**2+1)
                # print(T_W*T_H*(rescale_rate)**2)
                ab_flag = True
            else:
                ab_flag = False
                break

        Ibg_pil = Image.fromarray(cv2.cvtColor(Ibg_cv, cv2.COLOR_BGRA2RGB))
        Ibg_tensor = self.transform_Ibg(Ibg_pil)

        Ibg_tensor_pad = Ibg_tensor.new_full((3, 768, 768), -1)
        # torch.zeros((3, 768, 768)) - 1.0
        Ibg_tensor_pad[:, :Ibg_tensor.shape[1], :Ibg_tensor.shape[2]].copy_(
            Ibg_tensor)

        ############### Ipt ################################################
        if random.random() < 0.6:  # 0.8
            seed = self.data_dict[bg_name][region_name]['seed']
            np.random.seed(seed)
        Ipt256, all_mask = self.Ipt_gen.get_colorize(I_textA256)
        Ipt256_pil = Image.fromarray(cv2.cvtColor(Ipt256, cv2.COLOR_BGRA2RGBA))
        Ipt256_tensor = self.transform_Ipt(Ipt256_pil)
        Ipt256_A = Ipt256_tensor[3:4, ...]
        Ipt256_QM = torch.zeros_like(Ipt256_A)
        x1, y1, x2, y1, x2, y2, x1, y2 = quad
        Ipt256_QM[:, y1:y2, x1:x2] = 1.0

        Ipt256_tensor = torch.cat((Ipt256_tensor, Ipt256_QM), 0)
        tl, br = np.array([cx - HALF_squa_len, cy - HALF_squa_len]), np.array(
            [cx + HALF_squa_len, cy + HALF_squa_len])
        rect = np.array([tl, br])

        return {
            'Ibg': Ibg_tensor_pad,
            'Ipt': Ipt256_tensor,
            'rect': rect,
            'bg_name': bg_name,
            'region': region_name,
            'flag': ab_flag,
            'quad': quad,
            'text': text
        }

    def __len__(self):

        return len(self.img_list)
