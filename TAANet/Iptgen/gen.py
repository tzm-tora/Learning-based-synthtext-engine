# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from pygame import freetype
from . import render_text_mask
from Iptgen.colorize import Colorize
from . import data_cfg
import random
import pickle as cp

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def is_font_file(filename):
    return any(
        filename.endswith(extension) for extension in
        [".ttf", ".TTF", ".otf", ".OTF", ".ttc", ".TTC", ".fon"])


class Ipt_gen():
    def __init__(self):

        freetype.init()
        cur_file_path = data_cfg.data_dir  # os.path.dirname(__file__)
        # chinese text
        chinese_font_dir = os.path.join(cur_file_path,
                                        data_cfg.chinese_font_dir)
        self.chinese_font_list = os.listdir(
            os.path.join(cur_file_path, data_cfg.chinese_font_dir))
        self.chinese_font_list = [
            os.path.join(chinese_font_dir, font_name)
            for font_name in self.chinese_font_list
        ]
        self.chinese_font_list = [
            x for x in self.chinese_font_list if is_font_file(x)
        ]

        chinese_text_filepath = os.path.join(cur_file_path,
                                             data_cfg.chinese_text_filepath)
        self.chinese_text_list = open(chinese_text_filepath,
                                      'r',
                                      encoding='utf-8').readlines()
        self.chinese_text_list = [
            text.strip() for text in self.chinese_text_list
        ]

        # english text
        english_font_dir = os.path.join(cur_file_path,
                                        data_cfg.english_font_dir)
        self.english_font_list = os.listdir(
            os.path.join(cur_file_path, data_cfg.english_font_dir))
        self.english_font_list = [
            os.path.join(english_font_dir, font_name)
            for font_name in self.english_font_list
        ]
        self.english_font_list = [
            x for x in self.english_font_list if is_font_file(x)
        ]

        english_text_filepath = os.path.join(cur_file_path,
                                             data_cfg.english_text_filepath)
        self.english_text_list = open(english_text_filepath,
                                      'r',
                                      encoding='utf-8').readlines()
        self.english_text_list = [
            text.strip() for text in self.english_text_list
        ]
        # np.random.shuffle(self.english_text_list)

        self.colorsRGB, self.colorsLAB = get_color_matrix(
            os.path.join(cur_file_path, data_cfg.color_filepath))
        self.colorize = Colorize()
        self.en_std_char = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''

    def check_char(self, text):
        if len(text) == 1:
            a = '0123456789PABCXYZ'
            NO = random.randint(0, len(a) - 1)
            text = a[NO]
        return text

    def check_text_font(self, text, font):

        try:
            for char in text:
                line_bounds = font.get_rect(char)
        except:
            # print(text, font)
            valid = False
        else:
            valid = True
        return valid

    def get_i_textA(self):

        while True:
            # choose font, text and bg
            if np.random.rand() < data_cfg.is_chinese_rate:
                font_name = np.random.choice(self.chinese_font_list)
                text = random.choice(self.chinese_text_list)
            else:
                font_name = np.random.choice(self.english_font_list)
                text = random.choice(self.english_text_list)

            freetype.init()
            font = freetype.Font(font_name)
            font.antialiased = True
            font.origin = True
            font.size = np.random.randint(data_cfg.font_size[0],
                                          data_cfg.font_size[1] + 1)
            font.strong = np.random.rand() < data_cfg.strong_rate

            if len(text) == 1:
                if np.random.rand() < 0.9:
                    text = random.choice(self.english_text_list)

            text = self.check_char(text)

            valid = self.check_text_font(text, font)
            if not valid:
                continue

            # text = 'synthesis'
            # render text to surf
            param = {
                # 'is_curve': np.random.rand() < data_cfg.is_curve_rate,
                'is_curve':
                random.random() < data_cfg.is_curve_rate,
                'is_rotate':
                np.random.rand() < data_cfg.is_rotate_rate,
                'curve_rate':
                data_cfg.curve_rate_param[0] * np.random.randn() +
                data_cfg.curve_rate_param[1],
                'curve_center':
                random.randint(
                    len(text) // 2 - len(text) // 6,
                    len(text) // 2 + len(text) // 6) if
                (len(text) // 2 - len(text) // 6) <
                (len(text) // 2 + len(text) // 6) else len(text) // 2
            }
            i_textA, bbs1 = render_text_mask.render_text(font, text, param)

            # loc = np.where(text > 0)
            black = np.zeros_like(i_textA)
            black[i_textA > 0] = 1
            if black.sum() < 70 or black.sum() / len(text) < 50:
                # print(text, font)
                continue
            # nonzero = np.where(i_textA > 0)
            # if len(nonzero[0]) < 50 or len(nonzero[0])/len(text) < 30:
            #     print(text, font)
            #     continue

            # cv2.imshow('i_textA', i_textA)
            # cv2.waitKey()

            break

        return [text, i_textA]

    def get_colorize(self, i_textA, bg_p=None):

        param = {
            'is_border':
            np.random.rand() < data_cfg.is_border_rate,
            'is_shadow':
            np.random.rand() < data_cfg.is_shadow_rate,
            'is_add_3D':
            np.random.rand() < data_cfg.is_add3D_rate,
            'shadow_angle':
            np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree) +
            data_cfg.shadow_angle_param[0] * np.random.randn(),
            'shadow_shift':
            data_cfg.shadow_shift_param[0, :] * np.random.randn(3) +
            data_cfg.shadow_shift_param[1, :],
            'shadow_opacity':
            data_cfg.shadow_opacity_param_real[0] * np.random.randn() +
            data_cfg.shadow_opacity_param_real[1],
            'shadow_scale':
            np.random.choice(data_cfg.shadow_scale) + 3,
            'add_3D_shift':
            data_cfg.add3D_shift_param[0, :] * np.random.randn(3) +
            data_cfg.add3D_shift_param[1, :],
        }

        # get font color
        # if np.random.rand() < data_cfg.use_random_color_rate:
        if bg_p == None:
            B, G, R = np.random.rand(3) * 255.
            fg_col = np.array([B, G, R], np.uint8)
            # print(fg_col)
        else:
            fg_col, bg_col = get_font_color(self.colorsRGB, self.colorsLAB,
                                            bg_p)

        i_ptRGB, i_ptA, border_mask, pseudo3D_mask = self.colorize.color(
            i_textA, fg_col, param)

        i_pt = cv2.merge([i_ptRGB, i_ptA])

        i_mask = cv2.merge([i_textA, border_mask, pseudo3D_mask])

        return [i_pt, i_mask]


class FontColor(object):
    def __init__(self, colorsRGB, colorsLAB):

        self.colorsRGB = colorsRGB
        self.colorsLAB = colorsLAB
        self.ncol = colorsRGB.shape[0]

    def sample_normal(self, col_mean, col_std):

        col_sample = col_mean + col_std * np.random.randn()
        return np.clip(col_sample, 0, 255).astype(np.uint8)

    def sample_from_data(self, bg_mat):

        bg_mat = cv2.cvtColor(bg_mat, cv2.COLOR_RGB2Lab)
        bg_mat = np.reshape(bg_mat, (np.prod(bg_mat.shape[:2]), 3))
        bg_mean = np.mean(bg_mat, axis=0)

        norms = np.linalg.norm(self.colorsLAB - bg_mean[None, :], axis=1)
        # choose a random color amongst the top 3 closest matches:
        # nn = np.random.choice(np.argsort(norms)[:3])
        nn = np.argmin(norms)

        # nearest neighbour color:
        data_col = self.colorsRGB[np.mod(nn, self.ncol), :]

        # color
        col1 = self.sample_normal(data_col[:3], data_col[3:6])
        col2 = self.sample_normal(data_col[6:9], data_col[9:12])

        if nn < self.ncol:
            return (col2, col1)
        else:
            # need to swap to make the second color close to the input backgroun color
            return (col1, col2)

    def mean_color(self, arr):

        col = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        col = np.reshape(col, (np.prod(col.shape[:2]), 3))
        col = np.mean(col, axis=0).astype(np.uint8)
        return np.squeeze(cv2.cvtColor(col[None, None, :], cv2.COLOR_HSV2RGB))

    def invert(self, rgb):

        rgb = 127 + rgb
        return rgb

    def complement(self, rgb_color):

        col_hsv = np.squeeze(
            cv2.cvtColor(rgb_color[None, None, :], cv2.COLOR_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128  # uint8 mods to 255
        col_comp = np.squeeze(
            cv2.cvtColor(col_hsv[None, None, :], cv2.COLOR_HSV2RGB))
        return col_comp

    def triangle_color(self, col1, col2):

        col1, col2 = np.array(col1), np.array(col2)
        col1 = np.squeeze(cv2.cvtColor(col1[None, None, :], cv2.COLOR_RGB2HSV))
        col2 = np.squeeze(cv2.cvtColor(col2[None, None, :], cv2.COLOR_RGB2HSV))
        h1, h2 = col1[0], col2[0]
        if h2 < h1:
            h1, h2 = h2, h1  # swap
        dh = h2 - h1
        if dh < 127:
            dh = 255 - dh
        col1[0] = h1 + dh / 2
        return np.squeeze(cv2.cvtColor(col1[None, None, :], cv2.COLOR_HSV2RGB))

    def change_value(self, col_rgb, v_std=50):

        col = np.squeeze(
            cv2.cvtColor(col_rgb[None, None, :], cv2.COLOR_RGB2HSV))
        x = col[2]
        vs = np.linspace(0, 1)
        ps = np.abs(vs - x / 255.0)
        ps /= np.sum(ps)
        v_rand = np.clip(
            np.random.choice(vs, p=ps) + 0.1 * np.random.randn(), 0, 1)
        col[2] = 255 * v_rand
        return np.squeeze(cv2.cvtColor(col[None, None, :], cv2.COLOR_HSV2RGB))


def get_font_color(colorsRGB, colorsLAB, bg_arr):
    font_color = FontColor(colorsRGB, colorsLAB)
    return font_color.sample_from_data(bg_arr)


def get_color_matrix(col_file):
    with open(col_file, 'rb') as f:
        colorsRGB = cp.load(f, encoding='latin1')
    ncol = colorsRGB.shape[0]
    colorsLAB = np.r_[colorsRGB[:, 0:3], colorsRGB[:, 6:9]].astype(np.uint8)
    colorsLAB = np.squeeze(
        cv2.cvtColor(colorsLAB[None, :, :], cv2.COLOR_RGB2Lab))
    return colorsRGB, colorsLAB
