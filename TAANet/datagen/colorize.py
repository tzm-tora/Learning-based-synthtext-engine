"""
Colorizing the text mask.
Change the original code to Python3 support and simplifed the code structure.
Original project: https://github.com/ankush-me/SynthText
Author: Ankush Gupta
Date: 2015
"""
import cv2
import numpy as np
import pickle as cp
from PIL import Image
import random


class Layer(object):

    def __init__(self, alpha, color):

        # alpha for the whole image:
        assert alpha.ndim == 2
        self.alpha = alpha
        [n, m] = alpha.shape[:2]

        color = np.atleast_1d(np.array(color)).astype(np.uint8)
        # color for the image:
        if color.ndim == 1:  # constant color for whole layer
            ncol = color.size
            if ncol == 1:  # grayscale layer
                self.color = color * np.ones((n, m, 3), dtype=np.uint8)
            if ncol == 3:
                self.color = np.ones(
                    (n, m, 3), dtype=np.uint8) * color[None, None, :]
        elif color.ndim == 2:  # grayscale image
            self.color = np.repeat(
                color[:, :, None], repeats=3, axis=2).copy().astype(np.uint8)
        elif color.ndim == 3:  # rgb image
            self.color = color.copy().astype(np.uint8)
        else:
            print(color.shape)
            raise Exception("color datatype not understood")


class FontColor(object):

    def __init__(self, colorsRGB, colorsLAB):

        self.colorsRGB = colorsRGB
        self.colorsLAB = colorsLAB
        self.ncol = colorsRGB.shape[0]

    def sample_normal(self, col_mean, col_std):

        col_sample = col_mean + col_std * np.random.randn()
        return np.clip(col_sample, 0, 255).astype(np.uint8)

    def sample_from_data(self, bg_mat):

        bg_orig = bg_mat.copy()
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

        col_hsv = np.squeeze(cv2.cvtColor(
            rgb_color[None, None, :], cv2.COLOR_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128  # uint8 mods to 255
        col_comp = np.squeeze(cv2.cvtColor(
            col_hsv[None, None, :], cv2.COLOR_HSV2RGB))
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

        col = np.squeeze(cv2.cvtColor(
            col_rgb[None, None, :], cv2.COLOR_RGB2HSV))
        x = col[2]
        vs = np.linspace(0, 1)
        ps = np.abs(vs - x / 255.0)
        ps /= np.sum(ps)
        v_rand = np.clip(np.random.choice(vs, p=ps) +
                         0.1 * np.random.randn(), 0, 1)
        col[2] = 255 * v_rand
        return np.squeeze(cv2.cvtColor(col[None, None, :], cv2.COLOR_HSV2RGB))


class Colorize(object):

    def __init__(self):
        pass

    def blur(self, alpha, size):
        if size % 2 == 0:
            size -= 1
            size = max(1, size)
        size = size + 2
        shadow = cv2.GaussianBlur(alpha, (size, size), 0)
        return shadow.astype(np.uint8)

    def shift(self, img, dy, dx, fill_value=0):
        img = np.roll(img, (dy, dx), axis=(0, 1))
        if dy < 0:
            img[dy:, :] = fill_value
        elif dy > 0:
            img[: dy, :] = fill_value
        if dx < 0:
            img[:, dx:] = fill_value
        elif dx > 0:
            img[:, : dx] = fill_value
        return img.astype(np.uint8)

    def drop_shadow(self, alpha, theta, shift, size, op):
        shadow = cv2.GaussianBlur(alpha, (size, size), 0)
        dx, dy = (shift * np.array(
            [-np.sin(theta), np.cos(theta)])).astype(np.int16)
        shadow = op*self.shift(shadow, dy, dx)
        shadow = np.clip(shadow, 0, 255)
        return shadow.astype(np.uint8)

    def blend(self, cf, cb, mode='normal'):
        return cf

    def merge_two(self, fore, back, blend_type=None):
        a_f = fore.alpha / 255.0
        a_b = back.alpha / 255.0
        c_f = fore.color
        c_b = back.color

        a_r = a_f + a_b - a_f * a_b
        if blend_type != None:
            c_blend = self.blend(c_f, c_b, blend_type)
            c_r = (((1 - a_f) * a_b)[:, :, None] * c_b
                   + ((1 - a_b) * a_f)[:, :, None] * c_f
                   + (a_f * a_b)[:, :, None] * c_blend)
        else:
            c_r = (((1 - a_f) * a_b)[:, :, None] * c_b
                   + a_f[:, :, None] * c_f)

        return Layer((255 * a_r).astype(np.uint8), c_r.astype(np.uint8))

    def merge_down(self, layers, blends=None):
        nlayers = len(layers)
        if nlayers > 1:
            [n, m] = layers[0].alpha.shape[:2]
            out_layer = layers[-1]
            for i in range(-2, -nlayers - 1, -1):
                blend = None
                if blends is not None:
                    blend = blends[i + 1]
                    out_layer = self.merge_two(
                        fore=layers[i], back=out_layer, blend_type=blend)
            return out_layer
        else:
            return layers[0]

    def resize_im(self, im, osize):
        return np.array(Image.fromarray(im).resize(osize[::-1], Image.BICUBIC))

    def color_border(self, col_text, col_bg, bordar_color_type, bordar_color_idx, bordar_color_noise):

        choice = np.random.choice(3)

        col_text = cv2.cvtColor(col_text, cv2.COLOR_RGB2HSV)
        col_text = np.reshape(col_text, (np.prod(col_text.shape[:2]), 3))
        col_text = np.mean(col_text, axis=0).astype(np.uint8)

        vs = np.linspace(0, 1)

        def get_sample(x):
            ps = np.abs(vs - x / 255.0)
            ps /= np.sum(ps)
            v_rand = np.clip(np.random.choice(vs, p=ps) +
                             0.1 * bordar_color_noise, 0, 1)
            return 255 * v_rand

        # first choose a color, then inc/dec its VALUE:
        if choice == 0:
            # increase/decrease saturation:
            col_text[0] = get_sample(col_text[0])  # saturation
            col_text = np.squeeze(cv2.cvtColor(
                col_text[None, None, :], cv2.COLOR_HSV2RGB))
        elif choice == 1:
            # get the complementary color to text:
            col_text = np.squeeze(cv2.cvtColor(
                col_text[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = self.font_color.complement(col_text)
        else:
            # choose a mid-way color:
            col_bg = cv2.cvtColor(col_bg, cv2.COLOR_RGB2HSV)
            col_bg = np.reshape(col_bg, (np.prod(col_bg.shape[:2]), 3))
            col_bg = np.mean(col_bg, axis=0).astype(np.uint8)
            col_bg = np.squeeze(cv2.cvtColor(
                col_bg[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = np.squeeze(cv2.cvtColor(
                col_text[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = self.font_color.triangle_color(col_text, col_bg)

        # now change the VALUE channel:
        col_text = np.squeeze(cv2.cvtColor(
            col_text[None, None, :], cv2.COLOR_RGB2HSV))
        col_text[2] = get_sample(col_text[2])  # value
        return np.squeeze(cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))

    def color_text(self, text_arr, bg_arr):

        fg_col, bg_col = self.font_color.sample_from_data(bg_arr)
        return Layer(alpha=text_arr, color=fg_col), fg_col, bg_col

    # def color(self, text_arr, bg_arr, fg_col, bg_col, colorsRGB, colorsLAB, min_h, param):

    def color(self, Iptmask, Itexture, textA, bg_cv, fg_col, param, W, H):

        text_arr, border_a, pseudo3D_mask = cv2.split(Iptmask)

        # cv2.imshow('text_arr', text_arr.astype(np.uint8))
        # cv2.imshow('border_a', border_a.astype(np.uint8))
        # cv2.imshow('border_a', border_a.astype(np.uint8))
        # self.font_color = FontColor(colorsRGB, colorsLAB)

        if param['is_alphaBlend']:
            alpha = param['alpha']
        else:
            alpha = 1
        # l_text = Layer(alpha=text_A*alpha, color=text_arr)
        l_text = Layer(alpha=text_arr*alpha, color=fg_col)
        # bg_col = np.mean(np.mean(bg_arr, axis=0), axis=0)
        all_mask = text_arr.copy()
        layers = []
        blends = []

        # print(border_a)
        # param['texture_alpha'] = border_a[0, 0]/10
        # print(param['texture_alpha'])
        # if border_a[0, 0] != 0:
        #     # param['texture_alpha'] =
        #     l_texture = Layer(param['texture_alpha']*l_text.alpha, Itexture)
        #     layers.append(l_texture)
        #     blends.append('normal')

        layers.append(l_text)
        blends.append('normal')

        # add border:
        if border_a.sum() > 255:

            l_border = Layer(border_a*alpha, color=(
                np.random.rand(3) * 255.).astype(np.uint8))
            layers.append(l_border)
            blends.append('normal')

        # # add is_add_3D:
        if pseudo3D_mask.sum() > 0:
            # shadow angle:
            # theta = param['shadow_angle']

            pseudo3D_color = (np.random.rand(3) * 255.).astype(np.uint8)

            # all_mask = cv2.add(all_mask, pseudo3D_mask)

            # if param['is_shadow']:
            #     op = param['shadow_opacity']
            #     sha_scale = param['shadow_scale']
            #     shadow = self.drop_shadow(
            #         l_text.alpha, theta, shift, sha_scale*bsz, op)
            #     # cv2.imshow('pseudo3D_mask', pseudo3D_mask)
            #     # cv2.imshow('shadow', shadow)
            #     shadow = np.where(shadow < pseudo3D_mask,
            #                       shadow, pseudo3D_mask)
            #     # cv2.imshow('shadowin3D', shadow)
            #     l_shadow = Layer(shadow, 0)
            #     layers.append(l_shadow)
            #     blends.append('normal')

            l_add_3D = Layer(pseudo3D_mask*alpha,
                             pseudo3D_color)  # random color
            layers.append(l_add_3D)
            blends.append('normal')

        if param['is_shadow']:
            shift = param['shift']  # 3 * np.random.randn() + 1
            op = param['op']  # 0.2 * np.random.randn() + 0.8
            bsz = param['bsz']  # np.random.choice([3, 5, 7, 9, 11, 13])
            theta = param['theta']
            shadow = self.drop_shadow(textA, theta, shift, bsz, op)
            l_shadow = Layer(shadow, 0)
            layers.append(l_shadow)
            blends.append('normal')

        l_bg = Layer(alpha=255 * np.ones_like(textA,
                                              dtype=np.uint8), color=bg_cv)
        layers.append(l_bg)
        blends.append('normal')

        l_out = self.merge_down(layers, blends)
        l_out = l_out.color
        l_out = self.add_blur(l_out, param['blur_rate'], param['blur1'], param[
            'blur2'], W, H)
        return l_out, all_mask

    def add_blur(self, l_out, blur_rate, blur1, blur2, W, H):
        # blur_rate = np.random.rand()
        if W < 80 or H < 50:
            if blur_rate < 0.05:
                l_out = self.blur(l_out, 3)
            elif blur_rate < 0.1:
                l_out = cv2.medianBlur(l_out, 3)
        else:
            if blur_rate < 0.01 and H > 50:
                l_out = cv2.GaussianBlur(
                    l_out, (blur1, 1), 7)
            elif blur_rate < 0.015 and H > 50:
                l_out = cv2.GaussianBlur(l_out, (1, 7), 7)
            elif blur_rate < 0.05:
                l_out = self.blur(l_out, blur2)
            elif blur_rate < 0.1:
                l_out = cv2.medianBlur(l_out, blur2)
        return l_out

    def make_shadow(self, alpha, theta, shift, size, op, fill_value=0):
        if size % 2 == 0:
            size -= 1
            size = max(1, size)
        shadow = cv2.GaussianBlur(alpha, (size, size), 0)
        dx, dy = (shift * np.array(
            [-np.sin(theta), np.cos(theta)])).astype(np.int16)
        shadow = op*np.roll(shadow, (dy, dx), axis=(0, 1))
        if dy < 0:
            shadow[dy:, :] = fill_value
        elif dy > 0:
            shadow[: dy, :] = fill_value
        if dx < 0:
            shadow[:, dx:] = fill_value
        elif dx > 0:
            shadow[:, : dx] = fill_value
        return shadow.astype(np.uint8)  # [..., np.newaxis].repeat(3, 2)


def get_color_matrix(col_file):
    with open(col_file, 'rb') as f:
        colorsRGB = cp.load(f, encoding='latin1')
    ncol = colorsRGB.shape[0]
    colorsLAB = np.r_[colorsRGB[:, 0:3], colorsRGB[:, 6:9]].astype(np.uint8)
    colorsLAB = np.squeeze(cv2.cvtColor(
        colorsLAB[None, :, :], cv2.COLOR_RGB2Lab))
    return colorsRGB, colorsLAB


def get_font_color(colorsRGB, colorsLAB, bg_arr):
    font_color = FontColor(colorsRGB, colorsLAB)
    return font_color.sample_from_data(bg_arr)


def colorize(Iptmask_path, Itexture, textA_cv, bg_cv, fg_col, param, W, H):
    c = Colorize()
    return c.color(Iptmask_path, Itexture, textA_cv, bg_cv, fg_col, param, W, H)
