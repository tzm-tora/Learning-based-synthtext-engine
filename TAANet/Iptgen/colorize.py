"""
Colorizing the text mask.
Change the original code to Python3 support and simplifed the code structure.
Original project: https://github.com/ankush-me/SynthText
Author: Ankush Gupta
Date: 2015
"""
import cv2
import numpy as np


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


class Colorize(object):

    def __init__(self):
        pass

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

    def drop_shadow(self, alpha, theta, shift, size, op=0.95):
        if size % 2 == 0:
            size -= 1
            size = max(1, size)
        shadow = cv2.GaussianBlur(alpha, (size, size), 0)
        # [dx, dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        dx, dy = (shift * np.array(
            [-np.sin(theta), np.cos(theta)])).astype(np.int16)
        shadow = op * self.shift(shadow,  dy, dx)
        shadow = np.clip(shadow, 0, 255)
        return shadow.astype(np.uint8)

    def add_pseudo3D(self, alpha, theta, shift):
        # [dx, dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        dx, dy = (shift * np.array(
            [-np.sin(theta), np.cos(theta)])).astype(np.int16)
        pseudo3D = self.shift(alpha, dy, dx)
        return pseudo3D.astype(np.uint8)

    def border(self, alpha, size, kernel_type='RECT'):
        kdict = {'RECT': cv2.MORPH_RECT, 'ELLIPSE': cv2.MORPH_ELLIPSE,
                 'CROSS': cv2.MORPH_CROSS}
        kernel = cv2.getStructuringElement(kdict[kernel_type], (size, size))
        border = cv2.dilate(alpha, kernel, iterations=1+np.random.randint(3),
                            borderValue=0)  # - alpha
        return border

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

    def color(self, text_arr, fg_col, param):
        min_h = 50
        if min_h <= 20:
            bsz = 1
        elif min_h <= 50:
            bsz = 3
        else:
            bsz = 5
        l_text = Layer(alpha=text_arr, color=fg_col)
        all_mask = l_text.alpha.copy()
        layers = []
        blends = []

        layers.append(l_text)
        blends.append('normal')

        # add border:' bordar_color': tuple(np.random.randint(0, 256, 3)),
        border_a = np.zeros_like(all_mask)
        if param['is_border']:
            border_a = self.border(l_text.alpha, size=bsz)
            all_mask = cv2.add(all_mask, border_a)
            border_col = (np.random.rand(3) * 255.).astype(np.uint8)
            l_border = Layer(border_a, color=border_col)

            layers.append(l_border)
            blends.append('normal')

        # # add is_add_3D:
        pseudo3D_mask = np.zeros_like(all_mask)
        if param['is_add_3D']:
            # shadow angle:
            theta = param['shadow_angle']
            # shadow shift:
            if min_h <= 20:
                shift = param['add_3D_shift'][0]
            elif min_h < 40:
                shift = param['add_3D_shift'][1]
            else:
                shift = param['add_3D_shift'][2]

            B, G, R = border_col if param['is_border'] else fg_col
            B_3D, G_3D, R_3D = np.random.randint(
                -1, B), np.random.randint(-1, G), np.random.randint(-1, R)
            pseudo3D_col = np.array([B_3D, G_3D, R_3D], dtype=np.uint8)

            pseudo3D_mask = np.zeros_like(all_mask, dtype=np.uint8)
            for i in range(int(shift)+2):
                # pseudo3D = self.add_pseudo3D(l_text.alpha, theta, i)
                pseudo3D = self.add_pseudo3D(all_mask, theta, i)
                pseudo3D_mask = cv2.add(pseudo3D_mask, pseudo3D)
            # pseudo3D_mask_50 = (pseudo3D_mask/255*50).astype(np.uint8)
            all_mask = cv2.add(all_mask, pseudo3D_mask)

            if param['is_shadow']:
                op = param['shadow_opacity']
                sha_scale = param['shadow_scale']
                shadow = self.drop_shadow(
                    l_text.alpha, theta, shift, sha_scale*bsz, op)
                # cv2.imshow('pseudo3D_mask', pseudo3D_mask)
                # cv2.imshow('shadow', shadow)
                shadow = np.where(shadow < pseudo3D_mask,
                                  shadow, pseudo3D_mask)
                # cv2.imshow('shadowin3D', shadow)
                l_shadow = Layer(shadow, 0)
                layers.append(l_shadow)
                blends.append('normal')

            l_add_3D = Layer(pseudo3D_mask, pseudo3D_col)  # random color
            layers.append(l_add_3D)
            blends.append('normal')

        black_layers = layers.copy()
        gray_blends = blends.copy()
        l_bg_black = Layer(alpha=255 * np.ones_like(text_arr,
                                                    dtype=np.uint8),
                           color=(0, 0, 0))
        black_layers.append(l_bg_black)
        gray_blends.append('normal')
        plain_text = self.merge_down(black_layers, gray_blends)

        return plain_text.color, all_mask, border_a, pseudo3D_mask

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


def colorize(surf1, fg_col, min_h, param):
    c = Colorize()
    return c.color(surf1, fg_col, min_h, param)
