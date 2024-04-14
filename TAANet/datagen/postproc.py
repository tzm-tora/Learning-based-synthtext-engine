import cv2
import numpy as np
import random
import os
# import cfg


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
            self.color = np.repeat(color[:, :, None], repeats=3,
                                   axis=2).copy().astype(np.uint8)
        elif color.ndim == 3:  # rgb image
            self.color = color.copy().astype(np.uint8)
        else:
            print(color.shape)
            raise Exception("color datatype not understood")


class Effect(object):
    def __init__(self, Ibg_dir):
        def is_image_file(filename):
            return any(
                filename.endswith(extension)
                for extension in [".png", ".jpg", ".jpeg"])

        self.Ibg_dir = Ibg_dir
        self.bg_list = [x for x in os.listdir(Ibg_dir) if is_image_file(x)]

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
            img[:dy, :] = fill_value
        if dx < 0:
            img[:, dx:] = fill_value
        elif dx > 0:
            img[:, :dx] = fill_value
        return img.astype(np.uint8)

    def drop_shadow(self, alpha, shift, size, op):
        theta = np.pi / 360 * np.random.choice(360)
        shadow = cv2.GaussianBlur(alpha, (size, size), 0)
        dx, dy = (shift *
                  np.array([-np.sin(theta), np.cos(theta)])).astype(np.int16)
        shadow = op * self.shift(shadow, dy, dx)
        shadow = np.clip(shadow, 0, 255)
        return shadow.astype(np.uint8)

    def drop_shadow_3D(self, alpha, theta, shift, size, op=0.95):
        if size % 2 == 0:
            size -= 1
        size = max(3, size)
        # print(size)
        shadow = cv2.GaussianBlur(alpha, (size, size), 0)
        # [dx, dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        dx, dy = (shift *
                  np.array([-np.sin(theta), np.cos(theta)])).astype(np.int16)
        shadow = op * self.shift(shadow, dy, dx)
        shadow = np.clip(shadow, 0, 255)
        return shadow.astype(np.uint8)

    def border(self, alpha, size, kernel_type='RECT'):
        kdict = {
            'RECT': cv2.MORPH_RECT,
            'ELLIPSE': cv2.MORPH_ELLIPSE,
            'CROSS': cv2.MORPH_CROSS
        }
        kernel = cv2.getStructuringElement(kdict[kernel_type], (size, size))
        border = cv2.dilate(alpha, kernel, iterations=1,
                            borderValue=0)  # - alpha
        return border.astype(np.uint8)

    def add_pseudo3D(self, alpha, theta, shift):
        # [dx, dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        dx, dy = (shift *
                  np.array([-np.sin(theta), np.cos(theta)])).astype(np.int16)
        pseudo3D = self.shift(alpha, dy, dx)
        return pseudo3D.astype(np.uint8)

    def merge_two(self, fore, back, blend_type=None):
        a_f = fore.alpha / 255.0
        a_b = back.alpha / 255.0
        c_f = fore.color
        c_b = back.color

        a_r = a_f + a_b - a_f * a_b
        if blend_type != None:
            c_blend = c_f
            c_r = (((1 - a_f) * a_b)[:, :, None] * c_b +
                   ((1 - a_b) * a_f)[:, :, None] * c_f +
                   (a_f * a_b)[:, :, None] * c_blend)
        else:
            c_r = (((1 - a_f) * a_b)[:, :, None] * c_b + a_f[:, :, None] * c_f)

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
                    out_layer = self.merge_two(fore=layers[i],
                                               back=out_layer,
                                               blend_type=blend)
            return out_layer
        else:
            return layers[0]

    def get_texture(self, I_Width, I_Height):
        bg_name = np.random.choice(self.bg_list)
        bg = cv2.imread(os.path.join(self.Ibg_dir, bg_name))
        texture_arr = cv2.resize(bg, (I_Width, I_Height))
        texture_arr = cv2.medianBlur(texture_arr, random.choice([5, 7, 9]))
        return texture_arr

    def drop_effect(self, text_arr, text_A, bg_arr, param, I_Width, I_Height,
                    HeightText):
        # min_h = HeightText  #np.random.randint(-1, 80)
        if HeightText <= 20:
            bsz = 1
        elif HeightText <= 40:
            bsz = 3
        elif HeightText <= 60:
            bsz = 5
        elif HeightText <= 100:
            bsz = 7
        else:
            bsz = 9

        if bsz >= 5:
            negposx = -1 if np.random.rand() <= 0.5 else 1
            bsz = bsz + negposx * 2
        # print(bsz)
        if param['is_alphaBlend']:
            alpha = param['alpha']
            param['is_shadow'] = False
            param['is_texture'] = False
        else:
            alpha = 1
        l_text = Layer(alpha=text_A * alpha, color=text_arr)
        all_mask = l_text.alpha.copy().astype(np.uint8)

        layers = []
        blends = []

        if param['is_texture']:
            texture_arr = self.get_texture(I_Width, I_Height)
            l_texture = Layer(param['texture_alpha'] * l_text.alpha,
                              texture_arr)
            layers.append(l_texture)
            blends.append('normal')

        layers.append(l_text)
        blends.append('normal')

        border_a = np.zeros_like(all_mask)
        if param['is_border'] and bsz > 1:
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
            if HeightText <= 20:
                shift = param['add_3D_shift'][0]
            elif HeightText < 40:
                shift = param['add_3D_shift'][1]
            else:
                shift = param['add_3D_shift'][2]

            fg_col = (np.random.rand(3) * 255.).astype(np.uint8)
            B, G, R = border_col if param['is_border'] and bsz > 1 else fg_col
            B_3D, G_3D, R_3D = np.random.randint(-1, B), np.random.randint(
                -1, G), np.random.randint(-1, R)
            pseudo3D_col = np.array([B_3D, G_3D, R_3D], dtype=np.uint8)

            pseudo3D_mask = np.zeros_like(all_mask, dtype=np.uint8)
            for i in range(int(shift) + 2):
                # pseudo3D = self.add_pseudo3D(l_text.alpha, theta, i)
                pseudo3D = self.add_pseudo3D(all_mask, theta, i)
                pseudo3D_mask = cv2.add(pseudo3D_mask, pseudo3D)
            # pseudo3D_mask_50 = (pseudo3D_mask/255*50).astype(np.uint8)
            all_mask = cv2.add(all_mask, pseudo3D_mask)

            if param['is_shadow']:
                op = param['shadow_opacity']
                sha_scale = param['shadow_scale']
                shadow = self.drop_shadow_3D(l_text.alpha, theta, shift,
                                             sha_scale * bsz, op)
                # cv2.imshow('pseudo3D_mask', pseudo3D_mask)
                # cv2.imshow('shadow', shadow)
                shadow = np.where(shadow < pseudo3D_mask, shadow,
                                  pseudo3D_mask)
                # cv2.imshow('shadowin3D', shadow)
                l_shadow = Layer(shadow, 0)
                layers.append(l_shadow)
                blends.append('normal')

            l_add_3D = Layer(pseudo3D_mask, pseudo3D_col)  # random color
            layers.append(l_add_3D)
            blends.append('normal')

        if param['is_shadow']:
            # shift = 3 * np.random.randn() + 1
            if HeightText <= 20:
                shift = param['shadow_shift'][0]
            elif HeightText < 40:
                shift = param['shadow_shift'][1]
            else:
                shift = param['shadow_shift'][2]
            op = 0.3 * np.random.randn() + 0.8
            bsz = bsz + np.random.choice([2, 4])  # * bsz
            shadow = self.drop_shadow(text_A, shift, bsz, op)
            all_mask = cv2.add(all_mask, shadow)
            l_shadow = Layer(shadow, 0)
            layers.append(l_shadow)
            blends.append('normal')

        l_bg = Layer(alpha=255 * np.ones_like(text_A, dtype=np.uint8),
                     color=bg_arr)
        layers.append(l_bg)
        blends.append('normal')

        l_out = self.merge_down(layers, blends)
        l_out = l_out.color

        # blur_rate = np.random.rand()
        # if H < 50:
        #     if blur_rate < 0.05:
        #         l_out = self.blur(l_out, 3)
        #     elif blur_rate < 0.1:
        #         l_out = cv2.medianBlur(l_out, 3)
        # else:
        #     if blur_rate < 0.01 and H > 50:
        #         l_out = cv2.GaussianBlur(
        #             l_out, (random.choice([5, 7, 9]), 1), 7)
        #     elif blur_rate < 0.015 and H > 50:
        #         l_out = cv2.GaussianBlur(l_out, (1, 7), 7)
        #     elif blur_rate < 0.05:
        #         l_out = self.blur(l_out, random.choice([3, 5]))
        #     elif blur_rate < 0.1:
        #         l_out = cv2.medianBlur(l_out, random.choice([3, 5]))
        return l_out, all_mask
