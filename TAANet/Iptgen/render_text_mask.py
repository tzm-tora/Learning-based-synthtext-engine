"""
Rendering text mask.
Change the original code to Python3 support and simplifed the code structure.
Original project: https://github.com/ankush-me/SynthText
Author: Ankush Gupta
Date: 2015
"""

import cv2
import math
import numpy as np
import pygame
import pygame.locals


def center2size(surf, size):
    canvas = np.zeros(size).astype(np.uint8)
    size_h, size_w = size
    surf_h, surf_w = surf.shape[:2]
    canvas[(size_h - surf_h) // 2:(size_h - surf_h) // 2 + surf_h,
           (size_w - surf_w) // 2:(size_w - surf_w) // 2 + surf_w] = surf
    return canvas


def crop_safe(arr, rect, bbs=[], pad=0):
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2 * pad
    v0 = [max(0, rect[0]), max(0, rect[1])]
    v1 = [min(arr.shape[0], rect[0] + rect[2]),
          min(arr.shape[1], rect[1] + rect[3])]
    arr = arr[v0[0]:v1[0], v0[1]:v1[1], ...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i, 0] -= v0[0]
            bbs[i, 1] -= v0[1]
        return arr, bbs
    else:
        return arr


def render_normal_rotate(font, text, angle):
    text_line = text.split('\n')
    # print(text_line)
    lengths = [len(l) for l in text_line]
    # print(text_line[0])
    # font parameters:
    line_spacing = font.get_sized_height() + 1  # text height

    # initialize the surface to proper size:
    line_bounds = font.get_rect(text_line[np.argmax(lengths)])
    fsize = (round(2 * line_bounds.width),
             round(3 * line_spacing))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
    # print('fsize', fsize)

    bbs = []
    space = font.get_rect('O')
    x, y = 1.5 * space.height, 1.2 * line_spacing
    # xy = []
    for char in text_line[0]:
        # for ch in l:  # render each character
        if char.isspace():  # just shift
            x += space.height
        else:
            # render the character 逐字渲染
            # xy.append((x, y))
            ch_bounds = font.render_to(surf, (x, y), char, rotation=angle)
            # ch_bounds.x = x + ch_bounds.x
            # ch_bounds.y = y - ch_bounds.y
            x += ch_bounds.width
            bbs.append(np.array(ch_bounds))

    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    # crop the surface to fit the text:
    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(
        pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=0)
    # surf_arr = pygame.surfarray.pixels_alpha(surf)
    surf_arr = surf_arr.swapaxes(0, 1)

    # jpg_img = cv2.cvtColor(np.asarray(surf_arr), cv2.COLOR_RGB2BGR)
    # for i in xy:
    #     # print(i)
    #     cv2.circle(jpg_img, (int(i[0]), int(i[1])), 3, (255), 5)
    # for bb in bbs:
    #     # print(bb)
    #     cv2.rectangle(jpg_img, (bb[0], bb[1]),
    #                   (bb[0]+bb[2], bb[1]+bb[3]), (255), 1)
    #     cv2.imshow('jpg_img', jpg_img)
    return surf_arr, bbs


def render_normal(font, text):

    line_spacing = font.get_sized_height() + 1  # text height
    # initialize the surface to proper size:

    # try:
    #     line_bounds = font.get_rect(text) if len(
    #         text) > 1 else font.get_rect('TTT')
    # except:
    #     print(text)
    #     line_bounds = font.get_rect('000000000')
    # else:
    #     line_bounds = font.get_rect(text) if len(
    #         text) > 1 else font.get_rect('TTT')

    line_bounds = font.get_rect(text) if len(
        text) > 1 else font.get_rect('TTT')
    fsize = (round(4 * line_bounds.width),
             round(4 * line_spacing))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
    # print('fsize', fsize)

    bbs = []
    space = font.get_rect('O')
    # x, y = space.width, line_spacing
    x, y = space.width*2, line_spacing
    interval = 1+0.1*np.random.rand()*space.width
    for char in text:
        if char.isspace():  # just shift
            x += space.width
        else:
            # render the character 逐字渲染
            ch_bounds = font.render_to(surf, (x, y), char, rotation=0)
            x += ch_bounds.width + interval
            bbs.append(np.array(ch_bounds))

    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    # crop the surface to fit the text:
    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(
        pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=3)

    surf_arr = surf_arr.swapaxes(0, 1)
    return surf_arr, bbs


def render_curved(font, text, curve_rate, curve_center=None):
    wl = len(text)
    curve_rate = np.clip(curve_rate, -0.8, 0.8)
    lspace = font.get_sized_height() + 1
    line_bounds = font.get_rect(text) if len(
        text) > 1 else font.get_rect('TTT')

    fsize = (round(4.0 * line_bounds.width), round(20.0 * lspace))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    # baseline state
    if curve_center is None:
        curve_center = wl // 2
    curve_center = max(curve_center, 0)
    curve_center = min(curve_center, wl - 1)
    mid_idx = curve_center  # wl//2
    curve = [curve_rate * (i - mid_idx) * (i - mid_idx) for i in range(wl)]
    curve[mid_idx] = -np.sum(curve) / max(wl - 1, 1)
    rots = [-int(math.degrees(math.atan(5 * curve_rate *
                                        (i - mid_idx) / (font.size / 2)))) for i in range(wl)]
    bbs = []

    # place middle char
    rect = font.get_rect(text[mid_idx])
    if text[mid_idx].isspace():  # just shift
        mid_idx += 1
    # else:
    rect.centerx = surf.get_rect().centerx
    rect.centery = surf.get_rect().centery  # + 4*rect.height
    rect.centery += curve[mid_idx]
    ch_bounds = font.render_to(
        surf, rect, text[mid_idx], rotation=rots[mid_idx])
    mid_ch_bb = np.array(ch_bounds)

    # render chars to the left and right:
    last_rect = rect
    ch_idx = []

    spacewidth = 0

    rand = 0.1+0.1*np.random.rand()
    for i in range(wl):
        # skip the middle character
        if i == mid_idx:
            bbs.append(mid_ch_bb)
            ch_idx.append(i)
            continue

        if i < mid_idx:  # left-chars
            i = mid_idx - 1 - i
        elif i == mid_idx + 1:  # right-chars begin
            last_rect = rect

        char = text[i]
        newrect = font.get_rect(char)
        if char.isspace():  # just shift
            space = font.get_rect('O')
            spacewidth = space.width
        else:
            ch_idx.append(i)
            newrect.y = last_rect.y

            interval = font.get_rect('O').width*rand * \
                math.cos(math.radians(rots[i]))**3+spacewidth
            intervaly = math.cos(math.radians(rots[i]))
            # print(char, '   ',curve[i] ,curve[i]*intervaly)
            if i > mid_idx:
                newrect.topleft = (
                    last_rect.topright[0]+interval, newrect.topleft[1])
            else:
                newrect.topright = (
                    last_rect.topleft[0] - interval, newrect.topleft[1])
            spacewidth = 0

            newrect.centery = newrect.centery + curve[i]*intervaly
            try:
                bbrect = font.render_to(surf, newrect, char, rotation=rots[i])
            except ValueError:
                bbrect = font.render_to(surf, newrect, char)
            except TypeError:
                print(newrect, char)
                bbrect = font.render_to(surf, newrect, char)
            bbs.append(np.array(bbrect))
            last_rect = newrect

    # correct the bounding-box order:
    bbs_sequence_order = [None for i in ch_idx]
    # print(ch_idx)
    ch_idx_sorted = ch_idx.copy()
    # ch_idx_sorted.sort()
    ch_idx_wo_space = [None for i in ch_idx]
    for idx, i in enumerate(ch_idx):
        ch_idx_wo_space[idx] = ch_idx_sorted.index(i)
    # print('after：', ch_idx_wo_space)
    for idx, i in enumerate(ch_idx_wo_space):
        bbs_sequence_order[i] = bbs[idx]
    bbs = bbs_sequence_order
    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    # crop the surface to fit the text:
    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(
        pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=3)

    surf_arr = surf_arr.swapaxes(0, 1)
    return surf_arr, bbs


def center_warpPerspective(img, H, center, size):
    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    img = cv2.warpPerspective(img, M, size,
                              cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return img


def render_text(font, text, param):
    if param['is_curve']:
        return render_curved(font, text, param['curve_rate'], param['curve_center'])
    if param['is_rotate']:
        angle = int(90 * np.random.choice([1, 1, 2, 3, 3], 1))
        return render_normal_rotate(font, text, angle)
    else:
        return render_normal(font, text)
