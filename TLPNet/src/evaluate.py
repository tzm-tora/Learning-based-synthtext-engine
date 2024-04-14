from torchvision import transforms
from src.utils import makedirs
import os
import numpy as np
import torch
import cv2
from PIL import Image
import os
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def tensor2jpg(tensor, W=256, H=256):
    tensor = tensor.detach().squeeze(0).cpu()
    tensor = (tensor * 0.5 + 0.5)
    jpg_img = transforms.ToPILImage()(tensor)
    jpg_img = cv2.cvtColor(np.asarray(jpg_img), cv2.COLOR_RGB2BGR)
    return jpg_img


def tensor2jpg_mask(tensor):
    tensor = tensor.detach().squeeze(0).cpu()
    jpg_img = transforms.ToPILImage()(tensor)
    jpg_img = cv2.cvtColor(np.asarray(jpg_img), cv2.COLOR_RGB2BGR)
    return jpg_img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def cv2tensor(input):
    if input.shape[2] == 3:
        input = Image.fromarray(cv2.cvtColor(
            input, cv2.COLOR_BGR2RGB))  # cvmat -> PILimage
        transform_list = [transforms.ToTensor(
        ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    elif input.shape[2] == 4:
        input = Image.fromarray(cv2.cvtColor(
            input, cv2.COLOR_BGRA2RGBA))  # cvmat -> PILimage
        transform_list = [transforms.ToTensor(
        ), transforms.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1))]

    transform = transforms.Compose(transform_list)
    input_tensor = transform(input)

    return input_tensor


def model_prediction(generator, device, bg):
    bg_tensor = cv2tensor(bg).to(device)
    bg_tensor_pad = bg_tensor.new_full((3, 768, 768), 0)
    bg_tensor_pad[:, :bg_tensor.shape[1], :bg_tensor.shape[2]].copy_(bg_tensor)
    bg_tensor_pad = bg_tensor_pad.unsqueeze(0)
    with torch.no_grad():
        I_hm_pad = generator(bg_tensor_pad)
        I_hm_pad = tensor2jpg_mask(I_hm_pad)
    return I_hm_pad


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files


def inference(generator, device, bg_dir, I_bg_save_path, I_Hm_save_path,
              display_save_path):
    generator.eval()

    img_list, _, txt_list, = list_files(bg_dir)
    print(len(img_list))
    a = []
    for j, bg_name in enumerate(tqdm(img_list[:])):
        bg = cv2.imread(bg_name)
        H, W, _ = bg.shape

        if H > W:
            re_H = 768 if H > 768 else (H//16)*16
            re_W = int(((768/H)*W//16)*16) if H > 768 else (W//16)*16
        else:
            # W =
            re_W = 768 if W > 768 else (W//16)*16
            re_H = int(((768/W)*H//16)*16) if W > 768 else (H//16)*16

        re_bg = cv2.resize(bg, (re_W, re_H))
        # re_bg = bg
        I_hm = model_prediction(generator, device, re_bg)
        I_hm = I_hm[:re_H, :re_W, :]

        # I_hmshow = cv2.applyColorMap(I_hm, cv2.COLORMAP_JET)

        # alpha = 0.5
        # beta = 1-alpha
        # gamma = 0

        # heatmap_blend = cv2.addWeighted(bg, alpha, I_hmshow, beta, gamma)
        # heatmap_blend = cv2.addWeighted(re_bg, alpha, I_hmshow, beta, gamma)
        # cv2.imwrite(f'./sobeldisplayshift/{img_name}'+'.jpg', display)
        # cv2.imwrite(f'./heatmap_ori/{img_name}'+'.png', heatmapuint8)

        post_I_hm = cv2.cvtColor(I_hm, cv2.COLOR_BGR2GRAY)
        _thre, post_I_hm = cv2.threshold(
            post_I_hm, 10, 255, cv2.THRESH_BINARY)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            post_I_hm)
        for i in range(1, nlabels):
            if stats[i][4] < 300:
                post_I_hm[labels == i, ] = 0
                labels[labels == i, ] = 0
        if (post_I_hm/255).sum() > 15000:
            a.append(bg_name)
            name = os.path.split(bg_name)[-1]
            cv2.imwrite(os.path.join(
                I_bg_save_path, name), re_bg)
            cv2.imwrite(os.path.join(
                I_Hm_save_path, name.replace('.jpg', '.png')), post_I_hm)

            post_I_hm_show = cv2.applyColorMap(post_I_hm, cv2.COLORMAP_JET)
            # heatmap_post_blend = cv2.addWeighted(re_bg, alpha, post_I_hm_show, beta, gamma)
            heatmap_post_blend = post_I_hm_show * 1.1 + re_bg * 0.5
            # superimposed_img_test = heatmap_test * 0.9 + test * 1.
            heatmap_post_blend = np.clip(heatmap_post_blend, 0, 255).astype(
                np.uint8)

            cv2.imwrite(os.path.join(
                display_save_path, name.replace('.jpg', '_1.jpg')), heatmap_post_blend)  # name.replace('.png', '_1.jpg'))

            # display = np.concatenate([re_bg, I_hmshow, heatmap_blend,heatmap_post_blend], axis=1)
            # cv2.imshow('post_I_hm', post_I_hm)
            # cv2.imshow('display', display)
            # cv2.waitKey()
    print(len(a))


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
