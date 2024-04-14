import shutil
import torch
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader
from datagen.dataload import DataIptIbg
from datagen.postproc import Effect
import threading
import numpy as np
from collections import OrderedDict
import os
import cv2
import time
import queue
from Iptgen.gen import Ipt_gen
import random
from _CHM.src.model import build_generator
from _GTM.src.model import SpatialTransformer, STM, STM_out
from _GTM.src.utils import makedirs, del_file, collate_fn
# 不受seed 控制的随机：postprocess dataload  Iptgen: text_choice

q = queue.Queue(maxsize=cfg.batch_size * 5)


class Generator(object):
    # Heatmap_dir, Ibg_dir, resume, model_path, save_path, batch_size
    def __init__(self, device, rank, is_distributed):
        self.Heatmap_dir = cfg.Heatmap_dir
        self.Ibg_dir = cfg.Ibg_dir
        self.ng_map_dir = cfg.ng_map_dir
        self.resume = cfg.G_resume
        self.save_path = cfg.save_path
        # self.img_save_dir = os.path.join(self.save_path, 'image')
        self.img_save_dir = os.path.join(self.save_path, 'imgs')
        # self.mask_save_dir = os.path.join(self.save_path, 'mask')
        self.seg_save_dir = os.path.join(self.save_path, 'seg')
        self.text_mask_save_dir = os.path.join(self.save_path, 'text_mask')
        self.ng_mask_save_dir = os.path.join(self.save_path, 'ng_mask')
        self.BB_map_dir = os.path.join(self.save_path, 'BB_map')
        self.bg_dir = os.path.join(self.save_path, 'bg')
        self.label_txt_dir = os.path.join(self.save_path, 'annotations')
        self.data_dict_file = os.path.join(self.save_path, 'data_dict.npy')
        self.batch_size = cfg.batch_size
        self.is_distributed = is_distributed
        self.global_step = 0
        self.device = device
        self.rank = rank
        self.data_dict = self.make_data_dict(cfg.num)

        self.GeoTranMod = self.init_net(SpatialTransformer(mode=cfg.mode))
        self.ColHarmMod = self.init_net(build_generator())
        self.Ipt_gen = Ipt_gen()

        GTM_checkpoint = torch.load(cfg.GTM_model_path, map_location=device)
        self.GeoTranMod.load_state_dict(
            self.fix_model_state_dict(GTM_checkpoint['STN']))
        CHM_checkpoint = torch.load(cfg.CHM_model_path, map_location=device)
        self.ColHarmMod.load_state_dict(
            self.fix_model_state_dict(CHM_checkpoint['net_G']))
        self.effect = Effect(self.Ibg_dir)

        t_list = []
        for i in range(self.batch_size * 2):
            t = threading.Thread(target=self.postprocess,
                                 args=(q, ),
                                 daemon=True)
            t.start()
            t_list.append(t)

    def step(self):
        if self.rank == 0:
            print('Start the generation')
        for epoch in range(0, 1000):
            if self.rank == 0:
                np.save(self.data_dict_file, self.data_dict)
            time.sleep(1)
            ##############################################################
            max_epoch = 0
            for bg_name in self.data_dict.keys():
                total_count = 0
                region_list = list(self.data_dict[bg_name])
                for region in region_list:
                    total_count = total_count + \
                        self.data_dict[bg_name][region]['count']
                    max_epoch = total_count if total_count > max_epoch else max_epoch
            if max_epoch == 0:
                break
            ##############################################################
            img_list = []
            for bg_name in self.data_dict.keys():
                region_list = list(self.data_dict[bg_name])
                for region in region_list:
                    if self.data_dict[bg_name][region]['count'] > 0:
                        region_name = region
                        img_list.append((bg_name, region_name))
                        self.data_dict[bg_name][region_name]['count'] -= 1
                        break
            self.IptIbg_loader = self.make_data_loader(img_list)
            ##############################################################

            if self.rank == 0:
                print(f'[max_epoch {max_epoch:>3}]')
                self.IptIbg_loader = tqdm(self.IptIbg_loader)
            if self.is_distributed:
                self.data_sampler.set_epoch(epoch)
            for step, batch_data in enumerate(self.IptIbg_loader):
                self.global_step += 1
                with torch.no_grad():
                    Ibg, Ipt, rects, bg_names, regions, flags, quads, texts = batch_data[
                        'Ibg'].to(self.device), batch_data['Ipt'].to(
                            self.device
                    ), batch_data['rect'].to(self.device), batch_data[
                            'bg_name'], batch_data['region'], batch_data[
                                'flag'], batch_data['quad'], batch_data['text']

                    Ipt_full = STM(Ipt, rects)
                    pmtrx = self.GeoTranMod(Ibg, Ipt_full)
                    Itt_full = STM_out(Ipt, rects, pmtrx)
                    I_out = self.ColHarmMod(Ibg, Itt_full)
                    IttAQM_full = Itt_full[:, 3:5, ...]

                    cpu_rect = rects.detach().cpu()
                    cpu_pmtrx = pmtrx.detach().cpu()
                    cpu_out = I_out.detach().cpu()
                    cpu_IttAQM_full = IttAQM_full.detach().cpu()
                    torch.cuda.empty_cache()

                    for i, img_set in enumerate(
                            zip(cpu_out, cpu_IttAQM_full, cpu_rect, cpu_pmtrx,
                                bg_names, regions, flags, quads, texts)):
                        if q.qsize() >= self.batch_size:
                            time.sleep(0.01)
                        elif q.full():
                            time.sleep(1)
                            print('queue is full', img_set[4])
                        q.put(img_set)

    def postprocess(self, q):
        while True:
            cpu_out, cpu_IttAQM_full, cpu_rect, cpu_torch_mtrx, bg_name, region, ab_flag, quad, text = q.get(
            )  # img_set
            if ab_flag:
                continue

            cpu_IttA_full, cpu_IttQM_full = cpu_IttAQM_full[
                0:1, ...], cpu_IttAQM_full[1:2, ...]
            rect = cpu_rect.numpy()
            W, H = rect[1] - rect[0]

            I_out_save_path = f"{self.img_save_dir}/{bg_name}.jpg"
            TS_img = cv2.imread(I_out_save_path)
            I_Height, I_Width, _ = TS_img.shape

            cpu_out = cpu_out[:, :I_Height, :I_Width]
            cpu_IttA_full, cpu_IttQM_full = cpu_IttA_full[:, :I_Height, :
                                                          I_Width], cpu_IttQM_full[:, :
                                                                                   I_Height, :
                                                                                   I_Width]

            param = {
                'is_shadow':
                np.random.rand() < cfg.drop_shadow_rate,
                'is_alphaBlend':
                np.random.rand() < cfg.is_alphaBlend_rate,
                'alpha':
                0.6 + 0.4 * np.random.rand(),
                'is_texture':
                np.random.rand() < cfg.is_texture_rate,
                'texture_alpha':
                cfg.texture_alpha[0] + cfg.texture_alpha[1] * np.random.rand(),
                'is_border':
                np.random.rand() < cfg.is_border_rate,
                'is_add_3D':
                np.random.rand() < cfg.is_add3D_rate,
                'shadow_angle':
                np.pi / 4 * np.random.choice(cfg.shadow_angle_degree) +
                cfg.shadow_angle_param[0] * np.random.randn(),
                'shadow_shift':
                cfg.shadow_shift_param[0] * np.random.randn(3) +
                cfg.shadow_shift_param[1],
                'shadow_opacity':
                cfg.shadow_opacity_param_real[0] * np.random.randn() +
                cfg.shadow_opacity_param_real[1],
                'shadow_scale':
                np.random.choice(cfg.shadow_scale) + 3,
                'add_3D_shift':
                cfg.add3D_shift_param[0] * np.random.randn(3) +
                cfg.add3D_shift_param[1],
            }

            ########################################
            ########## make ng_mask image##############
            ng_mask_save_path = f"{self.ng_mask_save_dir}/{bg_name}.png"
            IttA_cv_full = self.tensor2cv_mask(cpu_IttA_full)

            mask = cv2.imread(ng_mask_save_path, 0)
            overlap_num = (mask / 255 * IttA_cv_full != 0).sum()
            if overlap_num > 10 or (IttA_cv_full).sum() < 50 or (
                    IttA_cv_full).sum() / len(text) < 30:
                if random.random() < 0.9:
                    self.data_dict[bg_name][region]['count'] += 1
                continue

            IttA_cv_full255 = (IttA_cv_full * 255).astype(np.uint8)
            IttQM_cv256_full = self.tensor2cv_mask255(cpu_IttQM_full)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

            if cfg.type == 'regression':
                IttA_cv_full255_dila = cv2.dilate(IttA_cv_full255,
                                                  kernel,
                                                  iterations=1 +
                                                  np.random.randint(3),
                                                  borderValue=255)
            elif cfg.type == 'segementation':
                IttA_cv_full255_dila = cv2.dilate(IttQM_cv256_full,
                                                  kernel,
                                                  iterations=1 +
                                                  np.random.randint(3),
                                                  borderValue=255)
            mask_out = cv2.add(mask, IttA_cv_full255_dila)
            cv2.imwrite(ng_mask_save_path, mask_out)

            ########################################
            ########## make txt label##############
            label_txt_save_path = f"{self.label_txt_dir}/{bg_name}.txt"
            homo_cv = homo_torch2cv(cpu_torch_mtrx.numpy())
            quad = quad.numpy().reshape(4, 2)  # .mul(squa_len/256)
            quad = np.concatenate((quad, np.ones([4, 1])), 1)
            quad = np.array((homo_cv @ quad.T).T)
            quad = quad[:, :2] / quad[:, 2:]  # (4,2)
            quad[:, 0] = quad[:, 0] / 256 * W
            quad[:, 1] = quad[:, 1] / 256 * H
            quad = (quad + rect[0]).astype(np.int32)  # I_Height, I_Width
            quad[:, 0] = np.clip(quad[:, 0], 0, I_Width)
            quad[:, 1] = np.clip(quad[:, 1], 0, I_Height)
            x1, y1, x2, y2, x3, y3, x4, y4 = quad.flatten()
            with open(label_txt_save_path, 'a') as txt:
                txt.write(
                    f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{text}\r\n")

            ########################################
            ########## make output##############
            # quad = quad
            HeightText = cal_H_W(quad)
            I_out_cv255 = self.tensor2cv(cpu_out)
            TS_img_out, all_mask = self.effect.drop_effect(
                I_out_cv255, IttA_cv_full255, TS_img, param, I_Width, I_Height,
                HeightText)
            # print(HeightText)
            cv2.imwrite(I_out_save_path, TS_img_out)

            ########## make text_mask image##############
            text_mask_save_path = f"{self.text_mask_save_dir}/{bg_name}.png"
            # IttA_cv_full = self.tensor2cv_mask(cpu_IttA_full)
            text_mask = cv2.imread(text_mask_save_path, 0)
            IttA_cv_full255 = (IttA_cv_full * 255).astype(np.uint8)
            text_mask_out = cv2.add(text_mask, IttA_cv_full255)
            text_mask_out = cv2.add(text_mask, all_mask)
            cv2.imwrite(text_mask_save_path, text_mask_out)

            ########################################
            ########## make BBox ng map##############
            BB_map_save_path = f"{self.BB_map_dir}/{bg_name}.png"
            # IttQM_cv256_full = self.tensor2cv_mask255(cpu_IttQM_full)
            BB_map = cv2.imread(BB_map_save_path, 0)
            BB_map_out = cv2.add(BB_map, IttQM_cv256_full)
            cv2.imwrite(BB_map_save_path, BB_map_out)

    def tensor2cv(self, tensor):
        img_numpy = (tensor * 0.5 + 0.5).mul(255).byte().numpy().transpose(
            (1, 2, 0))
        img_cv = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        return img_cv

    def tensor2cv_mask(self, tensor):
        img_cv = tensor.squeeze(0).numpy()
        return img_cv

    def tensor2cv_mask255(self, tensor):
        img_cv = tensor.mul(255).byte().squeeze(0).numpy()
        return img_cv

    def tensor2cv_sp(self, tensor):
        img_cv = tensor.mul(255).byte().numpy().transpose((1, 2, 0))
        return img_cv

    def init_net(self, net):
        if self.is_distributed:
            print(f"Loading the net in GPU: {self.rank}")
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(
                self.device)
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.rank], find_unused_parameters=False)
        else:
            print("Loading the net in cpu")
            pass
        return net.eval()

    def make_data_loader(self, img_list):
        self.IptIbg_data = DataIptIbg(self.data_dict, img_list, self.Ipt_gen,
                                      self.img_save_dir, self.BB_map_dir,
                                      self.seg_save_dir)
        self.data_sampler = torch.utils.data.distributed.DistributedSampler(
            self.IptIbg_data, rank=self.rank, shuffle=False,
            drop_last=False) if self.is_distributed else None
        IptIbg_loader = DataLoader(
            self.IptIbg_data,
            batch_size=self.batch_size,
            sampler=self.data_sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )  # prefetch_factor=self.batch_size)  # , persistent_workers=True
        return IptIbg_loader

    def make_data_dict(self, num):
        if self.rank == 0:
            del_file(self.save_path, self.resume)
            makedirs(self.img_save_dir)
            makedirs(self.ng_mask_save_dir)
            makedirs(self.seg_save_dir)
            makedirs(self.BB_map_dir)
            makedirs(self.label_txt_dir)
            makedirs(self.text_mask_save_dir)
            makedirs(self.bg_dir)

        if self.resume:
            data_dict = np.load(self.data_dict_file,
                                allow_pickle='TRUE').item()

        else:

            def is_image_file(filename):
                return any(
                    filename.endswith(extension)
                    for extension in [".png", ".jpg", ".jpeg"])

            data_dict = {}
            # HF_list = [os.path.join(self.Heatmap_dir, x.replace('.png', '')) for x in os.listdir(
            #     self.Heatmap_dir) if is_image_file(x)]
            HF_list = [
                x.replace('.png', '') for x in os.listdir(self.Heatmap_dir)
                if is_image_file(x)
            ]
            # HF_list = HF_list[:2]
            print(len(HF_list))
            # print(HF_list)
            if num > len(HF_list):
                HF_list = np.random.choice(HF_list, num)
                re_HF_list = renameList(list(HF_list))
            else:
                re_HF_list = np.random.choice(HF_list, num, False)  # 10000

            seed = 0  # Seed must be between 0 and 2**32 - 1
            for i, one_heatmap in enumerate(tqdm(re_HF_list[:])):

                # HF_list[i]  # .replace('.png', '')
                old_img_name = one_heatmap
                new_img_name = one_heatmap  # .replace('.png', '')

                HF = cv2.imread(
                    os.path.join(self.Heatmap_dir, old_img_name + '.png'), 0)

                data_dict[f'{new_img_name}'] = {}
                if self.rank == 0:
                    ##############################################
                    old_path = os.path.join(self.Ibg_dir,
                                            old_img_name + '.jpg')
                    new_path = os.path.join(self.img_save_dir,
                                            new_img_name + '.jpg')
                    shutil.copyfile(old_path, new_path)
                    new_path = os.path.join(self.bg_dir, new_img_name + '.jpg')
                    shutil.copyfile(old_path, new_path)
                    ##########################################
                    BB_map_path = os.path.join(self.BB_map_dir,
                                               new_img_name + '.png')
                    cv2.imwrite(BB_map_path, np.zeros_like(HF))
                    ###########################################
                    ng_map_path = os.path.join(self.ng_map_dir,
                                               old_img_name + '.png')
                    ng_mask_save_path = os.path.join(self.ng_mask_save_dir,
                                                     new_img_name + '.png')
                    text_mask_save_path = os.path.join(self.text_mask_save_dir,
                                                       new_img_name + '.png')
                    if os.path.isfile(ng_map_path) == True:
                        shutil.copyfile(ng_map_path, ng_mask_save_path)
                    else:
                        cv2.imwrite(ng_mask_save_path, np.zeros_like(HF))
                        cv2.imwrite(text_mask_save_path, np.zeros_like(HF))
                    #################################################
                    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        HF)
                    # print(labels.dtype)
                    seg_save_path = os.path.join(self.seg_save_dir,
                                                 new_img_name + '.png')
                    cv2.imwrite(seg_save_path, labels)

                for i in range(1, nlabels):
                    if HF[labels == i, ].all() != 0:
                        if stats[i][4] > 50000:
                            count = np.random.randint(12, 20)  # stats[i][4]
                        elif stats[i][4] > 10000:
                            count = np.random.randint(8, 10)
                        elif stats[i][4] > 5000:
                            count = np.random.randint(4, 6)
                        else:
                            count = 2

                        data_dict[f'{new_img_name}'][f'{i}'] = {
                            'count': count,
                            'stats': stats[i],
                            'seed': seed
                        }
                        seed += 1
                        if seed >= 2**32 - 1:
                            seed -= 2**32 - 1

        return data_dict

    def fix_model_state_dict(self, state_dict):
        if self.device == torch.device('cpu'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            return new_state_dict
        if self.is_distributed:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                if name.startswith('rainNet.') or name.startswith('locnet.'):
                    # remove 'module.' of dataparallel
                    name = 'module.' + name[:]
                new_state_dict[name] = v
            return new_state_dict


def homo_torch2cv(homo_torch):
    N = np.array([[2 / 256, 0, -1], [0, 2 / 256, -1], [0, 0, 1]], np.float32)
    N_inv = np.linalg.inv(N)
    homo_cv = N_inv @ homo_torch @ N
    homo_cv = np.linalg.inv(homo_cv)
    homo_cv = np.array(homo_cv, np.float32)
    return homo_cv


def cal_H_W(pts):
    (tl, tr, br, bl) = pts
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB)) * 2
    return maxHeight


def printdict(data_dict):
    for bg_name in data_dict.keys():
        region_list = list(data_dict[bg_name])
        region_list.remove('labels')
        for region in region_list:
            total_count = data_dict[bg_name][region]['count']
            print(bg_name, region, total_count)


def renameList(list):
    counts = {}
    for index, key in enumerate(list):
        if key in counts:
            counts[key] += 1
            list[index] = f'{key}_{counts[key]}'
        else:
            counts[key] = 0
    return list
