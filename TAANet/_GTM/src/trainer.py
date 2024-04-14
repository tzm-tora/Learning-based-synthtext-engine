import torch
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import to_items, reduce_value
from torchvision.utils import make_grid
from .utils import create_ckpt_dir
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .model import build_discriminator, SpatialTransformer, STM, STM_out
from src.dataset import DecompST4GTM, DecompST4GTM_val
from src.loss import STNLoss
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, device, rank):
        self.flag_epoch = 0
        self.global_step = 0
        self.device = device
        self.rank = rank

        self.train_loader, self.val_dataset = self.make_data_loader()

        self.STN = self.init_net(
            SpatialTransformer(cfg.mode).to(self.device))
        self.net_D = self.init_net(build_discriminator().to(self.device))

        self.init_optimizer()
        self.STNLoss = STNLoss().to(self.device)

        if cfg.resume:
            self.resume_model()

        self.loss_dict = {}
        self.tb_writer = None
        self.best_dice = 0
        if self.rank == 0:
            self.save_dir, self.tensorboard_dir = create_ckpt_dir(
                cfg.ckpt_dir, cfg.resume)
            self.tb_writer = SummaryWriter(self.tensorboard_dir)

    def iterate(self):
        if self.rank == 0:
            print('Start the training')
        for epoch in range(self.flag_epoch, cfg.max_epoch + 1):
            if self.rank == 0:
                print(f'[epoch {epoch:>3}]')
                self.train_loader = tqdm(self.train_loader)
            self.train_sampler.set_epoch(epoch)
            for step, batch_data in enumerate(self.train_loader):
                self.global_step += 1
                bg, P_pt, gt_text, homo_gt, rect = batch_data['bg'].to(self.device), batch_data['P_pt'].to(
                    self.device), batch_data['gt_text'].to(self.device), batch_data['homo_gt'].to(self.device), batch_data['rect'].to(self.device)

                # Gtext_roi_t = apply_transform_to_batch(P_pt, homo_gt)
                # Gtext_t = STM(Gtext_roi_t, rect)
                Gtext_t = STM_out(P_pt, rect, homo_gt)
                Geo_real = Gtext_t[:, :3, ...] * Gtext_t[:,
                                                         3:4, ...] + bg * (1 - Gtext_t[:, 3:4, ...])
                Geo_realA = Gtext_t[:, 3:4, ...]
                I_comp, I_tt, Ipt = self.train_step(
                    bg,  P_pt, gt_text, Geo_real, Geo_realA, homo_gt, rect)

                # log the loss and img
                if self.rank == 0 and self.global_step % (cfg.log_interval) == 1:
                    self.log_loss()

            # and epoch > 0
            if self.rank == 0 and epoch % (cfg.log_img_interval) == 0:
                self.log_img(I_comp, Geo_real, bg, Ipt, gt_text, I_tt)

            if epoch % 20 == 0 and epoch > 0:
                self.lr_scheduler_D.step()
                # self.lr_scheduler_G.step()
                self.lr_scheduler_STN.step()

            # # validation
            if self.rank == 0 and cfg.val == True and epoch % cfg.val_interval == 0:
                self.evaluate(epoch)
            # save the model
            if self.rank == 0 and epoch % cfg.save_model_interval == 0 and epoch >= 1:
                self.save_model(epoch, note=epoch)

    def train_step(self, bg, P_pt, gt_text, Geo_real, Geo_realA, homo_gt, rect):
        self.STN.train()

        Ipt = STM(P_pt, rect)
        pmtrx = self.STN(bg, Ipt)
        I_tt = STM_out(P_pt, rect, pmtrx)
        I_comp = I_tt[:, :3, ...] * I_tt[:, 3:4, ...] + \
            bg * (1 - I_tt[:, 3:4, ...])
        I_compA = I_tt[:, 3:4, ...]
        self.optimizer_D.zero_grad()
        self.backward_D(I_comp, I_compA, Geo_real, Geo_realA)
        self.optimizer_D.step()
        # update STN
        self.optimizer_STN.zero_grad()
        self.backward_STN(I_comp, I_compA, pmtrx,
                          homo_gt, I_tt, gt_text)
        self.optimizer_STN.step()

        return I_comp, I_tt, Ipt

    def backward_D(self, I_comp, I_compA, Geo_real, Geo_realA):

        D_real = self.net_D(Geo_real, Geo_realA)
        D_fake = self.net_D(I_comp.detach(), I_compA.detach())
        D_loss = torch.mean(F.relu(1. - D_real)) + \
            torch.mean(F.relu(1. + D_fake))
        self.loss_dict['disc_fake'] = D_fake.mean()
        self.loss_dict['discriminator'] = reduce_value(D_loss, average=True)

        D_loss.backward()  # retain_graph=True

    def backward_STN(self, I_comp, I_compA, pmtrx, homo_gt, I_tt, gt_text):
        MSE1_loss, STN_loss_dict = self.STNLoss(
            pmtrx, homo_gt, I_tt, gt_text)
        D_fake = self.net_D(I_comp, I_compA)
        STN_loss = MSE1_loss - torch.mean(D_fake)
        STN_loss.backward()

        STN_loss_dict['STN'] = reduce_value(STN_loss, average=True)
        self.loss_dict.update(STN_loss_dict)

    def init_net(self, net):
        if self.rank == 0:
            print(f"Loading the net in GPU")
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            net).to(self.device)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[self.rank], find_unused_parameters=False)
        return net

    def init_optimizer(self):
        self.optimizer_STN = torch.optim.Adam(
            self.STN.parameters(), lr=cfg.initial_lr, betas=(cfg.beta1, cfg.beta2))
        self.lr_scheduler_STN = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_STN, gamma=0.9)
        self.optimizer_D = torch.optim.Adam(
            self.net_D.parameters(), lr=cfg.initial_lr_D, betas=(cfg.beta1, cfg.beta2))
        self.lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_D, gamma=0.9)

    def make_data_loader(self):
        if self.rank == 0:
            print("Loading Dataset...")
        train_dataset = DecompST4GTM(
            dataset_dir=cfg.train_data_root, is_for_train=True)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(
            self.train_sampler, cfg.batch_size, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=8, pin_memory=True, prefetch_factor=cfg.batch_size)

        val_dataset = DecompST4GTM_val(
            dataset_dir=cfg.train_data_root)
        return train_loader, val_dataset

    def report(self, epoch, step, batch_num, loss_dict):
        print('[epoch {:>3}] | [STEP: {:>4}/{:>4d}] | Total Loss: {:.4f}'.format(
            epoch, step, batch_num, loss_dict['total']))

    def log_loss(self):
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(
                'detail_loss', self.loss_dict, self.global_step)
            self.tb_writer.add_scalar(
                'total_Loss/discriminator', self.loss_dict['discriminator'], self.global_step)
            self.tb_writer.add_scalar(
                'LR/lr', self.optimizer_STN.state_dict()['param_groups'][0]['lr'], self.global_step)

    def log_img(self, I_comp, Geo_real, bg, Ipt, gt_text, I_tt):
        if self.tb_writer is not None:
            dis_row = 3
            IptRGB, IptA = Ipt[:, :3, ...], Ipt[:, 3:4, ...]
            I_ttRGB, I_ttBM = I_tt[:, :3, ...], I_tt[:, 4:5, ...]

            # t_textRGB, t_textA = gt_text[:, :3, ...], gt_text[:, 3:4, ...]
            dir_comp = IptRGB * IptA + bg * (1 - IptA)

            images = torch.cat((IptRGB[0:dis_row, ...], I_ttRGB[0:dis_row, ...], bg[0:dis_row, ...],
                                dir_comp[0:dis_row, ...], I_comp[0:dis_row, ...], Geo_real[0:dis_row, ...]), 0)

            images = images*0.5 + 0.5

            I_ttBM = torch.cat(
                (I_ttBM, I_ttBM, I_ttBM), 1)
            gt_textBM = torch.cat(
                (gt_text[:, 4:, ...], gt_text[:, 4:, ...], gt_text[:, 4:, ...]), 1)
            images = torch.cat(
                (images, I_ttBM[0:dis_row, ...], gt_textBM[0:dis_row, ...]), 0)
            grid = make_grid(images, nrow=dis_row, padding=10,
                             pad_value=100)  # , normalize=True
            self.tb_writer.add_image('train', grid, self.global_step)

    def save_model(self, epoch, note=''):
        print('Saving the model...')
        save_files = {
            'STN': self.STN.module.state_dict(),
            'net_D': self.net_D.module.state_dict(),
            'optimizer_STN': self.optimizer_STN.state_dict(),
            'lr_scheduler_STN': self.lr_scheduler_STN.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'lr_scheduler_D': self.lr_scheduler_D.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step}
        torch.save(save_files, f'{self.save_dir}/{note}.pth')

    def resume_model(self):
        print("Loading the trained params and the state of optimizer...")
        checkpoint = torch.load(cfg.resume_path, map_location=self.device)
        self.STN.module.load_state_dict(checkpoint['STN'])
        self.net_D.module.load_state_dict(checkpoint['net_D'])
        self.optimizer_STN.load_state_dict(checkpoint['optimizer_STN'])
        self.lr_scheduler_STN.load_state_dict(checkpoint['lr_scheduler_STN'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.lr_scheduler_D.load_state_dict(checkpoint['lr_scheduler_D'])
        self.flag_epoch = int(checkpoint['epoch']) + 1
        self.global_step = checkpoint['global_step']
        print(f"Resuming from epoch: {self.flag_epoch}")

    def start_from_pretrain(self):
        print(f"Loading the trained params from {cfg.pretrain_path}")
        checkpoint = torch.load(cfg.pretrain_path, map_location=self.device)
        self.STN.module.load_state_dict(checkpoint['STN'])
        self.net_D.module.load_state_dict(checkpoint['net_D'])
        print(f"Resuming from pretrain: {cfg.pretrain_path}")

    def evaluate(self, epoch):
        self.STN.eval()
        val_loader = DataLoader(
            self.val_dataset, batch_size=cfg.val_batch, num_workers=8, shuffle=True)
        dice_list = []
        diceA_list = []
        if self.rank == 0:
            for step, batch_data in enumerate(val_loader):
                bg, P_pt, gt_text, rect = batch_data['bg'].to(self.device), batch_data['P_pt'].to(
                    self.device), batch_data['gt_text'].to(self.device), batch_data['rect'].to(self.device)
                with torch.no_grad():
                    I_pt = STM(P_pt, rect)
                    pmtrx = self.STN(bg, I_pt)
                    # warp_text_roi = apply_transform_to_batch(P_pt, pmtrx)
                    I_tt = STM_out(P_pt, rect, pmtrx)
                I_ttBM = I_tt[:, 4:5, ...]
                t_textBM = gt_text[:, 4:5, ...]
                dice = cal_dice(I_ttBM, t_textBM)
                dice_list.append(dice.cpu().numpy())
                I_ttA = I_tt[:, 3:4, ...]
                t_textA = gt_text[:, 3:4, ...]
                diceA = cal_dice(I_ttA, t_textA)
                diceA_list.append(diceA.cpu().numpy())
                # if step//1000000 == 0:
                #     self.log_img_test(warp_text1, bg, text, gt)
            ave_dice = np.mean(dice_list)
            ave_diceA = np.mean(diceA_list)

            if ave_dice > self.best_dice:
                self.best_dice = ave_dice
                self.save_model(epoch=epoch, note='best')

            if self.tb_writer is not None:
                self.tb_writer.add_scalar(
                    'metrics/DICEQM', ave_dice, self.global_step)
                self.tb_writer.add_scalar(
                    'metrics/DICEA', ave_diceA, self.global_step)

    def log_img_test(self, I_tt, bg, I_pt, real):
        if self.tb_writer is not None:
            dis_row = 2
            I_ttRGB, I_ttA = I_tt[:, :3, ...], I_tt[:, 3:4, ...]
            I_comp = I_ttRGB * I_ttA + bg * (1 - I_ttA)
            I_ptRGB, I_ptA = I_pt[:, :3, :, :], I_pt[:, 3:, :, :]
            images = torch.cat((I_ptRGB[0:dis_row, ...], bg[0:dis_row, ...],
                                I_comp[0:dis_row, ...], real[0:dis_row, ...]), 0)
            images = images*0.5 + 0.5
            grid = make_grid(images, nrow=dis_row,
                             padding=10)  # , normalize=True
            self.tb_writer.add_image('test', grid, self.global_step)


def cal_dice(x_t, x_o):
    epsilon = 1e-8
    N = x_t.shape[0]
    x_t_flat = x_t.reshape(N, -1)
    x_o_flat = x_o.reshape(N, -1)
    intersection = (x_t_flat * x_o_flat).sum(1)  # N, H*W -> N, 1 -> scolar
    union = x_t_flat.sum(1) + x_o_flat.sum(1)
    dice = ((2. * intersection + epsilon)/(union + epsilon)).mean()
    return dice
