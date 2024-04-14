import torch
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import to_items, reduce_value
from torchvision.utils import make_grid
from .utils import create_ckpt_dir, makedirs
from torch.utils.tensorboard import SummaryWriter
import piq
import numpy as np
from .model import build_generator, build_discriminator
from src.dataset import DecompST4CHM, DecompST4CHM_val
from src.loss import DetailLoss
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, device, rank):
        self.flag_epoch = 0
        self.global_step = 0
        self.device = device
        self.rank = rank

        self.train_loader, self.val_dataset = self.make_data_loader()

        self.net_G = self.init_net(
            build_generator().to(self.device))
        self.net_D = self.init_net(build_discriminator().to(self.device))

        self.optimizer_G, self.lr_scheduler_G, self.optimizer_D, self.lr_scheduler_D = self.init_optimizer()
        self.DetailLoss = DetailLoss().to(self.device)

        if cfg.resume:
            self.resume_model()

        self.loss_dict = {}
        self.tb_writer = None
        self.sec_PSNR = 0
        self.best_PSNR = 0
        if self.rank == 0:
            self.save_dir, self.tensorboard_dir = create_ckpt_dir(cfg.ckpt_dir)
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
                bg,  I_tt, text_t, real = batch_data['bg'].to(self.device), batch_data['I_tt'].to(
                    self.device), batch_data['gt_text'].to(self.device), batch_data['real'].to(self.device)

                output = self.train_step(bg, I_tt, text_t, real)

                # log the loss and img
                if self.rank == 0 and self.global_step % (cfg.log_interval) == 1:
                    self.log_loss()
                if self.rank == 0 and self.global_step % (cfg.log_img_interval) == 1:
                    self.log_img(output, bg, I_tt, text_t, real)
                    # self.evaluate(epoch)

            if epoch % 20 == 0 and epoch > 0:
                self.lr_scheduler_D.step()
                self.lr_scheduler_G.step()

            # validation
            if self.rank == 0 and cfg.val == True and epoch % cfg.val_interval == 0:
                self.evaluate(epoch)
            # save the model
            if self.rank == 0 and epoch % cfg.save_model_interval == 0:
                self.save_model(epoch, note=epoch)

    def train_step(self, bg, I_tt, text_t, real):
        self.net_G.train()
        output, output_real = self.net_G(bg, I_tt, text_t)
        # update D
        self.optimizer_D.zero_grad()
        self.backward_D(output, real, text_t)
        self.optimizer_D.step()
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G(output, output_real, real, text_t)
        self.optimizer_G.step()
        return output

    def backward_D(self, output, gt, text_t):
        t_textA = text_t[:, 3:4, :, :]
        D_real = self.net_D(gt, t_textA)
        D_fake = self.net_D(output.detach(), t_textA)
        D_loss = torch.mean(F.relu(1. - D_real)) + \
            torch.mean(F.relu(1. + D_fake))  # SN-patch-GAN loss

        self.loss_dict['disc_real'] = D_real.mean()
        self.loss_dict['disc_fake'] = D_fake.mean()
        self.loss_dict['discriminator'] = reduce_value(D_loss, average=True)

        self.real_dis = F.interpolate(
            D_real.detach(), size=[512, 512], mode="nearest")  # “bilinear”
        self.fake_dis = F.interpolate(
            D_fake.detach(), size=[512, 512], mode="nearest")

        D_loss.backward()  # retain_graph=True

    def backward_G(self, output, output_real, real, text_t):
        Detail_loss, loss_dict = self.DetailLoss(
            output, output_real, real, text_t)
        t_textA = text_t[:, 3:4, :, :]
        D_fake = self.net_D(output, t_textA)
        # D_fake = torch.mean(D_fake)
        G_Loss = Detail_loss - torch.mean(D_fake)
        G_Loss.backward()

        loss_dict['generator'] = reduce_value(G_Loss, average=True)
        self.loss_dict.update(loss_dict)

    def init_net(self, net):
        if self.rank == 0:
            print(f"Loading the net in GPU")
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            net).to(self.device)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[self.rank], find_unused_parameters=False)
        return net

    def init_optimizer(self):
        optimizer_G = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.net_G.parameters()), lr=cfg.initial_lr, betas=(cfg.beta1, cfg.beta2))
        lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_G, gamma=0.95)
        optimizer_D = torch.optim.Adam(
            self.net_D.parameters(), lr=cfg.initial_lr_D, betas=(cfg.beta1, cfg.beta2))
        lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_D, gamma=0.95)
        return optimizer_G, lr_scheduler_G, optimizer_D, lr_scheduler_D

    def make_data_loader(self):
        if self.rank == 0:
            print("Loading Dataset...")
        train_dataset = DecompST4CHM(cfg.train_data_root)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(
            self.train_sampler, cfg.batch_size, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, shuffle=False, num_workers=8, pin_memory=True)
        val_dataset = DecompST4CHM_val(cfg.train_data_root)
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
                'total_Loss/generator', self.loss_dict['generator'], self.global_step)
            self.tb_writer.add_scalar(
                'LR/lr', self.optimizer_G.state_dict()['param_groups'][0]['lr'], self.global_step)
            self.tb_writer.add_scalar(
                'LR/lr_D', self.optimizer_D.state_dict()['param_groups'][0]['lr'], self.global_step)

    def log_img(self, output, bg, I_tt, text_t, real):
        if self.tb_writer is not None:
            dis_row = 2
            I_ttRGB, I_ttA = I_tt[:, :3, :, :], I_tt[:, 3:4, :, :]
            t_textRGB, t_textA = text_t[:, :3, :, :], text_t[:, 3:4, :, :]
            out_textRGB = output * I_ttA + torch.ones_like(bg) * (1. - I_ttA)
            images = torch.cat((I_ttRGB[0:dis_row, ...], bg[0:dis_row, ...],
                                out_textRGB[0:dis_row, ...], t_textRGB[0:dis_row, ...],
                                output[0:dis_row, ...], real[0:dis_row, ...]), 0)
            images = images*0.5 + 0.5
            D_real = torch.cat(
                (self.real_dis, self.real_dis, self.real_dis), 1)
            D_fake = torch.cat(
                (self.fake_dis, self.fake_dis, self.fake_dis), 1)
            images = torch.cat(
                (images, D_fake[0:dis_row, ...], D_real[0:dis_row, ...]), 0)
            grid = make_grid(images, nrow=dis_row,
                             padding=10, pad_value=100)  # , normalize=True
            self.tb_writer.add_image('train', grid, self.global_step)

    def save_model(self, epoch, note=''):
        print('Saving the model...')
        save_files = {'net_G': self.net_G.module.state_dict(),
                      'net_D': self.net_D.module.state_dict(),
                      'optimizer_G': self.optimizer_G.state_dict(),
                      'lr_scheduler_G': self.lr_scheduler_G.state_dict(),
                      'optimizer_D': self.optimizer_D.state_dict(),
                      'lr_scheduler_D': self.lr_scheduler_D.state_dict(),
                      'epoch': epoch,
                      'global_step': self.global_step}
        torch.save(save_files, f'{self.save_dir}/{note}.pth')

    def resume_model(self):
        print("Loading the trained params and the state of optimizer...")
        checkpoint = torch.load(cfg.resume_path, map_location=self.device)
        self.net_G.module.load_state_dict(checkpoint['net_G'])
        self.net_D.module.load_state_dict(checkpoint['net_D'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.lr_scheduler_D.load_state_dict(checkpoint['lr_scheduler_D'])
        self.flag_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.net_G.train()
        print(f"Resuming from epoch: {self.flag_epoch}")

    def evaluate(self, epoch):
        self.net_G.eval()
        val_loader = DataLoader(
            self.val_dataset, batch_size=cfg.val_batch, num_workers=8, shuffle=True)
        psnr_list = []
        fMSE_list = []
        if self.rank == 0:
            for step, batch_data in enumerate(val_loader):
                bg, I_tt, real = batch_data['bg'].to(self.device), batch_data['I_tt'].to(
                    self.device), batch_data['real'].to(self.device)
                t_textA = I_tt[:, 3:4, :, :]
                with torch.no_grad():
                    output = self.net_G(bg, I_tt)
                output01 = output * 0.5 + 0.5
                real01 = real * 0.5 + 0.5
                psnr = piq.psnr(output01, real01, data_range=1.0).cpu()
                fMSE = torch.mean(
                    torch.sum(torch.abs(output - real) * t_textA) / torch.sum(t_textA)).cpu()
                psnr_list.append(psnr.numpy())
                fMSE_list.append(fMSE.numpy())
            ave_psnr = np.mean(psnr_list)
            ave_fMSE = np.mean(fMSE_list)
            if ave_psnr > self.best_PSNR:
                self.sec_PSNR = self.best_PSNR
                self.best_PSNR = ave_psnr
                self.save_model(epoch=epoch, note='best')
            elif ave_psnr > self.sec_PSNR:
                self.sec_PSNR = ave_psnr
                self.save_model(epoch=epoch, note='sec')
            # self.log_img_test(output, bg, text, real)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(
                    'metrics/PSNR', ave_psnr, self.global_step)
                self.tb_writer.add_scalar(
                    'metrics/fMSE', ave_fMSE, self.global_step)

    def log_img_test(self, output, bg, text, real):
        if self.tb_writer is not None:
            dis_row = 4
            textRGB, textA = text[:, :3, :, :], text[:, 3:, :, :]
            images = torch.cat((textRGB[0:dis_row, ...], bg[0:dis_row, ...],
                                output[0:dis_row, ...], real[0:dis_row, ...]), 0)
            images = images*0.5 + 0.5
            grid = make_grid(images, nrow=dis_row,
                             padding=10)  # , normalize=True
            self.tb_writer.add_image('test', grid, self.global_step)
