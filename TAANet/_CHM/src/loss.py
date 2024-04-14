import torch
import torch.nn as nn
from torchvision import models
import cfg


class DetailLoss(nn.Module):
    def __init__(self):
        super(DetailLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, output_real, real, text_t):

        t_textRGB, t_textA = text_t[:, :3, :, :], text_t[:, 3:4, :, :]
        # l1_bg_loss = torch.mean(
        #     torch.sum(torch.abs(output - real) * (1. - t_textA)) / torch.sum((1. - t_textA)))
        l1_text_loss = torch.mean(
            torch.sum(torch.abs(output - real) * t_textA) / torch.sum(t_textA))

        l1_real_loss = self.l1(output_real, real)

        # l1_bg_loss *= 100
        l1_text_loss *= 5
        l1_real_loss *= 5

        loss = l1_text_loss + l1_real_loss  # + l1_bg_loss

        # ,'l1_real_loss': l1_real_loss, 'l1_bg': l1_bg_loss
        return loss, {'l1_text': l1_text_loss, 'l1_real_loss': l1_real_loss, }
