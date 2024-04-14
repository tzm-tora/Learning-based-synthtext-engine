import torch
import torch.nn as nn
from torchvision import models
import cfg


class DetailLoss(nn.Module):
    def __init__(self):
        super(DetailLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.bceloss = nn.BCELoss()

    def forward(self, HF, HF_GT):

        # l2_loss = self.l2(HF, HF_GT)
        bceloss = self.bceloss(HF, HF_GT)
        diceA_loss = build_dice_loss(HF, HF_GT)

        loss = 10*bceloss + diceA_loss   # + l1_bg_loss + l1_real_loss l2_loss+

        # ,'l1_real_loss': l1_real_loss, 'l1_bg': l1_bg_loss 'l2_loss': l2_loss,
        return loss, {'bceloss': bceloss, 'dice_loss': diceA_loss}


def build_dice_loss(x_t, x_o):
    epsilon = 1e-8
    N = x_t.shape[0]
    x_t_flat = x_t.reshape(N, -1)
    x_o_flat = x_o.reshape(N, -1)
    intersection = (x_t_flat * x_o_flat).sum(1)  # N, H*W -> N, 1 -> scolar
    union = x_t_flat.sum(1) + x_o_flat.sum(1)
    dice_loss = 1. - ((2. * intersection + epsilon)/(union + epsilon)).mean()
    return dice_loss
