import torch.nn as nn
import cfg


class STNLoss(nn.Module):
    def __init__(self):
        super(STNLoss, self).__init__()
        self.l1 = nn.SmoothL1Loss()
        # self.l2 = nn.MSELoss()

    def forward(self, pmtrx, homo_gt, I_tt, gt_text):

        I_ttBM = I_tt[:, 4:5, ...]
        gt_textBM = gt_text[:, 4:5, ...]
        diceBM_loss = build_dice_loss(I_ttBM, gt_textBM)
        I_ttA = I_tt[:, 3:4, ...]
        gt_textA = gt_text[:, 3:4, ...]
        diceA_loss = build_dice_loss(I_ttA, gt_textA)
        dice_loss = diceBM_loss + diceA_loss

        homo = pmtrx  # mtrxmul(mtrxLS)
        L1_loss = self.l1(homo.flatten(1)[:, :8], homo_gt.flatten(1)[:, :8])

        L1_loss *= cfg.l1_coef
        dice_loss *= cfg.dice_coef

        loss = L1_loss + dice_loss

        return loss, {f'SML1': L1_loss, f'diceBM': diceBM_loss, f'diceA': diceA_loss}


def build_dice_loss(x_t, x_o):
    epsilon = 1e-8
    N = x_t.shape[0]
    x_t_flat = x_t.reshape(N, -1)
    x_o_flat = x_o.reshape(N, -1)
    intersection = (x_t_flat * x_o_flat).sum(1)  # N, H*W -> N, 1 -> scolar
    union = x_t_flat.sum(1) + x_o_flat.sum(1)
    dice_loss = 1. - ((2. * intersection + epsilon)/(union + epsilon)).mean()
    return dice_loss
