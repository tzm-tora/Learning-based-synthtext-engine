import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet18, resnet34


def mtrxmul(mtrxLS):
    homo = mtrxLS[0]
    for i in range(len(mtrxLS)-1):
        homo = homo.bmm(mtrxLS[i+1])  # .detach()
    return homo


def STM(patchs, rect, img_length=768):
    N, C, H, W = patchs.shape
    device = patchs.device
    lengths = img_length/(rect[:, -1] - rect[:, 0])
    # lengths = lengths
    ctr = (rect[:, -1] + rect[:, 0])/2  # x,y
    ctr_t = (img_length/2-ctr)/(img_length/2)
    eye = torch.tensor(
        [[1., 0, 0], [0, 1., 0], [0, 0, 1.]]).view(-1, 3, 3).repeat(N, 1, 1).to(device)
    eye[:, 0, 2] = ctr_t[:, 0]  # x shift
    eye[:, 1, 2] = ctr_t[:, 1]
    eye[:, 0] = eye[:, 0]*lengths[:, 0].view(-1, 1)
    eye[:, 1] = eye[:, 1]*lengths[:, 1].view(-1, 1)
    patchs_in_img = homo_grid_sample(patchs, eye, tar_size=img_length)
    return patchs_in_img


def STM_out(patchs, rect, STN_out, img_length=768):
    N, C, H, W = patchs.shape
    device = patchs.device
    lengths = img_length/(rect[:, -1] - rect[:, 0])
    # lengths = lengths
    ctr = (rect[:, -1] + rect[:, 0])/2  # x,y
    ctr_t = (img_length/2-ctr)/(img_length/2)
    eye = torch.tensor(
        [[1., 0, 0], [0, 1., 0], [0, 0, 1.]]).view(-1, 3, 3).repeat(N, 1, 1).to(device)
    eye[:, 0, 2] = ctr_t[:, 0]  # x shift
    eye[:, 1, 2] = ctr_t[:, 1]
    eye[:, 0] = eye[:, 0]*lengths[:, 0].view(-1, 1)
    eye[:, 1] = eye[:, 1]*lengths[:, 1].view(-1, 1)
    # homo = eye.bmm(STN_out)
    homo = STN_out.bmm(eye)
    patchs_in_img = homo_grid_sample(patchs, homo, tar_size=img_length)
    return patchs_in_img


def homo_grid_sample(im_batch_tensor, transform_tensor, tar_size=768):
    """ apply a geometric transform to a batch of image tensors
    args
        im_batch_tensor -- torch float tensor of shape (N, C, H, W)
        transform_tensor -- torch float tensor of shape (1, 3, 3)
    returns
        transformed_batch_tensor -- torch float tensor of shape (N, C, H, W)
    """
    N, C, H, W = im_batch_tensor.shape
    device = im_batch_tensor.device
    # torch.nn.functional.grid_sample takes a grid in [-1,1] and interpolates;
    # construct grid in homogeneous coordinates
    x, y = torch.meshgrid(
        [torch.linspace(-1, 1, tar_size), torch.linspace(-1, 1, tar_size)])
    x, y = x.flatten(), y.flatten()
    xy_hom = torch.stack([x, y, torch.ones(x.shape[0])],
                         dim=0).unsqueeze(0).to(device)
    # tansform the [-1,1] homogeneous coords
    # (N, 3, 3) matmul (N, 3, H*W) > (N, 3, H*W)
    xy_transformed = transform_tensor.matmul(xy_hom)
    # convert to inhomogeneous coords -- cf Szeliski eq. 2.21
    grid = xy_transformed[:, :2, :] / \
        (xy_transformed[:, 2, :].unsqueeze(1) + 1e-9)
    # (N, H, W, 2); cf torch.functional.grid_sample
    # grid = grid.permute(0, 2, 1).reshape(-1, H, W, 2)
    grid = grid.permute(0, 2, 1).reshape(-1, tar_size, tar_size, 2)
    grid = grid.expand(N, *grid.shape[1:])  # expand to minibatch
    transformed_batch = F.grid_sample(
        im_batch_tensor, grid, mode='bilinear', align_corners=True)
    transformed_batch.transpose_(3, 2)

    return transformed_batch


def apply_transform_to_batch(im_batch_tensor, transform_tensor):
    """ apply a geometric transform to a batch of image tensors
    args
        im_batch_tensor -- torch float tensor of shape (N, C, H, W)
        transform_tensor -- torch float tensor of shape (1, 3, 3)
    returns
        transformed_batch_tensor -- torch float tensor of shape (N, C, H, W)
    """
    N, C, H, W = im_batch_tensor.shape
    device = im_batch_tensor.device
    # torch.nn.functional.grid_sample takes a grid in [-1,1] and interpolates;
    # construct grid in homogeneous coordinates
    x, y = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])
    x, y = x.flatten(), y.flatten()
    xy_hom = torch.stack([x, y, torch.ones(x.shape[0])],
                         dim=0).unsqueeze(0).to(device)
    # tansform the [-1,1] homogeneous coords
    # (N, 3, 3) matmul (N, 3, H*W) > (N, 3, H*W)
    xy_transformed = transform_tensor.matmul(xy_hom)
    # convert to inhomogeneous coords -- cf Szeliski eq. 2.21
    grid = xy_transformed[:, :2, :] / \
        (xy_transformed[:, 2, :].unsqueeze(1) + 1e-9)
    # (N, H, W, 2); cf torch.functional.grid_sample
    grid = grid.permute(0, 2, 1).reshape(-1, H, W, 2)
    grid = grid.expand(N, *grid.shape[1:])  # expand to minibatch
    transformed_batch = F.grid_sample(
        im_batch_tensor, grid, mode='bilinear', align_corners=True)
    transformed_batch.transpose_(3, 2)

    return transformed_batch


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_act_conv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.Conv2d(dims_in, dims_out, kernel_size=kernel,
                          stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*conv)


def get_act_dconv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.ConvTranspose2d(dims_in, dims_out,
                                   kernel_size=kernel, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*conv)


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class SN_ConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SN_ConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class build_discriminator(nn.Module):
    def __init__(self):
        super(build_discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            SN_ConvWithActivation(4, 64, 4, 2, padding=1),
            SN_ConvWithActivation(64,  64,  4, 2, padding=1),
            SN_ConvWithActivation(64,  128, 4, 2, padding=1),
            SN_ConvWithActivation(128, 128, 4, 2, padding=1),
            SN_ConvWithActivation(128, 256, 4, 2, padding=1),
            SN_ConvWithActivation(256, 512, 4, 1, padding=1),
            # SN_ConvWithActivation(256, 256, 4, 2, padding=1),
            nn.Conv2d(512, 1, kernel_size=4),
            # nn.Sigmoid()
        )

    def forward(self, outputG, outputGA):
        cat = torch.cat((outputG, outputGA), 1)
        # print(cat.shape)
        # cat = outputG
        global_feat = self.discriminator(cat)
        output = global_feat  # .view(input.size()[0], -1)
        # print(output.shape)
        return output


# class LocalizationNetwork(nn.Module):
#     def __init__(self, num_output, in_channels):
#         super(LocalizationNetwork, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(64), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
#             nn.Conv2d(64, 128, 5, 1, 0, bias=False),
#             nn.BatchNorm2d(128), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
#             nn.Conv2d(128, 256, 5, 1, 0, bias=False), nn.BatchNorm2d(
#                 256), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
#             nn.Conv2d(256, 512, 3, 1, 0, bias=False), nn.BatchNorm2d(
#                 512), nn.ReLU(True),
#             # nn.MaxPool2d(2, 2),
#             # nn.Conv2d(512, 512, 3, 1, 0, bias=False), nn.BatchNorm2d(
#             #     512), nn.ReLU(True),
#             nn.AdaptiveAvgPool2d(1)  # batch_size x 512
#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(512, 256), nn.ReLU(True))
#         self.fc2 = nn.Linear(256, num_output)

#         self.fc2.weight.data.fill_(0)
#         self.fc2.bias.data = torch.eye(3).flatten()[:num_output]

#     def forward(self, x):
#         batch_size = x.size(0)
#         features = self.conv(x).view(batch_size, -1)
#         loc_out = self.fc2(self.fc1(features))
#         return loc_out

class ResNet(nn.Module):
    def __init__(self, model=resnet34(pretrained=True), in_channels=7):
        super(ResNet, self).__init__()
        # 取掉model的后1层
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        # self.Linear_layer = nn.Linear(512, outnum) #加上一层参数修改好的全连接层

    def forward(self, x):
        x = self.resnet_layer(x)
        # x = x.view(x.size(0), -1)
        # x = self.Linear_layer(x)
        return x

# resnet = resnet34(pretrained=False)
# # print(resnet)
# model = Net(resnet, 6)
# print(model)

class LocalizationNetwork(nn.Module):
    def __init__(self, num_output, in_channels):
        super(LocalizationNetwork, self).__init__()
        self.conv = ResNet(in_channels=in_channels)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True))
        self.fc2 = nn.Linear(256, num_output)

        self.fc2.weight.data.fill_(0)
        self.fc2.bias.data = torch.eye(3).flatten()[:num_output]

    def forward(self, x):
        # batch_size = x.size(0)
        features = self.conv(x).flatten(start_dim=1)  # .view(batch_size, -1)
        loc_out = self.fc2(self.fc1(features))
        return loc_out


class SpatialTransformer(nn.Module):

    def __init__(self, mode='homo', in_channels=7):
        super(SpatialTransformer, self).__init__()
        if mode == 'affine':
            self.num_output = 6
        elif mode == 'homo':
            self.num_output = 8
        self.locnet = LocalizationNetwork(self.num_output, in_channels)

    def forward(self, bg, text):

        textbg = torch.cat((text[:, :4, ...], bg), 1)
        loc_out = self.locnet(textbg)

        if self.num_output == 6:
            pmtrx = loc_out.view(-1, 2, 3)  # to the 2x3 matrix
            affine_grid = F.affine_grid(
                pmtrx, text.size(), align_corners=True)
            warp_text = F.grid_sample(
                text, affine_grid, mode='bilinear', align_corners=True)

        elif self.num_output == 8:
            device = bg.device
            homo = torch.cat(
                (loc_out, torch.ones(loc_out.size(0), 1).to(device)), 1)
            pmtrx = homo.view(-1, 3, 3)  # change it to the 3x3 matrix
            # warp_text = apply_transform_to_batch(text, pmtrx)

        return pmtrx
