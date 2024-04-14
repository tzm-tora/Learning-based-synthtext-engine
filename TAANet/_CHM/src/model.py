import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .normalize import RAIN


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


class UnetBlockCodec(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False,
                 norm_layer=RAIN, use_attention=False, enc=True, dec=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetBlockCodec) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
            enc (bool) -- if use give norm_layer in encoder part.
            dec (bool) -- if use give norm_layer in decoder part.
        """
        super(UnetBlockCodec, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_attention = use_attention
        use_bias = False
        input_nc = outer_nc
        self.norm_namebuffer = [
            'RAIN', 'RAIN_Method_Learnable', 'RAIN_Method_BN']
        if outermost:
            self.down = nn.Conv2d(
                input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.submodule = submodule
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        elif innermost:
            self.up = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                          stride=2, padding=1, bias=use_bias),
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            )
            self.upnorm = norm_layer(
                outer_nc) if dec else nn.InstanceNorm2d(outer_nc)
        else:
            self.down = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                          stride=2, padding=1, bias=use_bias),
            )
            self.downnorm = norm_layer(
                inner_nc) if enc else nn.InstanceNorm2d(inner_nc)
            self.submodule = submodule
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            )
            self.upnorm = norm_layer(
                outer_nc) if dec else nn.InstanceNorm2d(outer_nc)

        if use_attention:
            attention_conv = nn.Conv2d(
                outer_nc+input_nc, outer_nc+input_nc, kernel_size=1)
            attention_sigmoid = nn.Sigmoid()
            self.attention = nn.Sequential(
                *[attention_conv, attention_sigmoid])

    def forward(self, x, mask):
        if self.outermost:
            x = self.down(x)
            x = self.submodule(x, mask)
            ret = self.up(x)
            return ret
        elif self.innermost:
            ret = self.up(x)
            if self.upnorm._get_name() in self.norm_namebuffer:
                ret = self.upnorm(ret, mask)
            else:
                ret = self.upnorm(ret)
            ret = torch.cat([x, ret], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret
        else:
            ret = self.down(x)
            if self.downnorm._get_name() in self.norm_namebuffer:
                ret = self.downnorm(ret, mask)
            else:
                ret = self.downnorm(ret)
            ret = self.submodule(ret, mask)
            ret = self.up(ret)
            if self.upnorm._get_name() in self.norm_namebuffer:
                ret = self.upnorm(ret, mask)
            else:
                ret = self.upnorm(ret)
            # if self.use_dropout:    # only works for middle features
            #     ret = self.dropout(ret)
            ret = torch.cat([x, ret], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret


class RainNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 norm_type_indicator=[0, 0, 0, 0, 1, 1, 1, 1],
                 use_attention=True):
        super(RainNet, self).__init__()
        self.input_nc = input_nc
        self.norm_namebuffer = ['RAIN']
        self.use_attention = use_attention
        norm_type_list = [nn.InstanceNorm2d, RAIN]
        # -------------------------------Network Settings-------------------------------------
        self.model_layer0 = nn.Conv2d(
            input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.model_layer1 = get_act_conv(
            nn.LeakyReLU(0.2, True), ngf, ngf*2, 4, 2, 1, False)
        self.model_layer1norm = norm_type_list[norm_type_indicator[0]](ngf*2)

        self.model_layer2 = get_act_conv(nn.LeakyReLU(
            0.2, True), ngf*2, ngf*4, 4, 2, 1, False)
        self.model_layer2norm = norm_type_list[norm_type_indicator[1]](ngf*4)

        self.model_layer3 = get_act_conv(nn.LeakyReLU(
            0.2, True), ngf*4, ngf*8, 4, 2, 1, False)
        self.model_layer3norm = norm_type_list[norm_type_indicator[2]](ngf*8)

        self.unet_block = UnetBlockCodec(ngf * 8, ngf * 8, submodule=None, norm_layer=RAIN,
                                         innermost=True, enc=norm_type_indicator[3], dec=norm_type_indicator[4])

        self.model_layer11 = get_act_dconv(
            nn.ReLU(True), ngf*16, ngf*4, 4, 2, 1, False)
        self.model_layer11norm = norm_type_list[norm_type_indicator[5]](ngf*4)
        if use_attention:
            self.model_layer11att = nn.Sequential(
                nn.Conv2d(ngf*8, ngf*8, kernel_size=1, stride=1), nn.Sigmoid())

        self.model_layer12 = get_act_dconv(
            nn.ReLU(True), ngf*8, ngf*2, 4, 2, 1, False)
        self.model_layer12norm = norm_type_list[norm_type_indicator[6]](ngf*2)
        if use_attention:
            self.model_layer12att = nn.Sequential(
                nn.Conv2d(ngf*4, ngf*4, kernel_size=1, stride=1), nn.Sigmoid())

        self.model_layer13 = get_act_dconv(
            nn.ReLU(True), ngf*4, ngf, 4, 2, 1, False)
        self.model_layer13norm = norm_type_list[norm_type_indicator[7]](ngf)
        if use_attention:
            self.model_layer13att = nn.Sequential(
                nn.Conv2d(ngf*2, ngf*2, kernel_size=1, stride=1), nn.Sigmoid())

        # self.model_out = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(
        #     ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), nn.Tanh())
        self.model_out = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),)
        self.model_out2 = nn.Sequential(
            nn.Conv2d(64+3, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, output_nc, 3, 1, 1),
            nn.Tanh())

    def forward(self, x, mask):
        x0 = self.model_layer0(x)
        x1 = self.model_layer1(x0)
        if self.model_layer1norm._get_name() in self.norm_namebuffer:
            x1 = self.model_layer1norm(x1, mask)
        else:
            x1 = self.model_layer1norm(x1)
        x2 = self.model_layer2(x1)
        if self.model_layer2norm._get_name() in self.norm_namebuffer:
            x2 = self.model_layer2norm(x2, mask)
        else:
            x2 = self.model_layer2norm(x2)
        x3 = self.model_layer3(x2)
        if self.model_layer3norm._get_name() in self.norm_namebuffer:
            x3 = self.model_layer3norm(x3, mask)
        else:
            x3 = self.model_layer3norm(x3)

        ox3 = self.unet_block(x3, mask)
        ox2 = self.model_layer11(ox3)
        if self.model_layer11norm._get_name() in self.norm_namebuffer:
            ox2 = self.model_layer11norm(ox2, mask)
        else:
            ox2 = self.model_layer11norm(ox2)
        ox2 = torch.cat([x2, ox2], 1)
        if self.use_attention:
            ox2 = self.model_layer11att(ox2) * ox2

        ox1 = self.model_layer12(ox2)
        if self.model_layer12norm._get_name() in self.norm_namebuffer:
            ox1 = self.model_layer12norm(ox1, mask)
        else:
            ox1 = self.model_layer12norm(ox1)
        ox1 = torch.cat([x1, ox1], 1)
        if self.use_attention:
            ox1 = self.model_layer12att(ox1) * ox1

        ox0 = self.model_layer13(ox1)
        if self.model_layer13norm._get_name() in self.norm_namebuffer:
            ox0 = self.model_layer13norm(ox0, mask)
        else:
            ox0 = self.model_layer13norm(ox0)
        ox0 = torch.cat([x0, ox0], 1)
        if self.use_attention:
            ox0 = self.model_layer13att(ox0) * ox0

        out = self.model_out(ox0)
        xout = torch.cat((out, x), 1)
        out = self.model_out2(xout)

        return out


class build_generator(nn.Module):
    def __init__(self):
        super(build_generator, self).__init__()
        self.rainNet = RainNet()

    def forward(self, bg, text, text_t=None):
        textRGB, textA = text[:, :3, :, :], text[:, 3:4, :, :]
        comp = textRGB * textA + bg * (1 - textA)
        out_textRGB = self.rainNet(comp, textA)
        out = out_textRGB * textA + bg * (1 - textA)
        if text_t is not None:
            textRGB_t, textA_t = text_t[:, :3, ...], text_t[:, 3:4, ...]
            comp_real = textRGB_t * textA_t + bg * (1 - textA_t)
            out_real = self.rainNet(comp_real, textA_t)
            return out, out_real
        else:
            return out


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class SN_ConvWithActivation(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(SN_ConvWithActivation, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.SNconv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.SNconv2d(input)
        return self.activation(x)


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
            nn.Conv2d(512, 1, kernel_size=4),
            # nn.Sigmoid()
        )

    def forward(self, outputG, text_QM):
        cat = torch.cat((outputG, text_QM), 1)
        global_feat = self.discriminator(cat)
        output = global_feat
        return output
