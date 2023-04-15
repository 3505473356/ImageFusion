import random
import torch
import torch as t
from torch.nn.functional import conv2d, conv1d
import torch.nn as nn
import math
from torch.nn import functional as F
from copy import deepcopy
import os
import errno


def create_conv_block(in_channel, out_channel, kernel, stride, padding, pooling, bn=True, relu=True,
                      pool_layer=False):
    net_list = [t.nn.Conv2d(in_channel, out_channel, kernel, stride, padding, padding_mode="circular")]
    if bn:
        net_list.append(t.nn.BatchNorm2d(out_channel))
    if relu:
        net_list.append(t.nn.ReLU())
    if pooling[0] > 0 or pooling[1] > 0:
        if not pool_layer:
            net_list.append(t.nn.MaxPool2d(pooling, pooling))
        else:
            net_list.append(t.nn.Conv2d(out_channel, out_channel, pooling, pooling))
    return t.nn.Sequential(*net_list)


def create_residual_block(fs, ch, pad):
    return t.nn.Sequential(create_conv_block(ch, ch, fs, 1, pad, (0, 0), bn=True, relu=True),
                           t.nn.ReLU(),
                           create_conv_block(ch, ch, fs, 1, pad, (0, 0), bn=False, relu=False))


class CNNOLD(t.nn.Module):

    def __init__(self, ch_in, lp, fs, ech, res, legacy=True):
        super(CNNOLD, self).__init__()
        pad = fs // 2
        self.res = res
        # self.backbone = t.nn.Sequential(*[create_conv_block(3, 16, fs, 1, pad, (2, 2), pool_layer=lp),
        #                                   create_conv_block(16, 64, fs, 1, pad, (2, 2), pool_layer=lp),
        #                                   create_conv_block(64, 256, fs, 1, pad, (2, 2), pool_layer=lp),
        #                                   create_conv_block(256, 512, fs, 1, pad, (2, 1), pool_layer=lp),
        #                                   create_conv_block(512, ech, fs, 1, pad, (3, 1), bn=False, relu=False,
        #                                                     pool_layer=lp)])
        self.l1 = create_conv_block(ch_in, 16, fs, 1, pad, (2, 2), pool_layer=lp, relu=False)
        self.l2 = create_conv_block(16, 64, fs, 1, pad, (2, 2), pool_layer=lp, relu=False)
        self.l3 = create_conv_block(64, 256, fs, 1, pad, (2, 2), pool_layer=lp, relu=False)
        self.l4 = create_conv_block(256, 512, fs, 1, pad, (2, 1), pool_layer=lp, relu=False)
        self.l5 = create_conv_block(512, ech, fs, 1, pad, (2, 1), relu=False, pool_layer=lp, bn=False)
        if self.res == 1:
            # residual blocks
            self.l1res = create_residual_block(fs, 16, pad)
            self.l2res = create_residual_block(fs, 64, pad)
            self.l3res = create_residual_block(fs, 256, pad)
            self.l4res = create_residual_block(fs, 512, pad)
        if self.res == 2:
            # Squeeze and excitation blocks
            self.l1res = SE_Block(16)
            self.l2res = SE_Block(64)
            self.l3res = SE_Block(256)
            self.l4res = SE_Block(512)
        if self.res == 3:
            self.l1res = t.nn.Sequential(create_residual_block(fs, 16, pad), SE_Block(16))
            self.l2res = t.nn.Sequential(create_residual_block(fs, 64, pad), SE_Block(64))
            self.l3res = t.nn.Sequential(create_residual_block(fs, 256, pad), SE_Block(256))
            self.l4res = t.nn.Sequential(create_residual_block(fs, 512, pad), SE_Block(512))

    def forward(self, x):
        if self.res > 0:
            x_tmp = self.l1(x)
            x = self.l1res(x_tmp) + x_tmp
            x = t.relu(x)
            x_tmp = self.l2(x)
            x = self.l2res(x_tmp) + x_tmp
            x = t.relu(x)
            x_tmp = self.l3(x)
            x = self.l3res(x_tmp) + x_tmp
            x = t.relu(x)
            x_tmp = self.l4(x)
            x = self.l4res(x_tmp) + x_tmp
            x = t.relu(x)
            x = self.l5(x)
        else:
            x = self.l1(x)
            x = t.relu(x)
            x = self.l2(x)
            x = t.relu(x)
            x = self.l3(x)
            x = t.relu(x)
            x = self.l4(x)
            x = t.relu(x)
            x = self.l5(x)
        return x


def get_custom_CNN(ch_in, lp, fs, ech, res):
    return CNNOLD(ch_in, lp, fs, ech, res)


class SE_Block(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class Siamese(t.nn.Module):

    def __init__(self, backboneRGB, backboneIR, padding=3):
        super(Siamese, self).__init__()
        self.backboneRGB = backboneRGB
        self.backboneIR = backboneIR
        self.padding = padding
        self.out_batchnorm = t.nn.BatchNorm2d(1)
        # self.attn_layer = t.nn.TransformerEncoderLayer(256, 8, 512)
        # self.height_attn = t.nn.TransformerEncoder(self.attn_layer, 4)
        # self.pe = PositionalEncoding(256)

    def apply_attention(self, x):
        B, CH, H, W = x.shape
        x = self.pe(x.view(H, W * B, CH))
        x = self.height_attn(x).view(B, CH, H, W)
        return x

    def forward(self, source, target, padding=None, displac=None):
        # source = self.apply_attention(self.backbone(source))
        # target = self.apply_attention(self.backbone(target))

        if source.shape[1] == 3:
            source = self.backboneRGB(source)
            target = self.backboneIR(target)
        else:
            source = self.backboneIR(source)
            target = self.backboneRGB(target)
        if displac is None:
            # regular walk through
            score_map = self.match_corr(target, source, padding=padding)
            score_map = self.out_batchnorm(score_map)
            return score_map.squeeze(1).squeeze(1)
        else:
            # for importance visualisation
            shifted_target = t.roll(target, -displac, -1)
            score = source * shifted_target
            score = t.sum(score, dim=[1])
            return score

    def match_corr(self, embed_ref, embed_srch, padding=None):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].
        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.
        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        if padding is None:
            padding = self.padding
        b, c, h, w = embed_srch.shape
        _, _, h_ref, w_ref = embed_ref.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.

        if self.training:
            match_map = conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b, padding=(0, padding))
            match_map = match_map.permute(1, 0, 2, 3)
        else:
            match_map = F.conv2d(F.pad(embed_srch.view(1, b * c, h, w), pad=(padding, padding, 1, 1), mode='circular'),
                                 embed_ref, groups=b)

            match_map = t.max(match_map.permute(1, 0, 2, 3), dim=2)[0].unsqueeze(2)
        return match_map

    def get_repr_rgb(self, img):
        return self.backboneRGB(img)

    def get_repr_ir(self, img):
        return self.backboneIR(img)

    def conv_repr(self, repr1, repr2):
        return self.out_batchnorm(self.match_corr(repr1, repr2, padding=repr1.size(-1)//2)).squeeze(1).squeeze(1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def get_parametrized_model_IR(lp, fs, ech, res, pad, device):
    backboneRGB = get_custom_CNN(3, lp, fs, ech, res)
    backboneIR = get_custom_CNN(1, lp, fs, ech, res)
    model = Siamese(backboneRGB, backboneIR, padding=pad).to(device)
    return model


def save_model(model, name, epoch, optimizer=None):
    t.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None
    }, "./results_" + name + "/model_" + str(epoch) + ".pt")
    print("Model saved to: " + "./results_" + name + "/model_" + str(epoch) + ".pt")


def load_model(model, path, optimizer=None):
    checkpoint = t.load(path, map_location=t.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model at", path)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    else:
        return model


def jit_save(model, name, epoch, arb_in, args):
    # save model arguments
    filename = "./results_" + name + "/model.info"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write(str(args))

    # save actual model
    t.jit.save(t.jit.trace(model, arb_in), "./results_" + name + "/model_" + str(epoch) + ".jit")


def jit_load(path, device):
    return t.jit.load(path, map_location=device)
