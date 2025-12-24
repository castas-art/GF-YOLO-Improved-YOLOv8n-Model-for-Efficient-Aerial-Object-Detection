# 修复BiFPN类的初始化方法
# 保存到ultralytics/nn/AddModel/fusion.py
import torch.nn.functional as F
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

import torch.nn as nn
import torch.nn as nn
import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


# SE
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)



def Upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, c1, c2, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out


class ConvSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSEBlock, self).__init__()
        # 1×1卷积用于特征融合和降维
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # SE模块

    def forward(self, x):
        # 假设x是拼接后的特征
        x = self.conv(x)  # 通过1×1卷积融合和降维
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimFusion_3in_to_max(nn.Module):
    def __init__(self, in_channel_list, out_channels, multi_sample=True, sample_steps=2):
        super().__init__()
        self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                0] != out_channels else nn.Identity()
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                2] != out_channels else nn.Identity()
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        self.multi_sample = multi_sample  # 是否使用多次采样
        self.sample_steps = sample_steps  # 采样步骤数

        # 移除DySample上采样模块，使用普通上采样

        self.ConvSEBlock = ConvSEBlock(in_channel_list[0] + in_channel_list[1] + in_channel_list[2], out_channels)
        self.conv = nn.Conv2d(in_channel_list[0] + in_channel_list[1] + in_channel_list[2], out_channels, kernel_size=1)

        self.CBAM = CBAM(out_channels, out_channels)
        self.CBAM_1 = CBAM(in_channel_list[0], in_channel_list[0])
        self.CBAM_2 = CBAM(in_channel_list[1], in_channel_list[1])
        self.CBAM_3 = CBAM(in_channel_list[2], in_channel_list[2])

    def normal_upsample(self, x, target_size):
        _, _, H, W = x.shape
        target_H, target_W = target_size

        # 如果已经是目标尺寸，直接返回
        if (H, W) == target_size:
            return x

        # 使用双线性插值进行上采样
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x

    def forward(self, x):
        # 获取三个特征图的尺寸
        _, _, H1, W1 = x[0].shape
        _, _, H2, W2 = x[1].shape
        _, _, H3, W3 = x[2].shape

        # 找到最大的尺寸
        max_H = max(H1, H2, H3)
        max_W = max(W1, W2, W3)
        output_size = (max_H, max_W)

        # 调整所有特征图到最大尺寸，使用普通上采样
        x1 = self.normal_upsample(x[0], output_size)
        x2 = self.normal_upsample(x[1], output_size)
        x3 = self.normal_upsample(x[2], output_size)



        # 应用ConvSEBlock融合特征  self.conv(x)
        x = self.ConvSEBlock(torch.cat((x1, x2, x3), dim=1))
        # x = self.se(x)
        # x = self.ca(x)
        #x=torch.cat((x1, x2, x3), dim=1)
        #x=torch.cat((x1, x2, x3),dim=1)

        return self.CBAM(x)


class SimFusion_3in_to_mid(nn.Module):
    def __init__(self, in_channel_list, out_channels, multi_sample=True, sample_steps=2):
        super().__init__()
        self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                0] != out_channels else nn.Identity()
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                2] != out_channels else nn.Identity()

        # 移除DySample上采样模块

        self.ConvSEBlock = ConvSEBlock(in_channel_list[0] + in_channel_list[1] + in_channel_list[2], out_channels)
        self.conv = nn.Conv2d(in_channel_list[0] + in_channel_list[1] + in_channel_list[2], out_channels, kernel_size=1)
        self.CBAM = CBAM(out_channels, out_channels)
        self.CBAM_1 = CBAM(in_channel_list[0], in_channel_list[0])
        self.CBAM_2 = CBAM(in_channel_list[1], in_channel_list[1])
        self.CBAM_3 = CBAM(in_channel_list[2], in_channel_list[2])

    def normal_sample(self, x, target_size):
        """使用普通的采样方法：上采样用双线性插值，下采样用自适应池化"""
        _, _, H, W = x.shape
        target_H, target_W = target_size

        # 如果已经是目标尺寸，直接返回
        if (H, W) == target_size:
            return x

        # 判断是上采样还是下采样
        if H * W < target_H * target_W:  # 需要上采样
            # 使用双线性插值进行上采样
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        else:  # 需要下采样
            # 使用自适应平均池化进行下采样
            x = nn.functional.adaptive_avg_pool2d(x, target_size)

        return x

    def forward(self, x):
        # 获取三个特征图的尺寸
        _, _, H1, W1 = x[0].shape
        _, _, H2, W2 = x[1].shape
        _, _, H3, W3 = x[2].shape

        # 计算每个特征图的面积
        areas = [(H1 * W1), (H2 * W2), (H3 * W3)]
        # 找出中间大小的特征图索引
        mid_idx = areas.index(sorted(areas)[1])

        # 获取中间大小特征图的尺寸
        mid_H, mid_W = [(H1, W1), (H2, W2), (H3, W3)][mid_idx]
        output_size = (mid_H, mid_W)

        # 调整所有特征图到中间尺寸，使用普通采样
        x1 = self.normal_sample(x[0], output_size)
        x2 = self.normal_sample(x[1], output_size)
        x3 = self.normal_sample(x[2], output_size)


        x=self.ConvSEBlock(torch.cat((x1, x2, x3), dim=1))




        # 应用ConvSEBlock融合特征 self.conv(x)


        return self.CBAM(x)
