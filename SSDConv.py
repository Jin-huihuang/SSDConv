import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, in_ratio=0.7, out_ratio=0.5, exp_times=2, reduction=16):
        super(SSDConv, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.in_ratio = in_ratio
        self.need_down = False

        base_out_channels = int(math.ceil(out_channels * out_ratio))
        diversity_out_channels = out_channels - base_out_channels
        exp_out_channels = diversity_out_channels * exp_times
        self.main_in = int(math.ceil(in_channels * in_ratio))
        exp_in = in_channels - self.main_in
        diversity_in = self.main_in + exp_out_channels

        self.base_branch = nn.Conv2d(self.main_in, base_out_channels, kernel_size, stride, kernel_size // 2, bias=False)

        self.expand_operation = nn.Conv2d(exp_in, exp_out_channels, kernel_size=1, stride=1, padding=0, bias=False) if exp_in != 0 else None
        if exp_in == 0:
            diversity_in = self.main_in

        if exp_out_channels != self.out_channels:
            self.need_down = True
            self.down = nn.Conv2d(exp_out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, groups=base_out_channels, bias=False)

        diversity_groups = math.gcd(diversity_out_channels, self.main_in)
        self.diversity_groups = diversity_groups
        self.diversity_branch = nn.Conv2d(diversity_in, diversity_out_channels, kernel_size, 1, kernel_size // 2, groups=diversity_groups, bias=False) if diversity_out_channels != 0 else None

        self.bn1 = nn.BatchNorm2d(diversity_in)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.avgpool_s2_detail = nn.AvgPool2d(2, 2)
        self.avgpool_s2_expand = nn.AvgPool2d(2, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, max(2, out_channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(2, out_channels // reduction), out_channels),
            nn.Sigmoid()
        )
        self.epsilon = 1e-5

    def dca9_2re(self, x):
        b, c, _, _ = x.size()
        m = self.avg_pool(x).view(b, c)
        v = x.flatten(2).softmax(dim=-1)
        v = F.normalize(v, dim=-1)
        s = (v @ v.transpose(-2, -1))
        s = (s.add(1).div(2)).sum(dim=2, keepdim=True).div(c).view(b, c)
        w = self.fc((1. - s) * m).view(b, c, 1, 1)
        return w

    def jc_concat(self, x1, x2):
        b, c1, h, w = x1.size()
        _, c2, _, _ = x2.size()
        c1 //= self.diversity_groups
        c2 //= self.diversity_groups
        x1 = x1.view(b, c1, self.diversity_groups, h, w)
        x2 = x2.view(b, c2, self.diversity_groups, h, w)
        return torch.cat([x1, x2], dim=1).view(b, (c1 + c2) * self.diversity_groups, h, w)

    def forward(self, x):
        x_m = x[:, :self.main_in, :, :]
        x_e = x[:, self.main_in:, :, :]
        y_sc = self.base_branch(x_m)

        if self.stride == 2:
            x_m = self.avgpool_s2_detail(x_m)
            x_e = self.avgpool_s2_expand(x_e)

        y_e = self.expand_operation(x_e) if self.expand_operation else 0

        if self.diversity_branch:
            x_gwc = torch.cat([x_m, y_e], dim=1)
            x_gwc = self.bn1(x_gwc)
            y_gwc = self.diversity_branch(x_gwc)
            y_m = torch.cat([y_sc, y_gwc], dim=1)
        else:
            y_m = y_sc

        y_m = self.bn2(y_m)
        w = self.dca9_2re(y_m)
        y_m = y_m * w

        if self.need_down:
            y_e = self.down(y_e)

        return y_m + y_e
