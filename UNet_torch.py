import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(out_channels, out_channels, 3, padding=1),
                               nn.ReLU(inplace=True))
    return conv_block


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.conv_block_down1 = double_conv(1, 64)
        self.conv_block_down2 = double_conv(64, 128)
        self.conv_block_down3 = double_conv(128, 256)
        self.conv_block_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_block_up3 = double_conv(256 + 512, 256)
        self.conv_block_up2 = double_conv(128 + 256, 128)
        self.conv_block_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)


    def forward(self, x):
        convb1 = self.conv_block_down1(x)
        x = self.maxpool(convb1)

        convb2 = self.conv_block_down2(x)
        x = self.maxpool(convb2)

        convb3 = self.conv_block_down3(x)
        x = self.maxpool(convb3)

        x = self.conv_block_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, convb3], dim=1)

        x = self.conv_block_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, convb2], dim=1)

        x = self.conv_block_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, convb1], dim=1)

        x = self.conv_block_up1(x)

        out = self.conv_last(x)

        return out




def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()