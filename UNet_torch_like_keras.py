import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(out_channels, out_channels, 3, padding=1),
                               nn.ReLU(inplace=True))
    return conv_block

def conv_transpose(in_channels, out_channels):
    transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3,3), padding=1, stride=2)
    return transpose

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_block_down1 = double_conv(1, 32)
        self.conv_block_down2 = double_conv(32, 64)
        self.conv_block_down3 = double_conv(64, 128)
        self.conv_block_down4 = double_conv(128, 256)
        self.conv_block_down5 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)

#        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_transpose6 = conv_transpose(512, 256)
        self.conv_block_up6 = double_conv(512+256, 256)

        self.conv_transpose7 = conv_transpose(256, 128)
        self.conv_block_up7 = double_conv(128 + 256, 128)

        self.conv_transpose8 = conv_transpose(128, 64)
        self.conv_block_up8 = double_conv(64+128, 64)

        self.conv_transpose9 = conv_transpose(32 , 32)
        self.conv_block_up9 = double_conv(32+ 64, 32)

        self.conv_last = nn.Conv2d(32, 1, 1)


    def forward(self, x):
        convb1 = self.conv_block_down1(x)
        x = self.maxpool(convb1)

        convb2 = self.conv_block_down2(x)
        x = self.maxpool(convb2)

        convb3 = self.conv_block_down3(x)
        x = self.maxpool(convb3)

        convb4 = self.conv_block_down4(x)
        x = self.maxpool(convb4)

        convb5 = self.conv_block_down5(x)

        up6 = self.conv_transpose6(convb5)
        x = torch.cat([up6, convb4], dim=1)

        convb6 = self.conv_block_up6(x)

        up7 = self.conv_transpose7(convb6)
        x = torch.cat([up7, convb3], dim=1)

        convb7 = self.conv_block_up7(x)

        up8 = self.conv_transpose8(convb7)
        x = torch.cat([up8, convb2], dim=1)

        convb8 = self.conv_block_up8(x)

        up9 = self.conv_transpose9(convb8)
        x = torch.cat([up9, convb1], dim=1)

        convb9 = self.conv_block_up9(x)

        out = self.conv_last(convb9)

        return out




def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()