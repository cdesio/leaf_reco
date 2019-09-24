import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(out_channels, out_channels, 3, padding=1),
                               nn.ReLU(inplace=True))
    return conv_block


def conv_transpose(in_channels, out_channels, out_padding):
    transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3,3), padding=1, stride=2, output_padding=out_padding)
    return transpose


class cUNet(nn.Module):

    def __init__(self, out_size=4):
        super().__init__()

        self.conv_block_down1 = double_conv(1, 16)
        self.conv_block_down2 = double_conv(16, 32)
        self.conv_block_down3 = double_conv(32, 64)
        self.conv_block_down4 = double_conv(64, 128)
        self.conv_block_down5 = double_conv(128, 256)

        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)

        self.conv_transpose6 = conv_transpose(256, 128, out_padding=1)
        self.conv_block_up6 = double_conv(256, 128)

        self.conv_transpose7 = conv_transpose(128, 64, out_padding=1)
        self.conv_block_up7 = double_conv(128, 64)

        self.conv_transpose8 = conv_transpose(64, 32, out_padding=0)
        self.conv_block_up8 = double_conv(64, 32)

        self.conv_transpose9 = conv_transpose(32, 16, out_padding=1)
        self.conv_block_up9 = double_conv(32, 16)

        self.fc_linear = nn.Linear(123904, out_size)

        self.conv_last = nn.Sequential(nn.Conv2d(16, 1, 1),
                                       nn.Sigmoid())

    def forward(self, x):
    
        convb1 = self.conv_block_down1(x)
        #print('convb1: {}'.format(convb1.size()))

        pool1 = self.maxpool(convb1)
        #print('pool1: {}'.format(pool1.size()))

        convb2 = self.conv_block_down2(pool1)
        #print('convb2: {}'.format(convb2.size()))

        pool2 = self.maxpool(convb2)
        #print('pool2: {}'.format(pool2.size()))

        convb3 = self.conv_block_down3(pool2)
        #print('convb3: {}'.format(convb3.size()))

        pool3 = self.maxpool(convb3)
        #print('pool3: {}'.format(pool3.size()))

        convb4 = self.conv_block_down4(pool3)
        #print('convb4: {}'.format(convb4.size()))

        pool4 = self.maxpool(convb4)
        #print('pool4: {}'.format(pool4.size()))

        convb5 = self.conv_block_down5(pool4)
        #print('convb5: {}'.format(convb5.size()))
        flatt = convb5.view(convb5.size(0), -1)
        #print('flatten: {}'.format(flatt.size()))
        fc = self.fc_linear(flatt)
        #print('fc : {}'.format(fc.size()))

        up6 = self.conv_transpose6(convb5)
        #print('up6: {}'.format(up6.size()))
        x = torch.cat([up6, convb4], dim=1)
        #print('cat: {}'.format(x.size()))
        convb6 = self.conv_block_up6(x)
        #print('convb6: {}'.format(convb6.size()))

        up7 = self.conv_transpose7(convb6)
        #print('up7: {}'.format(up7.size()))
        x = torch.cat([up7, convb3], dim=1)
        #print('cat: {}'.format(x.size()))
        convb7 = self.conv_block_up7(x)
        #print('convb7: {}'.format(convb7.size()))

        up8 = self.conv_transpose8(convb7)
        #print('up8: {}'.format(up8.size()))
        x = torch.cat([up8, convb2], dim=1)
        #print('cat: {}'.format(x.size()))
        convb8 = self.conv_block_up8(x)
        #print('convb8: {}'.format(convb8.size()))

        up9 = self.conv_transpose9(convb8)
        #print('up9: {}'.format(up9.size()))

        x = torch.cat([up9, convb1], dim=1)
        #print('cat: {}'.format(x.size()))

        convb9 = self.conv_block_up9(x)
        #print('convb9: {}'.format(convb9.size()))

        mask = self.conv_last(convb9)
        #print("mask: {}".format(mask.size()))
        return mask, fc



def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (- ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()