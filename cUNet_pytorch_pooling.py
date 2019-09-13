import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(out_channels, out_channels, 3, padding=1),
                               nn.ReLU(inplace=True))
    return conv_block


def dilated_conv(in_channels, out_channels):
    transpose = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, stride=1,
                          dilation=1)
    return transpose


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_block_down1 = double_conv(1, 16)
        self.conv_block_down2 = double_conv(16, 32)
        self.conv_block_down3 = double_conv(32, 64)
        self.conv_block_down4 = double_conv(64, 128)
        self.conv_block_down5 = double_conv(128, 256)

        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)

        self.conv_transpose6 = dilated_conv(256, 128)
        self.conv_block_up6 = conv(256, 128)

        self.conv_transpose7 = dilated_conv(128, 64)
        self.conv_block_up7 = conv(128, 64)

        self.conv_transpose8 = dilated_conv(64, 32)
        self.conv_block_up8 = conv(64, 32)

        self.conv_transpose9 = dilated_conv(32, 16)
        self.conv_block_up9 = conv(32, 16)

        self.fc_classifier = nn.Linear(112896, 4)

        self.conv_last = nn.Sequential(nn.Conv2d(32, 1, 1),
                                       nn.Sigmoid())

    def forward(self, x):
        convb1 = self.conv_block_down1(x)
        print('convb1: {}'.format(convb1.size()))

        pool1 = self.maxpool(convb1)
        print('pool1: {}'.format(pool1.size()))

        convb2 = self.conv_block_down2(pool1)
        print('convb2: {}'.format(convb2.size()))

        pool2 = self.maxpool(convb2)
        print('pool2: {}'.format(pool2.size()))

        convb3 = self.conv_block_down3(pool2)
        print('convb3: {}'.format(convb3.size()))

        pool3 = self.maxpool(convb3)
        print('pool3: {}'.format(pool3.size()))

        convb4 = self.conv_block_down4(pool3)
        print('convb4: {}'.format(convb4.size()))

        pool4 = self.maxpool(convb4)
        print('pool4: {}'.format(pool4.size()))

        convb5 = self.conv_block_down5(pool4)
        print('convb5: {}'.format(convb5.size()))
        flatt = convb5.view(convb5.size(0), -1)
        print('flatten: {}'.format(flatt.size()))
        fc = self.fc_classifier(flatt)
        print('fc : {}'.format(fc.size()))
        classification = nn.Softmax(fc)

        up6 = self.conv_transpose6(convb5)
        print('up6: {}'.format(up6.size()))
        x = torch.cat([up6, convb4], dim=1)
        print('cat: {}'.format(x.size()))
        convb6 = self.conv_block_up6(x)
        print('convb6: {}'.format(convb6.size()))

        up7 = self.conv_transpose7(convb6)
        print('up7: {}'.format(up7.size()))
        x = torch.cat([up7, convb3], dim=1)
        print('cat: {}'.format(x.size()))
        convb7 = self.conv_block_up7(x)
        print('convb7: {}'.format(convb7.size()))

        up8 = self.conv_transpose8(convb7)
        print('up8: {}'.format(up8.size()))
        x = torch.cat([up8, convb2], dim=1)
        print('cat: {}'.format(x.size()))
        convb8 = self.conv_block_up8(x)
        print('convb8: {}'.format(convb8.size()))

        up9 = self.conv_transpose9(convb8)
        print('up9: {}'.format(up9.size()))

        x = torch.cat([up9, convb1], dim=1)
        print('cat: {}'.format(x.size()))

        convb9 = self.conv_block_up9(x)
        print('convb9: {}'.format(convb9.size()))

        mask = self.conv_last(convb9)
        print(mask.size())
        return mask, classification



def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (- ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()