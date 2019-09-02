import torch
import torch.nn as nn

IMG_WIDTH = 1400
IMG_HEIGHT = 1400


def double_conv(in_channels, out_channels):
    conv_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
	nn.ReLU(inplace=True),
	nn.Conv2d(out_channels, out_channels, 3, padding=1),
	nn.ReLU(inplace=True))
    return conv_block


class UNet(nn.Module):

    
    	
