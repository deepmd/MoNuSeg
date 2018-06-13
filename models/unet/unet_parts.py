import torch
import torch.nn as nn
import torch.nn.functional as F


class conv(nn.Module):
    '''(conv => BN => ReLU) * n'''
    def __init__(self, in_ch, out_ch, n):
        super(conv, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                  nn.BatchNorm2d(out_ch),
                  nn.ReLU(inplace=True)]
        for i in range(n-1):
            layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, n):
        super(down, self).__init__()
        self.conv = conv(in_ch, out_ch, n)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        before_pool = self.conv(x)
        x = self.down(before_pool)
        return x, before_pool


class up(nn.Module):
    def __init__(self, in_ch, out_ch, n, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = conv(in_ch, out_ch, n)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
