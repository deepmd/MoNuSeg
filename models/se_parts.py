from common import *


class cSE_block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(cSE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class sSE_block(nn.Module):
    def __init__(self, channels):
        super(sSE_block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(channels, 1, 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        _, c, _, _ = x.shape
        y = self.conv(x).repeat((1, c, 1, 1))
        return x * y


class scSE_block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(scSE_block, self).__init__()
        self.cSE = cSE_block(channels, reduction)
        self.sSE = sSE_block(channels)

    def forward(self, x):
        y = self.cSE(x)
        z = self.sSE(x)
        return y + z
