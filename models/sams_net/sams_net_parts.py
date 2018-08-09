from common import *


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch+in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.conv2(torch.cat([inp, x], dim=1))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        before_pool = self.conv(x)
        x = self.down(before_pool)
        return x, before_pool


class UpBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch1, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + in_ch2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, is_deconv=True):
        super(OutBlock, self).__init__()
        self.in_channels = in_channels

        if scale_factor == 1:
            self.block = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif is_deconv:
            self.block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor*2, stride=scale_factor,
                                            padding=scale_factor//2, output_padding=scale_factor%2)
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        return self.block(x)
