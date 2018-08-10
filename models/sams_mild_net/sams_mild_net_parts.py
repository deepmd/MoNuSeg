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


class MILDBlock(nn.Module):
    def __init__(self, in_ch_inputs, in_ch_features, out_ch):
        super(MILDBlock, self).__init__()
        self.conv_inputs = nn.Sequential(
            nn.Conv2d(in_ch_inputs, out_ch//2, 3, padding=1),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(inplace=True)
        )
        self.conv_features1 = nn.Sequential(
            nn.Conv2d(in_ch_features, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_features2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_ch//2 + in_ch_features, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, features):
        inputs = self.conv_inputs(inputs)
        inputs = self.conv_cat(torch.cat([inputs, features], dim=1))
        features = self.conv_features1(features)
        features = self.conv_features2(features)
        x = self.relu(inputs + features)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch_inputs, in_ch_features, out_ch):
        super(DownBlock, self).__init__()
        self.mild = MILDBlock(in_ch_inputs, in_ch_features, out_ch) if in_ch_features != 0 else None
        self.conv = nn.Sequential(
                        nn.Conv2d(in_ch_inputs, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    ) if in_ch_features == 0 else None
        self.down = nn.MaxPool2d(2)

    def forward(self, inputs, features):
        before_pool = self.mild(inputs, features) if self.mild is not None else self.conv(inputs)
        features = self.down(before_pool)
        return features, before_pool


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
