from ..se_parts import *


class bottleneck(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self, in_ch, out_ch):
        super(bottleneck, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 1),
                  nn.BatchNorm2d(out_ch),
                  nn.ReLU(inplace=True)]
        self.bottleneck = nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck(x)
        return x


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
    def __init__(self, in_ch, out_ch, n, add_se=False):
        super(down, self).__init__()
        self.conv = conv(in_ch, out_ch, n)
        self.down = nn.MaxPool2d(2)
        self.se = scSE_block(out_ch, 2) if add_se else (lambda x: x)

    def forward(self, x):
        before_pool = self.conv(x)
        before_pool = self.se(before_pool)
        x = self.down(before_pool)
        return x, before_pool


class up(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, n, bilinear=True, add_se=False):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch1 + in_ch2, out_ch, 2, stride=2)
        self.conv = conv(in_ch1 + in_ch2, out_ch, n)
        self.se = scSE_block(out_ch, 2) if add_se else (lambda x: x)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_merge(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, n, add_se=False):
        super(down_merge, self).__init__()
        self.conv = conv(in_ch1, out_ch, n)
        self.bottleneck = bottleneck(out_ch + in_ch2, out_ch)
        self.down = nn.MaxPool2d(2)
        self.se = scSE_block(out_ch, 2) if add_se else (lambda x: x)

    def forward(self, x1, x2):
        x = self.conv(x1)
        x = torch.cat([x, x2], dim=1)
        before_pool = self.bottleneck(x)
        before_pool = self.se(before_pool)
        x = self.down(before_pool)
        return x, before_pool


class normalize(nn.Module):
    def forward(self, x):
        x[:, :2] = F.normalize(x[:, :2].clone(), p=2, dim=1) * 0.999999  # multiplying by 0.999999 prevents 'nan'!
        x[:, 2:] = F.normalize(x[:, 2:].clone(), p=2, dim=1) * 0.999999  # multiplying by 0.999999 prevents 'nan'!

        return x
