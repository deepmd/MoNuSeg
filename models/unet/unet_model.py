from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, config):
        super(UNet, self).__init__()
        if len(config['down']) != len(config['up']):
            raise ValueError('Length of \'down\' and \'up\' should be the same.')
        self.downs = []
        self.bases = []
        self.ups = []
        last_channels = n_channels
        for d_channels, n in config['down']:
            self.downs.append(down(last_channels, d_channels, n))
            last_channels = d_channels
        for b_channels, n in config['base']:
            self.bases.append(conv(last_channels, b_channels, n))
            last_channels = b_channels
        bilinear_up = config['up_method'] == 'bilinear'
        for (u_channels, n), (d_channels, _) in zip(config['up'], reversed(config['down'])):
            self.ups.append(up(last_channels + d_channels, u_channels, n, bilinear_up))
            last_channels = u_channels
        self.outc = outconv(last_channels, n_classes)
        self.downs = nn.ModuleList(self.downs)
        self.bases = nn.ModuleList(self.bases)
        self.ups = nn.ModuleList(self.ups)

    def forward(self, x):
        d_outs = []
        for down in self.downs:
            x, before_pool = down(x)
            d_outs.append(before_pool)
        for base in self.bases:
            x = base(x)
        for up, d_out in zip(self.ups, reversed(d_outs)):
            x = up(x, d_out)
        x = self.outc(x)
        return x
