from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        if len(config['down']) != len(config['up']):
            raise ValueError('Length of \'down\' and \'up\' should be the same.')
        self.downs = []
        self.bases = []
        self.ups = []
        add_se = config.get('add_se', False)
        last_channels = config['in_channels']
        for d_channels, n in config['down']:
            self.downs.append(down(last_channels, d_channels, n, add_se))
            last_channels = d_channels
        for b_channels, n in config['base']:
            self.bases.append(conv(last_channels, b_channels, n))
            last_channels = b_channels
        up_method = config['up_method']
        for (u_channels, n), (d_channels, _) in zip(config['up'], reversed(config['down'])):
            self.ups.append(up(last_channels, d_channels, u_channels, n, up_method, add_se))
            last_channels = u_channels
        self.outc = outconv(last_channels, config['out_channels'])
        self.downs = nn.ModuleList(self.downs)
        self.bases = nn.ModuleList(self.bases)
        self.ups = nn.ModuleList(self.ups)
        self.penultimate_output = config.get('penultimate_output', False)

    def forward(self, x):
        d_outs = []
        for down in self.downs:
            x, before_pool = down(x)
            d_outs.append(before_pool)
        for base in self.bases:
            x = base(x)
        for up, d_out in zip(self.ups, reversed(d_outs)):
            x = up(x, d_out)
        y = self.outc(x)

        return (x, y) if self.penultimate_output else y
