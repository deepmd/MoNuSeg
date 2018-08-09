from .sams_net_parts import *


class SAMS_Net(nn.Module):
    def __init__(self, config):
        super(SAMS_Net, self).__init__()
        if len(config['down']) != len(config['up']):
            raise ValueError('Length of \'down\' and \'up\' should be the same.')
        self.downs = []
        self.bases = []
        self.ups = []
        self.outs = []
        in_channels = config['in_channels']
        last_channels = 0
        for d_channels, n in config['down']:
            self.downs.append(DownBlock(last_channels+in_channels, d_channels))
            last_channels = d_channels
        for b_channels, n in config['base']:
            self.bases.append(ConvBlock(last_channels+in_channels, b_channels))
            last_channels = b_channels
        scale = (len(self.downs) - 1) * 2
        for (u_channels, n), (d_channels, _) in zip(config['up'], reversed(config['down'])):
            self.ups.append(UpBlock(last_channels, d_channels, u_channels))
            self.outs.append(OutBlock(u_channels, config['out_channels'], scale_factor=scale))
            last_channels = u_channels
            scale = scale // 2
        self.downs = nn.ModuleList(self.downs)
        self.bases = nn.ModuleList(self.bases)
        self.ups = nn.ModuleList(self.ups)
        self.outs = nn.ModuleList(self.outs)

    def forward(self, inputs):
        d_outs = []
        outputs = []
        for i, down in enumerate(self.downs):
            x = torch.cat([x, inputs[i]], dim=1) if i > 0 else inputs[0]
            x, before_pool = down(x)
            d_outs.append(before_pool)
        for i, base in enumerate(self.bases):
            x = torch.cat([x, inputs[len(self.downs)]], dim=1)
            x = base(x)
        for up, out, d_out in zip(self.ups, self.outs, reversed(d_outs)):
            x = up(x, d_out)
            outputs.append(out(x))

        return list(reversed(outputs))
