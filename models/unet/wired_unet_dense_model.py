from .unet_parts import *
from ..unet import UNet


class DenseWiredUNet(UNet):
    def __init__(self, config2, config1):
        super(DenseWiredUNet, self).__init__(config2)
        downs = []
        bases = []
        ups = []
        add_se = config2.get('add_se', False)
        last_channels = config2['in_channels']
        for (d_channels2, n), (d_channels1, _) in zip(config2['down'], config1['down']):
            downs.append(down_merge(last_channels, d_channels1, d_channels2, n, add_se, True))
            last_channels = d_channels1 + d_channels2
        for b_channels, n in config2['base']:
            bases.append(conv(last_channels, b_channels, n))
            last_channels = b_channels
        up_method = config2['up_method']
        for (u_channels, n), (d_channels2, _), (d_channels1, _) in zip(config2['up'], reversed(config2['down']), reversed(config1['down'])):
            ups.append(up(last_channels, d_channels1 + d_channels2, u_channels, n, up_method, add_se))
            last_channels = u_channels
        self.outc = outconv(last_channels, config2['out_channels'])
        self.downs = nn.ModuleList(downs)
        self.bases = nn.ModuleList(bases)
        self.ups = nn.ModuleList(ups)
