from .unet_parts import *
from ..unet import UNet


class UNet2(UNet):
    def __init__(self, config2, config1):
        super(UNet2, self).__init__(config2)

        downs = []
        add_se = config2.get('add_se', False)
        last_channels = config2['in_channels']
        for (d_channels2, n), (d_channels1, _) in zip(config2['down'], config1['down']):
            downs.append(down_merge(last_channels, d_channels1, d_channels2, n, add_se))
            last_channels = d_channels2
        self.downs = nn.ModuleList(downs)