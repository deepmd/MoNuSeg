from common import *
from ..unet import UNet


class DoubleUNet(nn.Module):
    def __init__(self, config):
        super(DoubleUNet, self).__init__()
        self.concat_input = config['concat_input']
        self.unet1 = UNet(config['unet1'])
        self.unet2 = UNet(config['unet2'])

    def forward(self, x):
        output1 = self.unet1(x)
        x = torch.cat([x, output1], dim=1) if self.concat_input else output1
        output2 = self.unet2(x)
        return output1, output2
