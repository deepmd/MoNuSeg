from ..unet import UNet


class DoubleUNet(nn.Module):
    def __init__(self, config):
        super(DoubleUNet, self).__init__()
        self.unet1 = UNet(config['unet1'])
        self.unet2 = UNet(config['unet2'])

    def forward(self, x):
        output1 = self.unet1(x)
        output2 = self.unet2(output1)
        return output1, output2
