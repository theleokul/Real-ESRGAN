from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x, layer_name_list=[]):
        output = {}

        x0 = self.conv0(x)
        if 'conv0' in layer_name_list:
            output['conv0'] = x0.clone()
        x0 = F.leaky_relu(x0, negative_slope=0.2, inplace=True)

        x1 = self.conv1(x0)
        if 'conv1' in layer_name_list:
            output['conv1'] = x1.clone()
        x1 = F.leaky_relu(x1, negative_slope=0.2, inplace=True)

        x2 = self.conv2(x1)
        if 'conv2' in layer_name_list:
            output['conv2'] = x2.clone()
        x2 = F.leaky_relu(x2, negative_slope=0.2, inplace=True)

        x3 = self.conv3(x2)
        if 'conv3' in layer_name_list:
            output['conv3'] = x3.clone()
        x3 = F.leaky_relu(x3, negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)

        x4 = self.conv4(x3)
        if 'conv4' in layer_name_list:
            output['conv4'] = x4.clone()
        x4 = F.leaky_relu(x4, negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)

        x5 = self.conv5(x4)
        if 'conv5' in layer_name_list:
            output['conv5'] = x5.clone()
        x5 = F.leaky_relu(x5, negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)

        x6 = self.conv6(x5)
        if 'conv6' in layer_name_list:
            output['conv6'] = x6.clone()
        x6 = F.leaky_relu(x6, negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra
        x7 = self.conv7(x6)
        if 'conv7' in layer_name_list:
            output['conv7'] = x7.clone()
        x7 = F.leaky_relu(x7, negative_slope=0.2, inplace=True)

        x8 = self.conv8(x7)
        if 'conv8' in layer_name_list:
            output['conv8'] = x8.clone()
        x8 = F.leaky_relu(x8, negative_slope=0.2, inplace=True)

        x9 = self.conv9(x8)
        if 'conv9' in layer_name_list:
            output['conv9'] = x9.clone()

        if len(output) == 0:
            output = x9
        else:
            # Just alias interface
            output['output'] = x9.clone()

        return output
