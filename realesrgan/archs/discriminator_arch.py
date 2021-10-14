from basicsr.utils.registry import ARCH_REGISTRY
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True, dis_in_bottleneck=False):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        self.dis_in_bottleneck = dis_in_bottleneck
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        if dis_in_bottleneck:
            self.bottleneck1 = norm(nn.Linear(num_feat * 8, num_feat * 2))
            self.bottleneck2 = nn.Linear(num_feat * 2, 1)

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

        # h, w
        x1 = self.conv1(x0)
        if 'conv1' in layer_name_list:
            output['conv1'] = x1.clone()
        x1 = F.leaky_relu(x1, negative_slope=0.2, inplace=True)

        # h / 2, w / 2
        x2 = self.conv2(x1)
        if 'conv2' in layer_name_list:
            output['conv2'] = x2.clone()
        x2 = F.leaky_relu(x2, negative_slope=0.2, inplace=True)

        # h / 4, w / 4
        x3 = self.conv3(x2)
        if 'conv3' in layer_name_list:
            output['conv3'] = x3.clone()

        if self.dis_in_bottleneck:
            x3_pooled = F.adaptive_avg_pool2d(x3, 1)[..., 0, 0]  # B, 512
            x3_pooled = self.bottleneck1(x3_pooled) # B, 128
            x3_pooled = F.leaky_relu(x3_pooled, negative_slope=0.2, inplace=True)
            x3_pooled = self.bottleneck2(x3_pooled)
            output['bottleneck'] = x3_pooled.clone()

        x3 = F.leaky_relu(x3, negative_slope=0.2, inplace=True)

        # h / 8, w / 8, if h = w = 256 -> 32, 32? Not to large for adaptive_avg_pool2d
        # upsample; middle
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)

        # h / 4, w / 4
        x4 = self.conv4(x3)
        if 'conv4' in layer_name_list:
            output['conv4'] = x4.clone()
        x4 = F.leaky_relu(x4, negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)

        # h / 2, w / 2
        x5 = self.conv5(x4)
        if 'conv5' in layer_name_list:
            output['conv5'] = x5.clone()
        x5 = F.leaky_relu(x5, negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)

        # h, w
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
            # Just alias interface for x9
            output['output'] = x9.clone()

        return output


@ARCH_REGISTRY.register()
class TwoHeadedUNetDiscriminatorSN(UNetDiscriminatorSN):

    def __init__(self, num_feat, *args, two_head=True, is_face_mask_subtask_on=False, **kwargs):
        super(TwoHeadedUNetDiscriminatorSN, self).__init__(num_feat=num_feat, *args, **kwargs)

        self.two_head = two_head
        if two_head:
            self.conv9_face = nn.Conv2d(num_feat, 1, 3, 1, 1)

        self.is_face_mask_subtask_on = is_face_mask_subtask_on

    def forward(self, x: torch.Tensor, face_mask: torch.Tensor=None, layer_name_list=[]):
        # binary_mask: 0 -- non-face regions, 1 -- face regions

        if 'conv8' not in layer_name_list:
            layer_name_list.append('conv8')
        if 'conv9' not in layer_name_list:
            layer_name_list.append('conv9')

        output = super(TwoHeadedUNetDiscriminatorSN, self).forward(x, layer_name_list=layer_name_list)

        x8 = F.leaky_relu(output['conv8'], negative_slope=0.2, inplace=True)
        x9_face = self.conv9_face(x8)

        x9 = output['conv9']  # Old general features

        if self.is_face_mask_subtask_on:
            # Multi-task: fake/real + face segmentation map
            output['face_mask'] = x9_face
        else:
            if face_mask.ndim == 3:
                # Add color channel
                face_mask.unsqueeze_(1)
            x9_combined = x9_face * face_mask + x9 * (1. - face_mask)
            output['conv9'] = x9_combined
            output['output'] = x9_combined

        return output
