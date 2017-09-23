import torch
import torch.nn as nn

'''
The generator network : 
    1. Use pixel_shuffle to scale the image by 4X 
    2. Since all layers are convolutional layers, so there is no requirement on the height and width to the images except for the channel of the images to be 3
    3. 
'''

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Adversarial_G(nn.Module):

    def __init__(self, layers):
        self.inplanes = 64
        super(Adversarial_G, self).__init__()

        self.conv1 = conv3x3(3, 64) # the number of channels of input images is 3
        self.relu1 = nn.ReLU(inplace=True)

        # residual part(layers : number of residual blocks)
        self.residual = self._make_layer(BasicBlock, 64, layers)

        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)

        # upscale
        self.conv3 = conv3x3(64, 64*2*2) # upsacle 2X 
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = conv3x3(64, 64*2*2) # upscale 2X
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = conv3x3(64, 3) # output

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        # residual part
        residual = out # save for future use
        out = self.residual(out)

        out = self.residual(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        # upscale
        out = self.conv3(out)
        out = nn.functional.pixel_shuffle(out, upscale_factor = 2)
        out = self.relu3(out)
        out = self.conv4(out)
        out = nn.functional.pixel_shuffle(out, upscale_factor = 2)
        out = self.relu4(out)

        out = self.conv5(out)

        return out
  

    def _make_layer(self, block, planes, blocks, stride=1):
        '''generate subnetwork combined with B residual blocks.
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

