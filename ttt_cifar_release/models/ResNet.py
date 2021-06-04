# Based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3
import functional_layers as L

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        x = x.double()
        residual = x 
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual

class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        print(f"avg pool stride: {stride}")
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn
        # print(f"nOut: {nOut}, nIn: {nIn}, stride: {stride}, expand ratio: {self.expand_ratio}")

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

class ResNetCifar(nn.Module):
    def __init__(self, depth, width=1, classes=10, channels=3, norm_layer=nn.BatchNorm2d):
        assert (depth - 2) % 6 == 0         # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetCifar, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(norm_layer, 16 * width)
        self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
        self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * width, classes)

        # Initialization
        self._init_weights()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def _make_layer(self, norm_layer, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
            print("downsample activated")
        layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self):
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                flattened_size = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / flattened_size))
                if module.bias is not None:
                    module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                elif isinstance(module, nn.Linear):
                    module.weight.data.normal_(0, 0.01)
                    module.bias.data = torch.ones(module.bias.data.size())

    def forward(self, x, weights=None, cuda=True):
        x = x.double()
        if weights == None:
            x = self.conv1(x)
            # print(f"SHAPE 1: {x.shape}")
            x = self.layer1(x)
            # print(f"SHAPE 2: {x.shape}")
            x = self.layer2(x)
            # print(f"SHAPE 3: {x.shape}")
            x = self.layer3(x)
            # print(f"SHAPE 4: {x.shape}")
            x = self.bn(x)
            x = self.relu(x)
            x = self.avgpool(x)
            # print(f"SHAPE 5: {x.shape}")
            x = x.view(x.size(0), -1)
            # print(f"SHAPE 6: {x.shape}")
            x = self.fc(x)
            return x
        else:
            # conv1, layer1, layer2
            output = x
            output = L.conv2d(output, weights['ext.0.weight'], cuda=cuda)
            for ext_seq_id in range(1, 3):
                identity = output
                for ext_block_id in range(4):
                    for sub_block_id in range(1, 3):
                        stride = 2 if (ext_seq_id == 2 and ext_block_id == 0 and sub_block_id == 1) else 1
                        # error on next line?
                        output = L.batch_norm(output,
                                              weights[f'ext.{ext_seq_id}.{ext_block_id}.bn{sub_block_id}.weight'],
                                              bias=weights[f'ext.{ext_seq_id}.{ext_block_id}.bn{sub_block_id}.bias'],
                                              momentum=0.1, cuda=cuda)
                        output = L.relu(output)
                        output = L.conv2d(output, weights[f'ext.{ext_seq_id}.{ext_block_id}.conv{sub_block_id}.weight'],
                                          stride=stride, cuda=cuda)

                # downsample
                if ext_seq_id == 2:
                    identity = L.avg_pool(identity, 2)
                    identity = torch.cat([identity] + [identity.mul(0)], 1)
                output += identity

            # assume shared up to layer 2
            x = self.layer3(output)
            # print(f"SHAPE 4: {x.shape}")
            x = self.bn(x)
            x = self.relu(x)
            x = self.avgpool(x)
            # print(f"SHAPE 5: {x.shape}")
            x = x.view(x.size(0), -1)
            # print(f"SHAPE 6: {x.shape}")
            x = self.fc(x)
            return x