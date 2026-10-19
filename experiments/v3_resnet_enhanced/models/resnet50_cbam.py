"""
models/resnet50_cbam.py
ResNet50 with Convolutional Block Attention Module (CBAM)

Implements Woo et al. (ECCV 2018) CBAM attention specifically inserted into
layer3 and layer4 of ResNet50:
- layer1: standard Bottleneck (low-level edges, intensity transitions)
- layer2: standard Bottleneck (mid-level speckle textures)
- layer3: CBAM-augmented Bottleneck (channel + spatial lesion attention)
- layer4: CBAM-augmented Bottleneck (high-level uterine anatomy & capsule features)
- fc: 2-class linear classifier (UF vs NUF)

Pretrained ImageNet weights are loaded into all standard ResNet layers,
ensuring rapid convergence and preventing overfitting on ultrasound data.
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tv_models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class BottleneckCBAM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super(BottleneckCBAM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(planes * self.expansion, ratio=16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet50_CBAM, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers 1 & 2: Standard Bottleneck (no CBAM)
        self.layer1 = self._make_layer(BottleneckCBAM, 64, 3, stride=1, use_cbam=False)
        self.layer2 = self._make_layer(BottleneckCBAM, 128, 4, stride=2, use_cbam=False)

        # Layers 3 & 4: CBAM-augmented Bottleneck
        self.layer3 = self._make_layer(BottleneckCBAM, 256, 6, stride=2, use_cbam=True)
        self.layer4 = self._make_layer(BottleneckCBAM, 512, 3, stride=2, use_cbam=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckCBAM.expansion, num_classes)

        # Initialize non-pretrained weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1, use_cbam=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):
        print("Loading ImageNet pretrained weights into ResNet50-CBAM backbone...")
        tv_res = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        tv_state = tv_res.state_dict()
        model_state = self.state_dict()

        loaded_keys = 0
        skipped_keys = 0
        for k, v in tv_state.items():
            if k.startswith("fc."):
                continue
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded_keys += 1
            else:
                skipped_keys += 1

        self.load_state_dict(model_state)
        print(f"Successfully loaded {loaded_keys} pretrained tensors into ResNet50-CBAM backbone (skipped {skipped_keys} new attention tensors).")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing ResNet50-CBAM on {device}...")
    model = ResNet50_CBAM(num_classes=2, pretrained=True).to(device)
    
    # Test forward pass with 256x256 resolution
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: [2, 2])")
    
    # Test loss and backward pass
    loss_fn = nn.CrossEntropyLoss()
    target = torch.tensor([0, 1]).to(device)
    loss = loss_fn(output, target)
    loss.backward()
    print(f"Backward pass succeeded! Loss: {loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
