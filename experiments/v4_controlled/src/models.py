#!/usr/bin/env python3
"""
Model Architectures for Controlled Experiments
Location: experiments/v4_controlled/src/models.py

Defines:
  1. ResNet50Baseline: Standard Torchvision ResNet50 classifier
  2. ResNet50AuxLocalization: ResNet50 with Auxiliary Bounding-Box Regression Head
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feat = self.avgpool(x)
        flat = torch.flatten(feat, 1)
        logits = self.fc(flat)
        return logits


class ResNet50AuxLocalization(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Primary Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(2048, num_classes)
        
        # Auxiliary Bounding-Box Regression Head
        self.aux_conv = nn.Conv2d(2048, 256, kernel_size=3, padding=1)
        self.aux_relu = nn.ReLU(inplace=True)
        self.aux_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.aux_fc = nn.Linear(256, 4)
        self.aux_sigmoid = nn.Sigmoid() # Outputs normalized coordinates [0, 1]^4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x) # [B, 2048, H/32, W/32]

        # Primary Head
        cls_pool = self.avgpool(feat_map)
        cls_flat = torch.flatten(cls_pool, 1)
        cls_drop = self.dropout(cls_flat)
        logits = self.fc(cls_drop)

        # Auxiliary Head
        aux_f = self.aux_conv(feat_map)
        aux_f = self.aux_relu(aux_f)
        aux_p = self.aux_pool(aux_f)
        aux_flat = torch.flatten(aux_p, 1)
        pred_boxes = self.aux_sigmoid(self.aux_fc(aux_flat)) # [B, 4]

        return logits, pred_boxes
