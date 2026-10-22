"""Architectures CIFAR-10: MLP, CNN, ResNet-18 et WideResNet-16-4."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, hidden: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, num_classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                  nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """CNN compact: blocs 32/64/128 canaux, réduction spatiale progressive."""
    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(ConvBlock(3, 32, dropout=dropout / 2), ConvBlock(32, 64, dropout=dropout),
                                      ConvBlock(64, 128, pool=False, dropout=dropout))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, num_classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity() if stride == 1 and in_planes == planes else nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    """ResNet-18 avec stem 3x3 adapté aux petites images CIFAR (sans max-pooling initial)."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    def _layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        layers += [BasicBlock(self.in_planes, planes) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.pool(x).flatten(1))


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.shortcut = nn.Identity() if in_planes == out_planes and stride == 1 else nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(self.dropout(torch.relu(self.bn2(out))))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """WideResNet 16-4; réseau avancé compact sur images 32x32."""
    def __init__(self, num_classes: int = 10, widen_factor: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.block1 = self._group(widths[0], widths[1], 2, 1, dropout)
        self.block2 = self._group(widths[1], widths[2], 2, 2, dropout)
        self.block3 = self._group(widths[2], widths[3], 2, 2, dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
    @staticmethod
    def _group(in_planes: int, out_planes: int, count: int, stride: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(WideBasicBlock(in_planes, out_planes, stride, dropout), *[
            WideBasicBlock(out_planes, out_planes, 1, dropout) for _ in range(count - 1)])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block3(self.block2(self.block1(self.conv1(x))))
        return self.fc(torch.nn.functional.adaptive_avg_pool2d(torch.relu(self.bn(x)), 1).flatten(1))


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower().replace("-", "_")
    models = {"mlp": MLP, "cnn": CustomCNN, "resnet18": ResNetCIFAR, "wide_resnet": WideResNet, "wideresnet": WideResNet}
    if name not in models:
        raise ValueError(f"Modèle inconnu: {name}; choix: {', '.join(models)}")
    return models[name](**kwargs)
