#!/usr/bin/env python3

import torch
import torch.nn as nn
from config import model_config

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        if bn_act:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )
        else:
            self.layer = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)


    def forward(self, x):
        return self.layer(x)



class ResBlock(nn.Module):
    def __init__(self, in_channels, num_repeats=1, use_res=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [nn.Sequential(
                CNNBlock(in_channels, in_channels // 2, kernel_size=1, padding=0),
                CNNBlock(in_channels // 2, in_channels , kernel_size=3, padding=1)
            )]
        self.use_res = use_res
        self.num_repeats = num_repeats


    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_res else layer(x)
        return x


class Prediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
            CNNBlock(in_channels*2, 3 * (num_classes + 5), bn_act=False, kernel_size=1, padding=0)

        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pred(x)
        x = x.reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3])
        # go from: (N, num_anchors, per_anchor, size, size)
        # to     : (N, num_anchors, size, size, per_anchor)
        return x.permute(0, 1, 3, 4, 2)



class Yolov3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_layers()

    def forward(self, x):
        outputs = []
        skips = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Prediction):
                outputs.append(layer(x))
                continue

            print(f"In {i}th layer")
            x = layer(x)

            if isinstance(layer, ResBlock) and layer.num_repeats == 8:
                skips.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, skips.pop()], dim=1)

        return outputs

    def _create_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in model_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding = 1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels
            elif isinstance(module, int):
                num_repeats = module
                layers.append(
                    ResBlock(in_channels, num_repeats=num_repeats)
                )
            elif isinstance(module, str):
                if module == "P":
                    layers += [
                        ResBlock(in_channels, use_res=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        Prediction(in_channels // 2, num_classes=self.num_classes)
                        ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(
                        scale_factor=2, mode="bilinear"
                    ))
                    in_channels = in_channels * 3
        return layers


def main():
    num_classes = 20
    IMAGE_SIZE = 416
    model = Yolov3(num_classes=num_classes)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE // 8 , IMAGE_SIZE // 8 , num_classes + 5)
    print("worked")


if __name__ == "__main__":
    main()
