import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetFlowInterp(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=[32, 64, 128]):
        super(UNetFlowInterp, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feature

        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        for i, up in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip = skips[-(i+1)]
            x = torch.cat((x, skip), dim=1)
            x = up(x)

        return self.final_conv(x)
