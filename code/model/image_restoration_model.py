"""
File: image_restoration_model.py
Description: Image Restoration Model (U-NET with 1 Encoder and 2 Decoders)
"""
from typing import Tuple
import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Maybe add a second conv? not sure
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out += identity 
        out = self.silu(out)
        return out
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.res1 = ResnetBlock(out_channels, out_channels)
        self.res2 = ResnetBlock(out_channels, out_channels)
        self.res3 = ResnetBlock(out_channels, out_channels)
        self.mpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.silu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        skip = out
        out = self.mpool(out)
        return skip, out
    
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

    def forward(self, x):
        skip1, out = self.enc1(x)
        skip2, out = self.enc2(out)
        skip3, out = self.enc3(out)
        skip4, out = self.enc4(out)
        return [skip1, skip2, skip3, skip4], out

# last encoder block, but without maxpool
class Center(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.res1 = ResnetBlock(out_channels, out_channels)
        self.res2 = ResnetBlock(out_channels, out_channels)
        self.res3 = ResnetBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.silu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.res1 = ResnetBlock(out_channels, out_channels)
        self.res2 = ResnetBlock(out_channels, out_channels)
        self.res3 = ResnetBlock(out_channels, out_channels)

    def forward(self, x, skip):
        # upsample by interpolating
        out = nn.functional.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = torch.cat((out, skip), dim=1)
        out = self.conv2(out)
        out = self.silu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return out


class DecoderImage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, 32)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=1)
        self.act = nn.Tanh()

    def forward(self, x, skips):
        out = self.dec1(x, skips[3])
        out = self.dec2(out, skips[2])
        out = self.dec3(out, skips[1])
        out = self.dec4(out, skips[0])
        out = self.conv1(out)
        out = self.act(out)
        return out
    

class DecoderMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, 32)
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x, skips):
        out = self.dec1(x, skips[3])
        out = self.dec2(out, skips[2])
        out = self.dec3(out, skips[1])
        out = self.dec4(out, skips[0])
        out = self.conv1(out)
        out = self.act(out)
        return out



class ImageRestorationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = Encoder()
        self.center = Center(256, 512)
        self.dec_img = DecoderImage()
        self.dec_mask = DecoderMask()

    def forward(
            self, corrupted_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Image Restoration Model.

        Given a `corrupted_image` with shape (B, C, H, W) where B = batch size, C = # channels,
        H = image height, W = image width and normalized values between -1 and 1,
        run the Image Restoration Model forward and return a tuple of two tensors:
        (`predicted_image`, `predicted_binary_mask`).

        The `predicted_image` should be the output of the Image Decoder (B, C, H, W). In the
        assignment this is referred to as x^{hat}. This is NOT the `reconstructed_image`,
        referred to as `x_{reconstructed}` in the assignment handout.

        The `predicted_binary_mask` should be the output of the Binary Mask Decoder (B, 1, H, W). This
        is `m^{hat}` in the assignment handout.
        """

        skips, encoding = self.enc(corrupted_image)
        encoding = self.center(encoding)
        out_img = self.dec_img(encoding, skips)
        out_mask = self.dec_mask(encoding, skips)

        return out_img, out_mask

