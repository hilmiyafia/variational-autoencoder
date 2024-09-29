
import torch
from torchinfo import summary


class Block(torch.nn.Module):

    def __init__(self, in_count, out_count, scale=0):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_count, in_count, 3, groups=in_count,
                padding=1, padding_mode="reflect"),
            torch.nn.InstanceNorm2d(in_count),
            torch.nn.Conv2d(in_count, 4*in_count, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4*in_count, out_count, 1))
        self.skip = torch.nn.Conv2d(in_count, out_count, 1)
        if scale > 0:
            self.scale = torch.nn.UpsamplingNearest2d(scale_factor=2)
        elif scale < 0:
            self.scale = torch.nn.MaxPool2d(2)
        else:
            self.scale = torch.nn.Identity()

    def forward(self, input):
        return self.scale(self.layers(input) + self.skip(input))


class Autoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            Block(3, 32, -1),
            Block(32, 32),
            Block(32, 64, -1),
            Block(64, 64),
            Block(64, 128, -1),
            Block(128, 128),
            Block(128, 256, -1),
            Block(256, 256),
            torch.nn.Conv2d(256, 256, 4, groups=256),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 512))
        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (256, 1, 1)),
            torch.nn.ConvTranspose2d(256, 256, 4, groups=256),
            Block(256, 256, 1),
            Block(256, 128),
            Block(128, 128, 1),
            Block(128, 64),
            Block(64, 64, 1),
            Block(64, 32),
            Block(32, 32, 1),
            Block(32, 3))

    def forward(self, input):
        mean, logvar = self.encoder(input).chunk(2, 1)
        buffer = mean + torch.randn_like(mean)*torch.exp(logvar/2)
        return self.decoder(buffer), mean, logvar

