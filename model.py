
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
            torch.nn.Conv2d(in_count, 2 * in_count, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(2 * in_count, out_count, 1),
            torch.nn.Conv2d(
                out_count, out_count, 3, groups=out_count,
                padding=1, padding_mode="reflect"),
            torch.nn.InstanceNorm2d(out_count),
            torch.nn.Conv2d(out_count, 2 * out_count, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(2 * out_count, out_count, 1))
        if scale > 0:
            self.scale = torch.nn.UpsamplingNearest2d(scale_factor=scale)
        elif scale < 0:
            self.scale = torch.nn.MaxPool2d(-scale)
        else:
            self.scale = torch.nn.Identity()
        self.skip = torch.nn.Conv2d(in_count, out_count, 1)

    def forward(self, input):
        return self.scale(self.layers(input) + self.skip(input))


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 1),
            Block(16, 32, -2),
            Block(32, 32),
            Block(32, 64, -2),
            Block(64, 64),
            Block(64, 128, -2),
            Block(128, 128),
            Block(128, 256, -2),
            Block(256, 256),
            torch.nn.Conv2d(256, 256, 4, groups=256),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 32))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.Unflatten(1, (256, 1, 1)),
            torch.nn.ConvTranspose2d(256, 256, 4, groups=256),
            Block(256, 256, 2),
            Block(256, 128),
            Block(128, 128, 2),
            Block(128, 64),
            Block(64, 64, 2),
            Block(64, 32),
            Block(32, 32, 2),
            Block(32, 16),
            torch.nn.Conv2d(16, 3, 1))
        
    def forward(self, input):
        mean, log_var = self.encoder(input).chunk(2, 1)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
        return self.decoder(latent), mean, log_var

