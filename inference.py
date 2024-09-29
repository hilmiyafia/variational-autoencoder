
import torch
import numpy
import torchvision
import matplotlib.pyplot as pyplot
from dataset import Dataset
from train import VariationalAutoencoder


ROWS = 8
COLS = 8
BATCH = 16
PATH = "lightning_logs/version_0/checkpoints/epoch=999-step=136000.ckpt"


if __name__ == "__main__":
    model = VariationalAutoencoder.load_from_checkpoint(PATH).eval()
    dataset = Dataset("faces")
    with torch.no_grad():
        codes = []
        for i in range(0, len(dataset), BATCH):
            j = min(i+BATCH, len(dataset))
            batch = torch.stack([dataset[k] for k in range(i, j)])
            batch = batch.type_as(model.autoencoder.encoder[0].layers[0].weight)
            codes.append(model.autoencoder.encoder(batch).cpu()[:, :256])
        codes = torch.concat(codes)
        mean = codes.mean(0, keepdim=True)
        std = codes.std(0, keepdim=True)
        random = mean + std*torch.randn(ROWS*COLS, 256)
        random = random.type_as(model.autoencoder.decoder[-1].layers[0].weight)
        random = model.autoencoder.decoder(random) * 255
        random = torchvision.utils.make_grid(random, nrow=ROWS, padding=0)
        random = random.clamp(min=0, max=255).to(torch.uint8).cpu()
        torchvision.io.write_jpeg(random, "result.jpg")
