
import torch
import numpy
import torchvision
import matplotlib.pyplot as pyplot
from dataset import Dataset
from model import Model


ROWS = 8
COLS = 8
BATCH = 16
DIM = 16

if __name__ == "__main__":
    model = Model().eval().cuda()
    model.load_state_dict(torch.load("model.pt", weights_only=False))
    dataset = Dataset("faces")
    with torch.no_grad():
        codes = []
        for i in range(0, len(dataset), BATCH):
            j = min(i + BATCH, len(dataset))
            batch = torch.stack([dataset[k] for k in range(i, j)]).cuda()
            codes.append(model.encoder(batch)[:, :DIM])
        codes = torch.concat(codes)
        mean = codes.mean(0, keepdim=True)
        std = codes.std(0, keepdim=True)
        random = mean + std * torch.randn(ROWS * COLS, DIM).cuda()
        random = model.decoder(random) * 255
        random = torchvision.utils.make_grid(random, nrow=ROWS, padding=0)
        random = random.clamp(min=0, max=255).to(torch.uint8).cpu()
        torchvision.io.write_jpeg(random, "result.jpg")
