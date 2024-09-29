
import torch
import torchvision
from train import AdversarialAutoencoder


ROWS = 4
COLS = 4
PATH = "lightning_logs/version_0/checkpoints/epoch=63-step=17408.ckpt"


if __name__ == "__main__":
    model = AdversarialAutoencoder.load_from_checkpoint(PATH).eval()
    with torch.no_grad():
        random = torch.randn(ROWS*COLS, 256)
        random = random.type_as(model.autoencoder.decoder[-1].layers[0].weight)
        random = model.autoencoder.decoder(random) * 255
        random = torchvision.utils.make_grid(random, nrow=ROWS, padding=0)
        random = random.clamp(min=0, max=255).to(torch.uint8).cpu()
        torchvision.io.write_jpeg(random, "result.jpg")

