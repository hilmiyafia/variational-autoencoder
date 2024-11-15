
import torch
import lightning
import torchvision
from dataset import Dataset
from model import Model


class VariationalAutoencoder(lightning.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.noises = torch.randn(16, 16)
        self.data = []

    def training_step(self, batch, batch_index):
        output, mean, log_var = self.model(batch)
        reconstruction = (output - batch).pow(2).sum((1, 2, 3)).mean()
        latent = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(1).mean()
        laplacian = (self.laplacian(output) - self.laplacian(batch)).pow(2).sum((1, 2, 3)).mean()
        self.log("loss/reconstruction", reconstruction)
        self.log("loss/latent", latent)
        self.log("loss/laplacian", laplacian)
        self.data.append(mean.detach())
        return reconstruction + latent + laplacian

    def laplacian(self, input):
        padded = torch.nn.functional.pad(input, (2, 2, 2, 2), mode="reflect")
        average = torch.nn.functional.avg_pool2d(padded, 5, 1)
        return input - average
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-4)

    def on_train_epoch_end(self):
        with torch.no_grad():
            data = torch.concat(self.data)
            mean = data.mean(0, keepdim=True)
            std = data.std(0, keepdim=True)
            self.data.clear()
            device = self.model.decoder[-1].weight.device
            noises = self.noises.to(device) * std + mean
            samples = self.model.decoder(noises)
            grid = torchvision.utils.make_grid(samples, nrow=4, padding=0)
            self.logger.experiment.add_image("samples", grid, self.current_epoch)
            torch.save(self.model.state_dict(), "model.pt")


if __name__ == "__main__":
    dataset = Dataset("faces")
    dataloader = torch.utils.data.DataLoader(dataset, 16, shuffle=True,
                                             num_workers=2, persistent_workers=True)
    model = Model()
    # model.load_state_dict(torch.load("model.pt", weights_only=False))
    autoencoder = VariationalAutoencoder(model)
    trainer = lightning.Trainer(max_epochs=1000)
    trainer.fit(autoencoder, dataloader)

