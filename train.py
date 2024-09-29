
import torch
import lightning
import torchvision
from dataset import Dataset
from model import Autoencoder


class VariationalAutoencoder(lightning.LightningModule):

    def __init__(self, lr=2e-4, beta=2e-3, validation=None):
        super().__init__()
        self.save_hyperparameters("lr", "beta")
        self.autoencoder = Autoencoder().train()
        if validation is None: self.validation = None
        else: self.validation = torch.stack([sample for sample in validation])

    def training_step(self, batch, batch_index):
        output, mean, logvar = self.autoencoder(batch)
        loss_reconstruction = (output-batch).pow(2).sum() / batch.shape[0]
        loss_kl = -0.5 * (1+logvar-mean.pow(2)-logvar.exp()).sum()
        self.log("loss/reconstruction", loss_reconstruction)
        self.log("loss/kl", loss_kl)
        return loss_reconstruction + self.hparams.beta*loss_kl
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.autoencoder.parameters(), lr=self.hparams.lr)

    def on_train_epoch_end(self):
        if self.validation is None: return
        batch = self.validation.type_as(self.autoencoder.encoder[0].layers[0].weight)
        grid = torchvision.utils.make_grid(self.autoencoder(batch)[0], padding=0)
        self.logger.experiment.add_image("reconstruction", grid, self.current_epoch)


if __name__ == "__main__":
    dataset = Dataset("faces")
    train, validation = torch.utils.data.random_split(dataset, [len(dataset) - 4, 4])
    dataloader = torch.utils.data.DataLoader(train, 16, shuffle=True,
                                             num_workers=2, persistent_workers=True)
    model = VariationalAutoencoder(validation=validation)
    trainer = lightning.Trainer(max_epochs=1000)
    trainer.fit(model, dataloader)
        
