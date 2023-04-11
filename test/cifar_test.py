from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models

from data.cifar_dm import CIFAR10DataModule


class CIFAR10LitModule(LightningModule):
    def __init__(self, num_classes=10, lr=0.001):
        super(CIFAR10LitModule, self).__init__()
        self.save_hyperparameters()

        self.model = models.swin_v2_t(num_classes=num_classes)

        self.loss_fn = CrossEntropyLoss()

        self.acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch

        pred = self(imgs)
        loss = self.loss_fn(pred, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        pred = self(imgs)
        loss = self.loss_fn(pred, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        imgs, labels = batch

        pred = self(imgs)
        loss = self.loss_fn(pred, labels)
        self.acc(pred, labels)
        self.log('test_loss', loss)
        self.log('test_acc', self.acc, on_step=False, on_epoch=True)
        return loss


if __name__ == '__main__':
    # args = parse_args()

    cifar10_dm = CIFAR10DataModule(batch_size=256)
    module = CIFAR10LitModule()
    print('module loaded')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=3,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
    )

    logger = TensorBoardLogger('../logs', name='cifar10')

    trainer = Trainer(
        max_epochs=3,
        precision='16-mixed',
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        fast_dev_run=True
    )

    trainer.fit(module, cifar10_dm)
    trainer.test(module, cifar10_dm)
