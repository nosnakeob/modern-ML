import torch
import torchvision.models
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner import Tuner
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy

from torchvision import models

from data.mnist_dm import MNISTDataModule
from models.mnist_model import TransNet, ConvNet

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


class MNISTLitModule(LightningModule):
    def __init__(self, num_classes=10, lr=0.001):
        super(MNISTLitModule, self).__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(num_classes=num_classes)

        self.loss_fn = CrossEntropyLoss()

        self.acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        # print(imgs.shape)

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

    mnist_dm = MNISTDataModule(batch_size=256)
    module = MNISTLitModule()
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

    logger = TensorBoardLogger('../logs', name='mnist')

    logger.log_hyperparams(module.hparams)

    trainer = Trainer(
        # default_root_dir='..',
        max_epochs=3,
        precision='16-mixed',
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        fast_dev_run=True
    )

    # tuner = Tuner(trainer)
    #
    # tuner.lr_find(module, datamodule=mnist_dm)
    # tuner.scale_batch_size(module, datamodule=mnist_dm)

    trainer.fit(module, mnist_dm)
    trainer.test(module, mnist_dm)
