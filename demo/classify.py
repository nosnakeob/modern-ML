import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from transformers import AutoModel

from data.dm_interface import BuildInDataModuleI
from models.classify_model import *


# 转成3通道
class CatImg:
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.cat([img, img, img], dim=0)


class FashionMNISTDataModule(BuildInDataModuleI):
    def __init__(self, data_dir: str = "../data", batch_size: int = 32, num_workers: int = 6):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize((0.1307,), (0.3081,)),
            CatImg()
        ])

        super().__init__(FashionMNIST, data_dir, transform, transform, batch_size, num_workers)


class ClassifyLitModule(LightningModule):
    def __init__(self, num_classes=10, lr=0.001):
        super(ClassifyLitModule, self).__init__()
        self.save_hyperparameters()

        # self.model = models.resnet18(num_classes=num_classes)
        # self.model = models.swin_v2_t(num_classes=num_classes)
        # self.model = ConvNet(num_classes=num_classes)
        self.backbone = AutoModel.from_pretrained('facebook/vit-mae-base')
        self.backbone.trainable = False
        self.pool = partial(torch.mean, dim=1)
        self.fc = nn.Linear(768, num_classes)

        self.loss_fn = CrossEntropyLoss()

        self.acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)['last_hidden_state']
        x = self.pool(x)
        x = self.fc(x)
        return x

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


def train(module, mnist_dm):
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

    logger = TensorBoardLogger('../logs', name='classify')

    logger.log_hyperparams(module.hparams)

    trainer = Trainer(
        # default_root_dir='..',
        max_epochs=3,
        precision='16-mixed',
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        val_check_interval=0
        # fast_dev_run=True
    )

    # tuner = Tuner(trainer)
    # tuner.lr_find(module, datamodule=mnist_dm)
    # tuner.scale_batch_size(module, datamodule=mnist_dm)

    trainer.fit(module, mnist_dm)
    trainer.test(module, mnist_dm)


# def infer(module):
#     module

if __name__ == '__main__':
    mnist_dm = FashionMNISTDataModule(batch_size=128)
    module = ClassifyLitModule()
    print('module loaded')

    train(module, mnist_dm)

    # module.load_from_checkpoint()
