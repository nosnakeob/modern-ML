import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import random_split, DataLoader


class BuildInDataModuleI(LightningDataModule):

    def __init__(self, DATASET, data_dir: str = "../data",
                 train_transform=None, test_transform=None,
                 batch_size: int = 32, num_workers: int = 6):
        super().__init__()
        self.DATASET = DATASET
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.ds_predict = None
        self.ds_test = None
        self.ds_val = None
        self.ds_train = None

    def prepare_data(self):
        # download
        self.DATASET(self.data_dir, train=True, download=True)
        self.DATASET(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            ds_full = self.DATASET(self.data_dir, train=True, transform=self.train_transform)
            self.ds_train, self.ds_val = random_split(ds_full, [len(ds_full) - 5000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.ds_test = self.DATASET(self.data_dir, train=False, transform=self.test_transform)

        # if stage == "predict":
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)
