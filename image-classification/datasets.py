import os
from util import transform
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class GroceryDataset(pl.LightningDataModule):
    def __init__(self, data_dir='/home/hoan/Desktop/pp_hermes/paddleclas/dataset/ILSVRC2012/', batch_size=8, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
    def prepare_data(self):
        self.train_dir = os.path.join(self.data_dir, 'train/')
        self.test_dir = os.path.join(self.data_dir, 'val/')
        self.val_dir = os.path.join(self.data_dir, 'val/')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = ImageFolder(root=self.train_dir, transform=self.transform)
            self.val_set = ImageFolder(root=self.val_dir, transform=self.transform)
            self.val_set.class_to_idx = self.train_set.class_to_idx.copy()
            self.classes = self.train_set.classes
        if stage == 'test' or stage is None:
            self.test_set = ImageFolder(root=self.test_dir, transform=self.transform)
            self.test_set.class_to_idx = self.train_set.class_to_idx.copy()
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
if __name__ == "__main__":
    GroceryData = GroceryDataset()
    GroceryData.prepare_data()
    GroceryData.setup()
    test = GroceryData.train_dataloader()
    print(GroceryData.classes)
    print(next(iter(test)))
