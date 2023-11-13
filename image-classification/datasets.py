import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from util import transform
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SmallNoiseImageAugmentation:
    def __init__(self):
        # Define a combination of augmentations
        self.transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MotionBlur(blur_limit=(3, 7), p=0.5),
                        A.MedianBlur(blur_limit=3, p=0.5),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        A.RandomGamma(gamma_limit=(0.8, 1.2), p=0.5),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                    ],
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
                ),
                A.OneOf(
                    [
                        A.RandomRain(p=0.5),
                        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=0.5),
                    ],
                    p=0.5,
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        # convert PIL image to numpy array
        image = np.array(image)
        augmented = self.transform(image=image)
        return augmented['image']


class GroceryDataset(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="/home/hoan/Desktop/pp_hermes/paddleclas/dataset/ILSVRC2012/",
        batch_size=8,
        num_workers=2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.augmentation = SmallNoiseImageAugmentation()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_data(self):
        self.train_dir = os.path.join(self.data_dir, "train/")
        self.test_dir = os.path.join(self.data_dir, "val/")
        self.val_dir = os.path.join(self.data_dir, "val/")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = ImageFolder(
                root=self.train_dir, transform=self.augmentation
            )
            self.val_set = ImageFolder(root=self.val_dir, transform=self.augmentation)
            self.val_set.class_to_idx = self.train_set.class_to_idx.copy()
            self.classes = self.train_set.classes
            # class_weights = torch.unique(
            #     torch.tensor(self.train_set.targets), return_counts=True
            # )[1].float()
            # self.per_class_weights = class_weights / class_weights.sum()
            # self.per_class_weights = 1.0 / (
            #     self.per_class_weights * len(self.per_class_weights)
            # )

        if stage == "test" or stage is None:
            self.test_set = ImageFolder(root=self.test_dir, transform=self.augmentation)
            self.test_set.class_to_idx = self.train_set.class_to_idx.copy()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )


if __name__ == "__main__":
    GroceryData = GroceryDataset()
    GroceryData.prepare_data()
    GroceryData.setup()
    test = GroceryData.train_dataloader()
    print(GroceryData.classes)
    print(next(iter(test)))
