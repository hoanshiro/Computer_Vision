import pytorch_lightning as pl
from datasets import GroceryDataset
from config import per_class_weights
from model import FastViTClassifier
from pytorch_lightning import loggers as pl_loggers

if __name__ == "__main__":
    GroceryData = GroceryDataset(batch_size=16)
    model = FastViTClassifier(12, class_weights=per_class_weights)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/")
    trainer = pl.Trainer(
        max_epochs=10, log_every_n_steps=5, logger=tb_logger, accelerator="cpu"
    )  # , devices=1)
    trainer.fit(model, GroceryData)
