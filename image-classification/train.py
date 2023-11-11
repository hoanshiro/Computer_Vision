from datasets import GroceryDataset
from model import FastViTClassifier
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

if __name__=="__main__":
    GroceryData = GroceryDataset()
    model = FastViTClassifier(12)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/")
    trainer = pl.Trainer(max_epochs=10, log_every_n_steps=5)
    trainer.fit(model, GroceryData)