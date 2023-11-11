import pytorch_lightning as pl
from datasets import GroceryDataset
from model import FastViTClassifier
from pytorch_lightning import loggers as pl_loggers

if __name__ == "__main__":
    GroceryData = GroceryDataset(batch_size=4)
    per_class_weights = [
        1.2489,
        1.8869,
        0.8007,
        0.2735,
        7.6773,
        0.7991,
        1.0250,
        12.4299,
        0.7284,
        1.0876,
        4.3993,
        1.2391,
    ]
    model = FastViTClassifier(12, class_weights=per_class_weights)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/")
    trainer = pl.Trainer(
        max_epochs=10, log_every_n_steps=5, logger=tb_logger, accelerator="cpu"
    )  # , devices=1)
    trainer.fit(model, GroceryData)
