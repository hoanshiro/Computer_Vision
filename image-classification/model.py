from typing import Any
import torch
import timm
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score
import torch.optim as optim

class FastViTClassifier(pl.LightningModule):
    def __init__( self, num_classes, lr: float = 0.0002):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes)
        self.fast_vit = timm.create_model('fastvit_s12.apple_in1k',
                                          pretrained=True,
                                          num_classes=self.num_classes)
        
    def forward(self, x):
        x = self.fast_vit(x)
        return x
    
    def training_step(self, batch):
        imgs, labels = batch
        outs = self.forward(imgs)
        # print(outs, labels)
        loss = F.cross_entropy(outs, labels)
        self.log_dict({'train_loss': loss,
                       "train_accuracy": self.accuracy(outs, labels),
                       "train_f1_score": self.f1_score(outs, labels)})
        return loss
    
    def validation_step(self, batch, idx):
        imgs, labels = batch
        outs = self.forward(imgs)
        loss = F.cross_entropy(outs, labels)
        self.log_dict({'val_loss': loss,
                       "val_accuracy": self.accuracy(outs, labels),
                       "val_f1_score": self.f1_score(outs, labels)})
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.fast_vit.parameters(), self.lr)
    
    def predict_step(self, batch: Any, batch_idx: int,) :
        self.fast_vit.eval()
        return self.fast_vit(batch)

if __name__ == "__main__":
    model = FastViTClassifier(4)
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)
