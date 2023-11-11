import torchvision
import timm
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class FastVit512(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast_vit = timm.create_model('fastvit_s12.apple_in1k', pretrained=True,num_classes=0, global_pool="" )
        self.out_channels = self.fast_vit.feature_info[-1]['num_chs']
        self.conv = nn.Conv2d(self.out_channels, 512, kernel_size=1)

    def forward(self, x):
        x = self.fast_vit(x)
        x = self.conv(x)
        return x


def create_model(num_classes):
    fast_vit = FastVit512()
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    model = FasterRCNN(backbone=fast_vit,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    print(model)
    return model

if __name__=='__main__':
    model = create_model(20)
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    # print(model.forward(x))
