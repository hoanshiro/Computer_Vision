import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def load_img(img_path):
    img = Image.open(img_path)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img