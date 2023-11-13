from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def convert(x):
    return transform(x).unsqueeze(0)


def load_img(img_path):
    img = Image.open(img_path).convert("RGB")
    return convert(img)


captcha_img = "test_img/moto.png"


def split_9_sub_img(captcha_img_path):
    img = Image.open(captcha_img_path).convert("RGB")
    if not img:
        raise Exception("Cannot load image")
    w, h = img.size
    sub_imgs = []
    for i in range(3):
        for j in range(3):
            sub_img = img.crop((j * w / 3, i * h / 3, (j + 1) * w / 3, (i + 1) * h / 3))
            sub_imgs.append(convert(sub_img))
    return sub_imgs
