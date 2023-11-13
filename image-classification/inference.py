import time
import numpy as np
import torch
from model import FastViTClassifier
from util import load_img, split_9_sub_img
from config import CLASSES, per_class_weights

if __name__ == "__main__":
    img_path = "test_img/Bicycle_16.png"
    checkpoint_path = "checkpoint/epoch=49-step=4900.ckpt"
    img = load_img(img_path)
    # Load model
    print(per_class_weights)
    model = FastViTClassifier(num_classes=12, class_weights=per_class_weights)
    model = model.load_from_checkpoint(
        checkpoint_path,
        num_classes=12,
        class_weights=per_class_weights,
        map_location="cpu",
    )
    # model.load_state_dict(checkpoint['state_dict'])
    # Predict
    img_path = "test_img/Bicycle_16.png"
    model.eval()
    # while True:
    #     img_path = input("Input image path: ")
    source_imgs = (
        "/home/hoan/Desktop/pp_hermes/paddleclas/dataset/ILSVRC2012/val/other/"
    )
    captcha_img = "test_img/moto.png"
    # for img_path in os.listdir(source_imgs):
    #     img = load_img(source_imgs+img_path)
    total_time = 0
    for img in split_9_sub_img(captcha_img):
        start = time.time()
        out = model(img)
        # outs = model.predict_step(img, 0)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        probs = probs.detach().numpy()
        # round 2 digit after decimal
        probs = np.round(probs, decimals=2)
        predict_idx = np.argmax(probs)
        total_time += time.time() - start
        # print(probs)
        print(
            f"Predict: {CLASSES[predict_idx]}, prob: {probs} time: {time.time()-start}"
        )
    print(f"Total Time: {total_time}")
