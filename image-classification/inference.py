import time

import numpy as np
from model import FastViTClassifier
from util import load_img

CLASSES = [
    "Bicycle",
    "Bridge",
    "Bus",
    "Car",
    "Chimney",
    "Crosswalk",
    "Hydrant",
    "Motorcycle",
    "Other",
    "Palm",
    "Stair",
    "TrafficLight",
]


if __name__ == "__main__":
    img_path = "test_img/Bicycle_16.png"
    checkpoint_path = "checkpoint/epoch=9-step=980.ckpt"
    img = load_img(img_path)
    # Load model
    model = FastViTClassifier(12)
    model = model.load_from_checkpoint(
        checkpoint_path, num_classes=12, map_location="cpu"
    )
    # model.load_state_dict(checkpoint['state_dict'])
    # Predict
    img_path = "test_img/Bicycle_16.png"
    model.eval()
    while True:
        img_path = input("Input image path: ")
        img = load_img(img_path)
        start = time.time()
        probs = model(img)
        # outs = model.predict_step(img, 0)
        predict_idx = np.argmax(probs.detach().numpy()[0])
        print(probs)
        print(CLASSES[predict_idx])
