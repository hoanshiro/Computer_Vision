from util import load_img
import numpy as np
from model import FastViTClassifier


CLASSES = ['Apple', 'Avocado', 'Banana', 'Kiwi', 'Lemon', 'Lime', 'Mango',
            'Melon', 'Nectarine', 'Orange', 'Papaya', 'Passion-Fruit', 'Peach',
            'Pear', 'Pineapple', 'Plum', 'Pomegranate', 'Red-Grapefruit', 'Satsumas']


if __name__ == "__main__":
    img_path = 'GroceryStoreDataset/dataset/train/Fruit/Kiwi/Kiwi_001.jpg'
    checkpoint_path = 'lightning_logs/version_1/checkpoints/epoch=9-step=1430.ckpt'
    img = load_img(img_path)
    # Load model
    model = FastViTClassifier(20)
    model = model.load_from_checkpoint(checkpoint_path, num_classes=20)
    # checkpoint = torch.load('lightning_logs/version_1/checkpoints/epoch=9-step=1430.ckpt')
    # model.load_state_dict(checkpoint['state_dict'])
    # Predict
    outs = model.predict_step(img, 0)
    predict_idx = np.argmax(outs.detach().numpy()[0])
    print(CLASSES[predict_idx])
