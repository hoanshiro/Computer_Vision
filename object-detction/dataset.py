import os
import pytorch_lightning as pl
from utils import PascalVOCParser
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
          'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class PascalVOCDataset(Dataset):
    def __init__(self, data_dir, image_set='train', transform=None):
        self.data_dir = data_dir
        self.image_set = image_set
        self.transform = transform
        self.parser = PascalVOCParser(img_size=(224, 224))
        self.xml_files = self._load_xml_files()

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_file = self.xml_files[idx]
        data = self.parser.parse(xml_file)

        image = data['image']
        annotations = data['annotations'][0]

        if self.transform:
            image = self.transform(image)
        # annotations['labels'] = [CLASSES.index(label) for label in annotations['label']]
        annotations['label'] = CLASSES.index(annotations['label'])

        return image, annotations

    def _load_xml_files(self):
        image_set_file = os.path.join(self.data_dir, f'ImageSets/Main/{self.image_set}.txt')
        self.annotations_dir = os.path.join(self.data_dir, f'Annotations/')
        with open(image_set_file, 'r') as f:
            image_ids = [line.strip() for line in f]
        xml_files = [os.path.join(self.annotations_dir, f'{img_id}.xml') for img_id in image_ids]
        return xml_files
    

class PascalVOC(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = PascalVOCDataset(data_dir=self.data_dir, image_set='train', transform=transform)
            self.val_set = PascalVOCDataset(data_dir=self.data_dir, image_set='val', transform=transform)
        if stage == 'test' or stage is None:
            self.test_set = PascalVOCDataset(data_dir=self.data_dir, image_set='val', transform=transform)
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    

if __name__=='__main__':
# Example usage:
    data_dir = '/home/hoan/Desktop/Computer_Vision/object-detction/VOC2012/'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    PascalVOC_dataset = PascalVOC(data_dir, batch_size=2)
    PascalVOC_dataset.prepare_data()
    PascalVOC_dataset.setup()
    test = PascalVOC_dataset.train_dataloader()
    # print(PascalVOC_dataset.classes)
    print(next(iter(test)))
