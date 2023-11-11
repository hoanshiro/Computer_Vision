import os
import xml.etree.ElementTree as ET
from PIL import Image

class PascalVOCParser:
    def __init__(self, base_dir='VOC2012/', img_size=(224, 224)):
        self.base_dir = base_dir
        self.img_size = img_size

    def parse(self, xml_file):
        # Parse XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find image path
        image_name = root.find('filename').text
        img_path = f'{self.base_dir}/JPEGImages/{image_name}'
        image = Image.open(img_path)

        annotations = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text) * (self.img_size[0] / image.size[0]))
            ymin = int(float(bbox.find('ymin').text) * (self.img_size[1] / image.size[1]))
            xmax = int(float(bbox.find('xmax').text) * (self.img_size[0] / image.size[0]))
            ymax = int(float(bbox.find('ymax').text) * (self.img_size[1] / image.size[1]))
            # Store annotation information
            annotation = {
                'label': label,
                'bbox': [xmin, ymin, xmax, ymax]
            }
            
            annotations.append(annotation)

        return {
            'image': image,
            'annotations': annotations
        }

if __name__=='__main__':
# Example usage:
    xml_file_path = 'VOC2012/Annotations/2007_000027.xml'
    parser = PascalVOCParser()
    data = parser.parse(xml_file_path)
    image = data['image']
    annotations = data['annotations']
    print(image)
    print(annotations)
# Now you have the image and annotations in the desired format.
