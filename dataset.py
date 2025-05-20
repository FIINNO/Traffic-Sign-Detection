import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from letterbox import letterbox

class YoloMapillaryDataset(Dataset):
    def __init__(self, images_path, labels_path, img_size=640, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".png"))]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_path, self.image_files[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load labels
        labels = []
        label_path = os.path.join(self.labels_path, os.path.splitext(self.image_files[index])[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        labels.append(parts)

        labels = np.array(labels, dtype=np.float32)

        if len(labels) > 0:
            classes = labels[:, 0]
            boxes = labels[:, 1:]
        else:
            classes = np.array([], dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.float32)

        img_padded, boxes = letterbox(img, boxes, new_shape=(self.img_size, self.img_size), normalized=True)

        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        assert img_tensor.shape == (3, 640, 640), f"Got {img_tensor.shape}"
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        classes_tensor = torch.tensor(classes, dtype=torch.int64)


        batch_idx_tensor = torch.full((len(boxes_tensor),), fill_value=0, dtype=torch.int64)


        return {
          'img': img_tensor,
          'bboxes': boxes_tensor,
          'cls': classes_tensor,
          'batch_idx': batch_idx_tensor
        }