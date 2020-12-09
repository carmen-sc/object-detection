import os
import numpy as np
import torch


class Dataset(object):
    def __init__(self, root, transforms):
        self.root = "/home/carmen/catkin_ws/src/AMOD/object-detection"
        self.transforms = transforms
        # load all npz files
        self.files = list(sorted(os.listdir(os.path.join(self.root, "data_collection/dataset"))))

    def __getitem__(self, idx):
        # load images, boxes and classes
        file_path = os.path.join(self.root, "data_collection/dataset", self.files[idx])

        data = np.load(file_path)
        img = data[f"arr_{0}"]
        boxes = []
        for box in data[f"arr_{1}"]:
            boxes.append(box)
        print(boxes)
        classes = data[f"arr_{2}"]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # define target
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.files)