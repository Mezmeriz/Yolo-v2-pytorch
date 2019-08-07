"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from torch.utils.data import Dataset
from src.data_augmentation import *
import pickle
import copy
import pandas as pd

from synthetic.Annotations import AnnotationsCombined as Annotations

class TwoShapesDataset(Dataset):
    def __init__(self, root_path="~/dataNeural/yolo1", mode="train", trainingSet = "annotations", image_size=448, is_training=True):
        if mode in ["train", "val"]:
            root_path = os.path.expanduser(root_path)
            self.image_path = os.path.join(root_path, "images")
            anno_path = os.path.join(root_path, "annotations", "{}.pkl".format(trainingSet))

            self.anno = Annotations(anno_path, subdir=mode)


        self.classes = ["circle", "rectangle"]
        self.class_ids = [0, 1]
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.anno)
        self.is_training = is_training
        self.trainingSet = trainingSet

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        imageFile = self.anno.getImageInfo(item)
        image = cv2.imread(imageFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image[image > 1] = 255
        # for idx in range(len(objects)):
        #     objects[idx][4] = self.class_ids.index(objects[idx][4])
        # if self.is_training:
        #     transformations = Compose([Resize(self.image_size)])
        # else:
        #     transformations = Compose([Resize(self.image_size)])
        # image, objects = transformations((image, objects))
        x = self.anno[item]
        # TODO go through the code and correctly adjust bx and by
        # objects = np.vstack([x["xc"], x["yc"], x["bx"], x["by"], x["catID"]]).T
        objects = self.anno.getBBoxes(item)
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
    

if __name__ == '__main__':
    c = TwoShapesDataset()
    x = c[1]