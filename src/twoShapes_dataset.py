"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from torch.utils.data import Dataset
from src.data_augmentation import *
import pickle
import copy
import pandas as pd
from synthetic.Annotations import Annotations

class TwoShapesDataset(Dataset):
    def __init__(self, root_path="data/TwoShapes", mode="train", trainingSet = "test1", image_size=224, is_training=True):
        if mode in ["train", "val"]:
            self.image_path = os.path.join(root_path, "images", "{}".format(mode))
            anno_path = os.path.join(root_path, "annotations", "{}_{}.pkl".format(trainingSet, mode))

            self.anno = Annotations(anno_path)


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
        image_path = os.path.join(self.image_path, self.trainingSet + "_{:05d}.png".format(item))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # for idx in range(len(objects)):
        #     objects[idx][4] = self.class_ids.index(objects[idx][4])
        # if self.is_training:
        #     transformations = Compose([Resize(self.image_size)])
        # else:
        #     transformations = Compose([Resize(self.image_size)])
        # image, objects = transformations((image, objects))
        x = self.anno[item]
        objects = np.vstack([x["xc"], x["yc"], x["bx"], x["by"], x["catID"]]).T
        objects[:,0:4] = objects[:,0:4] * 448
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
    
