"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from src.utils import *
from src.yolo_net import Yolo

CLASSES = ["circle", "rectangle"]


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.7)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/model_synth1")
    parser.add_argument("--input", type=str, default="test_images/boiler")
    parser.add_argument("--output", type=str, default="test_images/boiler")

    args = parser.parse_args()
    return args

class Yolo():

    def __init__(opt = get_args()):
        if torch.cuda.is_available():
            if opt.pre_trained_model_type == "model":
                model = torch.load(opt.pre_trained_model_path)
            else:
                model = Yolo(80)
                model.load_state_dict(torch.load(opt.pre_trained_model_path))
        else:
            if opt.pre_trained_model_type == "model":
                model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
            else:
                model = Yolo(80)
                model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))
        model.eval()
        self.model = model
        self.opt = opt

    def __call__(self, image):
            opt = self.opt
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
            image = image[None, :, :, :]
            data = Variable(torch.FloatTensor(image))
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                logits = model(data)
                predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                              opt.nms_threshold)
            return predictions



if __name__ == "__main__":
    opt = get_args()

