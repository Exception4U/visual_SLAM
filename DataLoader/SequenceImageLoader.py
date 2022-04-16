import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging
from utils.PinholeCamera import PinholeCamera
import os


class SequenceImageLoader(object):
    default_config = {
        "root_path": "/Users/tushar.vaidya/datasets/kitti/hiring-assignment-lt/",
        "img_folder": "images",
        "ground_truth_file": "ground_truth.txt",
        "start": 000000,
        "format": "png"
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Sequence image loader config: ")
        logging.info(self.config)

        self.cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)

        self.img_id = self.config["start"]
        self.img_N = len(glob.glob(pathname=self.config["root_path"] + \
                                            self.config["img_folder"]+\
                                             "/*." + \
                                             self.config["format"]))

        self.pose_path = self.config["root_path"] + \
                            self.config["ground_truth_file"]
        self.gt_poses = []
        with open(self.pose_path) as f:
            lines = f.readlines()
            for line in lines:
                ss = line.strip().split()
                pose = np.zeros((1, len(ss)))
                for i in range(len(ss)):
                    pose[0, i] = float(ss[i])

                pose.resize([3, 4])
                self.gt_poses.append(pose)

    def get_size(self):
        return self.img_N

    def get_cur_pose(self):
        return self.gt_poses[self.img_id - 1]

    def __getitem__(self, item):
        file_name = self.config["root_path"] +"/" + self.config["img_folder"] +"/" \
                    + str(item).zfill(6) + ".png"
        img = cv2.imread(file_name)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < self.img_N:
            img = self.__getitem__(self.img_id)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]
