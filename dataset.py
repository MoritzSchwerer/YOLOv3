#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
from iou import iou
from nms import nms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YoloDataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,
            label_dir,
            anchors,
            S=[13,26,52],
            num_classes=20,
            transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.iou_threshold = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        image_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # shape: [(3, 13, 13, 6),
        #         (3, 26, 26, 6
        #         (3, 52, 52, 6))]
        # shape of the target predictions
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]


        for box in bboxes:
            # 1 boxes * 9 anchors = 9
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchors_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale # 0, 1, 2
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # 0, 1, 2
                scale = self.S[scale_idx]
                i, j = int(scale*y), int(scale*x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # if anchor has not been taken and this scale has no anchor yet
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_index][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = scale * x - j, scale * y - i
                    width_cell, height_cell = width * scale, height * scale
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)
