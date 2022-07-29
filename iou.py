#!/usr/bin/env python3
import torch

def iou(pred_boxes, true_boxes, format="corner"):
    """
    IOU: Intersection Over Union

    ------------
    Parameters:
        pred_boxes: shape (N, 4)
            contains N predicted boxes with their x,y,w,h values

        true_boxes: shape (N, 4)
            contains the N actual boxes with their x,y,w,h values

        format: either "corner" or "center"
            tells us which format the boxes are stored in

    -----------
    returns:
        intersection devided by the union of 2 boxes
    """

    if format == "corner":
        b1x1 = pred_boxes[...,0:1]
        b1y1 = pred_boxes[...,1:2]
        b1x2 = pred_boxes[...,2:3]
        b1y2 = pred_boxes[...,3:4]

        b2x1 = true_boxes[...,0:1]
        b2y1 = true_boxes[...,1:2]
        b2x2 = true_boxes[...,2:3]
        b2y2 = true_boxes[...,3:4]

    elif format == "center":
        width1 = pred_boxes[...,2:3]
        height1 = pred_boxes[...,3:4]
        b1x1 = pred_boxes[...,0:1] - width1  / 2
        b1y1 = pred_boxes[...,1:2] - height1 / 2
        b1x2 = pred_boxes[...,0:1] + width1  / 2
        b1y2 = pred_boxes[...,1:2] + height1 / 2

        width2 = true_boxes[...,2:3]
        height2 = true_boxes[...,3:4]
        b2x1 = true_boxes[...,0:1] - width2  / 2
        b2y1 = true_boxes[...,1:2] - height2 / 2
        b2x2 = true_boxes[...,0:1] + width2  / 2
        b2y2 = true_boxes[...,1:2] + height2 / 2

    x1 = torch.max(b1x1,b2x1)
    y1 = torch.max(b1y1,b2y1)
    x2 = torch.min(b1x2,b2x2)
    y2 = torch.min(b1y2,b2y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    b1_area = abs((b1x2 - b1x1) * (b1y2 - b1y1))
    b2_area = abs((b2x2 - b2x1) * (b2y2 - b2y1))

    return intersection / (b1_area + b2_area - intersection)
