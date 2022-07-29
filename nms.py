#!/usr/bin/env python3
import torch
from iou import iou

def nms(predictions, iou_threshold, prob_threshold, format="corner"):
    """
    NMS: Non Max Suppression

    -----------
    Parameters:
        predictions: list(c, p_o, x1, y1, x2, y2)
            a list containing the class, the probability of there beeing
            an object and the box corrdinates

        iou_threshold: float
            the threshold for removing a bounding box because it is
            probably for the same object

        prob_threshold: float
            the threshold at which we consider the prediction

        format: either "corner" or "center"
            value is passed through to the iou function

    --------
    Returns:
        nms_bboxes: list(c, p_o, x1, y1, x2, y2)
            all the valid boxes after removing unlikely ones and duplicate ones
    """

    assert type(predictions) == list

    bboxes = [box for box in predictions if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    nms_bboxes = []

    while bboxes:
        current_box = bboxes.pop(0)

        bboxes = [
             box for box in bboxes
             if box[0] != current_box[0]
             or iou(torch.tensor(current_box[2:]),
                   torch.tensor(box[2:]),
                   format=format)
             < iou_threshold
        ]
        nms_bboxes.append(current_box)

    return nms_bboxes
