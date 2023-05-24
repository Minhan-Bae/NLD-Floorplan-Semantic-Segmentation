from typing import List
import numpy as np
from skimage.measure import label

def iou_score(output: np.ndarray, target: np.ndarray):
    intersection = np.logical_and(target, output)
    union = np.logical_or(target, output)
    return np.sum(intersection) / np.sum(union)

def mean_average_precision(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], iou_thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05)):
    mean_ious = np.mean([iou_score(pred_mask, gt_mask) for pred_mask, gt_mask in zip(pred_masks, gt_masks)], axis=0)
    return np.mean([np.sum(mean_ious > thr) / len(mean_ious) for thr in iou_thresholds])