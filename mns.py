import pdb

import numpy as np

def non_max_suppression(detections):
    """
    Perform non-maximum suppression to remove overlapping bounding boxes.

    Args:
        boxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        scores (list): List of confidence scores for each bounding box.
        threshold (float): Threshold value for considering overlapping boxes.

    Returns:
        list: List of indices to keep after NMS.
    """
    boxes = []
    scores = []
    cls = []
    threshold = 0.5
    # pdb.set_trace()
    for key in detections:
        boxes.append(key[0])
        scores.append(key[1])
        cls.append(key[2])


    if len(boxes) == 0:
        return []

    # Convert bounding boxes to (x1, y1, x2, y2) format
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate the area of each bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort bounding boxes by their scores
    idxs = np.argsort(scores)

    # Initialize list to store the indices of selected boxes
    selected_idxs = []

    while len(idxs) > 0:
        # Select the last (highest score) bounding box
        last = len(idxs) - 1
        i = idxs[last]
        selected_idxs.append(i)

        # Calculate the intersection area
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)

        intersection = width * height

        # Calculate the overlap ratio
        overlap = intersection / (area[i] + area[idxs[:last]] - intersection)

        # Remove indices of bounding boxes where overlap is greater than threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return selected_idxs
