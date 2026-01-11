import numpy as np
from typing import List


def computeIOU(box1, box2):
    """
    Compute the IoU of two bounding boxes.
    The bounding box is Serialized: xTL, yTL, xBR, yBR.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    intersection = (max(0, min(x2, x4) - max(x1, x3)) * 
                   max(0, min(y2, y4) - max(y1, y3)))
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union = box1_area + box2_area - intersection
    
    return intersection / union


INDEX_TRUE_POSITIVE = 0
INDEX_FALSE_POSITIVE = 1
INDEX_FALSE_NEGATIVE = 2
INDEX_TRUE_NEGATIVE = 3


def computeBBConfusionMatrix(target, output, num_classes, 
                             iou_threshold=0.5) -> List[List[int]]:
    """
    Compute the Confusion Matrix.

    Return:
        List of list with format 
            [[true_positive, false_positive, false_negative, true_negative]]
        The class number is the index.
    """
    boxesIoUs = np.zeros((len(output['boxes']), len(target['boxes'])))
    
    for ii, boxO in enumerate(output['boxes']):
        for jj, boxT in enumerate(target['boxes']):
            boxesIoUs[ii, jj] = computeIOU(boxO, boxT)

    # Filter the matrix to keep only max values above the iou_threshold
    boxesIoUsFilter = boxesIoUs.copy()
    
    for ii, row in enumerate(boxesIoUsFilter):
        max_val = np.max(row, initial=-np.inf)
        if max_val > iou_threshold:
            boxesIoUsFilter[ii, boxesIoUsFilter[ii, :] == max_val] = max_val
            boxesIoUsFilter[ii, boxesIoUsFilter[ii, :] != max_val] = -1
        else:
            boxesIoUsFilter[ii, :] = 0

    for jj in range(boxesIoUsFilter.shape[1]):
        max_val = np.max(boxesIoUsFilter[:, jj], initial=-np.inf)
        if max_val > iou_threshold:
            boxesIoUsFilter[boxesIoUsFilter[:, jj] == max_val, jj] = max_val
            boxesIoUsFilter[boxesIoUsFilter[:, jj] != max_val, jj] = -1
        else:
            boxesIoUsFilter[:, jj] = 0
    
    # print(boxesIoUs)
    # print(boxesIoUsFilter)

    metric = [[0, 0, 0, 0] for _ in range(num_classes)]

    for ii, row in enumerate(boxesIoUsFilter):
        indexMax = (int(np.argmax(row)) 
                   if row[int(np.argmax(row))] > 0 else -1)
        output_class = int(output['labels'][ii])
        
        if indexMax > -1 and int(target['labels'][indexMax]) == output_class:
            metric[output_class][INDEX_TRUE_POSITIVE] += 1
        else:
            metric[output_class][INDEX_FALSE_POSITIVE] += 1
    
    for jj in range(boxesIoUsFilter.shape[1]):
        max_val = np.max(boxesIoUsFilter[:, jj], initial=-np.inf)
        target_class = int(target['labels'][jj])
        
        if not max_val > 0:
            metric[target_class][INDEX_FALSE_NEGATIVE] += 1

    # print(metric)
    return metric


def computeConfusionMatrix(preds, target, num_classes, weight):
    """
    Compute confusion matrix with weighted and unweighted variants.
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    confusion_matrix_weighted = np.zeros((num_classes, num_classes))

    percent_correct = 0
    percent_weighted_correct = 0
    
    for i in range(len(preds)):
        percent_node_correct = 0
        percent_weighted_node_correct = 0
        
        for j in range(len(preds[i])):
            if preds[i][j] == target[i][j]:
                percent_node_correct += 1
                percent_weighted_node_correct += weight[i][j]
            
            confusion_matrix[target[i][j]][preds[i][j]] += 1
            confusion_matrix_weighted[target[i][j]][preds[i][j]] += weight[i][j]
        
        percent_correct += percent_node_correct / len(preds[i]) * 100
        percent_weighted_correct += percent_weighted_node_correct * 100

    percent_correct /= len(preds)
    percent_weighted_correct /= len(preds)

    return confusion_matrix, confusion_matrix_weighted


def computeSConfusionMatrix(preds, target, num_classes):
    """
    Compute simple confusion matrix.
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    percent_correct = 0
    
    for i in range(len(preds)):
        if preds[i] == target[i]:
            percent_correct += 1
        confusion_matrix[target[i]][preds[i]] += 1
    
    percent_correct /= len(preds)
    percent_correct *= 100

    return confusion_matrix