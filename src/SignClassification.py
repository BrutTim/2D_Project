from skimage import io
import numpy as np
import template_data as td
from scipy.ndimage import binary_fill_holes

def classify_type(outer_label,labels):

    # Rand zu einer ausgefüllten Schildform machen
    outer_filled = binary_fill_holes(outer_label)
    score = calculate_iou(outer_filled, td.get_triangle_template())

    print("Dreiecksähnlichkeit:", score)

def identify_sign(type, label):
    pass

def classify_sign(labels):

    inner_label, outer_label = split_labels(labels)
    signtype = classify_type(outer_label,labels)

    return identify_sign(signtype, inner_label)

def split_labels(labels):

    outer_label = labels
    outer_label[outer_label > 2] = 0
    inner_label = labels - outer_label

    return inner_label, outer_label

def calculate_iou(mask, template):
    intersection = np.logical_and(mask, template).sum()
    union = np.logical_or(mask, template).sum()

    return intersection / union if union > 0 else 0.0

