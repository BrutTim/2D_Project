import numpy as np

from imageprocessing.MorphologischesOpening import morphologisch_opening
from masks import template_data as td
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
from skimage.draw import disk, polygon
from imageprocessing import Scaling
import matplotlib.pyplot as plt
from imageprocessing.Regionlabeling import sequential_region_labeling
from imageprocessing.CornerDetection import count_corners


def calculate_iou(mask, template):
    mask = np.asarray(mask, dtype=bool)
    template = np.asarray(template, dtype=bool)

    intersection = np.logical_and(mask, template).sum()
    union = np.logical_or(mask, template).sum()

    return intersection / union if union > 0 else 0.0


def choose_shape(scores, color, corners):
    adjusted_scores = scores.copy()
    print(f'Ecken: ' + str(corners))

    if color == "red":

        if corners < 2:
            adjusted_scores["circle"] *= 1.20
        elif 2 <= corners <= 4:
            adjusted_scores["triangle"] *= 1.20
        elif 5 <= corners <= 10:
            adjusted_scores["octagon"] *= 1.20

        adjusted_scores["diamond"] *= 0.0
        adjusted_scores["square"] *= 0.0

    if color == "yellow":

        adjusted_scores["octagon"] *= 0.0
        adjusted_scores["triangle"] *= 0.0
        adjusted_scores["downwards_triangle"] *= 0.0
        adjusted_scores["square"] *= 0.0
        adjusted_scores["circle"] *= 0.0

    if color == "blue":
        if corners <= 2:
            adjusted_scores["circle"] *= 1.20
        elif 2 <= corners <= 4:
            adjusted_scores["square"] *= 1.20

        adjusted_scores["diamond"] *= 0.0
        adjusted_scores["octagon"] *= 0.0
        adjusted_scores["triangle"] *= 0.0
        adjusted_scores["downwards_triangle"] *= 0.0
        adjusted_scores["diamond"] *= 0.0

    best_shape = max(adjusted_scores, key=adjusted_scores.get)
    return best_shape, adjusted_scores[best_shape]

def classify_type_scores(mask):
    scores = {}

    forms = [
        ("triangle", td.get_triangle_template()),
        ("diamond", td.get_diamond_template()),
        ("circle", td.get_circle_template()),
        ("octagon", td.get_octagon_template()),
        ("square", td.get_square_template()),
        ("downwards_triangle", td.get_downwards_triangle_template()),
    ]

    for form_name, template in forms:
        template_normalized = Scaling.normalize_image(template, 128)
        scores[form_name] = calculate_iou(mask, template_normalized)

    return scores


def classify_sign(sign_candidate, color):

    filled_sign_candidate = binary_fill_holes(sign_candidate)
    normalized_sign_candidate = Scaling.normalize_image(filled_sign_candidate, 128)

    corners = count_corners(normalized_sign_candidate)

    plt.imshow(normalized_sign_candidate, cmap="nipy_spectral",)
    plt.title(f"normalized and filled sign candidate")
    plt.axis("off")
    plt.show()

    scores = classify_type_scores(normalized_sign_candidate)
    type, score = choose_shape(scores, color, corners)
    print(type + ':' + str(score))
    return type, score


def get_inner_Label(normalized_symbol):
    labels = morphologisch_opening(normalized_symbol,iter_num=1)
    labels = sequential_region_labeling(labels)
    plt.imshow(labels, cmap="nipy_spectral",)
    plt.title(f"labels for inner symbol")
    plt.show()


    # inner_label = labels.copy()
    # inner_label[inner_label <= 2] = 0

    if np.any(labels > 0):
        return labels

    # Fall 2: Inneres Symbol wurde nicht als eigenes Label erkannt
    inverted_outer = (labels != 2).astype(np.uint8)
    inner_label_region = sequential_region_labeling(inverted_outer)
    plt.imshow(inner_label_region, cmap="nipy_spectral",)
    plt.title(f"labels for outer symbol")
    plt.show()
    inner_label = inner_label_region.copy()
    inner_label[inner_label <= 2] = 0

    return inner_label



def identify_sign(outer_type, inner_label):
    if outer_type == "octagon":
        return "Halt. Vorfahrt gewaehren"

    if outer_type == "diamond":
        return "Vorfahrtstrasse"

    sign_name, score = classify_inner_label(inner_label, outer_type)

    if score < 0.3:
        return "Unbekanntes Schild"

    return sign_name

def classify_inner_label(inner_label, outer_type):
    templates = td.get_vorschriftzeichen_templates_for_type(outer_type)

    if len(templates) == 0:
        return "Unbekanntes Schild", 0.0

    inner_normalized = Scaling.normalize_image(inner_label, 128)

    best_name = "Unbekanntes Schild"
    best_score = 0.0
    best_template = None

    for template, sign_name in templates:
        template_normalized = Scaling.normalize_image(template, 128)

        score = calculate_iou(
            inner_normalized,
            template_normalized
        )

        print(f"Inner Score: {score:.3f}, Schild: {sign_name}")

        if score > best_score:
            best_score = score
            best_name = sign_name
            best_template = template_normalized

    if best_template is not None:
        plt.imshow(best_template, cmap="gray", vmin=0, vmax=1)
        plt.title(f"Bestes normalisiertes Template: {best_name}")
        plt.axis("off")
        plt.show()

    return best_name, best_score
