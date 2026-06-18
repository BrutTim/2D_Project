import numpy as np
from masks import template_data as td
from scipy.ndimage import binary_fill_holes
from imageprocessing import Scaling
import matplotlib.pyplot as plt
from imageprocessing.Regionlabeling import sequential_region_labeling

def choose_shape_with_color(scores, color):
    adjusted_scores = scores.copy()

    if color == "red":
        adjusted_scores["diamond"] *= 0.75
        adjusted_scores["octagon"] *= 1.10
        adjusted_scores["triangle"] *= 1.05

    if color == "yellow":
        adjusted_scores["diamond"] *= 1.20

    if color == "blue":
        adjusted_scores["circle"] *= 1.20

    best_shape = max(adjusted_scores, key=adjusted_scores.get)

    return best_shape, adjusted_scores[best_shape]

def calculate_iou(mask, template):
    mask = np.asarray(mask, dtype=bool)
    template = np.asarray(template, dtype=bool)

    intersection = np.logical_and(mask, template).sum()
    union = np.logical_or(mask, template).sum()

    return intersection / union if union > 0 else 0.0

def choose_shape_with_color(scores, color):
    adjusted_scores = scores.copy()

    if color == "red":
        adjusted_scores["diamond"] *= 0.75
        adjusted_scores["octagon"] *= 1.10
        adjusted_scores["triangle"] *= 1.05

    if color == "yellow":
        adjusted_scores["diamond"] *= 1.20

    if color == "blue":
        adjusted_scores["circle"] *= 1.20

    best_shape = max(adjusted_scores, key=adjusted_scores.get)

    return best_shape, adjusted_scores[best_shape]

def classify_type_scores(mask):
    scores = {}

    forms = [
        ("triangle", td.get_triangle_template()),
        ("diamond", td.get_diamond_template()),
        ("circle", td.get_circle_template()),
        ("octagon", td.get_octagon_template()),
    ]

    for form_name, template in forms:
        template_normalized = Scaling.normalize_image(template, 128)
        scores[form_name] = calculate_iou(mask, template_normalized)

    return scores


def classify_sign(sign_candidate, color):

    filled_sign_candidate = binary_fill_holes(sign_candidate)
    normalized_sign_candidate = Scaling.normalize_image(filled_sign_candidate, 128)

    plt.imshow(normalized_sign_candidate, cmap="nipy_spectral",)
    plt.title(f"normalized sign candidate")
    plt.axis("off")
    plt.show()

    scores = classify_type_scores(normalized_sign_candidate)
    type, score = choose_shape_with_color(scores, color)
    print(type + ':' + str(score))
    return type, score


def get_inner_Label(normalized_symbol):
    labels = sequential_region_labeling(normalized_symbol)

    inner_label = labels.copy()
    inner_label[inner_label <= 2] = 0

    if np.any(inner_label > 0):
        return inner_label

    # Fall 2: Inneres Symbol wurde nicht als eigenes Label erkannt
    inverted_outer = (labels != 2).astype(np.uint8)
    inner_label_region = sequential_region_labeling(inverted_outer)

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

    return best_name, best_score