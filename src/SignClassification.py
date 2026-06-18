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


def get_inner_Label(labels):
    # Fall 1: Inneres Symbol wurde schon als eigenes Label erkannt
    inner_label = (labels > 2).astype(np.uint8)

    if inner_label.sum() > 0:
        return inner_label

    # Fall 2: Inneres Symbol wurde nicht als eigenes Label erkannt
    inverted_outer = (labels != 2).astype(np.uint8)
    inner_label_region = sequential_region_labeling(inverted_outer)
    inner_label = (inner_label_region > 2).astype(np.uint8)

    return inner_label


def split_labels(labels):
    outer_label = (labels == 2).astype(np.uint8)
    inner_label = get_inner_Label(labels)

    plot_split_labels(labels, outer_label, inner_label)

    return inner_label, outer_label


def plot_split_labels(labels, outer_label, inner_label):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        labels,
        cmap="nipy_spectral",
        vmin=0,
        vmax=max(2, labels.max())
    )
    axes[0].set_title(f"Alle Labels: {np.unique(labels)}")
    axes[0].axis("off")

    axes[1].imshow(
        outer_label,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1].set_title(
        f"Outer Label - {outer_label.sum()} Pixel"
    )
    axes[1].axis("off")

    axes[2].imshow(
        inner_label,
        cmap="nipy_spectral",
        vmin=0,
        vmax=max(2, inner_label.max())
    )
    axes[2].set_title(
        f"Inner Labels: {np.unique(inner_label)}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

"""
def classify_type(outer_label,labels):


    # Rand zu einer ausgefüllten Schildform machen
    outer_filled = binary_fill_holes(outer_label>0)
    outer_normalized = Scaling.normalize_image(outer_filled, 128)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[1, 1].imshow(outer_normalized, cmap="nipy_spectral",
                      vmin=0, vmax=labels.max())
    axes[1, 1].set_title("Bild 02 vor Anpassung")
    axes[1, 1].axis("off")


    forms = [
        ("triangle", td.get_triangle_template()),
        ("diamond", td.get_diamond_template()),
        ("circle", td.get_circle_template()),
        ("octagon", td.get_octagon_template())
    ]

    best_score = 0.0
    best_form = "unknown"

    for form_name, template in forms:
        template_normalized = Scaling.normalize_image(template, 128)

        score = calculate_iou(
            outer_normalized,
            template_normalized
        )

        print(f"Score: {score:.3f}, Shape: {form_name}")

        if score > best_score:
            best_score = score
            best_form = form_name

    print(f"Beste Form: {best_form}, Score: {best_score:.3f}")

    return best_form
def identify_sign(type, label):
    pass

def classify_sign(labels):

    inner_label, outer_label = split_labels(labels)

    signtype = classify_type(outer_label,labels)

    return identify_sign(signtype, inner_label)


def get_inner_Label(labels):
    # Fall 1: Inneres Symbol wurde schon als eigenes Label erkannt
    inner_label = (labels > 2).astype(np.uint8)

    if inner_label.sum() > 0:
        return inner_label

    #Fall 2: Inneres Symbol wurde nicht als eigenes Label erkannt
    inverted_outer = (labels != 2).astype(np.uint8) # vertauscht vorder und hintergrund
    inner_label_region = sequential_region_labeling(inverted_outer) # neue Rgionen markierung der Vertauschten binaryimages
    inner_label = (inner_label_region > 2).astype(np.uint8)

    return inner_label

def split_labels(labels):
    outer_label = (labels == 2).astype(np.uint8)
    inner_label = get_inner_Label(labels)

    plot_split_labels(labels, outer_label, inner_label)


    return inner_label, outer_label

def calculate_iou(mask, template):
    mask = np.asarray(mask, dtype=bool)
    template = np.asarray(template, dtype=bool)

    intersection = np.logical_and(mask, template).sum()
    union = np.logical_or(mask, template).sum()

    return intersection / union if union > 0 else 0.0



def plot_split_labels(labels, outer_label, inner_label):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ursprüngliche Labelmatrix
    axes[0].imshow(
        labels,
        cmap="nipy_spectral",
        vmin=0,
        vmax=max(2, labels.max())
    )
    axes[0].set_title(f"Alle Labels: {np.unique(labels)}")
    axes[0].axis("off")

    # Äußere Schildform
    axes[1].imshow(
        outer_label,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1].set_title(
        f"Outer Label – {outer_label.sum()} Pixel"
    )
    axes[1].axis("off")

    axes[2].imshow(
        inner_label,
        cmap="nipy_spectral",
        vmin=0,
        vmax=max(2, inner_label.max())
    )
    axes[2].set_title(
        f"Inner Labels: {np.unique(inner_label)}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
"""
