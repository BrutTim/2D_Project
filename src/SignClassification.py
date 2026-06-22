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


def make_outer_shape_mask(shape, outer_type):
    height, width = shape
    mask = np.zeros(shape, dtype=bool)

    if outer_type == "circle":
        center = (height // 2, width // 2)
        radius = int(0.48 * min(height, width))
        rr, cc = disk(center, radius, shape=shape)
        mask[rr, cc] = True

    elif outer_type == "triangle":
        rows = np.array([0.04 * height, 0.96 * height, 0.96 * height])
        cols = np.array([0.50 * width, 0.04 * width, 0.96 * width])
        rr, cc = polygon(rows, cols, shape)
        mask[rr, cc] = True

    elif outer_type == "downwards_triangle":
        rows = np.array([0.04 * height, 0.04 * height, 0.96 * height])
        cols = np.array([0.04 * width, 0.96 * width, 0.50 * width])
        rr, cc = polygon(rows, cols, shape)
        mask[rr, cc] = True

    elif outer_type == "octagon":
        rows = np.array([
            0.04 * height, 0.04 * height, 0.30 * height, 0.70 * height,
            0.96 * height, 0.96 * height, 0.70 * height, 0.30 * height
        ])
        cols = np.array([
            0.30 * width, 0.70 * width, 0.96 * width, 0.96 * width,
            0.70 * width, 0.30 * width, 0.04 * width, 0.04 * width
        ])
        rr, cc = polygon(rows, cols, shape)
        mask[rr, cc] = True

    elif outer_type in ("square", "rectangle"):
        y_min = int(0.04 * height)
        y_max = int(0.96 * height)
        x_min = int(0.04 * width)
        x_max = int(0.96 * width)
        mask[y_min:y_max, x_min:x_max] = True

    return mask


def remove_outer_shape_from_symbol_mask(symbol_mask, outer_type):
    symbol_mask = np.asarray(symbol_mask, dtype=bool)
    outer_shape = make_outer_shape_mask(symbol_mask.shape, outer_type)

    plt.imshow(outer_shape, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Outer Shape Mask ({outer_type})")
    plt.axis("off")
    plt.show()

    inner_shape = binary_erosion(
        outer_shape,
        structure=np.ones((13, 13), dtype=bool)
    )

    plt.imshow(inner_shape, cmap="gray", vmin=0, vmax=1)
    plt.title("Eroded Inner Shape")
    plt.axis("off")
    plt.show()

    outer_border = outer_shape & ~inner_shape
    outer_border = binary_dilation(
        outer_border,
        structure=np.ones((5, 5), dtype=bool)
    )

    plt.imshow(outer_border, cmap="gray", vmin=0, vmax=1)
    plt.title("Outer Border Removed From Symbol")
    plt.axis("off")
    plt.show()

    inner_symbol = symbol_mask & ~outer_border
    inner_symbol = inner_symbol & inner_shape

    plt.imshow(inner_symbol, cmap="gray", vmin=0, vmax=1)
    plt.title("Raw Inner Symbol")
    plt.axis("off")
    plt.show()

    return inner_symbol.astype(np.uint8)


def keep_relevant_inner_components(inner_symbol):
    labels = sequential_region_labeling(inner_symbol)

    plt.imshow(labels, cmap="nipy_spectral")
    plt.title("Inner Symbol Component Labels")
    plt.axis("off")
    plt.show()

    cleaned = np.zeros_like(inner_symbol, dtype=np.uint8)
    total_area = inner_symbol.shape[0] * inner_symbol.shape[1]

    for label in np.unique(labels):
        if label == 0:
            continue

        component = labels == label
        area = component.sum()

        if area < 0.0008 * total_area:
            continue

        ys, xs = np.where(component)
        height = ys.max() - ys.min() + 1
        width = xs.max() - xs.min() + 1

        if height < 3 or width < 3:
            continue

        cleaned[component] = 1

    plt.imshow(cleaned, cmap="gray", vmin=0, vmax=1)
    plt.title("Relevant Inner Components")
    plt.axis("off")
    plt.show()

    return cleaned


def extract_inner_symbol(symbol_mask, outer_type):
    inner_symbol = remove_outer_shape_from_symbol_mask(symbol_mask, outer_type)
    cleaned_inner_symbol = keep_relevant_inner_components(symbol_mask)

    plt.imshow(cleaned_inner_symbol, cmap="gray", vmin=0, vmax=1)
    plt.title("Extracted Inner Symbol")
    plt.axis("off")
    plt.show()

    return cleaned_inner_symbol


def fix_inverted_inner_symbol_mask(symbol_mask):
    symbol_mask = np.asarray(symbol_mask, dtype=bool)

    if not np.any(symbol_mask):
        return symbol_mask

    foreground_ratio = symbol_mask.sum() / symbol_mask.size
    labels = sequential_region_labeling(symbol_mask.astype(np.uint8))
    foreground_labels = [label for label in np.unique(labels) if label != 0]

    if len(foreground_labels) == 0:
        return symbol_mask

    largest_label = max(
        foreground_labels,
        key=lambda label: (labels == label).sum()
    )
    largest_component = labels == largest_label
    largest_ratio = largest_component.sum() / symbol_mask.size

    ys, xs = np.where(largest_component)
    touches_border = (
        ys.min() == 0 or
        xs.min() == 0 or
        ys.max() == symbol_mask.shape[0] - 1 or
        xs.max() == symbol_mask.shape[1] - 1
    )

    is_probably_background = (
        foreground_ratio > 0.45 and
        largest_ratio > 0.35 and
        touches_border
    )

    if not is_probably_background:
        return symbol_mask

    filled_background = binary_fill_holes(largest_component)
    corrected_symbol = filled_background & ~largest_component

    plt.imshow(corrected_symbol, cmap="gray", vmin=0, vmax=1)
    plt.title("Corrected Inverted Inner Symbol")
    plt.axis("off")
    plt.show()

    if corrected_symbol.sum() == 0:
        return symbol_mask

    return corrected_symbol


def get_inner_Label(normalized_symbol):
    normalized_symbol = fix_inverted_inner_symbol_mask(normalized_symbol)

    plt.imshow(normalized_symbol, cmap="gray", vmin=0, vmax=1)
    plt.title("Inner Symbol Before Labeling")
    plt.axis("off")
    plt.show()

    labels = sequential_region_labeling(normalized_symbol)
    labels = morphologisch_opening(normalized_symbol,iter_num=1)
    labels = sequential_region_labeling(labels)
    plt.imshow(labels, cmap="nipy_spectral",)
    plt.title(f"labels for inner symbol")
    plt.show()


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
    templates = td.get_sign_templates_for_type(outer_type)

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
