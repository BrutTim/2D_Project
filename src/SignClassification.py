import numpy as np

from imageprocessing.MorphologischesOpening import morphologisch_opening
from imageprocessing.Scaling import normalize_image
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
        rows = np.array([0.04 * height, 0.95 * height, 0.95 * height])
        cols = np.array([0.50 * width, 0.04 * width, 0.95 * width])
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


def classify_diamond_sign(symbol_mask, sign_candidate):
    symbol_mask = np.asarray(symbol_mask, dtype=bool)
    sign_candidate = np.asarray(sign_candidate, dtype=bool)

    inner_area = binary_erosion(
        sign_candidate,
        structure=np.ones((7, 7), dtype=bool)
    )
    inner_markings = symbol_mask & inner_area
    inner_area_size = inner_area.sum()

    if inner_area_size == 0:
        return "Vorfahrtstrasse"

    plt.imshow(inner_markings, cmap="gray", vmin=0, vmax=1)
    plt.title("Diamond Inner Markings")
    plt.axis("off")
    plt.show()

    inner_marking_ratio = inner_markings.sum() / inner_area_size

    print(f"Diamond inner marking ratio: {inner_marking_ratio:.4f}")

    if inner_marking_ratio > 0.015:
        return "Ende der Vorfahrtstrasse"

    return "Vorfahrtstrasse"


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
    large_components = np.zeros_like(symbol_mask, dtype=bool)

    for label in foreground_labels:
        component = labels == label
        component_ratio = component.sum() / symbol_mask.size

        if component_ratio >= 0.01:
            large_components |= component

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

    filled_large_components = binary_fill_holes(large_components)
    hole_symbol = filled_large_components & ~large_components

    if hole_symbol.sum() > 0:
        hole_labels = sequential_region_labeling(hole_symbol.astype(np.uint8))
        hole_components = [label for label in np.unique(hole_labels) if label != 0]

        if len(hole_components) > 0:
            valid_holes = np.zeros_like(hole_symbol, dtype=bool)

            for hole_label in hole_components:
                hole_component = hole_labels == hole_label
                hole_area_ratio = hole_component.sum() / symbol_mask.size

                if hole_area_ratio < 0.002:
                    continue

                hole_y_min, hole_x_min, hole_y_max, hole_x_max = component_bbox(hole_component)
                hole_touches_border = (
                    hole_y_min == 0 or
                    hole_x_min == 0 or
                    hole_y_max == symbol_mask.shape[0] - 1 or
                    hole_x_max == symbol_mask.shape[1] - 1
                )

                if hole_touches_border:
                    continue

                valid_holes |= hole_component

            # Nur invertieren, wenn die aktuelle Maske wirklich eine grosse
            # gefuellte Flaeche ist (z.B. blauer Kreis mit weissem Pfeil als
            # Loch). Bei Gefahrzeichen waere sonst der Dreiecksinnenraum ein
            # falsches "Loch", weil der Rand als Vordergrund erkannt wird.
            is_filled_foreground_shape = (
                foreground_ratio > 0.35 and
                largest_ratio > 0.20
            )

            if is_filled_foreground_shape and valid_holes.sum() / symbol_mask.size > 0.01:
                corrected_symbol = valid_holes

                plt.imshow(corrected_symbol, cmap="gray", vmin=0, vmax=1)
                plt.title("Corrected Hole Inner Symbol")
                plt.axis("off")
                plt.show()

                return corrected_symbol

    if not is_probably_background:
        return symbol_mask

    corrected_symbol = hole_symbol

    plt.imshow(corrected_symbol, cmap="gray", vmin=0, vmax=1)
    plt.title("Corrected Inverted Inner Symbol")
    plt.axis("off")
    plt.show()

    if corrected_symbol.sum() == 0:
        return symbol_mask

    return corrected_symbol

def component_bbox(component):
    ys, xs = np.where(component)
    return ys.min(), xs.min(), ys.max(), xs.max()


def bbox_distance(first_bbox, second_bbox):
    first_y_min, first_x_min, first_y_max, first_x_max = first_bbox
    second_y_min, second_x_min, second_y_max, second_x_max = second_bbox

    y_distance = max(0, second_y_min - first_y_max, first_y_min - second_y_max)
    x_distance = max(0, second_x_min - first_x_max, first_x_min - second_x_max)

    return max(y_distance, x_distance)


def make_inner_shape_mask(shape, outer_type):
    outer_shape = make_outer_shape_mask(shape, outer_type)

    erosion_size_by_type = {
        "triangle": 19,
        "downwards_triangle": 19,
        "circle": 17,
        "octagon": 15,
        "square": 13,
        "rectangle": 13,
    }
    erosion_size = erosion_size_by_type.get(outer_type, 15)

    return binary_erosion(
        outer_shape,
        structure=np.ones((erosion_size, erosion_size), dtype=bool)
    )


def keep_relevant_inner_components(labels, outer_type=None):

    cleaned = np.zeros_like(labels, dtype=np.uint8)
    total_area = labels.shape[0] * labels.shape[1]
    inner_shape = None
    outer_border = None

    if outer_type is not None:
        outer_shape = make_outer_shape_mask(labels.shape, outer_type)
        inner_shape = make_inner_shape_mask(labels.shape, outer_type)
        outer_border = outer_shape & ~inner_shape
        outer_border = binary_dilation(
            outer_border,
            structure=np.ones((5, 5), dtype=bool)
        )

    components = []

    for label in np.unique(labels):
        if label == 0:
            continue

        component = labels == label
        area = component.sum()

        if area < 3:
            continue

        ys, xs = np.where(component)
        height = ys.max() - ys.min() + 1
        width = xs.max() - xs.min() + 1

        if height < 2 or width < 2:
            continue

        if inner_shape is not None:
            inner_overlap = np.logical_and(component, inner_shape).sum() / area
            border_overlap = np.logical_and(component, outer_border).sum() / area
            bbox_area_ratio = (height * width) / total_area
            component_area_ratio = area / total_area
            is_large_inner_candidate = (
                component_area_ratio >= 0.01 and
                bbox_area_ratio <= 0.50
            )

            # Der Aussenrand ist gross, liegt stark in der Randzone und hat
            # meistens eine riesige Bounding Box. Das innere Symbol nicht.
            if is_large_inner_candidate:
                if border_overlap > 0.75:
                    continue

                if inner_overlap < 0.20:
                    continue
            elif border_overlap > 0.35:
                continue

            elif inner_overlap < 0.55:
                continue

            elif bbox_area_ratio > 0.50:
                continue

        components.append({
            "component": component,
            "area": area,
            "bbox": component_bbox(component),
        })

    if len(components) == 0:
        plt.imshow(cleaned, cmap="gray", vmin=0, vmax=1)
        plt.title("Relevant Inner Components")
        plt.axis("off")
        plt.show()
        return cleaned

    largest_component = max(components, key=lambda item: item["area"])
    largest_bbox = largest_component["bbox"]
    min_standalone_area = max(8, int(0.0008 * total_area))

    for item in components:
        close_to_main_symbol = bbox_distance(item["bbox"], largest_bbox) <= 18

        if item["area"] < min_standalone_area and not close_to_main_symbol:
            continue

        cleaned[item["component"]] = 1

    plt.imshow(cleaned, cmap="gray", vmin=0, vmax=1)
    plt.title("Relevant Inner Components")
    plt.axis("off")
    plt.show()

    return cleaned

def get_inner_Label(symbol, outer_type=None):
    normalized_symbol = normalize_image(symbol, 128)
    normalized_symbol = fix_inverted_inner_symbol_mask(normalized_symbol)

    plt.imshow(normalized_symbol, cmap="gray", vmin=0, vmax=1)
    plt.title("Inner Symbol Before Labeling")
    plt.axis("off")
    plt.show()

    labels = sequential_region_labeling(normalized_symbol)
    plt.imshow(labels, cmap="nipy_spectral",)
    plt.title(f"labels for inner symbol")
    plt.show()

    cleaned = keep_relevant_inner_components(labels, outer_type)
    cleaned = normalize_image(cleaned, 128).astype(np.uint8)

    plt.imshow(cleaned, cmap="gray", vmin=0, vmax=1)
    plt.title("Normalized Inner Label Only")
    plt.axis("off")
    plt.show()

    return cleaned



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
