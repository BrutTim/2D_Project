from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes
import SignClassification as sc
from imageprocessing.Regionlabeling import sequential_region_labeling
from imageprocessing.MorphologischesOpening import morphologisch_opening
from imageprocessing.RGBtoHSV import rgb_to_hsv
from imageprocessing.Scaling import normalize_image
from masks.colour_masks import get_color_mask, get_sign_position_mask


def plot_image(axis, image, title, cmap="gray", vmin=None, vmax=None):
    axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")


def plot_debug_grid(image, hsv_image, position_mask, labels, best_candidate):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    plot_image(axes[0], image, "original rgb")
    plot_image(axes[1], hsv_image[:, :, 0], "hue", cmap="hsv", vmin=0, vmax=1)
    plot_image(axes[2], hsv_image[:, :, 1], "saettigung", vmin=0, vmax=1)
    plot_image(axes[3], position_mask, "saettigungsmaske vor opening", vmin=0, vmax=1)
    plot_image(axes[4], labels, "region labeling", cmap="nipy_spectral")
    plot_image(axes[5], best_candidate, "ausgewaehlter schildkandidat", vmin=0, vmax=1)

    labels_for_inner_symbol = sc.get_debug_image("labels_for_inner_symbol")
    relevant_inner_components = sc.get_debug_image("relevant_inner_components")
    best_normalized_template = sc.get_debug_image("best_normalized_template")

    debug_items = [
        (labels_for_inner_symbol, "labels for inner symbol", "nipy_spectral", None, None),
        (relevant_inner_components, "relevant inner components", "gray", 0, 1),
        (best_normalized_template, "bestes normalisiertes template", "gray", 0, 1),
    ]

    for axis, (debug_image, title, cmap, vmin, vmax) in zip(axes[6:], debug_items):
        if debug_image is None:
            debug_image = np.zeros_like(position_mask)
            cmap = "gray"
            vmin = 0
            vmax = 1

        axis.imshow(debug_image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def score_candidate(candidate):
    area = candidate.sum()

    if area == 0:
        return 0

    ys, xs = np.where(candidate)

    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1

    bbox_area = width * height
    fill_ratio = area / bbox_area
    aspect_ratio = width / height

    if not (0.45 <= aspect_ratio <= 2.2):
        return 0

    score = area * fill_ratio

    return score

def extract_light_symbol_mask(image, sign_candidate):
    gray = make_gray_image(image)

    symbol_mask = (
        (gray > 180) &
        (sign_candidate > 0)
    )

    return symbol_mask.astype(np.uint8)

def extract_dark_symbol_mask(image, sign_candidate):

    gray = make_gray_image(image)

    symbol_mask = (
        (gray < 80) &
        (sign_candidate > 0)
    )

    return symbol_mask.astype(np.uint8)


def select_top_sign_candidates(labels, top_n=3):
    candidates = []
    image_height, image_width = labels.shape
    image_area = image_height * image_width

    for label in np.unique(labels):
        if label == 0:
            continue

        region = labels == label
        area = region.sum()

        if area < 200:
            continue

        ys, xs = np.where(region)

        y_min = ys.min()
        y_max = ys.max()
        x_min = xs.min()
        x_max = xs.max()

        height = y_max - y_min + 1
        width = x_max - x_min + 1

        bbox_area = width * height
        bbox_area_ratio = bbox_area / image_area

        aspect_ratio = width / height

        touches_border = (
            y_min == 0 or
            x_min == 0 or
            y_max == image_height - 1 or
            x_max == image_width - 1
        )

        #Schildkandidaten am Bildrand verwerfen
        if touches_border:
            continue

        #  kleine oder riesige Regionen rauswerfen
        if bbox_area_ratio < 0.002:
            continue

        if bbox_area_ratio > 0.60:
            continue

        # Verkehrsschilder sind meist nicht extrem schmal/lang
        if not (0.45 <= aspect_ratio <= 2.2):
            continue

        # Region auffüllen: aus Rand wird volle Außenform
        filled = binary_fill_holes(region)
        filled_area = filled.sum()

        filled_fill_ratio = filled_area / bbox_area
        original_fill_ratio = area / bbox_area

        # Gefüllte Schildform sollte halbwegs kompakt sein
        if filled_fill_ratio < 0.25:
            continue

        # Wenn die Originalregion fast nichts im eigenen BBox-Bereich hat,
        # ist sie oft nur Rauschen oder eine dünne Linie
        if original_fill_ratio < 0.03:
            continue

        # Kandidaten bevorzugen:
        # - große, aber nicht riesige Region
        # - kompakte gefüllte Form
        # - nicht am Rand
        score = filled_area * filled_fill_ratio

        candidates.append({
            "label": label,
            "mask": filled.astype(np.uint8),
            "score": score,
            "area": area,
            "bbox": (x_min, y_min, x_max, y_max),
            "bbox_size": (width, height),
            "aspect_ratio": aspect_ratio,
            "original_fill_ratio": original_fill_ratio,
            "filled_fill_ratio": filled_fill_ratio,
            "bbox_area_ratio": bbox_area_ratio,
        })

    candidates = sorted(
        candidates,
        key=lambda candidate: candidate["score"],
        reverse=True
    )

    top_candidates = candidates[:top_n]

    for candidate in top_candidates:
        bbox_width, bbox_height = candidate["bbox_size"]
        print(
            f"Top Kandidat Label {candidate['label']}: "
            f"area={candidate['area']}, "
            f"bbox={bbox_width}x{bbox_height}, "
            f"aspect={candidate['aspect_ratio']:.2f}, "
            f"orig_fill={candidate['original_fill_ratio']:.2f}, "
            f"filled_fill={candidate['filled_fill_ratio']:.2f}, "
            f"bbox_ratio={candidate['bbox_area_ratio']:.3f}, "
            f"score={candidate['score']:.1f}"
        )

    return top_candidates


def select_best_sign_candidate(labels):
    top_candidates = select_top_sign_candidates(labels, top_n=1)

    if len(top_candidates) == 0:
        return np.zeros_like(labels, dtype=np.uint8)

    return top_candidates[0]["mask"].astype(np.uint8)


def make_binary_image(gray_image):
    binary = np.zeros(gray_image.shape, dtype=np.uint8)
    binary[gray_image < 240] = 1
    return binary

def make_gray_image(image):
    if image.ndim == 3:
        gray_img = (
                0.299 * image[..., 0] +
                0.587 * image[..., 1] +
                0.114 * image[..., 2]
        )
        return gray_img
    else:
        return image


def detect_candidate_color(
    hsv_image,
    sign_candidate,
    candidate_label=None,
    min_color_pixels=30,
    min_color_ratio=0.015
):
    masks = get_color_mask(hsv_image)
    candidate = sign_candidate > 0
    candidate_area = candidate.sum()

    if candidate_area == 0:
        print(f"Kandidat {candidate_label} aussortiert: leere Maske")
        return None

    color_scores = {
        "red": np.logical_and(masks[0] > 0, candidate).sum(),
        "yellow": np.logical_and(masks[1] > 0, candidate).sum(),
        "blue": np.logical_and(masks[2] > 0, candidate).sum(),
    }

    best_color = max(color_scores, key=color_scores.get)
    best_score = color_scores[best_color]
    best_ratio = best_score / candidate_area

    if best_score < min_color_pixels or best_ratio < min_color_ratio:
        print(
            f"Kandidat {candidate_label} aussortiert: keine klare Farbe "
            f"(red={color_scores['red']}, "
            f"yellow={color_scores['yellow']}, "
            f"blue={color_scores['blue']}, "
            f"best_ratio={best_ratio:.3f})"
        )
        return None

    print(
        f"Kandidat {candidate_label} Farbe: {best_color} "
        f"(red={color_scores['red']}, "
        f"yellow={color_scores['yellow']}, "
        f"blue={color_scores['blue']}, "
        f"best_ratio={best_ratio:.3f})"
    )

    return best_color


def build_yellow_diamond_candidate(hsv_image):
    yellow_mask = get_color_mask(hsv_image)[1] > 0
    closed = binary_closing(
        yellow_mask,
        structure=np.ones((13, 13), dtype=bool)
    )
    filled = binary_fill_holes(closed)
    labels = sequential_region_labeling(filled.astype(np.uint8))

    candidate = select_best_sign_candidate(labels)

    if candidate.sum() == 0:
        return None

    return candidate


def choose_best_colored_shape_candidate(hsv_image, candidates):
    valid_candidates = []

    for candidate in candidates:
        color = detect_candidate_color(
            hsv_image,
            candidate["mask"],
            candidate_label=candidate["label"]
        )

        if color is None:
            continue

        shape_type, shape_iou = sc.classify_shape_by_iou(candidate["mask"])

        valid_candidates.append({
            **candidate,
            "color": color,
            "shape_type": shape_type,
            "shape_iou": shape_iou,
        })

    if len(valid_candidates) == 0:
        print("Kein Schildkandidat mit klarer Farbe gefunden")
        return None

    best_candidate = max(
        valid_candidates,
        key=lambda candidate: (candidate["shape_iou"], candidate["score"])
    )

    print(
        f"Gewaehlter Kandidat Label {best_candidate['label']}: "
        f"shape={best_candidate['shape_type']}, "
        f"shape_iou={best_candidate['shape_iou']:.3f}, "
        f"color={best_candidate['color']}, "
        f"candidate_score={best_candidate['score']:.1f}"
    )

    return best_candidate


def main():
    sc.clear_debug_images()
    image = io.imread("../resources/mülltonne.jpeg")
    hsv_image = rgb_to_hsv(image)

    position_mask = get_sign_position_mask(hsv_image)
    clean_mask = morphologisch_opening(position_mask,3)
    labels = sequential_region_labeling(clean_mask)

    # fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # axes = axes.ravel()
    # plot_image(axes[0], position_mask, "Saettigungsmaske vor Opening", vmin=0, vmax=1)
    # plot_image(axes[1], clean_mask, "Saettigungsmaske nach Opening", vmin=0, vmax=1)
    # plot_image(axes[2], labels, "Region Labeling", cmap="nipy_spectral")
    # plt.tight_layout()
    # plt.show()

    top_candidates = select_top_sign_candidates(labels, top_n=3)
    selected_candidate = choose_best_colored_shape_candidate(hsv_image, top_candidates)

    if selected_candidate is None:
        plot_debug_grid(
            image,
            hsv_image,
            position_mask,
            labels,
            np.zeros_like(labels, dtype=np.uint8)
        )
        return

    best_candidate = selected_candidate["mask"]
    best_color = selected_candidate["color"]

    print("Beste Schildfarbe:", best_color)

    if best_color == "yellow":
        diamond_candidate = build_yellow_diamond_candidate(hsv_image)

        if diamond_candidate is not None:
            # plt.imshow(diamond_candidate, cmap="gray", vmin=0, vmax=1)
            # plt.title("Diamond Sonderbehandlung Kandidat")
            # plt.axis("off")
            # plt.show()

            best_candidate = diamond_candidate

    # plt.imshow(best_candidate, cmap="gray", vmin=0, vmax=1)
    # plt.title("Ausgewählter Schild-Kandidat")
    # plt.axis("off")
    # plt.show()

    type,score = sc.classify_sign(best_candidate, best_color)

    if type == "octagon":
        print(sc.identify_sign(type, None))
        plot_debug_grid(image, hsv_image, position_mask, labels, best_candidate)
        return

    symbol_mask = extract_dark_symbol_mask(image, best_candidate)
    # plt.imshow(symbol_mask, cmap="gray", vmin=0, vmax=1)
    # plt.title("Symbol Mask")
    # plt.axis("off")
    # plt.show()

    if type == "diamond":
        print(sc.classify_diamond_sign(symbol_mask, best_candidate))
        plot_debug_grid(image, hsv_image, position_mask, labels, best_candidate)
        return

    inner_symbol = sc.get_inner_Label(symbol_mask, type)
    # plt.imshow(inner_symbol, cmap="gray", vmin=0, vmax=1)
    # plt.title("Inner Symbol Mask")
    # plt.axis("off")
    # plt.show()
    print(sc.classify_inner_label(inner_symbol, type))
    plot_debug_grid(image, hsv_image, position_mask, labels, best_candidate)


if __name__ == "__main__":
    main()
