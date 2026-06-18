from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import SignClassification as sc
from imageprocessing.Regionlabeling import sequential_region_labeling
from imageprocessing.MorphologischesOpening import morphologisch_opening
from imageprocessing.RGBtoHSV import rgb_to_hsv
from imageprocessing.Scaling import normalize_image
from masks.colour_masks import get_color_mask


def print_label_stats(labels):
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = labels == label
        area = mask.sum()

        ys, xs = np.where(mask)
        height = ys.max() - ys.min() + 1
        width = xs.max() - xs.min() + 1

        aspect_ratio = width / height
        fill_ratio = area / (width * height)

        touches_border = (
            ys.min() == 0 or
            xs.min() == 0 or
            ys.max() == labels.shape[0] - 1 or
            xs.max() == labels.shape[1] - 1
        )

        print(
            f"Label {label}: "
            f"area={area}, "
            f"bbox={width}x{height}, "
            f"aspect={aspect_ratio:.2f}, "
            f"fill={fill_ratio:.2f}, "
            f"border={touches_border}"
        )
    pass


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


def select_best_sign_candidate(labels):
    best_label = None
    best_score = -1

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

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return np.zeros_like(labels, dtype=np.uint8)

    candidate = labels == best_label
    candidate = binary_fill_holes(candidate)

    print(
        f"Label {label}: "
        f"area={area}, "
        f"bbox={width}x{height}, "
        f"aspect={aspect_ratio:.2f}, "
        f"orig_fill={original_fill_ratio:.2f}, "
        f"filled_fill={filled_fill_ratio:.2f}, "
        f"bbox_ratio={bbox_area_ratio:.3f}, "
        f"border={touches_border}, "
        f"score={score:.1f}"
    )

    return candidate.astype(np.uint8)


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


def main():
    image = io.imread("resources/Vorgeschriebene-Fahrtrichtnug-geradeaus_IDEAl.jpg")
    hsv_image = rgb_to_hsv(image)

    masks = get_color_mask(hsv_image)

    timmi = []
    candidates = []
    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    i = 0
    for mask_name, mask in [
        ("red", masks[0]),
        ("yellow", masks[1]),
        ("blue", masks[2]),
    ]:
        axes[i,0].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i,0].set_title(f"Farbmaske" + str(mask_name) + "vor Opening")
        axes[i,0].axis("off")

        clean_mask = morphologisch_opening(mask)
        axes[i,1].imshow(clean_mask, cmap="gray", vmin=0, vmax=1)
        axes[i,1].set_title(f"Farbmaske" + str(mask_name) + " nach Opening")
        axes[i,1].axis("off")

        timmi.append(clean_mask)

        labels = sequential_region_labeling(clean_mask)
        axes[i, 2].imshow(labels, cmap="nipy_spectral")
        axes[i, 2].set_title("Region Labeling")
        axes[i, 2].axis("off")

        print_label_stats(labels)

        sign_candidate = select_best_sign_candidate(labels)
        candidates.append((mask_name, sign_candidate))
        i =+ 1


    best_color = None
    best_candidate = None
    best_score = 0

    for color, candidate in candidates:
        score = score_candidate(candidate)

        print(color, score)

        if score > best_score:
            best_score = score
            best_color = color
            best_candidate = candidate

    print("Beste Schildfarbe:", best_color)

    plt.imshow(best_candidate, cmap="gray", vmin=0, vmax=1)
    plt.title("Ausgewählter Schild-Kandidat")
    plt.axis("off")
    plt.show()

    type,score = sc.classify_sign(best_candidate, best_color)

    symbol_mask = extract_dark_symbol_mask(image, best_candidate)
    normalized_symbol = normalize_image(symbol_mask, 128)

    inner_label = sc.get_inner_Label(normalized_symbol)

    print(sc.classify_inner_label(inner_label, type))

    #plt.imshow(normalized_symbol, cmap="gray", vmin=0, vmax=1)
    #plt.title("Normalisiertes Symbol")
    #plt.axis("off")
    #plt.show()


if __name__ == "__main__":
    main()
