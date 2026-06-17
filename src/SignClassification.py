from skimage import io
import numpy as np
import template_data as td
from scipy.ndimage import binary_fill_holes
from imageprocessing import Scaling
import matplotlib.pyplot as plt


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

def split_labels(labels):
    outer_label = (labels == 2).astype(np.uint8)
    inner_label = (labels > 2).astype(np.uint8)

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

    # Innere Symbole
    axes[2].imshow(
        inner_label,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[2].set_title(
        f"Inner Labels – {inner_label.sum()} Pixel"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()