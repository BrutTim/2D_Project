from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import SignClassification as sc



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

def get_neighbour_labels(labels, x, y):
    neighbours = []

    # 4er-Nachbarschaft oder 8er Nachbarschaft aber nur bereits besuchte Pixel:
    for dy, dx in [(0, -1), (-1, -1), (-1, 0), (-1, 1)]:#[(0, -1), (-1, 0)]:
        ny = y + dy
        nx = x + dx

        if 0 <= ny < labels.shape[0] and 0 <= nx < labels.shape[1]:
            label = labels[ny, nx]
            if label > 1:
                neighbours.append(label)

    return neighbours


def merge_label_sets(label_sets, label_a, label_b):
    set_a = None
    set_b = None

    for label_set in label_sets:
        if label_a in label_set:
            set_a = label_set
        if label_b in label_set:
            set_b = label_set

    if set_a is not set_b:
        set_a.update(set_b)
        label_sets.remove(set_b)


def sequential_region_labeling(binary_image):
    labels = np.zeros(binary_image.shape, dtype=int)
    next_label = 2
    collisions = []

    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 0:
                continue

            neighbour_labels = get_neighbour_labels(labels, x, y)
            neighbour_labels = sorted(set(neighbour_labels))

            if len(neighbour_labels) == 0:
                labels[y, x] = next_label
                next_label += 1
            elif len(neighbour_labels) == 1:
                labels[y, x] = neighbour_labels[0]
            else:
                chosen_label = neighbour_labels[0]
                labels[y, x] = chosen_label

                for other_label in neighbour_labels[1:]:
                    collision = {chosen_label, other_label}
                    if collision not in collisions:
                        collisions.append(collision)

    label_sets = []
    for label in range(2, next_label):
        label_sets.append({label})

    for collision in collisions:
        label_a, label_b = list(collision)
        merge_label_sets(label_sets, label_a, label_b)

    label_map = {}
    for label_set in label_sets:
        new_label = min(label_set)
        for old_label in label_set:
            label_map[old_label] = new_label

    # 3. Durchlauf: alle aequivalenten Labels durch ein gemeinsames Label ersetzen
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if labels[y, x] > 1:
                labels[y, x] = label_map[labels[y, x]]

    return labels

def main():
    image = io.imread("resources/Vorfahrt_IDEAl.jpg")

    img_array = np.array(image)

    gray_img = make_gray_image(img_array)
    original = gray_img.astype(float)
    binary_image = make_binary_image(original)
    labels = sequential_region_labeling(binary_image)


    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(img_array, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Bild 01 vor Anpassung")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Bild 02 vor Anpassung")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(binary_image, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Bild 02 vor Anpassung")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(labels, cmap="nipy_spectral",
                      vmin=0, vmax=labels.max())
    axes[1, 0].set_title("Bild 02 vor Anpassung")
    axes[1, 0].axis("off")

    print(sc.classify_sign(labels))

    axes[1, 1].imshow(labels, cmap="nipy_spectral",
                      vmin=0, vmax=labels.max())
    axes[1, 1].set_title("Bild 02 vor Anpassung")
    axes[1, 1].axis("off")

    plt.show()

if __name__ == "__main__":
    main()