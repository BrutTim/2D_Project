from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import SignClassification as sc
from imageprocessing.Regionlabeling import sequential_region_labeling
from imageprocessing.MorphologischesOpening import morphologisch_opening


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
    image = io.imread("resources/Verbot-fuer-Fahrzeuge-ueber-3_8m.png")

    img_array = np.array(image)

    gray_img = make_gray_image(img_array)
    original = gray_img.astype(float)
    binary_image = make_binary_image(original)
    binary_image = morphologisch_opening(binary_image)
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