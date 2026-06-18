import numpy as np
import matplotlib.pyplot as plt

def show_rgb_to_hsv(hsv, image):

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(h, cmap="hsv", vmin=0, vmax=1)
    axes[0, 1].set_title("Hue / Farbton")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(s, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Saturation / Sättigung")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(v, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Value / Helligkeit")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
    pass

def rgb_to_hsv(image):
    orig_image = image
    image = np.asarray(image, dtype=float)

    # Falls das Bild einen Alpha-Kanal hat, wird dieser entfernt
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    c_max = 255.0

    c_high = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta_c = c_high - c_min

    h = np.zeros_like(c_high)
    s = np.zeros_like(c_high)
    v = np.zeros_like(c_high)

    v = c_high / c_max

    # Saturation: S = Delta C / Chigh, falls Chigh > 0
    non_black = c_high > 0
    s[non_black] = delta_c[non_black] / c_high[non_black]

    chromatic = delta_c > 0

    r_rel = np.zeros_like(r)
    g_rel = np.zeros_like(g)
    b_rel = np.zeros_like(b)

    r_rel[chromatic] = (c_high[chromatic] - r[chromatic]) / delta_c[chromatic]
    g_rel[chromatic] = (c_high[chromatic] - g[chromatic]) / delta_c[chromatic]
    b_rel[chromatic] = (c_high[chromatic] - b[chromatic]) / delta_c[chromatic]

    red_is_max = chromatic & (r == c_high)
    green_is_max = chromatic & (g == c_high)
    blue_is_max = chromatic & (b == c_high)

    h_dash = np.zeros_like(h)

    h_dash[red_is_max] = b_rel[red_is_max] - g_rel[red_is_max]
    h_dash[green_is_max] = r_rel[green_is_max] - b_rel[green_is_max] + 2
    h_dash[blue_is_max] = g_rel[blue_is_max] - r_rel[blue_is_max] + 4

    h[chromatic] = h_dash[chromatic] / 6.0
    h[h < 0] += 1.0

    hsv = np.stack((h, s, v), axis=2)

    show_rgb_to_hsv(hsv, orig_image)
    return hsv