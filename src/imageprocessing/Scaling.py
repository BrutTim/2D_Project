import numpy as np
from skimage.transform import resize


def normalize_image(labels, size=128):
    mask = np.asarray(labels, dtype=bool)

    # Koordinaten aller Vordergrundpixel
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.zeros((size, size), dtype=bool)

    # Leeren Rand entfernen
    cropped = mask[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    height, width = cropped.shape

    # Quadratische Fläche erzeugen, damit nichts verzerrt wird
    side = max(height, width)
    canvas = np.zeros((side, side), dtype=np.uint8)

    # Objekt mittig einsetzen
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2

    canvas[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = cropped

    # Auf die gewünschte Größe skalieren
    normalized = resize(
        canvas,
        (size, size),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )

    return normalized > 0.5