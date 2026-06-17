import numpy as np
from skimage.draw import polygon


def get_triangle_template():
    # Dreieck
    triangle_template = np.zeros((128, 128), dtype=np.uint8)

    # Eckpunkte: oben, unten links, unten rechts
    rows = np.array([5, 122, 122])
    cols = np.array([64, 5, 122])

    rr, cc = polygon(rows, cols, triangle_template.shape)
    triangle_template[rr, cc] = 1

    return triangle_template