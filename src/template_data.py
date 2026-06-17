import numpy as np
from skimage.draw import polygon, disk


def get_triangle_template():
    # Dreieck
    triangle_template = np.zeros((128, 128), dtype=np.uint8)

    # Eckpunkte: oben, unten links, unten rechts
    rows = np.array([5, 122, 122])
    cols = np.array([64, 5, 122])

    rr, cc = polygon(rows, cols, triangle_template.shape)
    triangle_template[rr, cc] = 1

    return triangle_template

def get_diamond_template():
    diamond_template = np.zeros((128, 128), dtype=np.uint8)

    diamond_rows = np.array([5, 64, 122, 64])
    diamond_cols = np.array([64, 122, 64, 5])

    rr, cc = polygon(
        diamond_rows,
        diamond_cols,
        diamond_template.shape
    )

    diamond_template[rr, cc] = 1
    return diamond_template

def get_circle_template():

    circle_template = np.zeros((128, 128), dtype=np.uint8)
    rr, cc = disk(
        center=(64, 64),
        radius=59,
        shape=circle_template.shape
    )

    circle_template[rr, cc] = 1
    return circle_template

def get_octagon_template():
    octagon_template = np.zeros(
        (128, 128),
        dtype=np.uint8
    )

    octagon_rows = np.array([
        5, 5, 39, 88,
        122, 122, 88, 39
    ])

    octagon_cols = np.array([
        39, 88, 122, 122,
        88, 39, 5, 5
    ])

    rr, cc = polygon(
        octagon_rows,
        octagon_cols,
        octagon_template.shape
    )

    octagon_template[rr, cc] = 1

    return octagon_template