import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.draw import polygon, disk
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes

from masks.vorschriftzeichen_manifest import VORSCHRIFTZEICHEN_TEMPLATES


def get_triangle_template():
    # Dreieck
    triangle_template = np.zeros((128, 128), dtype=np.uint8)

    # Eckpunkte: oben, unten links, unten rechts
    rows = np.array([5, 122, 122])
    cols = np.array([64, 5, 122])

    rr, cc = polygon(rows, cols, triangle_template.shape)
    triangle_template[rr, cc] = 1

    return triangle_template


def get_downwards_triangle_template():
    downwards_triangle_template = np.zeros((128, 128), dtype=np.uint8)

    rows = np.array([5, 5, 122])
    cols = np.array([5, 122, 64])

    rr, cc = polygon(rows, cols, downwards_triangle_template.shape)
    downwards_triangle_template[rr, cc] = 1

    return downwards_triangle_template



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

def get_square_template():
    square_template = np.zeros((128, 128), dtype=np.uint8)

    square_rows = np.array([5, 5, 122, 122])
    square_cols = np.array([5, 122, 122, 5])

    rr, cc = polygon(
        square_rows,
        square_cols,
        square_template.shape
    )

    square_template[rr, cc] = 1

    return square_template


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





def load_binary_template(path, size=128, threshold=0.8):
    image = io.imread(path)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[..., :3]
        image = rgb2gray(image)
    else:
        image = image.astype(float)
        if image.max() > 1:
            image = image / 255.0

    normalized = resize(
        image,
        (size, size),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )

    return (normalized < threshold).astype(np.uint8)


def load_inner_symbol_template(path, outer_type, size=128):
    image = io.imread(path)

    if image.ndim == 2:
        return load_binary_template(path, size=size)

    if image.shape[2] == 4:
        image = image[..., :3]

    image = image.astype(float)
    if image.max() > 1:
        image = image / 255.0

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]

    blue_sign = (
        (blue > 0.35) &
        (blue > red + 0.12) &
        (blue > green + 0.04)
    )

    red_sign = (
        (red > 0.45) &
        (red > green + 0.12) &
        (red > blue + 0.12)
    )

    dark_symbol = (
        (red < 0.25) &
        (green < 0.25) &
        (blue < 0.25)
    )

    filled_blue_area = binary_fill_holes(blue_sign)
    white_symbol = (
        (red > 0.82) &
        (green > 0.82) &
        (blue > 0.82)
    )
    white_symbol_on_blue = filled_blue_area & white_symbol

    if white_symbol_on_blue.sum() > 20:
        symbol = white_symbol_on_blue
    elif dark_symbol.sum() > 20:
        symbol = dark_symbol
    else:
        symbol = load_binary_template(path, size=size)

    normalized = resize(
        symbol.astype(np.uint8),
        (size, size),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )

    return (normalized > 0.5).astype(np.uint8)


def get_vorschriftzeichen_templates_for_type(outer_type):
    templates = []

    for _code, sign_name, template_outer_type, path in VORSCHRIFTZEICHEN_TEMPLATES:
        if template_outer_type != outer_type:
            continue

        if not path.exists():
            continue

        templates.append((
            load_inner_symbol_template(path, outer_type),
            sign_name
        ))

    return templates


def get_all_vorschriftzeichen_templates():
    templates = []

    for _code, sign_name, _outer_type, path in VORSCHRIFTZEICHEN_TEMPLATES:
        if not path.exists():
            continue

        templates.append((
            load_inner_symbol_template(path, _outer_type),
            sign_name
        ))

    return templates
