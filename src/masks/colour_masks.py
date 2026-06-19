import numpy as np


def split_hsv(hsv):

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    return h, s, v

def get_red_mask(h, s, v):

    red_mask = (
        ((h <= 0.04) | (h >= 0.95)) &
        (s >= 0.35) &
        (v >= 0.20)
    )
    return red_mask.astype(np.uint8)

def get_yellow_mask(h, s, v):

    yellow_mask = (
        (h >= 0.10) & (h <= 0.18) &
        (s >= 0.30) &
        (v >= 0.30)
    )

    return yellow_mask.astype(np.uint8)

def get_blue_mask(h, s, v):

    blue_mask = (
            (h >= 0.55) & (h <= 0.72) &
            (s >= 0.50) &
            (v >= 0.20) & (v <= 0.90)
    )

    return blue_mask.astype(np.uint8)

def get_saturation_mask(h, s, v):

    saturation_mask = (
        (s >= 0.45) &
        (v >= 0.20)
    )

    return saturation_mask.astype(np.uint8)

def get_color_mask(hsv):

    h, s, v = split_hsv(hsv)

    return get_red_mask(h, s, v), get_yellow_mask(h, s, v), get_blue_mask(h, s, v)

def get_sign_position_mask(hsv):

    h, s, v = split_hsv(hsv)

    return get_saturation_mask(h, s, v)
