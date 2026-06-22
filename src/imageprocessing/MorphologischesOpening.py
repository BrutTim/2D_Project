import numpy as np


filter = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.uint8)
iter_num = 3

def dilate(in_image):
    image = np.asarray(in_image, dtype=float)
    struct_filter = np.asarray(filter, dtype=float)
    out_image = image.copy()

    center_y = struct_filter.shape[0] // 2
    center_x = struct_filter.shape[1] // 2
    active_positions = np.argwhere(struct_filter != 0)

    for _ in range(iter_num):
        padded = np.pad(
            out_image,
            ((center_y, struct_filter.shape[0] - center_y - 1),
             (center_x, struct_filter.shape[1] - center_x - 1)),
            mode="constant",
            constant_values=-np.inf
        )
        next_image = np.full_like(out_image, -np.inf, dtype=float)

        for filter_y, filter_x in active_positions:
            image_region = padded[
                filter_y:filter_y + out_image.shape[0],
                filter_x:filter_x + out_image.shape[1]
            ]
            next_image = np.maximum(
                next_image,
                image_region + struct_filter[filter_y, filter_x]
            )

        out_image = next_image

    return out_image


def erode(in_image):
    image = np.asarray(in_image, dtype=float)
    struct_filter = np.asarray(filter, dtype=float)
    out_image = image.copy()

    center_y = struct_filter.shape[0] // 2
    center_x = struct_filter.shape[1] // 2
    active_positions = np.argwhere(struct_filter != 0)

    for _ in range(iter_num):
        padded = np.pad(
            out_image,
            ((center_y, struct_filter.shape[0] - center_y - 1),
             (center_x, struct_filter.shape[1] - center_x - 1)),
            mode="constant",
            constant_values=np.inf
        )
        next_image = np.full_like(out_image, np.inf, dtype=float)

        for filter_y, filter_x in active_positions:
            image_region = padded[
                filter_y:filter_y + out_image.shape[0],
                filter_x:filter_x + out_image.shape[1]
            ]
            next_image = np.minimum(
                next_image,
                image_region - struct_filter[filter_y, filter_x]
            )

        out_image = next_image

    return out_image

def morphologisch_opening(in_image):
    image = erode(in_image)
    return dilate(image)


