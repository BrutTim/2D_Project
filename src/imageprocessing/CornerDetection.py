import numpy as np

from imageprocessing.MorphologischesOpening import dilate, erode


def pad_image(image, pad_width, constant_value=0):
    image = np.asarray(image)

    padded_shape = (
        image.shape[0] + 2 * pad_width,
        image.shape[1] + 2 * pad_width
    )
    padded = np.full(padded_shape, constant_value, dtype=image.dtype)
    padded[
        pad_width:pad_width + image.shape[0],
        pad_width:pad_width + image.shape[1]
    ] = image

    return padded


def reflect_index(index, size):
    if index < 0:
        return -index

    if index >= size:
        return 2 * size - index - 2

    return index


def gaussian_kernel(sigma):
    radius = max(1, int(np.ceil(3 * sigma)))
    positions = np.arange(-radius, radius + 1)
    kernel = np.exp(-(positions ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def convolve_vertical(image, kernel):
    image = np.asarray(image, dtype=float)
    result = np.zeros_like(image, dtype=float)
    radius = len(kernel) // 2

    for y in range(image.shape[0]):
        for offset, weight in enumerate(kernel):
            source_y = reflect_index(y + offset - radius, image.shape[0])
            result[y, :] += weight * image[source_y, :]

    return result


def convolve_horizontal(image, kernel):
    image = np.asarray(image, dtype=float)
    result = np.zeros_like(image, dtype=float)
    radius = len(kernel) // 2

    for x in range(image.shape[1]):
        for offset, weight in enumerate(kernel):
            source_x = reflect_index(x + offset - radius, image.shape[1])
            result[:, x] += weight * image[:, source_x]

    return result


def gaussian_filter(image, sigma):
    kernel = gaussian_kernel(sigma)
    smoothed = convolve_vertical(image, kernel)
    return convolve_horizontal(smoothed, kernel)


def maximum_filter(image, size):
    image = np.asarray(image, dtype=float)
    result = np.full_like(image, -np.inf, dtype=float)
    radius = size // 2

    for y in range(image.shape[0]):
        y_min = max(0, y - radius)
        y_max = min(image.shape[0], y + radius + 1)

        for x in range(image.shape[1]):
            x_min = max(0, x - radius)
            x_max = min(image.shape[1], x + radius + 1)
            result[y, x] = image[y_min:y_max, x_min:x_max].max()

    return result


def dilate_binary(mask):
    gray_mask = np.where(mask, 0.0, -1000.0)
    return dilate(gray_mask) > -500.0


def erode_binary(mask):
    gray_mask = mask.astype(float)
    eroded = erode(gray_mask)
    return eroded > -2.5


def get_contour_band(mask):
    dilated = dilate_binary(mask)
    eroded = erode_binary(mask)
    return dilated & ~eroded


def count_corners(
    mask,
    alpha=0.05,
    pre_sigma=0.8,
    sigma=1.5,
    threshold_ratio=0.08,
    min_distance=6,
    padding=12
):
    binary = np.asarray(mask, dtype=bool)
    binary = pad_image(binary, padding, constant_value=False)

    image = gaussian_filter(binary.astype(float), sigma=pre_sigma)

    iy, ix = np.gradient(image)

    a = gaussian_filter(ix * ix, sigma=sigma)
    b = gaussian_filter(iy * iy, sigma=sigma)
    c = gaussian_filter(ix * iy, sigma=sigma)

    det_m = a * b - c * c
    trace_m = a + b
    q = det_m - alpha * (trace_m ** 2)

    contour_band = get_contour_band(binary)
    q = q * contour_band

    max_q = q.max()
    if max_q <= 0:
        return 0

    threshold = threshold_ratio * max_q

    neighborhood_size = 2 * min_distance + 1
    local_max = q == maximum_filter(q, neighborhood_size)

    candidates = np.argwhere((q > threshold) & local_max)
    if len(candidates) == 0:
        return 0

    candidates = sorted(
        candidates,
        key=lambda point: q[point[0], point[1]],
        reverse=True
    )

    corners = []

    for y, x in candidates:
        is_far_enough = True

        for corner_y, corner_x in corners:
            distance = np.sqrt((y - corner_y) ** 2 + (x - corner_x) ** 2)

            if distance < min_distance:
                is_far_enough = False
                break

        if is_far_enough:
            corners.append((y, x))

    return len(corners)
