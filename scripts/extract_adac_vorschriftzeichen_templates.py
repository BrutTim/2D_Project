from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDERED_DIR = PROJECT_ROOT / "tmp" / "pdfs" / "rendered"
OUTPUT_DIR = PROJECT_ROOT / "resources" / "templates" / "vorschriftzeichen"


PAGE_06_CODES = [
    ["201", "205", "206", "208"],
    ["209", "209-30", "211", "214"],
    ["215", "220", "222", "223.1"],
    ["223.2", "223.3", "224", "229"],
    ["230", "237", "238", "239"],
    ["240", "241", "242.1", "242.2"],
    ["244.1", "244.2", "244.3", "244.4"],
    ["245", "250", "251", "253"],
    ["254", "255", "257-50", "257-51"],
]

PAGE_08_CODES = [
    ["259", "260", "261", "262"],
    ["263", "264", "265", "266"],
    ["267", "268", "269", "270.1"],
    ["270.2", "272", "273", "274"],
    ["274.1", "274.2", "275", "276"],
    ["277", "277.1", "278", "279"],
    ["280", "281", "281.1", "282"],
    ["283", "286", "290.1", "290.2"],
]


def filename_for_code(code):
    return code.replace(".", "_").replace("-", "_") + ".png"


def component_bbox(mask):
    labels, count = ndimage.label(mask)
    boxes = []

    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)

        if len(xs) == 0:
            continue

        area = len(xs)
        y_min = ys.min()
        y_max = ys.max()
        x_min = xs.min()
        x_max = xs.max()
        height = y_max - y_min + 1
        width = x_max - x_min + 1

        if area < 30:
            continue

        boxes.append((x_min, y_min, x_max, y_max, width, height, area))

    return boxes


def crop_sign(cell):
    arr = np.asarray(cell.convert("RGB"))
    non_white = np.any(arr < 245, axis=2)
    color_mask = (
        (arr.max(axis=2) - arr.min(axis=2) > 35) &
        (arr.min(axis=2) < 235)
    )

    boxes = component_bbox(non_white)
    if not boxes:
        return cell

    color_boxes = component_bbox(color_mask)
    color_boxes = [box for box in color_boxes if box[6] > 80]

    colored_sign_bbox = None
    if color_boxes:
        colored_sign_bbox = (
            min(box[0] for box in color_boxes),
            min(box[1] for box in color_boxes),
            max(box[2] for box in color_boxes),
            max(box[3] for box in color_boxes),
        )

    # The sign number is a small black component at the top of each grid cell.
    filtered = []
    for box in boxes:
        _x_min, _y_min, _x_max, y_max, width, height, area = box
        looks_like_code_above_sign = y_max < 80 and height < 45 and area < 4000

        if looks_like_code_above_sign:
            continue

        if colored_sign_bbox is not None:
            x_min, y_min, x_max, y_max = box[:4]
            sx_min, sy_min, sx_max, sy_max = colored_sign_bbox
            margin = 18
            overlaps_colored_sign = not (
                x_max < sx_min - margin or
                x_min > sx_max + margin or
                y_max < sy_min - margin or
                y_min > sy_max + margin
            )

            if not overlaps_colored_sign:
                continue

        if width < 3 or height < 3:
            continue

        filtered.append(box)

    if not filtered:
        filtered = boxes

    x_min = min(box[0] for box in filtered)
    y_min = min(box[1] for box in filtered)
    x_max = max(box[2] for box in filtered)
    y_max = max(box[3] for box in filtered)

    padding = 12
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(cell.width - 1, x_max + padding)
    y_max = min(cell.height - 1, y_max + padding)

    cropped = cell.crop((x_min, y_min, x_max + 1, y_max + 1))

    side = max(cropped.width, cropped.height)
    square = Image.new("RGB", (side, side), "white")
    square.paste(
        cropped,
        ((side - cropped.width) // 2, (side - cropped.height) // 2)
    )

    return square.resize((256, 256), Image.Resampling.LANCZOS)


def extract_page(image_path, codes):
    image = Image.open(image_path).convert("RGB")

    x_edges = [110, 365, 625, 885, 1145]
    y_edges = np.linspace(170, 2210, len(codes) + 1).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for row_index, row in enumerate(codes):
        for col_index, code in enumerate(row):
            cell = image.crop((
                x_edges[col_index],
                y_edges[row_index],
                x_edges[col_index + 1],
                y_edges[row_index + 1],
            ))

            sign = crop_sign(cell)
            target = OUTPUT_DIR / filename_for_code(code)
            sign.save(target)
            print(target)


def main():
    extract_page(RENDERED_DIR / "vorschrift-06.png", PAGE_06_CODES)
    extract_page(RENDERED_DIR / "vorschrift-08.png", PAGE_08_CODES)


if __name__ == "__main__":
    main()
