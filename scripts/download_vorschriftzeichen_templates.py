import argparse
import csv
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from masks.vorschriftzeichen_manifest import VORSCHRIFTZEICHEN_TEMPLATES


def read_url_map(csv_path):
    url_map = {}

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            code = row["code"].strip()
            url = row["url"].strip()
            if code and url:
                url_map[code] = url

    return url_map


def download_file(url, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request) as response:
        target_path.write_bytes(response.read())


def main():
    parser = argparse.ArgumentParser(
        description="Download template images for Vorschriftzeichen."
    )
    parser.add_argument(
        "csv",
        help="CSV with columns: code,url"
    )

    args = parser.parse_args()
    url_map = read_url_map(args.csv)

    missing = []

    for code, sign_name, _outer_type, path in VORSCHRIFTZEICHEN_TEMPLATES:
        url = url_map.get(code)

        if not url:
            missing.append(code)
            continue

        print(f"Downloading {code}: {sign_name}")
        download_file(url, path)

    if missing:
        print("No URL in CSV for:", ", ".join(missing))


if __name__ == "__main__":
    main()
