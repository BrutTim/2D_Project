from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "resources" / "templates" / "richtzeichen"


def template_path(code):
    filename = code.replace(".", "_").replace("-", "_") + ".png"
    return TEMPLATE_DIR / filename


# code, name, outer_type, image_path
RICHTZEICHEN_TEMPLATES = [
    ("301", "Vorfahrt", "triangle", template_path("301")),
    ("306", "Vorfahrtstrasse", "diamond", template_path("306")),
    ("307", "Ende der Vorfahrtstrasse", "diamond", template_path("307")),
    ("331.1", "Kraftfahrstrasse", "rectangle", template_path("331.1")),
    ("331.2", "Ende der Kraftfahrstrasse", "rectangle", template_path("331.2")),
]
