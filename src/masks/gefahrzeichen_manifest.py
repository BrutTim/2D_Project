from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "resources" / "templates" / "gefahrzeichen"


def template_path(code):
    filename = code.replace(".", "_").replace("-", "_") + ".png"
    return TEMPLATE_DIR / filename


# code, name, outer_type, image_path
GEFAHRZEICHEN_TEMPLATES = [
    ("101", "Gefahrstelle", "triangle", template_path("101")),
    ("101-10", "Flugbetrieb", "triangle", template_path("101-10")),
    ("101-11", "Fussgaengerueberweg", "triangle", template_path("101-11")),
    ("101-12", "Viehtrieb, Tiere", "triangle", template_path("101-12")),
    ("101-15", "Steinschlag", "triangle", template_path("101-15")),
    ("101-51", "Schnee- oder Eisglaette", "triangle", template_path("101-51")),
    ("101-52", "Splitt, Schotter", "triangle", template_path("101-52")),
    ("101-53", "Ufer", "triangle", template_path("101-53")),
    ("101-54", "Unzureichendes Lichtraumprofil", "triangle", template_path("101-54")),
    ("101-55", "Bewegliche Bruecke", "triangle", template_path("101-55")),
    ("102", "Kreuzung oder Einmuendung mit Vorfahrt von rechts", "triangle", template_path("102")),
    ("103", "Kurve rechts", "triangle", template_path("103")),
    ("105", "Doppelkurve zunaechst rechts", "triangle", template_path("105")),
    ("108", "Gefaelle 10 Prozent", "triangle", template_path("108")),
    ("110", "Steigung 12 Prozent", "triangle", template_path("110")),
    ("112", "Unebene Fahrbahn", "triangle", template_path("112")),
    ("114", "Schleuder- oder Rutschgefahr", "triangle", template_path("114")),
    ("117", "Seitenwind", "triangle", template_path("117")),
    ("120", "Verengte Fahrbahn", "triangle", template_path("120")),
    ("121", "Einseitig rechts verengte Fahrbahn", "triangle", template_path("121")),
    ("123", "Arbeitsstelle", "triangle", template_path("123")),
    ("124", "Stau", "triangle", template_path("124")),
    ("125", "Gegenverkehr", "triangle", template_path("125")),
    ("131", "Lichtzeichenanlage", "triangle", template_path("131")),
    ("133", "Fussgaenger", "triangle", template_path("133")),
    ("136", "Kinder", "triangle", template_path("136")),
    ("138", "Radverkehr", "triangle", template_path("138")),
    ("142", "Wildwechsel", "triangle", template_path("142")),
    ("151", "Bahnuebergang", "triangle", template_path("151")),
    ("156", "Dreistreifige Bake etwa 240 m vor Bahnuebergang", "bake", template_path("156")),
    ("159", "Zweistreifige Bake etwa 160 m vor Bahnuebergang", "bake", template_path("159")),
    ("162", "Einstreifige Bake etwa 80 m vor Bahnuebergang", "bake", template_path("162")),
]
