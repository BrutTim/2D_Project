from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "resources" / "templates" / "vorschriftzeichen"


def template_path(code):
    filename = code.replace(".", "_").replace("-", "_") + ".png"
    return TEMPLATE_DIR / filename


# code, name, outer_type, image_path
# Nur Templates eintragen, die im aktiven Template-Ordner vorhanden und
# als Einzelbild brauchbar sind. Entfernte/kaputte Templates liegen im Backup.
VORSCHRIFTZEICHEN_TEMPLATES = [
    ("201", "Andreaskreuz, dem Schienenverkehr Vorrang gewaehren", "cross", template_path("201")),
    ("205", "Vorfahrt gewaehren", "triangle", template_path("205")),
    ("206", "Halt. Vorfahrt gewaehren", "octagon", template_path("206")),
    ("208", "Vorrang des Gegenverkehrs", "circle", template_path("208")),
    ("209", "Vorgeschriebene Fahrtrichtung rechts", "circle", template_path("209")),
    ("209-30", "Vorgeschriebene Fahrtrichtung geradeaus", "circle", template_path("209-30")),
    ("211", "Vorgeschriebene Fahrtrichtung hier rechts", "circle", template_path("211")),
    ("214", "Vorgeschriebene Fahrtrichtung geradeaus oder rechts", "circle", template_path("214")),
    ("215", "Kreisverkehr", "circle", template_path("215")),
    ("220", "Einbahnstrasse", "rectangle", template_path("220")),
    ("222", "Vorgeschriebene Vorbeifahrt rechts vorbei", "circle", template_path("222")),
    ("223.1", "Seitenstreifen befahren", "rectangle", template_path("223.1")),
    ("223.2", "Seitenstreifen nicht mehr befahren", "rectangle", template_path("223.2")),
    ("223.3", "Seitenstreifen raeumen", "rectangle", template_path("223.3")),
    ("224", "Haltestelle Linienverkehr und Schulbusse", "rectangle", template_path("224")),
    ("229", "Taxenstand", "rectangle", template_path("229")),
    ("230", "Ladebereich", "rectangle", template_path("230")),
    ("237", "Radweg", "circle", template_path("237")),
    ("238", "Reitweg", "circle", template_path("238")),
    ("259", "Verbot fuer Fussgaenger", "circle", template_path("259")),
    ("260", "Verbot fuer Kraftfahrzeuge", "circle", template_path("260")),
    ("261", "Verbot fuer kennzeichnungspflichtige Kraftfahrzeuge mit gefaehrlichen Guetern", "circle", template_path("261")),
    ("262", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Masse", "circle", template_path("262")),
    ("263", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Achslast", "circle", template_path("263")),
    ("264", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Breite", "circle", template_path("264")),
    ("265", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Hoehe", "circle", template_path("265")),
    ("266", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Laenge", "circle", template_path("266")),
    ("280", "Ende des Ueberholverbots fuer Kraftfahrzeuge aller Art", "circle", template_path("280")),
    ("281", "Ende des Ueberholverbots fuer Kraftfahrzeuge ueber 3,5 t", "circle", template_path("281")),
    ("282", "Ueberholverbots fuer Kraftfahrzeuge ueber 3,5 t", "circle", template_path("282")),
    ("283", "Tempolimit 30", "circle", template_path("283")),
    ("284", "Autobahn", "square", template_path("284")),

]
