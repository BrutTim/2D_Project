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
    ("239", "Gehweg", "circle", template_path("239")),
    ("240", "Gemeinsamer Geh- und Radweg", "circle", template_path("240")),
    ("241", "Getrennter Rad- und Gehweg", "circle", template_path("241")),
    ("242.1", "Beginn einer Fussgaengerzone", "rectangle", template_path("242.1")),
    ("242.2", "Ende einer Fussgaengerzone", "rectangle", template_path("242.2")),
    ("244.1", "Beginn einer Fahrradstrasse", "rectangle", template_path("244.1")),
    ("244.2", "Ende einer Fahrradstrasse", "rectangle", template_path("244.2")),
    ("244.3", "Beginn einer Fahrradzone", "rectangle", template_path("244.3")),
    ("244.4", "Ende einer Fahrradzone", "rectangle", template_path("244.4")),
    ("245", "Bussonderfahrstreifen", "circle", template_path("245")),
    ("250", "Verbot fuer Fahrzeuge aller Art", "circle", template_path("250")),
    ("251", "Verbot fuer Kraftwagen und sonstige mehrspurige Kraftfahrzeuge", "circle", template_path("251")),
    ("253", "Verbot fuer Kraftfahrzeuge ueber 3,5 t", "circle", template_path("253")),
    ("254", "Verbot fuer Radverkehr", "circle", template_path("254")),
    ("255", "Verbot fuer Kraftraeder", "circle", template_path("255")),
    ("257-50", "Verbot fuer Mofas", "circle", template_path("257-50")),
    ("257-51", "Verbot fuer Reiter", "circle", template_path("257-51")),
    ("259", "Verbot fuer Fussgaenger", "circle", template_path("259")),
    ("260", "Verbot fuer Kraftfahrzeuge", "circle", template_path("260")),
    ("261", "Verbot fuer kennzeichnungspflichtige Kraftfahrzeuge mit gefaehrlichen Guetern", "circle", template_path("261")),
    ("262", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Masse", "circle", template_path("262")),
    ("263", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Achslast", "circle", template_path("263")),
    ("264", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Breite", "circle", template_path("264")),
    ("265", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Hoehe", "circle", template_path("265")),
    ("266", "Verbot fuer Fahrzeuge ueber angegebener tatsaechlicher Laenge", "circle", template_path("266")),
    ("267", "Verbot der Einfahrt", "circle", template_path("267")),
    ("268", "Schneeketten vorgeschrieben", "circle", template_path("268")),
    ("269", "Verbot fuer Fahrzeuge mit wassergefaehrdender Ladung", "circle", template_path("269")),
    ("270.1", "Beginn einer Verkehrsverbotszone zur Verminderung schaedlicher Luftverunreinigungen", "rectangle", template_path("270.1")),
    ("270.2", "Ende einer Verkehrsverbotszone zur Verminderung schaedlicher Luftverunreinigungen", "rectangle", template_path("270.2")),
    ("272", "Verbot des Wendens", "circle", template_path("272")),
    ("273", "Verbot des Unterschreitens des angegebenen Mindestabstands", "circle", template_path("273")),
    ("274", "Zulaessige Hoechstgeschwindigkeit", "circle", template_path("274")),
    ("274.1", "Beginn einer Tempo-30-Zone", "rectangle", template_path("274.1")),
    ("274.2", "Ende einer Tempo-30-Zone", "rectangle", template_path("274.2")),
    ("275", "Vorgeschriebene Mindestgeschwindigkeit", "circle", template_path("275")),
    ("276", "Ueberholverbot fuer Kraftfahrzeuge aller Art", "circle", template_path("276")),
    ("277", "Ueberholverbot fuer Kraftfahrzeuge ueber 3,5 t", "circle", template_path("277")),
    ("277.1", "Verbot des Ueberholens von einspurigen Fahrzeugen", "circle", template_path("277.1")),
    ("278", "Ende der zulaessigen Hoechstgeschwindigkeit", "circle", template_path("278")),
    ("279", "Ende der vorgeschriebenen Mindestgeschwindigkeit", "circle", template_path("279")),
    ("280", "Ende des Ueberholverbots fuer Kraftfahrzeuge aller Art", "circle", template_path("280")),
    ("281", "Ende des Ueberholverbots fuer Kraftfahrzeuge ueber 3,5 t", "circle", template_path("281")),
    ("281.1", "Ende des Verbots des Ueberholens von einspurigen Fahrzeugen", "circle", template_path("281.1")),
    ("282", "Ende saemtlicher streckenbezogener Geschwindigkeitsbeschraenkungen und Ueberholverbote", "circle", template_path("282")),
    ("283", "Absolutes Haltverbot", "circle", template_path("283")),
    ("286", "Eingeschraenktes Haltverbot", "circle", template_path("286")),
    ("290.1", "Beginn eines eingeschraenkten Haltverbots fuer eine Zone", "rectangle", template_path("290.1")),
    ("290.2", "Ende eines eingeschraenkten Haltverbots fuer eine Zone", "rectangle", template_path("290.2")),
]
