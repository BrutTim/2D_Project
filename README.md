# Erkennung und Klassifikation von Verkehrsschildern

Dieses Projekt entstand im Rahmen des Moduls **2D Computer Vision**. Ziel ist es, Verkehrsschilder in Bildern mithilfe klassischer Bildverarbeitungsverfahren zu erkennen und anhand ihrer äußeren Form zu klassifizieren. Die Verarbeitung erfolgt schrittweise und macht die einzelnen Zwischenergebnisse sichtbar.

## Aktueller Stand

Alle im Ordner `resources` enthaltenen **Idealbilder** funktionieren mit der implementierten Verarbeitung. Dazu gehören die ideal aufgenommenen Varianten der folgenden Schildformen:

- Dreieck
- Raute
- Kreis
- Achteck

Damit können unter anderem Vorfahrt-gewähren-, Vorfahrtsstraßen-, Geradeaus- und Stoppschilder anhand ihrer äußeren Form unterschieden werden.

Die Bilder mit realen Außenaufnahmen dienen als anspruchsvollere Testfälle. Bei ihnen können Beleuchtung, Hintergrund, Perspektive und weitere Bildinhalte die Segmentierung beeinflussen. Die Erkennung des konkreten Schildinhalts beziehungsweise des inneren Symbols ist im aktuellen Stand noch nicht implementiert.

## Verwendete Bildverarbeitungsschritte

### 1. Grauwertumwandlung

Farbbilder werden zunächst mit einer gewichteten Kombination der RGB-Kanäle in Grauwertbilder umgewandelt:

`0,299 · Rot + 0,587 · Grün + 0,114 · Blau`

### 2. Binarisierung

Das Grauwertbild wird mit einem festen Schwellwert von `240` binarisiert. Dunklere Pixel werden als Vordergrund und hellere Pixel als Hintergrund behandelt.

### 3. Morphologisches Opening

Zur Reduzierung kleiner Störungen wird ein morphologisches Opening durchgeführt. Dieses besteht aus einer Erosion mit anschließender Dilatation. Verwendet wird ein `3 × 3` großes Strukturelement über zwei Iterationen.

### 4. Region Labeling

Die zusammenhängenden Vordergrundregionen werden mit einem selbst implementierten sequenziellen Region-Labeling-Verfahren markiert. Dabei wird die bereits untersuchte 8er-Nachbarschaft betrachtet. Auftretende Label-Kollisionen werden gespeichert und anschließend zusammengeführt.

### 5. Trennung von Außenform und inneren Regionen

Die gelabelten Regionen werden in die äußere Schildform und die inneren Symbole aufgeteilt. Die äußere Kontur wird aufgefüllt, damit für den späteren Vergleich eine geschlossene Form vorliegt.

### 6. Normalisierung

Die erkannte Schildform wird auf ihren belegten Bereich zugeschnitten, mittig in eine quadratische Fläche eingesetzt und auf `128 × 128` Pixel skaliert. Durch die quadratische Zwischenfläche bleibt das Seitenverhältnis erhalten.

### 7. Formklassifikation

Für die Klassifikation werden künstlich erzeugte Referenzvorlagen für Dreieck, Raute, Kreis und Achteck verwendet. Jede normalisierte Schildform wird mit diesen Vorlagen über die **Intersection over Union (IoU)** verglichen:

`IoU = Schnittfläche / Vereinigungsfläche`

Die Vorlage mit dem höchsten IoU-Wert bestimmt die erkannte Schildform.

## Projektstruktur

```text
2D_Project/
├── resources/                      Test- und Idealbilder
├── src/
│   ├── main.py                     Ablauf der Bildverarbeitung
│   ├── SignClassification.py       Aufteilung und Formklassifikation
│   ├── template_data.py            Referenzformen
│   └── imageprocessing/
│       ├── MorphologischesOpening.py
│       ├── Regionlabeling.py
│       └── Scaling.py
└── README.md
```

## Verwendete Bibliotheken

- Python
- NumPy
- scikit-image
- SciPy
- Matplotlib

## Ausführung

Das zu verarbeitende Bild wird in `src/main.py` ausgewählt. Anschließend kann das Programm aus dem Projektverzeichnis gestartet werden:

```bash
python src/main.py
```

Während der Ausführung werden die IoU-Werte der einzelnen Referenzformen sowie die am besten passende Form ausgegeben. Zusätzlich zeigt Matplotlib das Originalbild und die Zwischenergebnisse der Verarbeitung an.

## Mögliche Erweiterungen

- robustere Segmentierung für Außenaufnahmen
- automatische Erkennung des Schildes innerhalb größerer Szenen
- Perspektivkorrektur bei schräg aufgenommenen Schildern
- Klassifikation der inneren Symbole
- Kombination von Form-, Farb- und Konturmerkmalen
