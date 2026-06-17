# 2D_Project

1 Schild im Gesamtbild finden:
- Farbe, geschlossene Konturen und geometrische Formen verwenden.

2 Perspektive korrigieren (optional)
- Ein schräg fotografiertes Schild in eine standardisierte Frontalansicht umwandeln.

3 In Grauwert oder Binärbild umwandeln
- Beleuchtung mit adaptivem Thresholding ausgleichen.

4 Außenkontur erkennen
- Zum Beispiel Dreieck, Kreis oder Rechteck.

5 Innere Regionen erkennen
- Etwa einen Pfeil oder ein anderes Symbol als zweite Region.

6 Schild klassifizieren
- Das normalisierte Symbol mit Referenzvorlagen oder Formmerkmalen vergleichen.

Zur Schild klassifizieren

- Nicht jedes Foto, aber normalerweise jeder Schildtyp einmal. Dafür gibt es mehrere Möglichkeiten:
Template Matching: Ein sauberes Referenzbild pro Schildtyp speichern und mit dem normalisierten Schild vergleichen.
Formmerkmale: Anzahl der Ecken, Flächenverhältnis, Rundheit, Hu-Momente und Konturhierarchie vergleichen.

Zur Schild im Gesamtbild finden

Zuerst über typische Schildfarben Kandidatenregionen finden,
nur geschlossene Konturen einer passenden Größe untersuchen,
geometrische Formen erkennen,
gefundene Schilder ausschneiden,
Perspektive und Größe normalisieren,
erst danach das innere Symbol analysieren.
