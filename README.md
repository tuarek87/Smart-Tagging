# Sound Library Tagger

Ein kleines, lokal laufendes macOS-Tool zum Prüfen und Schreiben von Kommentaren
in WAV- und MP3-Metadaten.

## Start

Auf macOS muss Python mit einem aktuellen Tcl/Tk installiert sein. Das
mitgelieferte Command-Line-Tools-Python verwendet auf manchen Systemen Tk 8.5
und kann beim Start der GUI abstürzen. Mit Homebrew:

```sh
brew install python-tk@3.11
brew install ollama
brew services start ollama
ollama pull qwen3:4b
python3.11 -m venv .venv-tk
source .venv-tk/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Falls bereits eine `.venv` mit dem Command-Line-Tools-Python angelegt wurde,
bleibt sie unverändert; verwende für die GUI stattdessen `.venv-tk`.

## macOS-App erstellen

Nach der Einrichtung von `.venv-tk` kannst du eine Finder-App erstellen:

```sh
bash build_app.sh
```

Das Bundle liegt anschließend unter `dist/Sound Library Tagger.app`. Ziehe es
zum Installieren in den Programme-Ordner. Der Build ist für die Architektur des
Macs gedacht, auf dem das Skript ausgeführt wird. Das App-Icon wird beim Build
aus `assets/sound-library-icon.svg` erzeugt. Ollama muss separat installiert
und gestartet sein; das Modell `qwen3:4b` muss ebenfalls verfügbar sein.
WAV- und MP3-Dateien können anschließend über „Öffnen mit…“ in die App geladen
werden. Bereits laufende Instanzen nehmen ebenfalls geöffnete Dateien entgegen.
In Keyboard Maestro kannst du dafür die Aktion „Execute Shell Script“ mit
folgendem Befehl verwenden:

```sh
open -a "/Applications/Sound Library Tagger.app" "$KMVAR_TriggerValue"
```

Dabei muss das App-Bundle im Programme-Ordner liegen. Der übergebene Pfad wird
in der Dateiliste ausgewählt und automatisch für KI-Tagvorschläge verarbeitet.
Die Audiodatei wird dabei nicht abgespielt oder verändert.

Ziehe Audiodateien oder einen Ordner in das Fenster oder wähle sie über die
Schaltflächen aus. Unterstützt werden WAV und MP3; Ordner werden rekursiv
durchsucht. „KI-Tags erzeugen“ analysiert Dateinamen und vorhandene Kommentare
mit dem lokalen Ollama-Modell und schlägt neue, semantisch verwandte Tags vor.
Über das Menü „Sprachmodell“ kannst du das eingetragene Ollama-Modell auf
Updates prüfen und bei Bedarf aktualisieren. Ollama lädt dafür gegebenenfalls
Modelldaten herunter; die Tagger-App bleibt geöffnet und muss nicht neu
gestartet werden.
Der UCS-Katalog war zuvor nur eine lose Prompt-Orientierung. Jetzt verwendet
die App die offizielle UCS-8.2.1-Kategorien-/Unterkategorienliste samt
englischen Synonymen aus `assets/ucs/ucs-8.2.1.tsv`, um die englischen
KI-Vorschläge gegen die echte Taxonomie abzugleichen. Nur bei klarer
Übereinstimmung ergänzt sie Kategorie und Unterkategorie als Tags;
Archivkategorien und unsichere Treffer werden ausgeschlossen. Die Daten
stammen aus den öffentlichen UCS-8.2.1-Ressourcen, die UCS als Public Domain
beschreibt (https://universalcategorysystem.com/).
Die KI zielt auf 10 bis 15 Suchkonzepte pro Datei, jeweils als
englisch-deutsches Begriffspaar. Sie übersetzt klare Begriffe aus dem
Dateinamen und den vorhandenen Tags und ergänzt zusätzliche thematisch
naheliegende Suchbegriffe. Ein paar weniger passende Vorschläge sind dabei
möglich und können vor dem Speichern manuell gelöscht werden. Die App filtert
solche Vorschläge nicht mehr anhand zusätzlicher Kontextregeln heraus;
bereinigt werden nur leere Tags und Wiederholungen. Deutsche Tags sollen
verständliche, gebräuchliche Übersetzungen sein. Bereits vorhandene Tags und
Duplikate werden entfernt; dadurch kann die sichtbare Anzahl unter zwanzig
einzelnen Sprach-Tags liegen. Sie liest zusammengesetzte Dateinamen im Kontext,
vermeidet exakte Wiederholungen und wiederholt englische Beschreibungswörter
nicht in mehreren Tags (aus `heavy movement` wird nach `heavy door` nur
`movement`), während wichtige Begriffe wie `door` wiederholt werden dürfen.
Deutsche Übersetzungen bleiben vollständig erhalten. Pro Suchkonzept soll sie
nur ein Tagpaar ausgeben.
Wenn für eine Datei keine neuen Tags übrig bleiben, zeigt die Tabelle
„Keine neuen Tags“ statt eines modalen Fehlerfensters; andere Dateien desselben
Batches werden trotzdem verarbeitet.
Objekte und Handlungen dürfen thematisch ergänzt werden; konkrete hörbare
Klangdetails kann die KI ohne Audioanalyse nicht erkennen. Tags mit
Unterstrichen werden als normale Wortgruppen angezeigt und gespeichert
(z. B. `machine_gun` als `machine gun`); Kommas trennen die Tags.
Eigennamen oder Begriffe ohne sinnvolle Übersetzung werden nicht künstlich
übertragen. Die KI verarbeitet bis zu drei Dateien pro Anfrage und nutzt Qwen3
im schnellen Nicht-Denkmodus. Der Prompt priorisiert zuerst die Übersetzung
wichtiger Quellbegriffe; vorhandene deutsche Tags sollen die englische
Entsprechung nicht verdrängen. Deutsche Übersetzungen werden als vollständige
Begriffe erhalten, statt vorhandene Wortbestandteile herauszukürzen.
Wenn du einen KI-Vorschlag bearbeitest und auf „Tags auf Auswahl anwenden“
klickst, merkt sich die App hinzugefügte und entfernte Tags lokal. Bei späteren
Vorschlägen für Dateien mit ähnlichen Namen und vorhandenen Tags fließen diese
Korrekturen als Vorlieben bzw. Ausschlüsse ein. Seltene, aussagekräftige
Kontextbegriffe zählen stärker als allgemeine Wörter; nur schwach ähnliche
Dateien erhalten die Korrektur daher nicht automatisch. Widersprüchliche
Korrekturen für dasselbe Tag werden zurückgehalten statt als feste Vorliebe
oder Ablehnung weitergegeben. Gespeichert werden nur die Kontextbegriffe und
Tag-Korrekturen, keine Audiodaten oder Dateipfade. Auf dem Mac liegt der
Lernspeicher unter
`~/Library/Application Support/Sound Library Tagger/learning.sqlite3`.
Die Tabelle zeigt bestehende Kommentare und neue KI-Tags getrennt. Wähle die
Dateien aus und bearbeite bei Bedarf die neuen Tags. „Tags auf Auswahl
anwenden“ übernimmt sie zunächst in die ausgewählten Zeilen, speichert sie aber
noch nicht. Erst nach Klick auf „Tags in Dateien speichern…“ und Bestätigung
werden die Tags als Kommentare in die Audiodateien geschrieben. Vorhandene
Tags bleiben erhalten und werden mit den neuen Tags kombiniert. Wenn die
Dateien bereits in Sononym geladen sind, aktualisiere dort anschließend die
Bibliothek. Mit „Tags mit KI ergänzen“ kannst du eigene Tags aus dem Eingabefeld
als thematische Vorlage verwenden: Die KI schlägt passende zusätzliche Tags vor
und übernimmt Vorlage sowie Ergänzungen in den Entwurf. Auch diese Änderungen
werden erst nach „Tags auf Auswahl anwenden“ und anschließendem Speichern in
die Audiodateien geschrieben.
Falls noch kein Kommentar vorhanden ist, zeigt die Spalte „Ursprüngliche Tags“
die aus dem Dateinamen erkannten Begriffe an.
Mit dem Player über der Tabelle kannst du die ausgewählte WAV- oder MP3-Datei
direkt in der App abspielen, stoppen und über den Schieberegler durchsuchen.
Per Rechtsklick auf einen Eintrag kannst du ihn im Finder anzeigen, direkt im
Player abspielen, Pfad oder Dateiname kopieren oder den Eintrag aus der Liste
entfernen. Dabei werden keine Audiodateien gelöscht.

Die Kommentare werden als ID3-`COMM`-Frame (englisch, Standardbeschreibung)
geschrieben. Nur Dateiname und bestehender Kommentar werden zur lokalen
Ollama-Instanz auf diesem Rechner gesendet; Audiodateien selbst werden nicht
hochgeladen oder akustisch analysiert. Das Modell muss vorher installiert sein
und Ollama unter `localhost:11434` laufen.

## Tests

```sh
python -m unittest discover -s tests
```
