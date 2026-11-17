from collections import Counter
import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path


IGNORED_CONTEXT_TERMS = {
    "audio", "file", "sample", "sound", "wav", "mp3", "the", "and", "for",
    "with", "from", "und", "der", "die", "das", "ein", "eine", "mit", "von",
    "effect", "effects", "recording", "field", "library", "tag", "tags",
    "sfx", "fx",
}


class LearningStoreError(RuntimeError):
    pass


def _default_store_path():
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "Sound Library Tagger" / "learning.sqlite3"


def _normalize_tag(tag):
    return re.sub(r"\s+", " ", tag.replace("_", " ")).strip(" ,;\t\r\n").casefold()


def split_tags(value):
    return {
        _normalize_tag(tag)
        for tag in re.split(r"[,;\n]+", value or "")
        if _normalize_tag(tag)
    }


def _context_terms(filename, original_tags):
    text = "{} {}".format(Path(filename).stem.replace("_", " "), original_tags or "")
    terms = {
        term
        for term in re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        if len(term) > 1 and term not in IGNORED_CONTEXT_TERMS
    }
    return terms


def _context_similarity(first, second, term_weights):
    overlap = first.intersection(second)
    if not overlap:
        return 0.0
    numerator = sum(term_weights[term] ** 2 for term in overlap)
    first_norm = sum(term_weights[term] ** 2 for term in first) ** 0.5
    second_norm = sum(term_weights[term] ** 2 for term in second) ** 0.5
    return numerator / (first_norm * second_norm)


def _connect(db_path=None):
    path = Path(db_path) if db_path else _default_store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=10)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tag_feedback (
                context_terms TEXT NOT NULL,
                tag TEXT NOT NULL,
                polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
                count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (context_terms, tag, polarity)
            )
            """
        )
        return connection
    except (OSError, sqlite3.Error) as error:
        raise LearningStoreError(
            "Der lokale Lernspeicher konnte nicht geöffnet werden: {}".format(error)
        ) from error


def record_tag_feedback(items, db_path=None):
    feedback = []
    for filename, original_tags, previous_tags, updated_tags in items:
        terms = _context_terms(filename, original_tags)
        if not terms:
            continue
        context = json.dumps(sorted(terms), ensure_ascii=False)
        previous = split_tags(previous_tags)
        updated = split_tags(updated_tags)
        feedback.extend(
            (context, tag, 1)
            for tag in updated - previous
        )
        feedback.extend(
            (context, tag, -1)
            for tag in previous - updated
        )

    if not feedback:
        return 0

    connection = _connect(db_path)
    try:
        with connection:
            connection.executemany(
                """
                INSERT INTO tag_feedback (context_terms, tag, polarity, count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT (context_terms, tag, polarity)
                DO UPDATE SET count = MIN(tag_feedback.count + 1, 8)
                """,
                feedback,
            )
    except sqlite3.Error as error:
        raise LearningStoreError(
            "Nutzerkorrekturen konnten nicht gespeichert werden: {}".format(error)
        ) from error
    finally:
        connection.close()
    return len(feedback)


def get_relevant_feedback(filename, original_tags="", db_path=None, limit=8):
    terms = _context_terms(filename, original_tags)
    if not terms:
        return {"prefer": [], "avoid": []}

    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT context_terms, tag, polarity, count FROM tag_feedback"
        ).fetchall()
    except sqlite3.Error as error:
        raise LearningStoreError(
            "Nutzerkorrekturen konnten nicht gelesen werden: {}".format(error)
        ) from error
    finally:
        connection.close()

    parsed_rows = []
    documents = set()
    for context_json, tag, polarity, count in rows:
        try:
            example_terms = frozenset(
                term
                for term in json.loads(context_json)
                if isinstance(term, str) and term not in IGNORED_CONTEXT_TERMS
            )
        except (json.JSONDecodeError, TypeError) as error:
            raise LearningStoreError(
                "Der lokale Lernspeicher enthält ungültige Kontextdaten."
            ) from error
        if not example_terms:
            continue
        parsed_rows.append((example_terms, tag, polarity, count))
        documents.add(example_terms)

    document_frequency = Counter(
        term for document in documents for term in document
    )
    document_count = len(documents)
    term_weights = {
        term: math.log(
            (document_count + 1) / (document_frequency.get(term, 0) + 1)
        )
        + 1
        for term in terms.union(document_frequency)
    }

    scores = {}
    for example_terms, tag, polarity, count in parsed_rows:
        similarity = _context_similarity(terms, example_terms, term_weights)
        if similarity < 0.3:
            continue
        strength = min(1.5, 0.6 + 0.15 * count)
        evidence = similarity * strength
        net, total = scores.setdefault(tag, [0.0, 0.0])
        scores[tag] = [net + polarity * evidence, total + evidence]

    ranked = {"prefer": [], "avoid": []}
    for tag, (net, total) in scores.items():
        confidence = abs(net) / total if total else 0
        if abs(net) < 0.3 or confidence < 0.5:
            continue
        ranked["prefer" if net > 0 else "avoid"].append((abs(net), tag))

    return {
        key: [
            tag
            for _, tag in sorted(values, key=lambda item: (-item[0], item[1]))[:limit]
        ]
        for key, values in ranked.items()
    }
