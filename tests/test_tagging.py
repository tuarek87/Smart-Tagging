import json
import queue
import unittest
import tempfile
import wave
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from ai import (
    OllamaError,
    _load_ucs_catalog,
    _match_ucs_category,
    _is_contradictory_door_state_tag,
    _is_unrelated_bullet_tag,
    _is_unrelated_stair_tag,
    pull_ollama_model,
    suggest_semantic_tags,
    suggest_semantic_tags_batch,
)
from app import (
    TaggingApp,
    combine_tags,
    read_comment,
    suggest_filename_tags,
    write_comment,
)


def set_ollama_responses(urlopen, generation_body):
    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.body).encode("utf-8")

    def respond(request, timeout):
        return Response(generation_body)

    urlopen.side_effect = respond


class OllamaModelUpdateTests(unittest.TestCase):
    @patch("ai.urlopen")
    def test_pull_updates_model_and_reports_progress(self, urlopen):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter(
                    [
                        b'{"status":"pulling manifest"}\n',
                        b'{"status":"success"}\n',
                    ]
                )

        urlopen.return_value = Response()
        progress = Mock()

        pull_ollama_model("qwen3:4b", progress)

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/pull")
        self.assertEqual(json.loads(request.data), {"name": "qwen3:4b"})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 30)
        self.assertEqual(
            [call.args[0] for call in progress.call_args_list],
            ["pulling manifest", "success"],
        )

    @patch("ai.urlopen")
    def test_pull_raises_ollama_error_when_model_update_fails(self, urlopen):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([b'{"error":"model not found"}\n'])

        urlopen.return_value = Response()

        with self.assertRaisesRegex(OllamaError, "model not found"):
            pull_ollama_model("unknown:model")


class AudioMetadataTests(unittest.TestCase):
    def test_open_document_event_adds_and_selects_file(self):
        app = TaggingApp.__new__(TaggingApp)
        app.root = Mock()
        app.files = {"/sounds/door opening.wav": {}}
        app.tree = Mock()
        app._add_paths = Mock()
        app._generate_ai_tags = Mock()

        app._handle_open_document("file:///sounds/door%20opening.wav")

        callback, paths = app.root.after_idle.call_args.args
        self.assertEqual(paths, ("file:///sounds/door%20opening.wav",))
        callback(paths)
        app._add_paths.assert_called_once_with(["/sounds/door opening.wav"])
        app.tree.selection_set.assert_called_once_with(
            ["/sounds/door opening.wav"]
        )
        app.tree.focus.assert_called_once_with("/sounds/door opening.wav")
        app.tree.see.assert_called_once_with("/sounds/door opening.wav")
        app._generate_ai_tags.assert_called_once_with()

    def test_formats_player_time(self):
        self.assertEqual(TaggingApp._format_player_time(125.9), "2:05")
        self.assertEqual(TaggingApp._format_player_time(-1), "0:00")

    def test_programmatic_slider_update_does_not_recurse(self):
        app = TaggingApp.__new__(TaggingApp)
        app.player_path = "/tmp/sample.wav"
        app.player_duration = 20
        app.player_seeking = False
        app.player_scale_updating = False
        app.player_time_label = Mock()
        app.player_scale = Mock()
        app.player_scale.set.side_effect = lambda value: app._on_seek_change(value)

        app._update_player_time(5)

        app.player_scale.set.assert_called_once_with(5)
        app.player_time_label.configure.assert_called_once_with(text="0:05 / 0:20")

    def test_apply_tags_updates_selected_rows_and_explains_they_are_not_saved(self):
        app = TaggingApp.__new__(TaggingApp)
        app.tree = Mock()
        app.tree.selection.return_value = ["/sounds/one.wav", "/sounds/two.wav"]
        app.tags_input = Mock()
        app.tags_input.get.return_value = "soldier, ritter"
        app.status_label = Mock()
        app.files = {
            path: {"tags": "", "status": "KI-Vorschlag"}
            for path in app.tree.selection.return_value
        }

        app._apply_draft()

        for path in app.files:
            self.assertEqual(app.files[path]["tags"], "soldier, ritter")
            self.assertEqual(app.files[path]["status"], "Zu schreiben")
            app.tree.set.assert_any_call(path, "new_tags", "soldier, ritter")
            app.tree.set.assert_any_call(path, "status", "Zu schreiben")
        app.status_label.configure.assert_called_once_with(
            text="Tags auf 2 Datei(en) angewendet. Noch nicht gespeichert: "
            "Klicke auf „Tags in Dateien speichern…“."
        )

    @patch("app.record_tag_feedback", return_value=2)
    def test_apply_tags_saves_differences_as_local_learning_feedback(
        self, record_feedback
    ):
        path = "/sounds/pferd_satteln.wav"
        app = TaggingApp.__new__(TaggingApp)
        app.tree = Mock()
        app.tree.selection.return_value = [path]
        app.tags_input = Mock()
        app.tags_input.get.return_value = "horse, saddle"
        app.status_label = Mock()
        app.files = {
            path: {
                "original_tags": "pferd, sattel",
                "tags": "horse, stairs",
                "status": "KI-Vorschlag",
                "learnable": True,
            }
        }

        app._apply_draft()

        record_feedback.assert_called_once_with(
            [
                (
                    "pferd_satteln.wav",
                    "pferd, sattel",
                    "horse, stairs",
                    "horse, saddle",
                )
            ]
        )

    @patch("app.suggest_semantic_tags_batch")
    @patch("app.get_relevant_feedback")
    def test_ai_worker_passes_similar_context_feedback_to_tagging(
        self, get_feedback, suggest_tags
    ):
        app = TaggingApp.__new__(TaggingApp)
        app.ai_queue = queue.Queue()
        get_feedback.return_value = {"prefer": ["hoof"], "avoid": ["stairs"]}
        suggest_tags.return_value = {0: ["horse", "hoof"]}

        app._run_ai(
            [
                (
                    "/sounds/pferd_satteln.wav",
                    "pferd_satteln.wav",
                    "pferd, sattel",
                )
            ],
            "qwen3:4b",
        )

        suggest_tags.assert_called_once_with(
            [(0, "pferd_satteln.wav", "pferd, sattel")],
            "qwen3:4b",
            learning_preferences={
                0: {"prefer": ["hoof"], "avoid": ["stairs"]}
            },
        )
        self.assertEqual(app.ai_queue.get()[0], "success")
        self.assertEqual(app.ai_queue.get()[0], "progress")
        self.assertEqual(app.ai_queue.get()[0], "done")

    @patch("app.suggest_semantic_tags_batch")
    @patch("app.get_relevant_feedback")
    def test_ai_worker_uses_manual_tags_as_a_template_and_keeps_them(
        self, get_feedback, suggest_tags
    ):
        app = TaggingApp.__new__(TaggingApp)
        app.ai_queue = queue.Queue()
        get_feedback.return_value = {"prefer": [], "avoid": []}
        suggest_tags.return_value = {0: ["campfire", "bonfire"]}

        app._run_ai(
            [
                (
                    "/sounds/fire.wav",
                    "fire.wav",
                    "flame, Feuer",
                )
            ],
            "qwen3:4b",
            "campfire, Lagerfeuer",
        )

        suggest_tags.assert_called_once_with(
            [(0, "fire.wav", "flame, Feuer")],
            "qwen3:4b",
            learning_preferences={0: {"prefer": [], "avoid": []}},
            seed_tags={0: "campfire, Lagerfeuer"},
        )
        self.assertEqual(
            app.ai_queue.get(),
            (
                "success",
                "/sounds/fire.wav",
                ["campfire", "Lagerfeuer", "bonfire"],
                True,
            ),
        )

    @patch("app.messagebox.showerror")
    def test_empty_ai_suggestions_are_reported_without_a_popup(self, showerror):
        path = "/sounds/opaque_name.wav"
        app = TaggingApp.__new__(TaggingApp)
        app.ai_queue = queue.Queue()
        app.ai_queue.put(("success", path, []))
        app.ai_queue.put(("done",))
        app.ai_running = True
        app.ai_button = Mock()
        app.ai_errors = []
        app.ai_success_count = 0
        app.ai_no_tags_count = 0
        app.root = Mock()
        app.status_label = Mock()
        app.tree = Mock()
        app.files = {path: {"tags": "", "learnable": False}}

        app._poll_ai_results()

        self.assertEqual(app.files[path]["status"], "Keine neuen Tags")
        self.assertEqual(app.ai_no_tags_count, 1)
        app.status_label.configure.assert_called_once_with(
            text="Für 1 Datei(en) wurden keine neuen Tags vorgeschlagen."
        )
        showerror.assert_not_called()

    def test_extracts_original_tags_from_filename(self):
        self.assertEqual(
            suggest_filename_tags("DarkForest_Wind-loop_01.wav"),
            ["dark", "forest", "wind", "loop"],
        )

    def test_combines_new_tags_without_replacing_or_duplicating_originals(self):
        self.assertEqual(
            combine_tags("Elephant; wildlife", "animal, elephant"),
            "Elephant, wildlife, animal",
        )

    def test_replaces_underscores_in_original_and_new_tags(self):
        self.assertEqual(
            combine_tags("sound_effect, single_shot", "machine_gun"),
            "sound effect, single shot, machine gun",
        )

    def test_writes_and_reads_a_wav_comment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(44100)
                audio.writeframes(b"\x00\x00" * 100)

            write_comment(str(path), "Elephant, wildlife")
            write_comment(
                str(path),
                combine_tags(read_comment(str(path)), "animal, elephant"),
            )

            self.assertEqual(
                read_comment(str(path)),
                "Elephant, wildlife, animal",
            )


class SemanticTagTests(unittest.TestCase):
    def test_matches_confident_categories_from_the_ucs_catalog(self):
        catalog = _load_ucs_catalog()

        self.assertEqual(
            _match_ucs_category(["stone", "secret passage"], catalog),
            "DOORSton",
        )
        self.assertEqual(catalog["DOORSton"][:2], ("DOORS", "STONE"))
        self.assertIsNone(_match_ucs_category(["water", "plumbing"], catalog))
        self.assertIsNone(_match_ucs_category(["door", "open"], catalog))
        self.assertIsNone(
            _match_ucs_category(
                ["hit", "impact", "penetration", "strike", "body"],
                catalog,
                source_context="pfeile_treffer_körper.wav",
            )
        )
        self.assertNotIn("ADR", catalog)

    def test_interprets_door_opening_without_inventing_stairs(self):
        filename = "scheune_kirche_tür_alt_auf.wav"
        original_tags = "scheune, kirche, tür, alt, auf"

        self.assertTrue(
            _is_unrelated_stair_tag("stairs", filename, original_tags)
        )
        self.assertTrue(
            _is_unrelated_stair_tag("hoch", filename, original_tags)
        )
        self.assertFalse(
            _is_unrelated_stair_tag("stairs", "church_stairs.wav", "")
        )
        self.assertTrue(
            _is_contradictory_door_state_tag(
                "closed door", filename, original_tags
            )
        )
        self.assertTrue(
            _is_contradictory_door_state_tag(
                "door closing", filename, original_tags
            )
        )
        self.assertFalse(
            _is_contradictory_door_state_tag(
                "closed door", "door_closed.wav", ""
            )
        )
        self.assertTrue(
            _is_unrelated_bullet_tag(
                "bullet impact", "pfeile_treffer.wav", "pfeile, treffer"
            )
        )
        self.assertFalse(
            _is_unrelated_bullet_tag(
                "bullet impact", "gun_bullet.wav", "gun, bullet"
            )
        )

    @patch("ai.urlopen")
    def test_removes_overlapping_and_opposite_door_tags(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "door", "german": "Tür"},
                                    {"english": "open", "german": "offen"},
                                    {
                                        "english": "door opening",
                                        "german": "Türöffnung",
                                    },
                                    {
                                        "english": "closed door",
                                        "german": "geschlossene Tür",
                                    },
                                    {
                                        "english": "door closing",
                                        "german": "Tür schließen",
                                    },
                                ],
                            }
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        self.assertEqual(
            suggest_semantic_tags("door_auf.wav", original_tags="Tür, auf"),
            [
                "door",
                "open",
                "offen",
                "door opening",
                "türöffnung",
                "closed door",
                "geschlossene tür",
                "door closing",
                "tür schliessen",
            ],
        )

    @patch("ai.urlopen")
    def test_rejects_successful_empty_tags_when_model_repeats_existing_terms(
        self, urlopen
    ):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "aufsatteln", "german": "aufsatteln"},
                                    {"english": "pferd", "german": "pferd"},
                                    {"english": "aufsteigen", "german": "aufsteigen"},
                                ],
                            }
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        with self.assertRaisesRegex(OllamaError, "keine neuen Tags"):
            suggest_semantic_tags(
                "aufsatteln_pferd_aufsteigen.wav",
                original_tags="aufsatteln, pferd, aufsteigen",
            )
        self.assertEqual(
            suggest_semantic_tags_batch(
                [
                    (
                        0,
                        "aufsatteln_pferd_aufsteigen.wav",
                        "aufsatteln, pferd, aufsteigen",
                    )
                ]
            ),
            {0: []},
        )

    @patch("ai.urlopen")
    def test_requests_semantic_tags_and_removes_existing_duplicates(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "Elephant", "german": "Elefant"},
                                    {"english": "animal", "german": "Tier"},
                                    {
                                        "english": "wildlife_sound",
                                        "german": "Wildtierklang",
                                    },
                                    {"english": "wildlife", "german": "Wildtier"},
                                    {"english": "animal", "german": "Tier"},
                                    {"english": "to", "german": "zu"},
                                ],
                            }
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        tags = suggest_semantic_tags(
            "elephant.wav", original_tags="Elephant, field recording"
        )

        self.assertEqual(
            tags,
            [
                "elefant",
                "animal",
                "tier",
                "wildlife sound",
                "wildtierklang",
            ],
        )
        request = urlopen.call_args_list[0].args[0]
        payload = json.loads(request.data.decode("utf-8"))
        prompt = payload["messages"][1]["content"]
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertIn("elephant.wav", prompt)
        self.assertIn("field recording", prompt)
        self.assertIn("möglichst 10 bis 15 verschiedene", prompt)
        self.assertIn("Der Nutzer prüft und löscht unpassende Vorschläge selbst", prompt)
        self.assertIn("gebräuchliche, verständliche deutsche Übersetzung", prompt)
        self.assertIn("/no_think", payload["messages"][1]["content"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], 1200)

    @patch("ai.urlopen")
    def test_keeps_all_generated_tags_for_manual_review(
        self, urlopen
    ):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {
                                        "english": "closed door",
                                        "german": "geschlossene Tür",
                                    },
                                    {
                                        "english": "residence",
                                        "german": "Wohnsitz",
                                    },
                                    {
                                        "english": "sci-fi",
                                        "german": "Science-Fiction",
                                    },
                                ],
                            }
                        ]
                    }
                )
            }
        }
        set_ollama_responses(urlopen, body)

        tags = suggest_semantic_tags(
            "tür_zu_wohnung.wav", original_tags="tür, zu, wohnung"
        )

        self.assertEqual(
            tags,
            [
                "closed door",
                "geschlossene tür",
                "residence",
                "wohnsitz",
                "sci-fi",
                "science-fiction",
            ],
        )
        self.assertEqual(urlopen.call_count, 1)

    @patch("ai.urlopen")
    def test_removes_english_words_repeated_across_tags(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "heavy door", "german": "schwere Tür"},
                                    {
                                        "english": "heavy movement",
                                        "german": "schwere Bewegung",
                                    },
                                ],
                            }
                        ]
                    }
                )
            }
        }
        set_ollama_responses(urlopen, body)

        tags = suggest_semantic_tags("heavy_door_movement.wav")

        self.assertEqual(
            tags,
            ["heavy door", "schwere tür", "movement", "schwere bewegung"],
        )
        prompt = json.loads(urlopen.call_args.args[0].data)["messages"][1]["content"]
        self.assertIn("ein späteres heavy movement nur movement lauten", prompt)

    @patch("ai.urlopen")
    def test_prioritizes_source_translation_over_related_tags(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "liquor", "german": "Schnaps"},
                                    {"english": "alcohol", "german": "Alkohol"},
                                    {"english": "drinking", "german": "trinken"},
                                ],
                            }
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        tags = suggest_semantic_tags(
            "schnaps_raus_auf_einfüllen_trinken.wav",
            "schnaps, raus, auf, einfüllen, trinken",
        )

        self.assertIn("liquor", tags)
        self.assertNotIn("schnaps", tags)
        prompt = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))[
            "messages"
        ][1]["content"]
        self.assertIn("möglichst 10 bis 15 verschiedene", prompt)
        self.assertIn(
            "wiederhole aber vorhandene Tags nicht als neue deutsche Vorschläge",
            prompt,
        )
        self.assertIn("Gib nicht nur die Übersetzungen der Wörter", prompt)

    @patch("ai.urlopen")
    def test_keeps_natural_phrases_when_words_overlap(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "mud", "german": "Schlamm"},
                                    {
                                        "english": "slip on mud",
                                        "german": "rutschen im Schlamm",
                                    },
                                ],
                            }
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        tags = suggest_semantic_tags("person_slips_in_mud.wav")

        self.assertEqual(
            tags,
            ["mud", "schlamm", "slip on mud", "rutschen im schlamm"],
        )

    @patch("ai.urlopen")
    def test_does_not_repeat_existing_sicherung_words_in_suggestions(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "bottle", "german": "Flasche"},
                                    {"english": "unlock", "german": "Sicherung öffnen"},
                                    {"english": "safeguard", "german": "Sicherung"},
                                    {"english": "release", "german": "Sicherung ab"},
                                    {
                                        "english": "bottle security",
                                        "german": "Sicherung Sicherung",
                                    },
                                ],
                            }
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        tags = suggest_semantic_tags(
            "flasche_öffnen_sicherung_ab.wav",
            "flasche, öffnen, sicherung, ab",
        )

        self.assertNotIn("sicherung", tags)
        self.assertNotIn("sicherung sicherung", tags)
        self.assertEqual(len(tags), len(set(tags)))
        prompt = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))[
            "messages"
        ][1]["content"]
        self.assertIn("Erhalte vollständige natürliche Phrasen", prompt)

    @patch("ai.urlopen")
    def test_suggests_tags_for_multiple_files_in_one_request(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "animal", "german": "tier"}
                                ],
                            },
                            {
                                "id": 1,
                                "tags": [
                                    {"english": "rain", "german": "regen"}
                                ],
                            },
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        tags = suggest_semantic_tags_batch(
            [
                (0, "Elephant.wav", ""),
                (1, "Rain.wav", "weather"),
            ],
            learning_preferences={
                0: {"prefer": ["hoof"], "avoid": ["stairs"]}
            },
        )

        self.assertEqual(tags, {0: ["animal", "tier"], 1: ["rain", "regen"]})
        self.assertEqual(urlopen.call_count, 1)
        request = urlopen.call_args_list[0].args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["options"]["num_predict"], 1800)
        prompt = payload["messages"][1]["content"]
        self.assertIn('"prefer": ["hoof"]', prompt)
        self.assertIn('"avoid": ["stairs"]', prompt)

    @patch("ai.urlopen")
    def test_seed_tags_guide_suggestions_and_are_not_repeated(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {"english": "dragon", "german": "Drache"},
                                    {"english": "fire", "german": "Feuer"},
                                ],
                            }
                        ]
                    }
                )
            }
        }
        set_ollama_responses(urlopen, body)

        with patch("ai._match_ucs_category", return_value=None):
            tags = suggest_semantic_tags_batch(
                [(0, "dragon.wav", "")],
                seed_tags={0: "dragon"},
            )

        self.assertEqual(tags, {0: ["fire", "feuer"]})
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        prompt = payload["messages"][1]["content"]
        self.assertIn('"user_seed_tags": "dragon"', prompt)
        self.assertIn("thematische Vorlage", prompt)

    @patch("ai.urlopen")
    def test_recovers_when_model_returns_one_based_file_ids(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 1,
                                "tags": [
                                    {"english": "shield", "german": "Schild"}
                                ],
                            }
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        self.assertEqual(
            suggest_semantic_tags(
                "hit_schlag_schild_mittelalter.wav",
                "hit, schlag, schild, mittelalter",
            ),
            ["shield"],
        )

    @patch("ai.urlopen")
    def test_limits_generated_tags_to_fifteen_translated_concepts(self, urlopen):
        body = {
            "message": {
                "content": json.dumps(
                    {
                        "files": [
                            {
                                "id": 0,
                                "tags": [
                                    {
                                        "english": "englishword{}".format(i),
                                        "german": "deutschwort{}".format(i),
                                    }
                                    for i in range(15)
                                ],
                            }
                        ]
                    }
                )
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        tags = suggest_semantic_tags("sample.wav")

        self.assertEqual(len(tags), 30)
        self.assertEqual(sum(tag.startswith("englishword") for tag in tags), 15)
        self.assertEqual(sum(tag.startswith("deutschwort") for tag in tags), 15)
        self.assertFalse(any("_" in tag for tag in tags))

    @patch("ai.urlopen")
    def test_rejects_batch_response_missing_a_file(self, urlopen):
        body = {"message": {"content": json.dumps({"files": []})}}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        set_ollama_responses(urlopen, body)

        with self.assertRaisesRegex(OllamaError, "nicht für alle Dateien"):
            suggest_semantic_tags_batch([(0, "Elephant.wav", "")])

    @patch("ai.urlopen")
    def test_rejects_malformed_model_response(self, urlopen):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"message":{"content":"not json"}}'

        urlopen.return_value = Response()

        with self.assertRaisesRegex(OllamaError, "keine gültige Tag-Liste"):
            suggest_semantic_tags("elephant.wav")


if __name__ == "__main__":
    unittest.main()
