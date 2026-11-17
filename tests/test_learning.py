import tempfile
import unittest
from pathlib import Path

from learning import get_relevant_feedback, record_tag_feedback


class LearningStoreTests(unittest.TestCase):
    def test_persists_additions_and_removals_for_similar_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "learning.sqlite3"
            record_tag_feedback(
                [
                    (
                        "pferd_satteln.wav",
                        "pferd, sattel",
                        "horse, saddle, stairs",
                        "horse, saddling, hoof",
                    )
                ],
                db_path=store_path,
            )

            feedback = get_relevant_feedback(
                "pferd_satteln_alt.wav",
                "pferd, sattel",
                db_path=store_path,
            )
            unrelated = get_relevant_feedback(
                "regen_wind.wav", "wetter", db_path=store_path
            )

        self.assertIn("saddling", feedback["prefer"])
        self.assertIn("hoof", feedback["prefer"])
        self.assertIn("stairs", feedback["avoid"])
        self.assertNotIn("stairs", unrelated["avoid"])

    def test_ignores_unchanged_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "learning.sqlite3"
            record_tag_feedback(
                [("door_open.wav", "door, open", "open, door", "door, open")],
                db_path=store_path,
            )

            feedback = get_relevant_feedback(
                "door_open.wav", "door, open", db_path=store_path
            )

        self.assertEqual(feedback, {"prefer": [], "avoid": []})

    def test_conflicting_feedback_is_not_repeated_as_a_preference_or_avoidance(self):
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "learning.sqlite3"
            example = ("door_hinge.wav", "door, hinge", "", "hinge")
            record_tag_feedback([example], db_path=store_path)
            record_tag_feedback(
                [
                    (
                        "door_hinge.wav",
                        "door, hinge",
                        "hinge",
                        "",
                    )
                ],
                db_path=store_path,
            )

            feedback = get_relevant_feedback(
                "door_hinge_close.wav",
                "door, hinge",
                db_path=store_path,
            )

        self.assertNotIn("hinge", feedback["prefer"])
        self.assertNotIn("hinge", feedback["avoid"])

    def test_rare_context_terms_outweigh_a_common_word(self):
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "learning.sqlite3"
            record_tag_feedback(
                [
                    (
                        "door_hinge_squeak.wav",
                        "door, hinge, squeak",
                        "",
                        "hinge creak",
                    )
                ],
                db_path=store_path,
            )
            unrelated_examples = [
                (
                    "door_car_{}.wav".format(index),
                    "door, car, vehicle{} ".format(index),
                    "",
                    "generic door tag",
                )
                for index in range(8)
            ]
            record_tag_feedback(unrelated_examples, db_path=store_path)

            related = get_relevant_feedback(
                "door_hinge_noise.wav",
                "door, hinge",
                db_path=store_path,
            )
            weak_match = get_relevant_feedback(
                "door_window.wav",
                "door, window",
                db_path=store_path,
            )

        self.assertIn("hinge creak", related["prefer"])
        self.assertNotIn("hinge creak", weak_match["prefer"])
        self.assertNotIn("generic door tag", weak_match["prefer"])


if __name__ == "__main__":
    unittest.main()
