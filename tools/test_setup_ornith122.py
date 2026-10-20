"""Tests for setup.py's Ornith 1.5 122B family (docs/BACKENDS.md): the file layout of Unsloth's repository at the
pinned revision, the sizes' estimates, the default size and pin_of(), which reads a file's size and SHA-256 from the
Hub's LFS pointer.  Nothing here touches the network."""
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REPO = "unsloth/Ornith-1.5-122B-GGUF"
# size: (files, folder) as the repository lists them: one file in the root, or numbered files in a folder
LAYOUT = {
    "UD-IQ1_M": (["Ornith-1.5-122B-UD-IQ1_M.gguf"], ""),
    "UD-IQ2_XXS": (["Ornith-1.5-122B-UD-IQ2_XXS.gguf"], ""),
    "UD-Q2_K_XL": (["Ornith-1.5-122B-UD-Q2_K_XL.gguf"], ""),
    "UD-IQ3_XXS": (["Ornith-1.5-122B-UD-IQ3_XXS.gguf"], ""),
    "UD-Q3_K_XL": ([f"Ornith-1.5-122B-UD-Q3_K_XL-0000{i}-of-00003.gguf" for i in (1, 2, 3)], "UD-Q3_K_XL/"),
    "UD-Q4_K_XL": ([f"Ornith-1.5-122B-UD-Q4_K_XL-0000{i}-of-00003.gguf" for i in (1, 2, 3)], "UD-Q4_K_XL/"),
    "UD-Q5_K_XL": ([f"Ornith-1.5-122B-UD-Q5_K_XL-0000{i}-of-00003.gguf" for i in (1, 2, 3)], "UD-Q5_K_XL/"),
    "UD-Q6_K_XL": ([f"Ornith-1.5-122B-UD-Q6_K_XL-0000{i}-of-00004.gguf" for i in (1, 2, 3, 4)], "UD-Q6_K_XL/"),
    "Q8_0": ([f"Ornith-1.5-122B-Q8_0-0000{i}-of-00004.gguf" for i in (1, 2, 3, 4)], "Q8_0/"),
}
POINTER = (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"ab" * 32 + b"\nsize 12345\n")


class Layout(unittest.TestCase):
    fam = setup.FAMILIES["ornith122"]

    def test_the_sizes_are_the_repositorys_nine(self):
        self.assertEqual(sorted(setup.family_models("ornith122")), sorted(LAYOUT))
        self.assertEqual(self.fam["engine"], "poly")
        self.assertRegex(self.fam["hf"], rf"^https://huggingface.co/{REPO}/resolve/[0-9a-f]{{40}}/$")

    def test_file_names_shards_and_folders(self):
        for size, (files, folder) in LAYOUT.items():
            got = [setup.model_file(self.fam, size, i) for i in range(1, setup.model_shards(self.fam, size) + 1)]
            self.assertEqual(got, files, size)
            base = self.fam["models"][size].get("hf", self.fam["hf"])
            self.assertEqual(base, setup.hf(REPO) + folder, size)

    def test_estimates_hold_together(self):
        last = 0.0
        for size, d in setup.family_models("ornith122").items():
            self.assertGreater(d["download_gb"], last, size)                 # listed by size
            last = d["download_gb"]
            self.assertLess(d["arena_gb"], d["download_gb"], size)
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
            self.assertTrue(d["about"], size)

    def test_the_image_encoder_is_pinned(self):
        size, sha = self.fam["sha256"][self.fam["mmproj"]]
        self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertAlmostEqual(size / 1e9, self.fam["vision_gb"], delta=0.05)

    def test_the_tag_is_its_own(self):
        tags = [d["tag"] for d in setup.FAMILIES.values() if d["tag"]]
        self.assertEqual([b for b in tags if b.startswith(self.fam["tag"])], [self.fam["tag"]])
        self.assertEqual([b for b in tags if self.fam["tag"].startswith(b)], [self.fam["tag"]])

    def test_every_preferred_size_exists_once(self):
        self.assertEqual(sorted(self.fam["prefer"]), sorted(self.fam["models"]))


class Default(unittest.TestCase):
    sizes = setup.family_models("ornith122")

    def pick(self, ram, vram):
        fam = setup.FAMILIES["ornith122"]
        return setup.poly_default({m: self.sizes[m] for m in fam["prefer"]}, ram, vram)

    def test_ram_decides(self):
        self.assertEqual(self.pick(96, 16), "UD-Q4_K_XL")        # 96 GB takes 4 bits, not the 5 that wants all of it
        self.assertEqual(self.pick(128, 24), "UD-Q4_K_XL")
        self.assertEqual(self.pick(64, 16), "UD-Q3_K_XL")
        self.assertEqual(self.pick(48, 16), "UD-IQ3_XXS")
        self.assertEqual(self.pick(32, 16), "UD-IQ1_M")          # none fits: the one that wants the least


class Pins(unittest.TestCase):
    def reply(self, body):
        r = mock.MagicMock()
        r.__enter__.return_value = io.BytesIO(body)
        return r

    def test_pin_from_the_hubs_pointer(self):
        fam = {"hf": setup.hf(REPO), "hub_pins": True, "sha256": {}}
        with tempfile.TemporaryDirectory() as t, \
                mock.patch.object(setup.urllib.request, "urlopen", return_value=self.reply(POINTER)) as up:
            s = Path(t) / "Ornith-1.5-122B-UD-IQ1_M.gguf"
            self.assertEqual(setup.pin_of(fam, {}, s), (12345, "ab" * 32))
            self.assertEqual(setup.pin_of(fam, {}, s), (12345, "ab" * 32))      # read once
            self.assertEqual(up.call_count, 1)
            req = up.call_args[0][0]
            self.assertIn("/raw/", req.full_url)
            self.assertNotIn("/resolve/", req.full_url)
            self.assertTrue(req.full_url.endswith("Ornith-1.5-122B-UD-IQ1_M.gguf"))

    def test_a_pinned_file_asks_nobody(self):
        fam = {"hf": setup.hf(REPO), "hub_pins": True, "sha256": {"a.gguf": (1, "c" * 64)}}
        with mock.patch.object(setup.urllib.request, "urlopen", side_effect=AssertionError):
            self.assertEqual(setup.pin_of(fam, {}, Path("a.gguf")), (1, "c" * 64))

    def test_a_family_without_hub_pins_keeps_its_table(self):
        with self.assertRaises(KeyError):
            setup.pin_of({"hf": "x", "sha256": {}}, {}, Path("b.gguf"))

    def test_offline_a_verified_file_is_taken_at_its_mark(self):
        fam = {"hf": setup.hf(REPO), "hub_pins": True, "sha256": {}}
        with tempfile.TemporaryDirectory() as t:
            s = Path(t) / "x.gguf"
            s.write_bytes(b"12345")
            s.with_name("x.gguf.done").write_text("sha256 " + "cd" * 32, encoding="utf-8")
            with mock.patch.object(setup.urllib.request, "urlopen", side_effect=OSError("offline")), \
                    mock.patch.object(setup, "warn"):
                self.assertEqual(setup.pin_of(fam, {}, s), (5, "cd" * 32))

    def test_offline_without_a_mark_stops(self):
        fam = {"hf": setup.hf(REPO), "hub_pins": True, "sha256": {}}
        with tempfile.TemporaryDirectory() as t, \
                mock.patch.object(setup.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaises(SystemExit):
                setup.pin_of(fam, {}, Path(t) / "y.gguf")

    def test_an_answer_that_is_not_a_pointer_stops(self):
        fam = {"hf": setup.hf(REPO), "hub_pins": True, "sha256": {}}
        with tempfile.TemporaryDirectory() as t, \
                mock.patch.object(setup.urllib.request, "urlopen", return_value=self.reply(b"<html>")):
            with self.assertRaises(SystemExit):
                setup.pin_of(fam, {}, Path(t) / "z.gguf")


if __name__ == "__main__":
    unittest.main()
