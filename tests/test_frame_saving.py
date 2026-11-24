import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from helpers import functions, script


class FrameSavingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src, self.dst = self.root / "render.exr", self.root / "final.exr"
        self.src.write_bytes(b"new complete frame")
        self.dst.write_bytes(b"previous good frame")
        self.ns = functions(script("WORKER"), ["move_into_place"],
                            dict(os=os, shutil=shutil, tempfile=tempfile, AWAY=False))

    def test_replaces_existing_frame(self):
        self.ns["move_into_place"](str(self.src), str(self.dst))
        self.assertEqual(self.dst.read_bytes(), b"new complete frame")
        self.assertFalse(self.src.exists())

    def test_copy_failure_preserves_existing_frame_and_source(self):
        def interrupted(src, part):
            Path(part).write_bytes(b"half")
            raise OSError("connection lost")
        with patch.object(os, "replace", side_effect=OSError("different drives")), \
                patch.object(os, "rename", side_effect=OSError("different drives")), \
                patch.object(shutil, "copyfile", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.ns["move_into_place"](str(self.src), str(self.dst))
        self.assertEqual(self.dst.read_bytes(), b"previous good frame")
        self.assertTrue(self.src.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["final.exr", "render.exr"])

    def test_cross_drive_copy_publishes_only_complete_file(self):
        replace = os.replace
        def cross_drive(src, dst):
            if str(src) == str(self.src):
                raise OSError("different drives")
            self.assertEqual(self.dst.read_bytes(), b"previous good frame")
            self.assertEqual(Path(src).read_bytes(), b"new complete frame")
            return replace(src, dst)
        for away in (False, True):
            with self.subTest(away=away):
                self.src.write_bytes(b"new complete frame")
                self.dst.write_bytes(b"previous good frame")
                self.ns["AWAY"] = away
                with patch.object(os, "replace", side_effect=cross_drive):
                    self.ns["move_into_place"](str(self.src), str(self.dst))
                self.assertEqual(self.dst.read_bytes(), b"new complete frame")
                self.assertFalse(self.src.exists())

    def test_failed_publish_preserves_existing_frame(self):
        with patch.object(os, "replace", side_effect=PermissionError("locked")), \
                patch.object(os, "rename", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.ns["move_into_place"](str(self.src), str(self.dst))
        self.assertEqual(self.dst.read_bytes(), b"previous good frame")
        self.assertTrue(self.src.exists())


if __name__ == "__main__":
    unittest.main()
