import ast
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

from helpers import SOURCE, functions, script


class SaveTests(unittest.TestCase):
    def test_cancelled_save_rolls_back_token(self):
        for old in (None, "previous-token"):
            with self.subTest(old=old):
                scene = {} if old is None else {"token": old}
                bpy = types.SimpleNamespace(
                    data=types.SimpleNamespace(scenes={"Scene": scene}),
                    context=types.SimpleNamespace(window_manager=None),
                    ops=types.SimpleNamespace(wm=types.SimpleNamespace(save_mainfile=lambda: {'CANCELLED'})))
                ns = functions(SOURCE.read_text(encoding="utf-8"), ["_save_now"],
                               dict(bpy=bpy, TOKEN="token", _TEST={}, _A={"token_unsaved": True}))
                self.assertIn("cancelled", ns["_save_now"]("new-token", "Scene"))
                self.assertEqual(scene.get("token"), old)
                self.assertFalse(ns["_A"]["saving"])
                self.assertTrue(ns["_A"]["token_unsaved"])


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.blend = Path(self.temp.name) / "project.blend"
        self.blend.write_bytes(b"saved scene")
        self.clock = 100
        self.state = dict(sync=dict(t0=100, next=100, token="new", scene="Scene",
                                    blend=str(self.blend), addr="host:47800", job="job1"))
        self.save = Mock(return_value=None)
        self.rpc = Mock(return_value={"ok": True})
        self.ns = functions(SOURCE.read_text(encoding="utf-8"), ["_sync_tick"], dict(
            _A=self.state, time=types.SimpleNamespace(time=lambda: self.clock), os=os,
            bpy=types.SimpleNamespace(data=types.SimpleNamespace(filepath=str(self.blend))),
            _same_file=lambda a, b: a == b, _save_now=self.save, _prefs=lambda: None,
            _computer_id=lambda: "pc1", rpc=self.rpc, threading=threading))

    def tick(self):
        self.ns["_sync_tick"]()

    def finish_request(self):
        # Production notify runs in a real thread; wait only for that thread's completion.
        s = self.state["sync"]
        self.tick()
        import time
        deadline = time.monotonic() + 2
        while "reply" not in s and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertIn("reply", s)
        self.tick()

    def test_lost_acknowledgement_retries_without_resaving(self):
        self.rpc.side_effect = [OSError("timeout"), {"ok": True}]
        self.tick()
        self.finish_request()
        self.assertIsNotNone(self.state["sync"])
        self.assertIn("retrying automatically", self.state["msg"])
        self.clock += 3
        self.finish_request()
        self.assertIsNone(self.state["sync"])
        self.assertEqual(self.save.call_count, 1)
        self.assertEqual(self.rpc.call_count, 2)
        self.assertEqual(self.rpc.call_args_list[0], self.rpc.call_args_list[1])

    def test_host_rejection_is_not_reported_as_success(self):
        self.rpc.return_value = {"error": "the render changed while saving"}
        self.tick()
        self.finish_request()
        self.assertIsNone(self.state["sync"])
        self.assertIn("couldn't sync", self.state["msg"])

    def test_slow_acknowledgement_does_not_block_ui_tick(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def slow(*args):
            started.set()
            release.wait(2)
            return {"ok": True}
        self.rpc.side_effect = slow
        self.tick()
        self.tick()
        self.assertTrue(started.wait(1))
        self.assertNotIn("reply", self.state["sync"])
        self.assertEqual(self.save.call_count, 1)
        release.set()


class HostSyncTests(unittest.TestCase):
    def setUp(self):
        method = next(node for node in ast.walk(ast.parse(script("COORD")))
                      if isinstance(node, ast.FunctionDef) and node.name == "op_sync_end")
        self.fn = functions(ast.unparse(method), ["op_sync_end"],
                            dict(now=lambda: 100, FILES_MAX=20000))["op_sync_end"]
        self.host = types.SimpleNamespace(lock=threading.Lock(), dirty=False, event=Mock(), job=dict(
            id="job1", blend={"token": "old"}, sync=0, sync_pending=True))

    def test_repeated_acknowledgement_does_not_reload_twice(self):
        req = dict(job="job1", token="new", size=123)
        self.assertEqual(self.fn(self.host, req, ""), {"ok": True})
        self.assertEqual(self.fn(self.host, req, ""), {"ok": True})
        self.assertEqual(self.host.job["sync"], 1)
        self.assertFalse(self.host.job["sync_pending"])
        self.assertTrue(self.host.dirty)
        self.host.event.assert_called_once()

    def test_old_job_retry_cannot_modify_new_render(self):
        self.assertIn("error", self.fn(self.host, dict(job="old-job", token="new"), ""))
        self.assertEqual(self.host.job["blend"]["token"], "old")


if __name__ == "__main__":
    unittest.main()
