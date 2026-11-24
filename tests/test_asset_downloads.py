import ast
import base64
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from helpers import functions, script


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "a" / "b" / "c" / "d"
        self.data = b"scene texture" * 20000
        self.mtime = 1700000000.0
        self.requests, self.messages = [], []
        self.fail_offsets = set()
        self.change_at = None
        self.ns = functions(script("WORKER"), ["fetch_one", "_drop_quiet"], dict(
            os=os, base64=base64, shutil=shutil, json=json, threading=threading,
            time=types.SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda _: None),
            host_addr=lambda: "host:47800", local_name=lambda _: "texture.bin",
            cache_dir=lambda _: str(self.cache), strip_tail=lambda p, n: p,
            FILE_CHUNK=1 << 20, FILE_MAX=4 << 30, COPIED=[0, 0], PACE=[64 << 10],
            log=self.messages.append, put=lambda **kw: self.messages.append(kw), rpc=self.rpc))

    def rpc(self, addr, req, timeout):
        off, n = req.get("off", 0), req.get("n", 0)
        if n:
            self.requests.append((off, n))
            if off in self.fail_offsets:
                raise OSError("connection dropped")
        changed = self.change_at is not None and n and off >= self.change_at
        return dict(ok=True, size=len(self.data), mtime=self.mtime + bool(changed),
                    stamp=[len(self.data), int((self.mtime + bool(changed)) * 1e9)],
                    data=base64.b64encode(self.data[off:off + n]).decode())

    def fetch(self):
        return self.ns["fetch_one"]("/project/texture.bin")

    def test_interruption_resumes_after_worker_restart(self):
        self.fail_offsets.add(64 << 10)
        self.assertIsNone(self.fetch())
        part = self.cache / "texture.bin.part"
        self.assertEqual(part.read_bytes(), self.data[:64 << 10])
        self.assertFalse((self.cache / "texture.bin").exists())
        self.fail_offsets.clear()
        self.requests.clear()
        self.assertIsNotNone(self.fetch())
        self.assertEqual(self.requests[0][0], 64 << 10)
        self.assertEqual((self.cache / "texture.bin").read_bytes(), self.data)
        self.assertFalse((self.cache / "texture.bin.lock").exists())
        self.assertFalse((self.cache / "texture.bin.part.json").exists())

    def test_transient_failure_retries_same_offset(self):
        rpc = self.rpc
        attempts = []
        def flaky(addr, req, timeout):
            if req.get("n") and not attempts:
                attempts.append(req["off"])
                raise OSError("timeout")
            return rpc(addr, req, timeout)
        self.ns["rpc"] = flaky
        self.assertIsNotNone(self.fetch())
        self.assertEqual(attempts, [0])
        self.assertEqual(self.requests[0], (0, 32 << 10))

    def test_same_size_changed_source_is_never_published(self):
        self.change_at = 64 << 10
        self.assertIsNone(self.fetch())
        self.assertFalse((self.cache / "texture.bin").exists())
        self.assertFalse((self.cache / "texture.bin.part.json").exists())

    def test_changed_metadata_discards_old_partial(self):
        self.fail_offsets.add(64 << 10)
        self.assertIsNone(self.fetch())
        self.mtime += 5
        self.data = b"X" * len(self.data)
        self.fail_offsets.clear()
        self.requests.clear()
        self.assertIsNotNone(self.fetch())
        self.assertEqual(self.requests[0][0], 0)
        self.assertEqual((self.cache / "texture.bin").read_bytes(), self.data)

    def test_oversized_chunk_is_rejected(self):
        rpc = self.rpc
        def bad(addr, req, timeout):
            result = rpc(addr, req, timeout)
            if req.get("n"):
                result["data"] = base64.b64encode(b"X" * (req["n"] + 1)).decode()
            return result
        self.ns["rpc"] = bad
        self.assertIsNone(self.fetch())
        self.assertFalse((self.cache / "texture.bin").exists())

    def test_cache_does_not_accept_same_size_file_one_second_old(self):
        self.cache.mkdir(parents=True)
        dst = self.cache / "texture.bin"
        dst.write_bytes(b"X" * len(self.data))
        os.utime(dst, (self.mtime - 1, self.mtime - 1))
        self.assertIsNotNone(self.fetch())
        self.assertEqual(dst.read_bytes(), self.data)

    def test_lock_heartbeat_runs_while_network_request_is_blocked(self):
        # Exercise the production thread immediately, without waiting 90 seconds.
        event = threading.Event()
        real_wait = event.wait
        event.wait = lambda _: real_wait(0.01)
        rpc = self.rpc
        observed = []
        def blocked(addr, req, timeout):
            if req.get("n") and not observed:
                lock = self.cache / "texture.bin.lock"
                os.utime(lock, (1, 1))
                deadline = time.monotonic() + 2
                while lock.stat().st_mtime == 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                observed.append(lock.stat().st_mtime > 1)
            return rpc(addr, req, timeout)
        self.ns["rpc"] = blocked
        self.ns["threading"] = types.SimpleNamespace(Event=lambda: event, Thread=threading.Thread)
        self.assertIsNotNone(self.fetch())
        self.assertEqual(observed, [True])

    def test_complete_partial_is_revalidated_before_publish(self):
        # A complete copy left behind by an interrupted run is checked against the host once more.
        self.cache.mkdir(parents=True)
        (self.cache / "texture.bin.part").write_bytes(self.data)
        (self.cache / "texture.bin.part.json").write_text(json.dumps({
            "path": "/project/texture.bin", "size": len(self.data), "mtime": self.mtime,
            "stamp": [len(self.data), int(self.mtime * 1e9)]}), encoding="utf-8")
        rpc = self.rpc
        calls = []
        def changed_at_end(addr, req, timeout):
            r = rpc(addr, req, timeout)
            calls.append(req)
            if not req.get("n") and len(calls) > 1:
                r["mtime"] += 1
            return r
        self.ns["rpc"] = changed_at_end
        self.assertIsNone(self.fetch())
        self.assertFalse((self.cache / "texture.bin").exists())

    def test_slow_responses_reduce_next_chunk(self):
        ticks = iter([0, 40] * 100)
        self.ns["time"].monotonic = lambda: next(ticks)
        self.assertIsNotNone(self.fetch())
        self.assertEqual(self.requests[0][1], 64 << 10)
        self.assertEqual(self.requests[1][1], 16 << 10)


class HostFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "texture.bin"
        self.path.write_bytes(b"texture")
        method = next(node for node in ast.walk(ast.parse(script("COORD")))
                      if isinstance(node, ast.FunctionDef) and node.name == "op_file")
        self.ns = functions(ast.unparse(method), ["op_file"],
                            dict(os=os, base64=base64, FILE_CHUNK=1 << 20, file_key=lambda p: p))
        self.host = types.SimpleNamespace(job=dict(opts={"copy": True}, files={"texture": str(self.path)}))

    def request(self, **kw):
        return self.ns["op_file"](self.host, dict(path="texture", **kw), "127.0.0.1")

    def test_rejects_stale_version_even_when_file_size_matches(self):
        meta = self.request()
        stat = self.path.stat()
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        self.assertIn("error", self.request(n=4, stamp=meta["stamp"]))

    def test_returns_validated_chunk_and_version(self):
        meta = self.request()
        r = self.request(off=1, n=3, stamp=meta["stamp"])
        self.assertEqual(base64.b64decode(r["data"]), b"ext")
        self.assertEqual(r["stamp"], meta["stamp"])

    def test_rejects_file_changed_while_reading(self):
        real_fstat = os.fstat
        calls = []
        def changed(fd):
            stat = real_fstat(fd)
            calls.append(fd)
            if len(calls) == 2:
                return types.SimpleNamespace(st_size=stat.st_size, st_mtime_ns=stat.st_mtime_ns + 1)
            return stat
        with patch.object(os, "fstat", side_effect=changed):
            self.assertIn("error", self.request(n=4))


if __name__ == "__main__":
    unittest.main()
