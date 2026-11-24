bl_info = {
    "name": "MultiGPU",
    "author": "klosixgaming-arch",
    "version": (2, 6, 0),
    "blender": (4, 2, 0),
    "location": "Properties > Output > MultiGPU  |  3D Viewport > Sidebar > MultiGPU",
    "description": "Render an animation on every graphics card of one or more computers on your network. "
                   "One computer hands out the frames, so no frame is ever rendered twice",
    "category": "Render",
}

"""
MultiGPU 2
==========
Press Render and every ticked card - on this computer and on any computer that joined it - renders
the animation. One computer (the host) hands out the frames one at a time, so two cards can never
get the same frame, a faster card simply takes more of them, and a card that crashes or loses the
network has its frame handed to another card.

Setting up two computers
* Host (the computer with the .blend open): MultiGPU > Network > Host. It shows the address the
  others type in (and its Tailscale address, if Tailscale is running). Allow the port (47800 by
  default) through that computer's firewall.
* Other computer: Network > Join, type the host's address or name, press Join. Leave Blender open
  there. Hosts you joined are remembered, with a name you can change. Joining also works while the
  host is already rendering: the new computer just starts taking frames.
* Press Render on the host. Each computer finds the .blend on the NAS by itself (it matches the
  end of the host's path against this computer's network drives and mounts) and converts the
  host's paths to its own. If it can't, set "NAS folder on this computer" in the preferences.
* Best: set the output to a path relative to the .blend, e.g. //../renders/ ("//" is the folder the
  .blend is in), then every computer saves to the same NAS folder with nothing to convert.

Before a render starts
* Every computer checks that it can find the .blend, the files it uses and the output folder, and
  that the add-ons the scene needs are there. Anything wrong is listed for you to confirm first.

How a frame is made
* The card renders into a hidden ".multigpu-partial" folder next to the frames, then moves the
  finished file into place in one step, so a frame on the NAS is either complete or not there.
* Placeholders (on by default): a 0 KB file marks a frame as taken while it renders.
* Overwrite off (default): frames that already exist are skipped, so Render carries on after Stop.
* Every card proves it sees the same output folder as the host before it takes a frame, and the
  host checks every finished frame really shows up there.
* A frame that fails goes to a card that has rendered others fine. Failing on three of those, it is
  given up on and listed, instead of looping forever. A card that fails two frames in a row stops.
* A frame that takes far longer than any other (a stuck card) is also given to an idle card.
* A card that runs out of GPU memory restarts and renders in tiles half the size (again if needed,
  down to 256 px). Tiles save the memory the image itself takes, not what the scene's data takes.

While it renders
* Saving the .blend (Ctrl+S) doesn't change what's being rendered. Sync Changes does: it saves, and
  each card finishes its frame, then reloads. Until then a card that restarts waits.
* Stop or resume a single card from its row. Stop All stops every card on every computer.
* Only one Blender per computer runs its cards; a second one open at the same time just watches.
* The computers are kept awake while they render.

If the host stops answering mid-render (it crashed, was switched off or lost its network)
* The joined computers keep rendering the frames they are on. Every few seconds the host sends each
  of them a copy of the render's progress, over the network, so each one can carry it on.
* After a minute one of them (the one with the most cards) becomes the new host. The 0 KB
  placeholders the old host's cards left behind are removed and those frames are handed out again.
* Only one computer can take over: the small marker file the host keeps in the output folder shows
  whether the host is really gone, and who the host is now. No two computers ever host one render.
* When the old host comes back it sees that, and joins the new host as a helper by itself.
* "Take Over if the Host Stops" (on a joined computer) switches this off.

Also
* Show Render: each card opens a Blender window showing its frame rendering live. Arrange Windows
  tiles your Blender window and the card windows in a grid (Windows and Linux).
* Advanced: output format (OpenEXR / PNG / JPEG and their settings), overwrite, placeholders, and the
  scene's Post Processing (Compositing, Sequencer), which every card uses.
* Cycles only. A CPU row lets computers without a supported graphics card take part.
"""

import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zlib

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty

VERSION = "2.6.0"
DEFAULT_PORT = 47800
TOKEN = "_multigpu_token"
_BACKENDS = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")
_LABEL = {"OPTIX": "OptiX", "CUDA": "CUDA", "HIP": "HIP", "ONEAPI": "oneAPI", "METAL": "Metal", "CPU": "CPU"}
_INTEGRATED = ("(TM) Graphics", "Radeon Graphics", "UHD Graphics", "Iris")
_ROOT = os.path.join(tempfile.gettempdir(), "multigpu2")
_ENUM_CACHE = {}
_TEST = {}      # testing hooks: {"cmd": [...], "computer": id, "cpu": True, "nosave": True}


# ======================================================================================
# Code shared by the add-on, the host process and every card (each runs its own copy)
# ======================================================================================
SHARED = r'''
import http.client, json, math, os, re, sys, time


def device_base(dev_id):
    """A Cycles device id without its address: the same for one card wherever it is plugged in."""
    d = dev_id[:-len("_OptiX")] if dev_id.endswith("_OptiX") else dev_id
    return d.rsplit("_", 1)[0] if "_" in d else d


def pattern_name(pat, f):
    """The file name of frame f, from the way the render names its frames."""
    return pat["head"] + str(f).zfill(pat["pad"]) + pat["tail"]


def to_ranges(frames):
    """[1, 2, 3, 7, 8] -> [[1, 3], [7, 8]]"""
    out = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def from_ranges(ranges):
    return [f for a, b in ranges or [] for f in range(a, b + 1)]


FILE_CHUNK = 1 << 20           # files only the host has are sent to the other computers in pieces of this size
FILE_MAX = 4 << 30             # ... and never if they are bigger than this
FILES_MAX = 20000              # the most files of one scene that are checked and can be sent


def file_key(p):
    """One spelling for a path, whichever computer wrote it."""
    return (p or "").replace("\\", "/").lower()


WALK = ('SKIP_PACKED', 'SKIP_WEAK_REFERENCES')
WALK_ALL = WALK + ('EXPAND_SEQUENCES', 'EXPAND_TOKENS', 'EXPAND_CACHES')
TOKENS = ("<UDIM>", "<UVTILE>")


def path_exists(p):
    """A file is there. For a tiled image (name.<UDIM>.png): at least one of its tiles is."""
    if any(t in p for t in TOKENS):
        d, name = os.path.split(p)
        rx = re.escape(name).replace(re.escape("<UDIM>"), r"\d{4}").replace(re.escape("<UVTILE>"), r"u\d+_v\d+")
        try:
            return any(re.match("^" + rx + "$", n, re.I) for n in os.listdir(d))
        except OSError:
            return False
    return os.path.exists(p)


def scene_paths(bpy, expand=False, sequencer=False):
    """Every file outside the .blend that the open scene uses, as Blender itself lists them: textures, movies,
    linked .blend files (and the files those use), fonts, volumes, caches, IES and OSL files...
    Returns [(kind, name, library, absolute path, one of several)]. With expand, an image sequence, a tiled
    image or a volume sequence comes as all of its files. None when this Blender is too old to tell."""
    walk = getattr(bpy.data, "file_path_foreach", None)
    if walk is None or os.environ.get("MULTIGPU_OLD_FILE_LIST"):
        return None
    out = []

    def visit(idb, path, meta):
        kind = type(idb).__name__
        if not path or kind in ("Sound", "Text") or (kind == "Scene" and not sequencer):
            return None
        if kind == "Image" and not idb.users:
            return None
        lib = getattr(idb, "library", None)
        try:                                          # ".." worked out by name, the way Blender does it
            full = os.path.normpath(bpy.path.abspath(path, library=lib))
        except Exception:
            full = path
        out.append((kind, idb.name, lib.filepath if lib else "", full, bool(getattr(meta, "is_expanded", False))))
        return None
    try:
        walk(visit, flags=set(WALK_ALL if expand else WALK))
    except (TypeError, ValueError):
        return None
    return out


def norm_join(base_dir, rel):
    """base_dir with a relative path added, ".." worked out by name as Blender does. For a path of any
    computer; the result has "/" between its parts."""
    anchor, parts = split_any(base_dir)
    parts = list(parts)
    for x in (rel or "").replace("\\", "/").split("/"):
        if x in ("", "."):
            continue
        if x == "..":
            if parts:
                parts.pop()
        else:
            parts.append(x)
    return ("" if anchor == "/" else anchor) + "/" + "/".join(parts)


def bundled_here(p, datafiles):
    """A file that ships with Blender (.../datafiles/...), in this computer's own Blender. None if it has none."""
    s = (p or "").replace("\\", "/")
    i = s.lower().rfind("/datafiles/")
    if i < 0 or not datafiles:
        return None
    q = os.path.join(datafiles, *[x for x in s[i + len("/datafiles/"):].split("/") if x])
    return q if os.path.isfile(q) else None

PROTOCOL = 4


def rpc(addr, payload, timeout=4.0):
    """POST one JSON request to the host at 'ip:port' and return its JSON answer. Raises OSError."""
    if not addr:
        raise OSError("not connected to a host")
    conn = None
    try:
        host, _, port = addr.rpartition(":")
        conn = http.client.HTTPConnection(host.strip("[]"), int(port), timeout=timeout)
        conn.request("POST", "/api", json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        if r.status != 200:
            raise OSError("the host answered with HTTP %d" % r.status)
        return json.loads(data.decode("utf-8"))
    except (ValueError, http.client.HTTPException) as e:
        raise OSError("bad answer from the host: %s" % e)
    finally:
        if conn is not None:
            conn.close()


def proc_start(pid):
    """A value that identifies this run of process `pid` (its start time), or None if it can't be read.
    Process ids get reused, so a pid alone doesn't prove a process is still the one we started."""
    try:
        pid = int(pid)
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes as W
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
            k.OpenProcess.restype = W.HANDLE
            k.GetProcessTimes.argtypes = [W.HANDLE] + [ctypes.POINTER(W.FILETIME)] * 4
            k.CloseHandle.argtypes = [W.HANDLE]
            h = k.OpenProcess(0x1000, False, pid)
            if not h:
                return None
            try:
                c, e, kt, ut = W.FILETIME(), W.FILETIME(), W.FILETIME(), W.FILETIME()
                if not k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
                    return None
                return (c.dwHighDateTime << 32) | c.dwLowDateTime
            finally:
                k.CloseHandle(h)
        if sys.platform.startswith("linux"):
            with open("/proc/%d/stat" % pid) as f:
                return int(f.read().rsplit(")", 1)[1].split()[19])
        import subprocess
        out = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        return out or None
    except Exception:
        return None


class Awake(object):
    """Keeps the computer from going to sleep while switched on (best effort, on every system)."""

    def __init__(self):
        self.state = False
        self.child = None

    def on(self):
        if self.state:
            return
        self.state = True
        try:
            if os.name == "nt":
                import ctypes
                f = ctypes.windll.kernel32.SetThreadExecutionState
                f.argtypes = [ctypes.c_uint]
                f(0x80000000 | 0x00000001)          # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                return
            import shutil
            import subprocess
            pid = str(os.getpid())
            if sys.platform == "darwin" and shutil.which("caffeinate"):
                cmd = ["caffeinate", "-i", "-w", pid]
            elif shutil.which("systemd-inhibit") and shutil.which("tail"):
                cmd = ["systemd-inhibit", "--what=idle:sleep", "--who=MultiGPU", "--why=Rendering", "--mode=block",
                       "tail", "--pid=" + pid, "-f", "/dev/null"]      # ends by itself when this process does
            else:
                return
            self.child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, start_new_session=True)
        except Exception:
            pass

    def off(self):
        if not self.state:
            return
        self.state = False
        try:
            if os.name == "nt":
                import ctypes
                f = ctypes.windll.kernel32.SetThreadExecutionState
                f.argtypes = [ctypes.c_uint]
                f(0x80000000)
            elif self.child is not None:
                import signal
                os.killpg(self.child.pid, signal.SIGTERM)
                self.child = None
        except Exception:
            pass


# ---- paths: the same NAS folder has a different path on each computer -----------------
def _clean(parts):
    out = []
    for b in parts:
        if b in ("", "."):
            continue
        if b == "..":
            if out:
                out.pop()
            continue
        out.append(b)
    return out


def split_any(p):
    """(anchor, parts) of a Windows drive, Windows network (UNC) or Linux/macOS path, on any OS."""
    s = (p or "").replace("\\", "/")
    if s.startswith("//"):
        bits = [b for b in s[2:].split("/") if b]
        return "//" + "/".join(bits[:2]).lower(), _clean(bits[2:])
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return s[:2].lower(), _clean(s[2:].split("/"))
    return "/", _clean(s.split("/"))


def map_path(p, maps):
    """Translate an absolute path from another computer into this computer's terms.
    maps: [(their_prefix, our_prefix)]. Blender-relative paths (//...) are returned unchanged."""
    if not p or not maps or p.startswith("//"):
        return p
    a, parts = split_any(p)
    low = [x.lower() for x in parts]
    best = None
    for src, dst in maps:
        if not src or not dst:
            continue
        sa, sp = split_any(src)
        n = len(sp)
        if sa == a and n <= len(parts) and low[:n] == [x.lower() for x in sp] and (best is None or n > best[0]):
            best = (n, dst)
    if best is None:
        return p
    rest = parts[best[0]:]
    return os.path.join(best[1], *rest) if rest else best[1]


def strip_tail(p, k):
    """p without its last k parts, keeping p's own separators."""
    s = p.rstrip("/\\")
    for _ in range(k):
        i = max(s.rfind("/"), s.rfind("\\"))
        if i < 0:
            return ""
        s = s[:i]
    if len(s) == 2 and s[1] == ":":
        s += "\\"
    return s or "/"


def candidate_roots(extra="", hint=""):
    """Folders on this computer the NAS might be under: the ones set in the preferences first,
    then network drives / mounts."""
    roots, seen = [], set()

    def add(r):
        if r and r not in seen:
            seen.add(r)
            roots.append(r)
    for r in (extra or "").replace("\n", ";").split(";"):
        add(r.strip().strip('"'))
    if os.name == "nt":
        m = re.search(r"smb-share:server=([^,/\\]+),share=([^,/\\]+)", hint or "")
        if m:
            add("\\\\%s\\%s" % (m.group(1), m.group(2)))
        try:
            import ctypes
            import string
            mask = ctypes.windll.kernel32.GetLogicalDrives()
            for i, letter in enumerate(string.ascii_uppercase):
                if mask >> i & 1:
                    add(letter + ":\\")
        except Exception:
            pass
    else:
        uid = os.getuid() if hasattr(os, "getuid") else 0
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        bases = ["/run/user/%d/gvfs" % uid, "/mnt", "/media", "/run/media", "/net", "/srv", "/Volumes"]
        if user:
            bases += ["/media/" + user, "/run/media/" + user]
        for b in bases:
            try:
                for e in sorted(os.listdir(b)):
                    add(os.path.join(b, e))
            except OSError:
                pass
        try:
            with open("/proc/mounts") as f:
                for line in f:
                    bits = line.split()
                    if len(bits) > 2 and bits[2] in ("cifs", "smb3", "smbfs", "nfs", "nfs4", "fuse.sshfs", "9p"):
                        add(bits[1].replace("\\040", " "))
        except OSError:
            pass
        add(os.path.expanduser("~"))
    return roots


def find_blend(host_path, size, extra=""):
    """Find the host's .blend on this computer by matching the end of its path under each root.
    Returns (local_path, host_prefix, local_prefix). Raises LookupError with a readable reason."""
    def look(p):
        try:
            if not os.path.isfile(p):
                return 0
            return 2 if (not size or os.path.getsize(p) == size) else 1
        except OSError:
            return 0
    _, parts = split_any(host_path)
    other = None
    mine = [r.strip().strip('"') for r in (extra or "").replace("\n", ";").split(";") if r.strip().strip('"')]
    for k in range(len(parts), 1, -1):              # folders set on this computer come first: its own copy,
        tail = parts[-k:]                           # kept in step by a sync program, beats a far-away original
        for r in mine:
            cand = os.path.join(r, *tail)
            got = look(cand)
            if got == 2 and split_any(cand) != split_any(host_path):
                return cand, strip_tail(host_path, k), r
            if got == 1 and other is None:
                other = cand
    if other:
        raise LookupError("found %s, but it isn't the same version as the host's yet" % other)
    if look(host_path) == 2:
        return host_path, "", ""
    roots = candidate_roots(extra, host_path)
    for k in range(len(parts), 1, -1):
        tail = parts[-k:]
        for r in roots:
            cand = os.path.join(r, *tail)
            got = look(cand)
            if got == 2:
                return cand, strip_tail(host_path, k), r
            if got == 1 and other is None:
                other = cand
    if other:
        raise LookupError("found %s, but it isn't the same version as the host's yet" % other)
    raise LookupError("couldn't find %s on this computer" % (parts[-1] if parts else host_path))


# ---- output format -----------------------------------------------------------------------
def apply_format(im, f):
    """Set Blender image settings `im` from MultiGPU's format dict. Each setting is tried on its own, so a
    Blender version without one of them still gets the rest."""
    if not f:
        return
    fmt = f.get("format", "OPEN_EXR")

    def put(attr, value):
        try:
            setattr(im, attr, value)
        except (TypeError, ValueError, AttributeError):
            pass
    put("file_format", fmt)
    if fmt != "OPEN_EXR_MULTILAYER":               # (a multilayer file keeps every pass as it is)
        color = f.get("color", "RGB")
        if fmt == "JPEG" and color == "RGBA":
            color = "RGB"                          # JPEG has no alpha
        put("color_mode", color)
        if im.color_mode != color:
            put("color_mode", "RGB")
    if fmt in ("OPEN_EXR", "OPEN_EXR_MULTILAYER"):
        put("color_depth", f.get("exr_depth", "32"))
        put("exr_codec", f.get("exr_codec", "DWAA"))
        put("quality", int(f.get("exr_quality", 90)))
    elif fmt == "PNG":
        put("color_depth", f.get("png_depth", "8"))
        put("compression", int(f.get("png_compression", 15)))
    elif fmt == "JPEG":
        put("quality", int(f.get("jpeg_quality", 90)))


# ---- windows: tile Blender and the card windows in a grid --------------------------------
def grid_cells(area, n):
    x, y, w, h = area
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / float(cols))))
    cw, ch = w // cols, h // rows
    return [[x + (i % cols) * cw, y + (i // cols) * ch, cw, ch] for i in range(n)]


class _Win(object):
    def __init__(self):
        import ctypes
        from ctypes import wintypes as W
        self.ct, self.W = ctypes, W
        u = self.u = ctypes.WinDLL("user32", use_last_error=True)
        self.PROC = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
        u.EnumWindows.argtypes = [self.PROC, W.LPARAM]
        u.GetWindowThreadProcessId.argtypes = [W.HWND, ctypes.POINTER(W.DWORD)]
        u.IsWindowVisible.argtypes = [W.HWND]
        u.GetWindow.argtypes = [W.HWND, ctypes.c_uint]
        u.GetWindow.restype = W.HWND
        u.GetClassNameW.argtypes = [W.HWND, W.LPWSTR, ctypes.c_int]
        u.GetWindowRect.argtypes = [W.HWND, ctypes.POINTER(W.RECT)]
        u.MonitorFromWindow.argtypes = [W.HWND, W.DWORD]
        u.MonitorFromWindow.restype = W.HANDLE
        u.MonitorFromPoint.argtypes = [W.POINT, W.DWORD]
        u.MonitorFromPoint.restype = W.HANDLE
        u.GetMonitorInfoW.argtypes = [W.HANDLE, ctypes.c_void_p]
        u.IsZoomed.argtypes = [W.HWND]
        u.IsIconic.argtypes = [W.HWND]
        u.ShowWindow.argtypes = [W.HWND, ctypes.c_int]
        u.SetWindowPos.argtypes = [W.HWND, W.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_uint]
        u.SetWindowTextW.argtypes = [W.HWND, W.LPCWSTR]
        u.GetSystemMetrics.argtypes = [ctypes.c_int]
        u.GetWindowPlacement.argtypes = [W.HWND, ctypes.c_void_p]
        u.SetWindowPlacement.argtypes = [W.HWND, ctypes.c_void_p]

        class WP(ctypes.Structure):
            _fields_ = [("length", W.UINT), ("flags", W.UINT), ("showCmd", W.UINT), ("ptMin", W.POINT),
                        ("ptMax", W.POINT), ("rc", W.RECT)]
        self.WP = WP
        try:
            self.dwm = ctypes.WinDLL("dwmapi")
            self.dwm.DwmGetWindowAttribute.argtypes = [W.HWND, W.DWORD, ctypes.c_void_p, W.DWORD]
        except OSError:
            self.dwm = None

    def windows(self, pid):
        ct, W, u = self.ct, self.W, self.u
        found = []

        def cb(h, l):
            p = W.DWORD()
            u.GetWindowThreadProcessId(h, ct.byref(p))
            if p.value == pid and u.IsWindowVisible(h) and not u.GetWindow(h, 4):
                buf = ct.create_unicode_buffer(64)
                u.GetClassNameW(h, buf, 64)
                r = W.RECT()
                u.GetWindowRect(h, ct.byref(r))
                found.append((buf.value.startswith("GHOST"), (r.right - r.left) * (r.bottom - r.top), h))
            return True
        u.EnumWindows(self.PROC(cb), 0)
        found.sort(key=lambda t: (not t[0], -t[1]))
        return [h for g, a, h in found if g] or [h for g, a, h in found]

    def _work(self, mon):
        ct, W = self.ct, self.W

        class MI(ct.Structure):
            _fields_ = [("cbSize", W.DWORD), ("rcMonitor", W.RECT), ("rcWork", W.RECT), ("dwFlags", W.DWORD)]
        mi = MI()
        mi.cbSize = ct.sizeof(MI)
        if not mon or not self.u.GetMonitorInfoW(mon, ct.byref(mi)):
            return None
        r = mi.rcWork
        return [r.left, r.top, r.right - r.left, r.bottom - r.top]

    def area(self, h):
        return self._work(self.u.MonitorFromWindow(h, 2))

    def primary_area(self):
        return self._work(self.u.MonitorFromPoint(self.W.POINT(0, 0), 1))

    def place(self, h, rect):
        ct, W, u = self.ct, self.W, self.u
        x, y, w, hh = rect
        if u.IsZoomed(h) or u.IsIconic(h):
            u.ShowWindow(h, 9)                     # SW_RESTORE
            time.sleep(0.15)
        l = t = r = b = 0
        if self.dwm:                               # Windows 10/11 draw an invisible resize border
            wr, ext = W.RECT(), W.RECT()
            if u.GetWindowRect(h, ct.byref(wr)) and \
                    self.dwm.DwmGetWindowAttribute(h, 9, ct.byref(ext), ct.sizeof(ext)) == 0:
                l, t, r, b = ext.left - wr.left, ext.top - wr.top, wr.right - ext.right, wr.bottom - ext.bottom
        u.SetWindowPos(h, None, x - l, y - t, w + l + r, hh + t + b, 0x0004 | 0x0010)
        return True

    def title(self, h, text):
        self.u.SetWindowTextW(h, text)

    def save(self, h):
        wp = self.WP()
        wp.length = self.ct.sizeof(self.WP)
        if not self.u.GetWindowPlacement(h, self.ct.byref(wp)):
            return None
        return {"show": wp.showCmd, "rc": [wp.rc.left, wp.rc.top, wp.rc.right, wp.rc.bottom],
                "max": [wp.ptMax.x, wp.ptMax.y]}

    def restore(self, h, s):
        W = self.W
        wp = self.WP()
        wp.length = self.ct.sizeof(self.WP)
        wp.showCmd = 3 if s.get("show") == 3 else 1          # maximized, or normal
        wp.ptMin = W.POINT(-1, -1)
        wp.ptMax = W.POINT(*s["max"])
        wp.rc = W.RECT(*s["rc"])
        self.u.SetWindowPlacement(h, self.ct.byref(wp))

    def screen_height(self):
        return self.u.GetSystemMetrics(1)

    def close(self):
        pass


class _X11(object):
    """Just enough Xlib (through ctypes) to find, move and title this process's windows."""

    def __init__(self):
        import ctypes
        import ctypes.util
        c = self.c = ctypes
        x = self.x = c.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        V, UL, I, UI = c.c_void_p, c.c_ulong, c.c_int, c.c_uint
        x.XOpenDisplay.argtypes = [c.c_char_p]
        x.XOpenDisplay.restype = V
        x.XDefaultRootWindow.argtypes = [V]
        x.XDefaultRootWindow.restype = UL
        x.XInternAtom.argtypes = [V, c.c_char_p, I]
        x.XInternAtom.restype = UL
        x.XGetWindowProperty.argtypes = [V, UL, UL, c.c_long, c.c_long, I, UL, c.POINTER(UL), c.POINTER(I),
                                         c.POINTER(UL), c.POINTER(UL), c.POINTER(V)]
        x.XGetWindowProperty.restype = I
        x.XFree.argtypes = [V]
        x.XSendEvent.argtypes = [V, UL, I, c.c_long, V]
        x.XMoveResizeWindow.argtypes = [V, UL, I, I, UI, UI]
        x.XFlush.argtypes = [V]
        x.XSync.argtypes = [V, I]
        x.XCloseDisplay.argtypes = [V]
        x.XTranslateCoordinates.argtypes = [V, UL, UL, I, I, c.POINTER(I), c.POINTER(I), c.POINTER(UL)]
        x.XGetGeometry.argtypes = [V, UL, c.POINTER(UL), c.POINTER(I), c.POINTER(I), c.POINTER(UI),
                                   c.POINTER(UI), c.POINTER(UI), c.POINTER(UI)]
        x.XChangeProperty.argtypes = [V, UL, UL, UL, I, I, c.c_char_p, I]
        x.XDefaultScreen.argtypes = [V]
        x.XDisplayWidth.argtypes = [V, I]
        x.XDisplayHeight.argtypes = [V, I]
        x.XSetErrorHandler.argtypes = [V]
        x.XSetErrorHandler.restype = V
        self._eh = c.CFUNCTYPE(I, V, V)(lambda d, e: 0)    # never let an X error kill Blender
        self.dpy = x.XOpenDisplay(None)
        if not self.dpy:
            raise OSError("can't open the X11 display")
        self.old_eh = x.XSetErrorHandler(c.cast(self._eh, V))
        self.root = x.XDefaultRootWindow(self.dpy)

    def close(self):
        if self.dpy:
            self.x.XSync(self.dpy, 0)
            self.x.XSetErrorHandler(self.old_eh)
            self.x.XCloseDisplay(self.dpy)
            self.dpy = None

    def atom(self, name):
        return self.x.XInternAtom(self.dpy, name.encode(), 0)

    def prop(self, w, name, n=4096):
        c = self.c
        t, f, ni, ba, data = c.c_ulong(), c.c_int(), c.c_ulong(), c.c_ulong(), c.c_void_p()
        if self.x.XGetWindowProperty(self.dpy, w, self.atom(name), 0, n, 0, 0, c.byref(t), c.byref(f),
                                     c.byref(ni), c.byref(ba), c.byref(data)) != 0 or not data.value:
            return []
        try:
            if f.value == 32:                       # 32-bit items come back as C longs
                arr = c.cast(data, c.POINTER(c.c_long))
                return [arr[i] for i in range(ni.value)]
            if f.value == 8:
                return c.string_at(data, ni.value)
            return []
        finally:
            self.x.XFree(data)

    def geom(self, w):
        c = self.c
        r, x0, y0 = c.c_ulong(), c.c_int(), c.c_int()
        ww, hh, bw, dp = c.c_uint(), c.c_uint(), c.c_uint(), c.c_uint()
        self.x.XGetGeometry(self.dpy, w, c.byref(r), c.byref(x0), c.byref(y0), c.byref(ww), c.byref(hh),
                            c.byref(bw), c.byref(dp))
        ax, ay, ch = c.c_int(), c.c_int(), c.c_ulong()
        self.x.XTranslateCoordinates(self.dpy, w, self.root, 0, 0, c.byref(ax), c.byref(ay), c.byref(ch))
        return [ax.value, ay.value, ww.value, hh.value]

    def windows(self, pid):
        out = [w for w in self.prop(self.root, "_NET_CLIENT_LIST") if self.prop(w, "_NET_WM_PID", 1) == [pid]]
        out.sort(key=lambda w: -(lambda g: g[2] * g[3])(self.geom(w)))
        return out

    def monitors(self):
        c = self.c
        try:
            import ctypes.util
            xr = c.CDLL(ctypes.util.find_library("Xrandr") or "libXrandr.so.2")

            class MI(c.Structure):
                _fields_ = [("name", c.c_ulong), ("primary", c.c_int), ("automatic", c.c_int),
                            ("noutput", c.c_int), ("x", c.c_int), ("y", c.c_int), ("width", c.c_int),
                            ("height", c.c_int), ("mwidth", c.c_int), ("mheight", c.c_int),
                            ("outputs", c.c_void_p)]
            xr.XRRGetMonitors.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.POINTER(c.c_int)]
            xr.XRRGetMonitors.restype = c.POINTER(MI)
            xr.XRRFreeMonitors.argtypes = [c.c_void_p]
            n = c.c_int()
            p = xr.XRRGetMonitors(self.dpy, self.root, 1, c.byref(n))
            if p and n.value > 0:
                mons = sorted(([p[i].x, p[i].y, p[i].width, p[i].height, p[i].primary] for i in range(n.value)),
                              key=lambda m: -m[4])
                xr.XRRFreeMonitors(c.cast(p, c.c_void_p))
                return [m[:4] for m in mons]
        except (OSError, AttributeError):
            pass
        s = self.x.XDefaultScreen(self.dpy)
        return [[0, 0, self.x.XDisplayWidth(self.dpy, s), self.x.XDisplayHeight(self.dpy, s)]]

    def _fit(self, m):
        wa = self.prop(self.root, "_NET_WORKAREA")
        desk = (self.prop(self.root, "_NET_CURRENT_DESKTOP") or [0])[0]
        if len(wa) >= 4 * (desk + 1):
            ax, ay, aw, ah = wa[4 * desk:4 * desk + 4]
            x0, y0 = max(m[0], ax), max(m[1], ay)
            x1, y1 = min(m[0] + m[2], ax + aw), min(m[1] + m[3], ay + ah)
            if x1 - x0 > 200 and y1 - y0 > 200:
                return [x0, y0, x1 - x0, y1 - y0]
        return list(m)

    def area(self, w):
        x, y, ww, hh = self.geom(w)
        cx, cy = x + ww // 2, y + hh // 2
        mons = self.monitors()
        m = next((m for m in mons if m[0] <= cx < m[0] + m[2] and m[1] <= cy < m[1] + m[3]), mons[0])
        return self._fit(m)

    def primary_area(self):
        return self._fit(self.monitors()[0])

    def _msg(self, w, name, data):
        c = self.c

        class CM(c.Structure):
            _fields_ = [("type", c.c_int), ("serial", c.c_ulong), ("send_event", c.c_int),
                        ("display", c.c_void_p), ("window", c.c_ulong), ("message_type", c.c_ulong),
                        ("format", c.c_int), ("l", c.c_long * 5)]

        class EV(c.Union):
            _fields_ = [("xclient", CM), ("pad", c.c_long * 24)]
        ev = EV()
        ev.xclient.type = 33                      # ClientMessage
        ev.xclient.send_event = 1
        ev.xclient.window = w
        ev.xclient.message_type = self.atom(name)
        ev.xclient.format = 32
        for i, v in enumerate(list(data)[:5]):
            ev.xclient.l[i] = v
        self.x.XSendEvent(self.dpy, self.root, 0, (1 << 19) | (1 << 20), c.addressof(ev))

    def place(self, w, rect):
        x, y, ww, hh = rect
        if self.prop(self.root, "_NET_SUPPORTING_WM_CHECK"):
            state = self.prop(w, "_NET_WM_STATE")
            mv, mh = self.atom("_NET_WM_STATE_MAXIMIZED_VERT"), self.atom("_NET_WM_STATE_MAXIMIZED_HORZ")
            fs = self.atom("_NET_WM_STATE_FULLSCREEN")
            if mv in state or mh in state:
                self._msg(w, "_NET_WM_STATE", [0, mv, mh, 2, 0])
            if fs in state:
                self._msg(w, "_NET_WM_STATE", [0, fs, 0, 2, 0])
            if mv in state or mh in state or fs in state:
                self.x.XSync(self.dpy, 0)
                time.sleep(0.25)
            ext = (list(self.prop(w, "_NET_FRAME_EXTENTS")) + [0, 0, 0, 0])[:4]
            l, r, t, b = ext
            flags = 1 | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (2 << 12)   # NW gravity, x y w h, pager
            self._msg(w, "_NET_MOVERESIZE_WINDOW", [flags, x, y, max(100, ww - l - r), max(100, hh - t - b)])
        else:
            self.x.XMoveResizeWindow(self.dpy, w, x, y, ww, hh)
        self.x.XSync(self.dpy, 0)
        return True

    def save(self, w):
        x, y, ww, hh = self.geom(w)
        l, r, t, b = (list(self.prop(w, "_NET_FRAME_EXTENTS")) + [0, 0, 0, 0])[:4]
        state = self.prop(w, "_NET_WM_STATE")
        mx = self.atom("_NET_WM_STATE_MAXIMIZED_VERT") in state or self.atom("_NET_WM_STATE_MAXIMIZED_HORZ") in state
        return {"rect": [x - l, y - t, ww + l + r, hh + t + b], "max": mx}

    def restore(self, w, s):
        self.place(w, s["rect"])
        if s.get("max"):
            time.sleep(0.2)
            self._msg(w, "_NET_WM_STATE", [1, self.atom("_NET_WM_STATE_MAXIMIZED_VERT"),
                                           self.atom("_NET_WM_STATE_MAXIMIZED_HORZ"), 2, 0])
            self.x.XSync(self.dpy, 0)

    def title(self, w, text):
        b = text.encode("utf-8")
        self.x.XChangeProperty(self.dpy, w, self.atom("_NET_WM_NAME"), self.atom("UTF8_STRING"), 8, 0, b, len(b))
        b2 = text.encode("latin-1", "replace")
        self.x.XChangeProperty(self.dpy, w, self.atom("WM_NAME"), self.atom("STRING"), 8, 0, b2, len(b2))
        self.x.XFlush(self.dpy)

    def screen_height(self):
        return self.x.XDisplayHeight(self.dpy, self.x.XDefaultScreen(self.dpy))


def screen_api():
    if os.name == "nt":
        return _Win()
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY"):
            raise OSError("no X11 display")
        return _X11()
    raise OSError("arranging windows isn't supported on this system")


def screen_info(pid):
    """(area of the monitor this process's window is on, screen height, found the window)."""
    api = screen_api()
    try:
        ws = api.windows(pid)
        area = (api.area(ws[0]) if ws else None) or api.primary_area()
        return area, api.screen_height(), bool(ws)
    finally:
        api.close()


def save_own():
    """Where this process's main window is now (and whether it's maximized), to put it back later."""
    api = screen_api()
    try:
        ws = api.windows(os.getpid())
        return api.save(ws[0]) if ws else None
    finally:
        api.close()


def restore_own(state):
    api = screen_api()
    try:
        ws = api.windows(os.getpid())
        if ws and state:
            api.restore(ws[0], state)
    finally:
        api.close()


def place_own(rect=None, title=None):
    """Move this process's main window to rect and/or set its title. False if the window isn't found."""
    api = screen_api()
    try:
        ws = api.windows(os.getpid())
        if not ws:
            return False
        if rect:
            api.place(ws[0], rect)
        if title:
            api.title(ws[0], title)
        return True
    finally:
        api.close()
'''
exec(compile(SHARED, "multigpu_shared", "exec"))


# ======================================================================================
# The host process: hands out frames. Runs on its own (Blender or its Python), so it keeps
# going if the host's Blender is closed, and is restarted from its saved state if it dies.
# ======================================================================================
COORD = r'''
import base64, heapq, threading, traceback, socket, uuid, zlib
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

LEASE = 90          # seconds without any sign of life before a card's frame goes to another card
LEASE_AWAY = 600    # the same for a computer with its own copy of the folder: its frame would be saved twice
VERIFY_GRACE = 150  # seconds a finished frame may take to show up in the host's output folder
HANG_MIN = 600      # a frame must have taken at least this long before an idle card also tries it
HOLD_MARGIN = 0.9   # a slow card leaves the last frames alone only if the others finish them clearly sooner
RESCUE_MARGIN = 1.25    # a free card also renders a slow card's frame only if it will clearly be done first
LOCK_EVERY = 15     # seconds between refreshes of the marker in the output folder
LOCK_TRUST = 40     # no frames are handed out once the marker couldn't be confirmed for this long
LIVE = ("loading", "ready", "rendering", "idle", "reloading", "restarting")


def now():
    return time.time()


def _norm(p):
    return os.path.normcase(os.path.normpath(p))


def scan_sizes(paths):
    out = {}
    for d in set(os.path.dirname(p) for p in paths):
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_file():
                            out[_norm(os.path.join(d, e.name))] = e.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return out


class Coordinator(object):
    def __init__(self, opts):
        self.lock = threading.RLock()
        self.opts = opts
        self.state_file = opts["state"]
        self.version = opts.get("version", "?")
        self.job = None
        self.computers = {}
        self.cards = {}
        self.events = []
        self.dirty = True
        self.host_seen = now()
        self.lock_written = 0
        self.cleanup = []
        self.server = None
        self.resumed_at = 0.0
        self.probes = []
        self.arrive_t = 0.0
        self.pre = None
        self.awake = Awake()
        self.no_live = None
        self.nas_down = False
        self.lock_ok = now()                       # when we last confirmed the output folder's marker is ours
        self.lock_n = 0
        self.lock_mine = None
        self.rep = (0.0, None, None)
        self.holdoff = float(opts.get("holdoff", 12))
        self.took_over = bool(opts.get("takeover"))
        self.orphans = False

    # ---- bookkeeping -------------------------------------------------------------------
    def event(self, text):
        self.events.append([now(), text])
        del self.events[:-40]
        self.dirty = True
        print(time.strftime("%H:%M:%S"), text)
        sys.stdout.flush()

    def load(self):
        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self.job = data.get("job")
        self.cards = data.get("cards") or {}
        self.computers = data.get("computers") or {}
        self.events = data.get("events") or []
        t = now()
        for c in self.cards.values():
            c["seen"] = t                  # cards still rendering get a full lease to check back in
        for c in self.computers.values():
            c["seen"] = 0
        j = self.job
        if j and j["state"] == "running" and not j["opts"].get("overwrite"):
            # frames finished after the last save: cards write them in one step, so a file there is complete
            sizes = scan_sizes(j["paths"])
            for f, p in zip(j["frames"], j["paths"]):
                st = self.fs(f)
                if st["s"] in ("todo", "busy") and sizes.get(_norm(p), 0) > 0:
                    st.update(s="done", v=True, card=None, backup=None)
        if j:
            for c in self.cards.values():          # forget assignments the saved state can't vouch for
                f = c.get("frame")
                st = j["fs"].get(str(f)) if f is not None else None
                if f is not None and (not st or st["s"] != "busy" or c["id"] not in (st.get("card"), st.get("backup"))):
                    c["frame"] = None
            # Frames handed out just before the restart aren't in the saved state: give the cards a
            # moment to say what they're rendering (they check in every 5 seconds) before handing out more.
            self.resumed_at = t
            if self.took_over:
                self.orphans = j["state"] == "running" and bool(j["opts"].get("placeholders", True))
            else:
                self.event("Host restarted; picking the render back up")

    def save(self):
        with self.lock:
            if not self.dirty:
                return
            text = json.dumps({"job": self.job, "cards": self.cards, "computers": self.computers,
                               "events": self.events})
            self.dirty = False
        tmp = self.state_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, self.state_file)
        except OSError:
            pass

    def label(self, card):
        comp = self.computers.get(card.get("computer")) or {}
        return "%s (%s)" % (card.get("label") or card.get("id"), comp.get("name") or "?")

    def fs(self, f):
        return self.job["fs"][str(f)]

    def active_cards(self):
        """Cards that are up and could take a frame right now."""
        return [c for c in self.cards.values() if c.get("use") and not c.get("stopped") and c.get("phase") in LIVE
                and not (self.computers.get(c.get("computer")) or {}).get("blocked")
                and now() - (self.computers.get(c.get("computer")) or {}).get("seen", 0) < 60]

    # ---- frames ----------------------------------------------------------------------
    def note_failure(self, f, card, error):
        """Record that `card` couldn't render frame f. True when the frame should be given up on:
        it failed on three cards that have rendered other frames fine, or six times in all."""
        st = self.fs(f)
        fails = st.setdefault("fails", [])
        if card["id"] not in fails:
            fails.append(card["id"])
        st["err"] = (error or "unknown error")[:300]
        st["nfail"] = st.get("nfail", 0) + 1
        if card.get("done", 0) > 0:
            st["gfail"] = st.get("gfail", 0) + 1
        return st.get("gfail", 0) >= 3 or st["nfail"] >= 6

    def release(self, card, fail=False, error=None):
        """Take the card's frame away from it (it stopped, crashed, failed or went silent)."""
        f = card.get("frame")
        card["frame"] = None
        card["frame_t"] = None
        card["backup"] = False
        if f is None or not self.job:
            return
        st = self.job["fs"].get(str(f))
        if not st or st["s"] != "busy":
            return
        kid = card["id"]
        self.dirty = True
        if st.get("backup") == kid:              # it was the second card on this frame; the first carries on
            st["backup"] = None
            if fail:
                self.note_failure(f, card, error)
            return
        if st.get("card") != kid:
            return
        give_up = self.note_failure(f, card, error) if fail else False
        b = self.cards.get(st.get("backup") or "")
        if b and b.get("frame") == f:            # the second card on this frame takes it over
            st.update(card=b["id"], backup=None, t=b.get("frame_t") or now())
            b["backup"] = False
        elif give_up:
            st.update(s="failed", card=None, backup=None)
            self.event("Frame %d failed %d times and was skipped: %s" % (f, st["nfail"], st["err"]))
        else:
            st.update(s="todo", card=None, backup=None, free_t=now())
            if fail:
                self.event("%s couldn't render frame %d (%s); another card will try it"
                           % (self.label(card), f, st["err"]))

    def pick(self, card):
        """Lowest frame still to do. A frame that failed on this card goes to a different card when there is
        one, and a frame that already failed somewhere is left to a card that has rendered something."""
        live = self.active_cards()
        live_ids = set(c["id"] for c in live)
        kid = card["id"]
        proven = card.get("done", 0) > 0
        proven_others = any(c.get("done", 0) > 0 and c["id"] != kid for c in live)
        retry = later = None
        t = now()
        for f in self.job["frames"]:
            st = self.fs(f)
            if st["s"] != "todo":
                continue
            fails = st.get("fails") or ()
            if kid in fails:
                if retry is None and (not (live_ids - set(fails)) or t - st.get("free_t", t) > min(60, LEASE)):
                    retry = f                      # nobody else is left to try it, or nobody has for a minute
                continue
            if fails and not proven and proven_others:
                if later is None:
                    later = f
                continue
            return f
        return later if later is not None else retry

    def hang_limit(self):
        slow = self.job.get("slowest")
        return max(HANG_MIN, 3.0 * slow) if slow else None

    # ---- the last frames go to whoever finishes them soonest ------------------------------
    def pace(self, c, t):
        """(seconds until the card is free, seconds it needs for a frame). None while that isn't known, or
        when its frame is far over its usual time and it can't be counted on."""
        a = c.get("avg")
        if not a:
            return None
        if c.get("frame") is not None and c.get("frame_t"):
            e = t - c["frame_t"]
            return None if e > 2 * a + 30 else (max(0.0, a - e), a)
        return (0.0, a)

    def others_finish(self, card, n, t, before):
        """When the other cards would have n more frames done between them, each frame going to the card that
        finishes it soonest. None if that isn't within `before` seconds."""
        heap = []
        for c in self.active_cards():
            p = None if c is card else self.pace(c, t)
            if p:
                heapq.heappush(heap, (p[0] + p[1], p[1]))
        end = 0.0
        for _ in range(n):
            if not heap:
                return None
            end, a = heapq.heappop(heap)
            if end >= before:
                return None
            heapq.heappush(heap, (end + a, a))
        return end

    def hold_back(self, card, t):
        """Near the end, a slow card doesn't start a frame when the faster cards will have finished every frame
        that is left before it could finish one: the render ends sooner without it. If they turn out not to
        (a card dropped out, the frames got heavier), the card goes back to work."""
        j, a = self.job, card.get("avg")
        if not j["opts"].get("smart_finish", True) or not a or card.get("hold_off"):
            return False
        n = sum(1 for f in j["frames"] if self.fs(f)["s"] == "todo")
        end = self.others_finish(card, n, t, HOLD_MARGIN * a)
        if end is None:
            card["held"] = None
            return False
        if not card.get("held"):
            card["held"] = t + end * 1.5 + 15
            self.event("%s leaves the last %d frame%s to the faster cards (it needs %s a frame, they'll be done in %s)"
                       % (self.label(card), n, "" if n == 1 else "s", _dur(a), _dur(end)))
        elif t > card["held"]:
            card.update(held=None, hold_off=True)
            self.event("%s is rendering again: the last frames are taking longer than expected" % self.label(card))
            return False
        return True

    def pick_rescue(self, card, t):
        """Nothing is left to hand out, but a slower card will need much longer for its frame than this free
        card would for the same frame: render it here too. Whichever finishes first counts."""
        a = card.get("avg")
        if not self.job["opts"].get("smart_finish", True) or not a:
            return None
        best = None
        for f in self.job["frames"]:
            st = self.fs(f)
            if st["s"] != "busy" or st.get("card") == card["id"] or card["id"] in (st.get("fails") or ()) \
                    or st.get("nbackup", 0) >= 2:
                continue
            b = self.cards.get(st.get("backup") or "")
            h = self.cards.get(st.get("card") or "")
            if h is None or (b is not None and b.get("frame") == f):
                continue
            if self.away(h.get("computer")) or self.away(card.get("computer")):
                continue                              # two folders: the frame would end up saved twice
            e = t - (st.get("t") or t)
            # what it still needs; with no pace known, or once it is over its usual time, how long it has been at it
            left = h["avg"] - e if h.get("avg") and e < h["avg"] else (e - h["avg"] if h.get("avg") else e)
            if left > a * RESCUE_MARGIN and (best is None or left > best[0]):
                best = (left, f)
        return best[1] if best else None

    def pick_backup(self, card, t):
        """Nothing is left to hand out: a frame another card has been on for far too long (it may be
        stuck) is also given to this idle card. Whichever finishes first counts."""
        limit = self.hang_limit()
        if not limit:
            return None
        for f in self.job["frames"]:
            st = self.fs(f)
            if st["s"] != "busy" or st.get("card") == card["id"] or card["id"] in (st.get("fails") or ()) \
                    or t - (st.get("t") or t) <= limit:
                continue
            b = self.cards.get(st.get("backup") or "")
            if b is not None and b.get("frame") == f:      # a second card is on it already
                if not (b.get("frame_t") and t - b["frame_t"] > limit) or st.get("nbackup", 0) >= 3:
                    continue                               # ...and it isn't overdue itself
            return f
        return None

    def check_finished(self):
        j = self.job
        if not j:
            return False
        if j["state"] != "running":
            return j["state"] == "finished"
        for f in j["frames"]:
            st = self.fs(f)
            if st["s"] in ("todo", "busy") or (st["s"] == "done" and not st.get("v")):
                return False
        j["state"] = "finished"
        j["ended"] = now()
        nf = sum(1 for f in j["frames"] if self.fs(f)["s"] == "failed")
        na = sum(1 for f in j["frames"] if self.fs(f).get("away"))
        self.event("Finished: %d frames in %s%s%s" % (len(j["frames"]), _dur(j["ended"] - j["started"]),
                                                     ", %d failed" % nf if nf else "",
                                                     ". %d are still on their way to this computer's folder" % na if na else ""))
        for c in self.cards.values():
            c["frame"] = None
            if c.get("phase") in LIVE:
                c["phase"] = "done"
        self.schedule_cleanup()
        return True

    def schedule_cleanup(self):
        j = self.job
        if not j:
            return
        leftovers = [p for f, p in zip(j["frames"], j["paths"]) if self.fs(f)["s"] != "done" and self.fs(f).get("claimed")]
        self.cleanup.append({"job": j["id"], "placeholders": leftovers, "outdirs": sorted(set(os.path.dirname(p) for p in j["paths"])),
                             "lock": os.path.join(j["outdir"], ".multigpu-lock.json")})

    def write_lock(self, j):
        """The marker in the output folder: stops a second host rendering into it, and lets every card
        prove it's looking at the same folder as the host."""
        path = os.path.join(j["outdir"], ".multigpu-lock.json")
        me = self.computers.get(j["host"]) or {}
        self.lock_n += 1
        data = {"host": j["host"], "host_name": j.get("host_name"), "job": j["id"], "t": time.time(), "state": "running",
                "term": j.get("term", 0), "n": "%d-%d" % (os.getpid(), self.lock_n),
                "addrs": me.get("addrs") or self.opts.get("addrs") or [],
                "port": (self.server.server_address[1] if self.server else None) or me.get("port") or self.opts.get("port")}
        try:
            os.makedirs(j["outdir"], exist_ok=True)
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(path + ".tmp", path)
            self.lock_mine = data["n"]
            return True
        except OSError:
            return False

    def read_lock(self, j):
        try:
            with open(os.path.join(j["outdir"], ".multigpu-lock.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def other_host(self, j, cur=None):
        """The marker of another computer that has taken this render over as host, or None."""
        other = cur or self.read_lock(j)
        if other and other.get("job") == j["id"] and other.get("host") != j["host"] and \
                (other.get("term") or 0) > j.get("term", 0):
            return other
        return None

    def deposed(self, other):
        """Another computer became the host while this one was out of reach: hand everything over to it."""
        with self.lock:
            j = self.job
            if not j or j["state"] != "running" or other.get("job") != j["id"]:
                return
            j["state"] = "moved"
            j["ended"] = now()
            j["moved"] = {"host": other.get("host"), "name": other.get("host_name") or "Another computer",
                          "addrs": other.get("addrs") or [], "port": other.get("port")}
            for c in self.cards.values():
                c.update(frame=None, frame_t=None, backup=False)
                if c.get("phase") in LIVE:
                    c["phase"] = "off"
            self.event("%s took over as host while this computer was out of reach. This computer now helps it"
                       % j["moved"]["name"])

    def clear_claims(self, j):
        try:
            for n in os.listdir(j["outdir"]):
                m = re.match(r"\.multigpu-takeover-(\d+)\.json$", n)
                if m and int(m.group(1)) <= j.get("term", 0):
                    os.remove(os.path.join(j["outdir"], n))
        except OSError:
            pass

    def standby(self, t):
        """Is another computer ready to take over as host? Then this one must keep proving it's alive."""
        return any(c.get("failover") and not c.get("host") and t - c.get("seen", 0) < 300 for c in self.computers.values())

    # ---- requests --------------------------------------------------------------------
    def handle(self, req, local, ip):
        op = req.get("op")
        fn = getattr(self, "op_" + str(op), None)
        if fn is None:
            return {"error": "unknown request %r" % op}
        if op in ("sync_begin", "sync_end", "sync_abort", "unsynced") and not local and self.job \
                and req.get("computer") and req.get("computer") == self.job.get("origin"):
            return fn(req, ip)                     # the computer the render was started on, after a take-over
        if op in ("start", "sync_begin", "sync_end", "sync_abort", "quit", "unsynced", "precheck") and not local:
            return {"error": "only the host's own Blender can do that"}
        return fn(req, ip)

    def mismatch(self, addon, name):
        if addon and addon != self.version:
            return ("MultiGPU versions don't match: the host has %s, %s has %s. Install the same version on "
                    "every computer." % (self.version, name or "the other computer", addon))
        return None

    def op_file(self, req, ip):
        """A piece of a file of this render that the asking computer can't find. Only files the scene uses.
        With "members": the files that belong to a path (all frames of a sequence, all tiles of an image)."""
        j = self.job
        if not j or not j["opts"].get("copy"):
            return {"error": "not a file of this render"}
        key = file_key(req.get("path"))
        if req.get("members"):
            names = (j.get("sets") or {}).get(key) or ([j["files"][key]] if key in (j.get("files") or {}) else [])
            names = [n for n in names if file_key(n) in j["files"]]
            return {"ok": True, "members": names} if names else {"error": "not a file of this render"}
        ent = (j.get("files") or {}).get(key)
        if not ent:
            return {"error": "not a file of this render"}
        try:
            off, n = max(0, int(req.get("off") or 0)), max(0, min(int(req.get("n") or 0), FILE_CHUNK))
            st = os.stat(ent)
            stamp = [st.st_size, st.st_mtime_ns]
            if req.get("stamp") is not None and req["stamp"] != stamp:
                return {"error": "the file changed during the copy"}
            data = b""
            if n:
                with open(ent, "rb") as f:
                    opened = os.fstat(f.fileno())
                    if [opened.st_size, opened.st_mtime_ns] != stamp:
                        return {"error": "the file changed during the copy"}
                    f.seek(off)
                    data = f.read(n)
                    after = os.fstat(f.fileno())
                current = os.stat(ent)
                if any([s.st_size, s.st_mtime_ns] != stamp for s in (after, current)):
                    return {"error": "the file changed during the copy"}
        except (OSError, ValueError, TypeError) as e:
            return {"error": "the host can't read it (%s)" % (getattr(e, "strerror", None) or e)}
        return {"ok": True, "size": st.st_size, "mtime": st.st_mtime, "stamp": stamp,
                "data": base64.b64encode(data).decode("ascii")}

    # ---- computers that work from their own copy of the folder (kept in step by a sync program) ----
    def away(self, comp_id):
        """True when that computer doesn't see the host's output folder itself: it renders into its own copy,
        and its frames reach the host's folder later."""
        j = self.job
        r = (self.computers.get(comp_id) or {}).get("resolve") or {}
        return bool(j and r.get("job") == j["id"] and r.get("away"))

    def mark_have(self, comp_id, frames):
        """Frames that computer already has in its own copy: they aren't rendered again."""
        j = self.job
        if not j or j["opts"].get("overwrite"):
            return 0
        n = 0
        for f in frames or []:
            st = j["fs"].get(str(f))
            if st is not None and st["s"] == "todo":
                st.update(s="done", v=True, by=None, comp=comp_id, t=now(), away=True, size=None)
                n += 1
        if n:
            self.dirty = True
            name = (self.computers.get(comp_id) or {}).get("name") or "Another computer"
            self.event("%s already has %d frame%s of this render: not rendered again" % (name, n, "" if n == 1 else "s"))
            self.check_finished()
        return n

    def op_have(self, req, ip):
        with self.lock:
            j = self.job
            if not j or j["id"] != req.get("job") or j["state"] != "running":
                return {"ok": False}
            return {"ok": True, "n": self.mark_have(req.get("computer"), req.get("frames"))}

    def op_probe(self, req, ip):
        """Put a small file into the output folder. A computer that sees it appear shares the folder with the host
        directly; one that doesn't has its own copy."""
        with self.lock:
            j = self.job
            if not j or j["id"] != req.get("job"):
                return {"error": "no such render"}
            d = j["outdir"]
        name = ".multigpu-probe-%s" % uuid.uuid4().hex[:8]
        try:
            with open(os.path.join(d, name), "w") as f:
                f.write("x")
        except OSError as e:
            return {"error": "the host can't write to its output folder (%s)" % (e.strerror or e)}
        with self.lock:
            self.probes.append((now(), os.path.join(d, name)))
        return {"ok": True, "name": name}

    def op_ping(self, req, ip):
        with self.lock:
            name = next((c.get("name") for c in self.computers.values() if c.get("host")), None)
            return {"ok": True, "protocol": PROTOCOL, "version": self.version, "job": self.job and self.job["id"],
                    "pid": os.getpid(), "pstart": proc_start(os.getpid()), "bind": self.opts.get("bind"),
                    "name": name, "running": bool(self.job and self.job["state"] == "running")}

    def op_hello(self, req, ip):
        c = req.get("computer") or {}
        cid = c.get("id")
        if not cid:
            return {"error": "no computer id"}
        bad = self.mismatch(c.get("addon"), c.get("name"))
        if bad or c.get("protocol") != PROTOCOL:
            return {"error": bad or "MultiGPU versions don't match. Install the same version on every computer."}
        with self.lock:
            t = now()
            if c.get("is_host"):
                self.host_seen = t
            if c.get("viewer"):                    # a second Blender on a computer: it only looks
                return {"ok": True, "now": t, "viewer": True, "job": self.job_brief(), "want": {}, "kill": [],
                        "maps": self.maps(cid), "status": self.status(), "uses": {}, "pre": None}
            comp = self.computers.setdefault(cid, {"id": cid, "bad": 0, "blocked": None})
            if t - comp.get("seen", 0) > 60 and comp.get("seen"):
                self.event("%s connected again" % (c.get("name") or cid))
            elif not comp.get("seen"):
                self.event("%s connected" % (c.get("name") or cid))
            comp.update(name=c.get("name") or cid, os=c.get("os"), blender=c.get("blender"), addon=c.get("addon"),
                        ip=ip, seen=t, host=bool(c.get("is_host")), resolve=c.get("resolve") or {},
                        port=c.get("port"), addrs=c.get("addrs") or [], failover=bool(c.get("failover")))
            if c.get("pre"):
                comp["pre"] = c["pre"]
            ids = []
            for k in c.get("cards") or []:
                card = self.cards.setdefault(k["id"], {"id": k["id"], "phase": "off", "done": 0, "avg": None,
                                                       "stopped": False, "streak": 0, "frame": None, "seen": 0,
                                                       "epoch": 0, "seq": 0})
                card.update(computer=cid, label=k.get("label"), backend=k.get("backend"))
                if (k.get("use_v") or 0) >= card.get("use_v", 0):   # it has seen the host's last change
                    card["use"] = bool(k.get("use"))
                    card["use_v"] = k.get("use_v") or 0
                ids.append(k["id"])
            comp["cards"] = ids
            for kid, p in (c.get("procs") or {}).items():
                card = self.cards.get(kid)
                if card and card.get("computer") == cid:
                    self.proc_report(card, p, t)
            self.dirty = True
            pre = self.pre if self.pre and t - self.pre["t"] < 120 and self.pre.get("host") != cid else None
            return {"ok": True, "now": t, "job": self.job_brief(), "want": self.want(cid), "kill": self.kill(cid),
                    "maps": self.maps(cid), "status": self.status(), "pre": pre,
                    "uses": {k: [bool(self.cards[k].get("use")), self.cards[k].get("use_v", 0)] for k in ids}}

    def proc_report(self, card, p, t):
        j = self.job
        if not j or p.get("job") != j["id"]:
            return
        if not p.get("alive"):
            card["restart"] = False                # a stuck card's Blender has been closed; it starts afresh
        if p.get("alive"):
            card["seen"] = max(card.get("seen", 0), t)
            if card.get("phase") in ("off", "restarting", "lost", "done") and card.get("at") != [j["id"], j["sync"]] \
                    and j["state"] == "running" and not card.get("stopped"):
                card["phase"] = "loading"
            return
        seq = p.get("seq") or 0
        if seq <= card.get("seq", 0):
            return
        card["seq"] = seq
        why = p.get("why")
        if why == "crash":
            f = card.get("frame")
            self.release(card, fail=True, error="Blender closed unexpectedly (crash?)")
            card["phase"] = "restarting"
            self.event("%s closed unexpectedly%s; restarting it" % (self.label(card), " on frame %d" % f if f is not None else ""))
        elif why in ("gave_up", "error"):
            self.release(card)
            said = card.get("phase") == "error"     # (the card itself may have told us already)
            card["phase"] = "error"
            card["err"] = p.get("error") or ("Blender closed unexpectedly 3 times in 10 minutes" if why == "gave_up" else "see its log")
            if not said:
                self.event("%s stopped: %s" % (self.label(card), card["err"]))
        elif why == "oom":
            self.out_of_memory(card, p.get("tile"))
        elif why in ("reload", "stale"):
            self.release(card)
            if card.get("phase") not in ("stopped", "error"):
                card["phase"] = "loading"
        else:
            self.release(card)
        self.dirty = True

    def job_brief(self):
        j = self.job
        if not j:
            return None
        out = {k: j.get(k) for k in ("id", "state", "blend", "scene", "sync", "sync_pending", "opts", "output",
                                      "outdir", "host", "host_name", "frames_n", "unsynced", "first", "origin",
                                      "term", "moved", "pattern")}
        if j.get("pattern"):
            if j.get("_ranges") is None:
                j["_ranges"] = to_ranges(j["frames"])
            out["ranges"] = j["_ranges"]
        return out

    def want(self, cid):
        j = self.job
        comp = self.computers.get(cid) or {}
        if not j or j["state"] != "running" or comp.get("blocked"):
            return {}
        out = {}
        for k in comp.get("cards", []):
            c = self.cards[k]
            if c.get("use") and not c.get("stopped") and c.get("phase") != "error":
                out[k] = c.get("epoch", 0)
        return out

    def kill(self, cid):
        j = self.job
        comp = self.computers.get(cid) or {}
        ids = comp.get("cards", [])
        if not j or j["state"] != "running" or comp.get("blocked"):
            return list(ids)
        return [k for k in ids if not self.cards[k].get("use") or self.cards[k].get("stopped")
                or self.cards[k].get("phase") == "error" or self.cards[k].get("restart")]

    def maps(self, cid):
        j = self.job
        if not j:
            return []
        res = {}
        for k, comp in self.computers.items():
            r = comp.get("resolve") or {}
            if r.get("job") == j["id"] and r.get("ok") and r.get("host_prefix") and r.get("local_prefix"):
                res[k] = (r["host_prefix"], r["local_prefix"])
        out = []
        if cid in res:
            hp, lp = res[cid]
            out.append([hp, lp])
            for k, (hpd, lpd) in res.items():
                if k != cid:
                    mine = map_path(hpd, [(hp, lp)])
                    if mine != hpd:
                        out.append([lpd, mine])
        elif cid == (j.get("origin") or j["host"]):
            for k, (hpd, lpd) in res.items():
                out.append([lpd, hpd])
        return out

    def op_card(self, req, ip):
        kid = req.get("card")
        want = req.get("want")
        if req.get("protocol") != PROTOCOL or self.mismatch(req.get("addon"), None):
            return {"exit": "error", "why": "this computer's MultiGPU version isn't the host's (%s)" % self.version}
        with self.lock:
            t = now()
            j = self.job
            card = self.cards.get(kid)
            if card is None:
                card = self.cards[kid] = {"id": kid, "phase": "off", "done": 0, "avg": None, "stopped": False,
                                          "streak": 0, "frame": None, "epoch": 0, "seq": 0, "use": True,
                                          "computer": req.get("computer"), "label": req.get("label")}
            card["seen"] = t
            if "warn" in req:
                card["warn"] = req.get("warn") or None
            self.dirty = True
            rep = req.get("report")
            if rep and j and req.get("job") == j["id"]:
                self.report(card, rep, t)
            if want == "bye":
                self.bye(card, req)
                return {"ok": True}
            if not j or req.get("job") != j["id"]:
                return {"exit": "over"}
            if j["state"] != "running":
                self.release(card)
                return {"exit": "done" if j["state"] == "finished" else "stopped"}
            comp = self.computers.get(card.get("computer")) or {}
            if card.get("stopped"):
                self.release(card)
                card["phase"] = "stopped"
                return {"exit": "stopped"}
            if not card.get("use"):
                self.release(card)
                return {"exit": "stopped"}
            if comp.get("blocked"):
                self.release(card)
                return {"exit": "error", "why": comp["blocked"]}
            if card.get("phase") == "error":
                self.release(card)
                return {"exit": "error", "why": card.get("err")}
            if want == "hello":
                card.update(at=[j["id"], req.get("sync")], phase="ready", err=None, restart=False)
                return {"ok": True}
            if want == "beat":
                f = req.get("frame")
                if f is not None and card.get("frame") != f:
                    st = j["fs"].get(str(f))
                    if st and st["s"] == "todo" and card.get("frame") is None:
                        # the host restarted and lost track of this card's frame: it's still this card's
                        st.update(s="busy", card=kid, backup=None, t=t, claimed=True)
                        card.update(frame=f, frame_t=t, phase="rendering", backup=False)
                return {"ok": True, "drop": f is not None and card.get("frame") != f}
            # want == "next"
            if card.get("frame") is not None:        # our answer got lost: hand the same frame back
                st = self.fs(card["frame"])
                if st["s"] == "busy" and kid in (st.get("card"), st.get("backup")):
                    return {"frame": card["frame"]}
                card["frame"] = None
            if t - self.resumed_at < min(self.holdoff, LEASE):
                card["phase"] = "idle"
                return {"wait": 2, "why": "the host just restarted"}
            if t - self.lock_ok > LOCK_TRUST and self.standby(t):
                card["phase"] = "idle"             # (another computer may be taking over as host right now)
                return {"wait": 3, "why": "checking the output folder"}
            if (req.get("sync") or 0) < j["sync"]:
                card["phase"] = "reloading"
                return {"exit": "reload"}
            if j.get("sync_pending"):
                card["phase"] = "idle"
                return {"wait": 2, "why": "waiting for the changes to be saved"}
            if card.get("streak", 0) >= 2:
                card["phase"] = "error"
                card["err"] = "2 frames in a row failed here: %s" % (card.get("last_err") or "see its log")
                self.event("%s stopped: %s" % (self.label(card), card["err"]))
                return {"exit": "error", "why": card["err"]}
            f = self.pick(card)
            if f is not None and self.hold_back(card, t):
                card["phase"] = "idle"
                return {"wait": 3, "why": "the faster cards will finish the rest sooner"}
            card["held"] = None
            if f is None:
                if self.check_finished():
                    card["phase"] = "done"
                    return {"exit": "done"}
                b = self.pick_backup(card, t)
                stuck = b is not None
                if b is None:
                    b = self.pick_rescue(card, t)
                if b is not None:
                    st = self.fs(b)
                    old = self.cards.get(st.get("backup") or "")
                    if old is not None and old.get("frame") == b:      # the earlier second card got stuck too
                        old.update(frame=None, frame_t=None, backup=False, restart=True)
                    st["backup"] = kid
                    st["nbackup"] = st.get("nbackup", 0) + 1
                    card.update(frame=b, frame_t=t, phase="rendering", backup=True, at=[j["id"], req.get("sync")])
                    first = self.cards.get(st.get("card") or "") or {}
                    if stuck:
                        self.event("Frame %d has taken %s on %s; %s is rendering it too in case it's stuck"
                                   % (b, _dur(t - (st.get("t") or t)), self.label(first), self.label(card)))
                    else:
                        self.event("Frame %d is slow on %s; %s is free and rendering it too, the first one done counts"
                                   % (b, self.label(first), self.label(card)))
                    return {"frame": b}
                card["phase"] = "idle"
                return {"wait": 3}
            self.fs(f).update(s="busy", card=kid, backup=None, t=t, claimed=True)
            card.update(frame=f, frame_t=t, phase="rendering", backup=False, at=[j["id"], req.get("sync")])
            return {"frame": f}

    def report(self, card, rep, t):
        j = self.job
        f = rep.get("frame")
        st = j["fs"].get(str(f))
        if st is None:
            return
        rid = rep.get("rid")
        if rid and card.get("last_rid") == rid:   # the same report sent again after a lost answer
            return
        card["last_rid"] = rid
        if rep.get("ok"):
            card["streak"] = 0
            card["last_fail"] = None
            card["done"] = card.get("done", 0) + 1
            secs = rep.get("secs")
            if secs:                              # a card's first frame includes loading the scene onto it:
                first = card["done"] <= 2          # it is forgotten as soon as there is a second one
                card["avg"] = secs if first or not card.get("avg") else card["avg"] * 0.7 + secs * 0.3
                j["slowest"] = max(j.get("slowest") or 0, secs)
            if st["s"] != "done":
                limit = self.hang_limit()
                for hid in (st.get("card"), st.get("backup")):   # anyone else still on this frame
                    h = self.cards.get(hid or "")
                    if h is None or h is card or h.get("frame") != f:
                        continue
                    if limit and h.get("frame_t") and t - h["frame_t"] > limit:
                        h["restart"] = True
                        self.event("%s was stuck on frame %d, which another card has now rendered; restarting it"
                                   % (self.label(h), f))
                    h.update(frame=None, frame_t=None, backup=False)
                gone = self.away(card.get("computer"))
                st.update(s="done", by=card["id"], comp=card.get("computer"), t=t, v=gone, away=gone, card=None, backup=None,
                          size=rep.get("size"))
            elif st.get("by") != card["id"]:
                st["size"] = None                 # written twice: either file is fine
            if card.get("frame") == f:
                card.update(frame=None, frame_t=None, backup=False)
            self.check_finished()
        elif rep.get("env"):                      # the output folder is unreachable: not the frame's or card's fault
            if card.get("frame") == f:
                self.release(card)
        else:
            if card.get("last_fail") != f:
                card["streak"] = card.get("streak", 0) + 1
            card["last_fail"] = f
            card["last_err"] = (rep.get("error") or "unknown error")[:200]
            if card.get("frame") == f:
                self.release(card, fail=True, error=rep.get("error"))

    def bye(self, card, req):
        why = req.get("why")
        if why in ("cancelled", "closed"):
            self.release(card)
            card["stopped"] = True
            card["phase"] = "stopped"
            card["stop_why"] = why
            self.event("%s stopped: %s" % (self.label(card), "its render was cancelled in its window"
                                           if why == "cancelled" else "its window was closed"))
        elif why == "error":
            self.release(card)
            card["phase"] = "error"
            card["err"] = req.get("error") or "see its log"
            self.event("%s stopped: %s" % (self.label(card), card["err"]))
        elif why == "oom":
            self.out_of_memory(card, req.get("tile"))
        else:
            self.release(card)

    def out_of_memory(self, card, tile):
        f = card.get("frame")
        self.release(card)                          # not the frame's fault, and not yet the card's
        if card.get("phase") not in ("stopped", "error"):
            card["phase"] = "restarting"
        if f is not None or card.get("tile") != tile:
            self.event("%s ran out of GPU memory%s; restarting it with %s px tiles"
                       % (self.label(card), " on frame %d" % f if f is not None else "", tile or "smaller"))
        card["tile"] = tile

    def op_precheck(self, req, ip):
        with self.lock:
            self.pre = dict(req.get("pre") or {}, t=now())
            return {"ok": True}

    def op_start(self, req, ip):
        spec = req.get("job") or {}
        frames, paths = spec.get("frames") or [], spec.get("paths") or []
        if not frames or len(frames) != len(paths):
            return {"error": "no frames to render"}
        with self.lock:
            if self.job and self.job["state"] == "running":
                return {"error": "already rendering"}
        outdir = os.path.dirname(paths[0])
        lockp = os.path.join(outdir, ".multigpu-lock.json")
        probe = os.path.join(outdir, ".multigpu-probe-%s" % uuid.uuid4().hex[:8])
        try:                                       # can we write there? (also gives us the NAS's own clock)
            os.makedirs(outdir, exist_ok=True)
            with open(probe, "w") as f:
                f.write("x")
            nas_now = os.path.getmtime(probe)
            os.remove(probe)
        except OSError as e:
            return {"error": "Can't write to the output folder %s (%s)" % (outdir, e.strerror or e)}
        try:
            with open(lockp, encoding="utf-8") as f:
                other = json.load(f)
            if other.get("host") != spec.get("host") and other.get("state") == "running" \
                    and nas_now - os.path.getmtime(lockp) < 50:
                return {"error": "%s is already rendering into this folder. Join it instead of starting "
                                 "a second render." % (other.get("host_name") or "Another computer")}
        except (OSError, ValueError):
            pass
        overwrite = bool((spec.get("opts") or {}).get("overwrite"))
        sizes = {} if overwrite else scan_sizes(paths)
        with self.lock:
            if self.job and self.job["state"] == "running":
                return {"error": "already rendering"}
            fs, existing = {}, 0
            for f, p in zip(frames, paths):
                if not overwrite and sizes.get(_norm(p), 0) > 0:
                    fs[str(f)] = {"s": "done", "v": True, "pre": True}
                    existing += 1
                else:
                    fs[str(f)] = {"s": "todo"}
            t = now()
            job = self.job = {"id": uuid.uuid4().hex[:12], "state": "running", "started": t, "ended": None,
                              "host": spec.get("host"), "host_name": spec.get("host_name"), "blend": spec.get("blend"),
                              "origin": spec.get("host"), "term": 0,
                              "scene": spec.get("scene"), "output": spec.get("output"), "outdir": outdir,
                              "opts": spec.get("opts") or {}, "frames": frames, "paths": paths, "fs": fs,
                              "existing": existing, "sync": 0, "sync_pending": False, "synced_at": None,
                              "frames_n": len(frames), "first": frames[0], "unsynced": False, "slowest": None,
                              "files": dict((file_key(f), f) for f in (spec.get("files") or [])[:FILES_MAX]
                                            if isinstance(f, str)),
                              "sets": dict((file_key(k), [f for f in v if isinstance(f, str)])
                                           for k, v in (spec.get("sets") or {}).items() if isinstance(v, list)),
                              "pattern": spec.get("pattern") if isinstance(spec.get("pattern"), dict) else None}
            for c in self.cards.values():
                c.update(frame=None, frame_t=None, phase="off", stopped=False, streak=0, done=0, err=None,
                         warn=None, at=None, last_rid=None, last_fail=None, last_err=None, backup=False,
                         restart=False, tile=None, avg=None, held=None, hold_off=False)
            for comp in self.computers.values():
                comp.update(bad=0, blocked=None)
            self.no_live = None
            self.nas_down = False
            self.pre = None
            self.event("Render started: %d frames%s" % (len(frames), ", %d already done" % existing if existing else ""))
            for comp_id, have in (spec.get("have") or {}).items():     # frames other computers hold in their own copy
                if isinstance(have, list):
                    self.mark_have(comp_id, from_ranges(have))
            self.check_finished()
            res = {"ok": True, "job": job["id"], "existing": existing, "status": self.status()}
        if job["state"] == "running":
            self.write_lock(job)
            self.lock_written = self.lock_ok = now()
        return res

    def op_stop(self, req, ip):
        with self.lock:
            j = self.job
            if not j or j["state"] != "running":
                return {"ok": True}
            j["state"] = "stopped"
            j["ended"] = now()
            for c in self.cards.values():
                self.release(c)
                if c.get("phase") in LIVE:
                    c["phase"] = "off"
            self.schedule_cleanup()
            self.event("Stopped (from %s)" % (req.get("by") or ip))
            return {"ok": True}

    def op_card_stop(self, req, ip):
        with self.lock:
            c = self.cards.get(req.get("card"))
            if not c:
                return {"error": "no such card"}
            self.release(c)
            c["stopped"] = True
            c["phase"] = "stopped"
            c["stop_why"] = None
            self.event("%s stopped by you" % self.label(c))
            return {"ok": True}

    def op_card_use(self, req, ip):
        """Tick or untick a card on any computer (from the host's panel)."""
        with self.lock:
            c = self.cards.get(req.get("card"))
            if not c:
                return {"error": "no such card"}
            c["use"] = bool(req.get("use"))
            c["use_v"] = c.get("use_v", 0) + 1
            if not c["use"]:
                self.release(c)
            self.event("%s %s" % (self.label(c), "ticked" if c["use"] else "unticked"))
            return {"ok": True}

    def op_card_resume(self, req, ip):
        with self.lock:
            c = self.cards.get(req.get("card"))
            if not c:
                return {"error": "no such card"}
            c.update(stopped=False, phase="off", err=None, streak=0, last_fail=None, epoch=c.get("epoch", 0) + 1)
            self.event("%s resumed" % self.label(c))
            return {"ok": True}

    def op_sync_begin(self, req, ip):
        with self.lock:
            if not self.job or self.job["state"] != "running":
                return {"error": "nothing is rendering"}
            self.job["sync_pending"] = True
            self.dirty = True
            return {"ok": True}

    def op_sync_end(self, req, ip):
        with self.lock:
            j = self.job
            if not j:
                return {"error": "nothing is rendering"}
            if req.get("job") is not None and req["job"] != j["id"]:
                return {"error": "the render changed while saving"}
            if req.get("token") == j["blend"].get("token"):
                return {"ok": True}               # a lost acknowledgement can safely be retried
            j["blend"]["token"] = req.get("token")
            j["blend"]["size"] = req.get("size")
            if isinstance(req.get("files"), list):         # the saved scene may use other files now
                j["files"] = dict((file_key(f), f) for f in req["files"][:FILES_MAX] if isinstance(f, str))
                j["sets"] = dict((file_key(k), [f for f in v if isinstance(f, str)])
                                 for k, v in (req.get("sets") or {}).items() if isinstance(v, list))
            j["sync"] += 1
            j["sync_pending"] = False
            j["unsynced"] = False
            j["synced_at"] = now()
            self.dirty = True
            self.event("Changes synced; each card reloads after its frame")
            return {"ok": True}

    def op_sync_abort(self, req, ip):
        with self.lock:
            if self.job:
                self.job["sync_pending"] = False
                self.dirty = True
            return {"ok": True}

    def op_unsynced(self, req, ip):
        """The .blend was saved while rendering, without Sync Changes."""
        with self.lock:
            j = self.job
            if j and j["state"] == "running" and not j.get("unsynced"):
                j["unsynced"] = True
                self.event("You saved the .blend. The cards keep rendering the earlier version until you press Sync Changes")
            return {"ok": True}

    def op_replica(self, req, ip):
        """Everything another computer needs to carry this render on as host, should this one stop answering."""
        with self.lock:
            j = self.job
            if not j or j["state"] != "running":
                return {"ok": True, "none": True}
            t = now()
            if self.rep[1] is not None and self.rep[2] == j["id"] and t - self.rep[0] < 4:
                return {"ok": True, "job": j["id"], "term": j.get("term", 0), "t": self.rep[0], "blob": self.rep[1]}
            text = json.dumps({"job": j, "cards": self.cards, "computers": self.computers, "events": self.events,
                               "t": t})
            jid, term = j["id"], j.get("term", 0)
        blob = base64.b64encode(zlib.compress(text.encode("utf-8"), 3)).decode("ascii")
        self.rep = (t, blob, jid)
        return {"ok": True, "job": jid, "term": term, "t": t, "blob": blob}

    def op_status(self, req, ip):
        with self.lock:
            return {"ok": True, "status": self.status()}

    def op_quit(self, req, ip):
        with self.lock:
            if self.job and self.job["state"] == "running" and not req.get("force"):
                return {"error": "still rendering"}
        threading.Thread(target=self.server.shutdown, daemon=True).start()
        return {"ok": True}

    # ---- what the panels show --------------------------------------------------------
    def phase(self, c, t):
        ph = c.get("phase") or "off"
        if ph in ("rendering", "ready", "idle", "loading") and t - c.get("seen", 0) > 30:
            return "lost"
        return ph

    def status(self):
        t = now()
        j = self.job
        limit = self.hang_limit() if j and j["state"] == "running" else None
        comps = []
        for comp in sorted(self.computers.values(), key=lambda c: (not c.get("host"), c.get("name") or "")):
            online = t - comp.get("seen", 0) < 15
            cards = []
            for kid in comp.get("cards", []):
                c = self.cards.get(kid)
                if not c:
                    continue
                secs = t - c["frame_t"] if c.get("frame_t") else None
                cards.append({"id": kid, "label": c.get("label"), "backend": c.get("backend"), "use": bool(c.get("use")),
                              "phase": self.phase(c, t), "frame": c.get("frame"), "secs": secs,
                              "done": c.get("done", 0), "avg": c.get("avg"), "stopped": c.get("stopped"),
                              "stop_why": c.get("stop_why") if c.get("stopped") else None,
                              "err": c.get("err"), "warn": c.get("warn"),
                              "backup": bool(c.get("backup") and c.get("frame") is not None),
                              "held": bool(c.get("held")),
                              "slow": bool(limit and secs and secs > limit)})
            if not online and t - comp.get("seen", 0) > 600 and not any(x["phase"] in LIVE for x in cards):
                continue
            r = comp.get("resolve") or {}
            pre = comp.get("pre") if self.pre and (comp.get("pre") or {}).get("id") == self.pre.get("id") else None
            comps.append({"id": comp["id"], "name": comp.get("name"), "os": comp.get("os"), "online": online,
                          "host": comp.get("host"), "blender": comp.get("blender"), "blocked": comp.get("blocked"),
                          "away": self.away(comp["id"]),
                          "resolve_error": r.get("error") if j and r.get("job") == j["id"] and not r.get("ok") else None,
                          "resolving": bool(j and r.get("job") == j["id"] and r.get("busy")), "cards": cards,
                          "pre": pre, "ip": comp.get("ip")})
        out = {"computers": comps, "events": self.events[-8:], "version": self.version, "now": t}
        if j:
            n = {"todo": 0, "busy": 0, "done": 0, "failed": 0}
            failed, unverified, err = [], 0, None
            for f in j["frames"]:
                st = self.fs(f)
                n[st["s"]] += 1
                if st["s"] == "failed":
                    failed.append(f)
                    err = err or st.get("err")
                elif st["s"] == "done" and not st.get("v"):
                    unverified += 1
            arriving = sum(1 for f in j["frames"] if self.fs(f).get("away"))
            rate = 0.0
            for c in self.cards.values():
                if c.get("avg") and c.get("phase") in ("rendering", "ready", "idle", "loading") \
                        and t - c.get("seen", 0) < 30:
                    rate += 1.0 / c["avg"]
            left = n["todo"] + n["busy"]
            running = j["state"] == "running"
            out["job"] = {"id": j["id"], "state": j["state"], "total": len(j["frames"]), "done": n["done"],
                          "busy": n["busy"], "todo": n["todo"], "failed": failed[:40], "nfailed": n["failed"],
                          "fail_err": err, "unverified": unverified, "existing": j.get("existing", 0), "arriving": arriving,
                          "eta": left / rate if rate > 0 and left else None, "started": j["started"],
                          "ended": j.get("ended"), "synced_at": j.get("synced_at"), "sync": j["sync"],
                          "sync_pending": j.get("sync_pending"), "outdir": j["outdir"],
                          "name": (j.get("blend") or {}).get("name"), "host": j.get("host"),
                          "blend": (j.get("blend") or {}).get("path"), "scene": j.get("scene"),
                          "first": j["frames"][0], "last": j["frames"][-1],
                          "unsynced": bool(j.get("unsynced")) and running, "nas_down": self.nas_down and running,
                          "stalled": bool(running and left and self.no_live and t - self.no_live > 45)}
        return out

    def text_status(self):
        with self.lock:
            s = self.status()
        lines = ["MultiGPU host %s" % self.version, ""]
        j = s.get("job")
        if j:
            lines.append("%s: %d / %d frames, %s" % (j["name"], j["done"], j["total"], j["state"]))
        for comp in s["computers"]:
            lines.append("")
            lines.append("%s%s%s" % (comp["name"], " (host)" if comp["host"] else "", "" if comp["online"] else " - offline"))
            for c in comp["cards"]:
                if c["use"]:
                    lines.append("   %-28s %-10s frame %-6s done %d" % (c["label"], c["phase"], c["frame"], c["done"]))
        lines.append("")
        lines += [time.strftime("%H:%M:%S ", time.localtime(t)) + e for t, e in s["events"]]
        return "\n".join(lines) + "\n"

    # ---- housekeeping every 2 seconds ------------------------------------------------
    def maintain(self):
        while True:
            time.sleep(2)
            try:
                self.maintain_once()
            except Exception:
                traceback.print_exc()

    def maintain_once(self):
        t = now()
        check, lockjob, cleanup, orphans, watchjob = [], None, None, [], None
        arriving, old_probes = [], []
        with self.lock:
            j = self.job
            running = bool(j and j["state"] == "running")
            old_probes = [q for q in self.probes if t - q[0] > 30]
            self.probes = [q for q in self.probes if t - q[0] <= 30]
            if j and t - self.arrive_t > 10:           # frames rendered into another computer's own copy of the
                self.arrive_t = t                       # folder: have they reached this one yet?
                arriving = [(f, q) for f, q in zip(j["frames"], j["paths"]) if self.fs(f).get("away")][:2000]
            if running:
                for c in self.cards.values():
                    if c.get("frame") is not None and \
                            t - c.get("seen", 0) > (LEASE_AWAY if self.away(c.get("computer")) else LEASE):
                        f = c["frame"]
                        self.release(c)
                        c["phase"] = "lost"
                        self.event("Lost contact with %s; frame %d goes to another card" % (self.label(c), f))
                for f, p in zip(j["frames"], j["paths"]):
                    st = self.fs(f)
                    if st["s"] == "done" and not st.get("v"):
                        check.append((f, p, st.get("t") or t))
                    elif st["s"] == "busy":       # safety net: a busy frame always has a card that holds it
                        o = self.cards.get(st.get("card") or "")
                        b = self.cards.get(st.get("backup") or "")
                        if b is not None and b.get("frame") != f:
                            st["backup"] = b = None
                        if not o or o.get("frame") != f:
                            if b is not None:
                                st.update(card=b["id"], backup=None)
                                b["backup"] = False
                            else:
                                st.update(s="todo", card=None, backup=None)
                            self.dirty = True
                live = any(self.phase(c, t) in LIVE for c in self.cards.values() if c.get("use"))
                self.no_live = None if live else (self.no_live or t)
                if t - self.lock_written > LOCK_EVERY:
                    lockjob = j
                elif self.standby(t) or self.took_over:
                    watchjob = j                   # another computer could take over: look at the marker often
                if self.orphans and t - self.resumed_at > min(self.holdoff, LEASE):
                    self.orphans = False           # frames nobody is rendering anymore: their 0 KB files go
                    orphans = [p for f, p in zip(j["frames"], j["paths"]) if self.fs(f)["s"] == "todo"]
            if self.cleanup:
                cleanup, self.cleanup = self.cleanup, []
        for _t, q in old_probes:
            try:
                os.remove(q)
            except OSError:
                pass
        if arriving:
            sizes = scan_sizes([q for _f, q in arriving])
            here = [f for f, q in arriving if sizes.get(_norm(q), 0) > 0]
            if here:
                with self.lock:
                    if self.job is j:
                        for f in here:
                            self.fs(f)["away"] = False
                        self.dirty = True
                        if j["state"] == "finished" and not any(self.fs(f).get("away") for f in j["frames"]):
                            self.event("Every frame has arrived in %s" % j["outdir"])
        if running:
            self.awake.on()
        else:
            self.awake.off()
        # file system work happens outside the lock: the NAS can be slow
        sizes, dirs = {}, {}
        for f, p, _ in check[:300]:
            try:
                sizes[f] = os.path.getsize(p)
            except OSError:
                sizes[f] = -1
                d = os.path.dirname(p)
                if d not in dirs:
                    dirs[d] = os.path.isdir(d)
        down = any(not ok for ok in dirs.values())
        if check:
            with self.lock:
                j = self.job
                if j and j["state"] == "running":
                    if down != self.nas_down:
                        self.nas_down = down
                        self.event("The host can't reach the output folder %s. Waiting for it to come back"
                                   % j["outdir"] if down else "The output folder is reachable again")
                    for f, p, td in check[:300]:
                        st = self.fs(f)
                        if st["s"] != "done" or st.get("v"):
                            continue
                        want = st.get("size")
                        if sizes.get(f, -1) > 0 and (not want or sizes[f] == want):
                            st["v"] = True
                            self.dirty = True
                        elif down:
                            st["t"] = t           # not the cards' fault: don't count this time against them
                        elif t - td > VERIFY_GRACE:
                            comp = self.computers.get(st.get("comp")) or {}
                            st["vfail"] = st.get("vfail", 0) + 1
                            if st["vfail"] >= 2:
                                st.update(s="failed", v=False, err="it was rendered twice, but the file never "
                                          "appeared at %s" % p)
                                self.event("Frame %d was rendered twice but never appeared in %s; skipped" % (f, j["outdir"]))
                                continue
                            st.update(s="todo", v=False, by=None, size=None)
                            comp["bad"] = comp.get("bad", 0) + 1
                            self.event("Frame %d from %s never showed up in %s; rendering it again"
                                       % (f, comp.get("name", "?"), j["outdir"]))
                            if comp.get("bad", 0) >= 2 and not comp.get("blocked") and not comp.get("host"):
                                comp["blocked"] = ("its frames don't show up in the host's output folder. Make sure "
                                                   "both computers save to the same NAS folder")
                                self.event("%s stopped: %s" % (comp.get("name"), comp["blocked"]))
                    self.check_finished()
        elif self.nas_down and running:
            with self.lock:
                self.nas_down = False
        if watchjob:
            cur = self.read_lock(watchjob)
            other = self.other_host(watchjob, cur)
            if other:
                self.deposed(other)
            elif cur is not None and cur.get("n") != self.lock_mine:
                lockjob = watchjob                 # an earlier host wrote over ours: put ours back right away
            elif cur is not None:
                self.lock_ok = now()
        if lockjob:
            other = self.other_host(lockjob)
            if other:
                self.deposed(other)
            elif self.write_lock(lockjob):
                self.lock_written = self.lock_ok = now()
                if self.took_over:
                    self.took_over = False
                    self.clear_claims(lockjob)
        removed = 0
        if orphans:
            sizes = scan_sizes(orphans)
            for p in orphans:
                if sizes.get(_norm(p)) == 0:
                    try:
                        os.remove(p)
                        removed += 1
                    except OSError:
                        pass
        if removed:
            with self.lock:
                self.event("Removed %d placeholder%s the earlier host's cards left behind" % (removed, "" if removed == 1 else "s"))
        for item in cleanup or []:
            self.do_cleanup(item)
        self.save()
        if not running and t - self.host_seen > 1200:
            print("No host Blender for 20 minutes; closing")
            self.save()
            os._exit(0)

    def do_cleanup(self, item):
        def still_over():                           # stop as soon as a new render starts
            with self.lock:
                return not (self.job and self.job["state"] == "running")
        removed = 0
        for p in item["placeholders"]:
            if not still_over():
                return
            try:
                if os.path.getsize(p) == 0:
                    os.remove(p)
                    removed += 1
            except OSError:
                pass
        for d in item["outdirs"]:
            try:
                for n in os.listdir(d):
                    if n.startswith(".multigpu-probe-"):
                        os.remove(os.path.join(d, n))
            except OSError:
                pass
            part = os.path.join(d, ".multigpu-partial")
            try:
                for sub in os.listdir(part):
                    sp = os.path.join(part, sub)
                    for n in os.listdir(sp):
                        try:
                            os.remove(os.path.join(sp, n))
                        except OSError:
                            pass
                    try:
                        os.rmdir(sp)
                    except OSError:
                        pass
                os.rmdir(part)
            except OSError:
                pass
        if still_over():
            try:
                os.remove(item["lock"])
            except OSError:
                pass
            try:
                for n in os.listdir(os.path.dirname(item["lock"])):
                    if re.match(r"\.multigpu-takeover-\d+\.json$", n):
                        os.remove(os.path.join(os.path.dirname(item["lock"]), n))
            except OSError:
                pass
        if removed:
            with self.lock:
                self.event("Removed %d unfinished placeholder%s" % (removed, "" if removed == 1 else "s"))


def _dur(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return "%dh %02dm" % (h, m) if h else ("%dm %02ds" % (m, s) if m else "%ds" % s)


C = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, C.text_status().encode("utf-8"), "text/plain; charset=utf-8")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if 0 < n < 20000000 else {}
        except (ValueError, OSError):
            return self._send(400, {"error": "bad request"})
        ip = self.client_address[0]
        local = ip in ("127.0.0.1", "::1", "::ffff:127.0.0.1")
        try:
            res = C.handle(req, local, ip)
        except Exception as e:
            traceback.print_exc()
            res = {"error": "host error: %s" % e}
        try:
            self._send(200, res)
        except OSError:
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"     # on Windows that would let two hosts share a port


def open_port(opts):
    """Listen on the port. If an earlier MultiGPU host still has it, reuse that one or ask it to leave."""
    bind, port = opts["bind"], int(opts["port"])
    try:
        return Server((bind, port), Handler), None
    except OSError as e:
        first = e
    other = None
    try:
        other = rpc("127.0.0.1:%d" % port, {"op": "ping"}, timeout=2)
    except OSError:
        pass
    if not (other and other.get("ok") and other.get("pid")):
        return None, {"error": "Port %d is already used by another program on this computer (%s). Pick another "
                               "port in the MultiGPU preferences, the same on every computer" % (port, first)}
    if other.get("version") == opts.get("version") and other.get("bind") == bind and not opts.get("takeover"):
        return None, {"pid": other["pid"], "pstart": other.get("pstart"), "port": port, "bind": bind, "t": time.time(),
                      "protocol": PROTOCOL, "version": other.get("version"), "adopted": True}
    try:
        rpc("127.0.0.1:%d" % port, {"op": "quit"}, timeout=2)
    except OSError:
        pass
    for _ in range(40):
        time.sleep(0.25)
        try:
            return Server((bind, port), Handler), None
        except OSError:
            pass
    return None, {"error": "An earlier MultiGPU host (%s) is still using port %d%s" % (
        other.get("version"), port, " and is rendering. Stop that render first" if other.get("running") else "")}


def main():
    global C, LEASE, VERIFY_GRACE, HANG_MIN, LOCK_EVERY, LOCK_TRUST
    opts = json.loads(sys.argv[sys.argv.index("--") + 1])
    LOCK_EVERY = opts.get("lock_every", LOCK_EVERY)
    LOCK_TRUST = opts.get("lock_trust", LOCK_TRUST)
    LEASE = opts.get("lease", LEASE)
    VERIFY_GRACE = opts.get("grace", VERIFY_GRACE)
    HANG_MIN = opts.get("hang", HANG_MIN)
    C = Coordinator(opts)
    if opts.get("resume"):
        C.load()
    srv, info = open_port(opts)
    if srv is None:
        info["pid"] = info.get("pid") or os.getpid()
        with open(opts["info"] + ".tmp", "w", encoding="utf-8") as f:
            json.dump(info, f)
        os.replace(opts["info"] + ".tmp", opts["info"])
        sys.exit(0 if info.get("adopted") else 2)
    C.server = srv
    info = {"pid": os.getpid(), "pstart": proc_start(os.getpid()), "port": srv.server_address[1], "bind": opts["bind"],
            "t": time.time(), "protocol": PROTOCOL, "version": C.version}
    with open(opts["info"] + ".tmp", "w", encoding="utf-8") as f:
        json.dump(info, f)
    os.replace(opts["info"] + ".tmp", opts["info"])
    print("MultiGPU host listening on %s:%d" % (opts["bind"], srv.server_address[1]))
    sys.stdout.flush()
    threading.Thread(target=C.maintain, daemon=True).start()
    srv.serve_forever(poll_interval=0.5)
    C.save()


main()
'''


# ======================================================================================
# The script each card's Blender runs: asks the host for a frame, renders it, repeats.
# ======================================================================================
WORKER = r'''
import atexit, base64, hashlib, shutil, tempfile, threading, traceback
import bpy

cfg = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))
ADDR, CARD, JOB = cfg["addr"], cfg["card"], cfg["job"]
OPTS = cfg.get("opts") or {}
MAPS = [list(m) for m in cfg.get("maps") or []]
TAG = re.sub(r"[^A-Za-z0-9_-]", "_", CARD)
ME = {"frame": None, "exiting": False, "tmps": set(), "n": 0, "log_pos": 0, "lock": None}
st = {"label": cfg["label"], "pid": os.getpid(), "state": "starting", "frame": None, "error": None}


def put(**kw):
    st.update(kw)
    st["t"] = time.time()
    tmp = cfg["status"] + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, cfg["status"])
    except OSError:
        pass


def log(msg):
    print("MultiGPU: " + msg)
    sys.stdout.flush()


def host_addr():
    """Where the host is now. This computer's MultiGPU notes it in a small file when another computer
    takes over as host mid-render, so a card that's rendering simply carries on with the new one."""
    p = cfg.get("addr_file")
    if p:
        try:
            m = os.path.getmtime(p)
            if m != ME.get("addr_m"):
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                ME["addr_m"] = m
                if d.get("job") == JOB and d.get("addr") and d["addr"] != ME.get("addr"):
                    ME["addr"] = d["addr"]
                    log("the host is now at %s" % d["addr"])
        except (OSError, ValueError):
            pass
    return ME.get("addr") or ADDR


def call(want, timeout=8.0, **kw):
    req = {"op": "card", "protocol": PROTOCOL, "addon": cfg.get("version"), "card": CARD,
           "computer": cfg["computer"], "label": cfg["label"], "job": JOB, "sync": cfg["sync"], "want": want}
    req.update(kw)
    return rpc(host_addr(), req, timeout=timeout)


def clear_tmp(d):
    try:
        for n in os.listdir(d):
            try:
                os.remove(os.path.join(d, n))
            except OSError:
                pass
    except OSError:
        pass


def leave(state, code, why=None, error=None, tell=True):
    ME["exiting"] = True
    put(state=state, frame=None, error=error)
    if tell:
        try:
            call("bye", timeout=3.0, why=why or state, error=error)
        except OSError:
            pass
    for d in ME["tmps"]:
        clear_tmp(d)
        for p in (d, os.path.dirname(d)):
            try:
                os.rmdir(p)
            except OSError:
                pass
    log("leaving: %s%s" % (state, " - " + error if error else ""))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def ask(want, **kw):
    """Talk to the host, retrying for up to 10 minutes while it can't be reached."""
    t0, delay = time.time(), 2.0
    while True:
        try:
            return call(want, **kw)
        except OSError as e:
            if time.time() - t0 > 600:
                leave("error", 4, error="lost the host for 10 minutes (%s)" % e, tell=False)
            put(state="waiting for host", error=str(e))
            log("can't reach the host (%s); trying again" % e)
            time.sleep(delay)
            delay = min(10.0, delay + 2.0)


def on_exit_reply(r):
    why = r.get("exit")
    if why == "reload":
        leave("reload", 5, tell=False)
    if why == "error":
        leave("error", 3, error=r.get("why"), tell=False)
    leave(why if why in ("done", "stopped", "over") else "done", 0, tell=False)


# ---- one Blender per card ----------------------------------------------------------------
def card_lock():
    """Hold an operating-system lock for this card for as long as this Blender lives. False if another
    Blender on this computer already renders as this card (so the card is never used twice at once)."""
    path = cfg.get("lock")
    if not path:
        return True
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = open(path, "a+b")
    except OSError:
        return True
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        try:
            with open(path + ".pid", encoding="utf-8") as g:
                other = json.load(g)
        except (OSError, ValueError):
            other = None
        if other and other.get("pid") and other["pid"] != os.getpid():
            cur = proc_start(other["pid"])
            if cur is not None and (other.get("pstart") is None or str(cur) == str(other["pstart"])):
                return False                       # that Blender is really there
        return True
    except Exception:
        return True
    ME["lock"] = f
    try:
        with open(path + ".pid", "w", encoding="utf-8") as g:
            json.dump({"pid": os.getpid(), "pstart": proc_start(os.getpid())}, g)
    except OSError:
        pass
    return True


if not card_lock():
    put(state="duplicate")
    log("another Blender on this computer is already rendering as this card; leaving it to that one")
    os._exit(7)

# ---- the right file? -------------------------------------------------------------------
if not bpy.data.filepath:
    leave("error", 3, error="Blender %s on this computer couldn't open the .blend. Was it saved with a newer "
          "Blender?" % bpy.app.version_string)
sc = bpy.context.scene
if cfg.get("scene") and sc.name != cfg["scene"]:
    other = bpy.data.scenes.get(cfg["scene"])
    if other is None:
        leave("error", 3, error="scene %r isn't in this copy of the file" % cfg["scene"])
    sc = other
    try:
        bpy.context.window.scene = sc
    except Exception:
        pass
rd = sc.render
if cfg.get("token") and sc.get("_multigpu_token") != cfg["token"]:
    put(state="stale")
    log("this computer still sees a different version of the file; trying again shortly")
    os._exit(6)

AWAKE = Awake()
AWAKE.on()                                  # the computer mustn't go to sleep mid-render

# ---- pick this card ------------------------------------------------------------------
on = []
if cfg["backend"] == "CPU":
    for s in bpy.data.scenes:
        try:
            s.cycles.device = "CPU"
        except Exception:
            pass
    on = ["CPU"]
else:
    # Cycles also remembers cards that are no longer there. Switching on one of those renders on the
    # CPU without a word, so only a card it can see right now may be used.
    want = cfg["device_id"]
    try:
        import _cycles
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = cfg["backend"]
        prefs.get_devices_for_type(cfg["backend"])
        here = [d[2] for d in _cycles.available_devices(cfg["backend"]) if len(d) >= 3 and d[1] == cfg["backend"]]
        if want not in here:
            # A card gets another address when a card is added, removed or stops working: find it by name
            same = [i for i in here if device_base(i) == device_base(want) and i not in (cfg.get("other_ids") or [])]
            if len(same) == 1:
                print("MultiGPU: %s is at a new address (%s, was %s)" % (cfg["label"], same[0], want), flush=True)
                want = same[0]
            else:
                want = None
        for d in prefs.devices:
            d.use = bool(want) and d.id == want and d.type == cfg["backend"]
            if d.use:
                on.append(d.name)
    except Exception as e:
        leave("error", 3, error="couldn't set up %s: %s" % (cfg["backend_label"], e))
    if not on:
        leave("error", 3, error="%s wasn't found with %s. Is the card still working? Press Find GPUs on this "
                                "computer to look again" % (cfg["label"], cfg["backend_label"]))
    for s in bpy.data.scenes:
        try:
            s.cycles.device = "GPU"
        except Exception:
            pass
rd.use_persistent_data = bool(OPTS.get("persistent", True))
for _k in ("compositing", "sequencer"):           # Post Processing, as the host's scene has it
    if OPTS.get(_k) is not None:
        try:
            setattr(rd, "use_" + _k, bool(OPTS[_k]))
        except Exception:
            pass
apply_format(rd.image_settings, OPTS.get("format"))
if OPTS.get("ext") is not None:
    rd.use_file_extension = bool(OPTS["ext"])
TILE = cfg.get("tile")                       # set after this card ran out of GPU memory: render in smaller tiles
if TILE:
    try:
        sc.cycles.use_auto_tile = True
        sc.cycles.tile_size = int(TILE)
    except Exception:
        TILE = None


def tile_now():
    try:
        return int(sc.cycles.tile_size) if sc.cycles.use_auto_tile else None
    except Exception:
        return None


def smaller_tile():
    """The next tile size to try after running out of GPU memory, or None when tiles can't get smaller."""
    try:
        side = max(rd.resolution_x, rd.resolution_y) * rd.resolution_percentage // 100
    except Exception:
        side = 2048
    base = tile_now() or 2048
    while base > 256 and base // 2 >= side:      # a tile as big as the image doesn't split anything
        base //= 2
    new = base // 2
    return new if new >= 256 else None


# ---- paths from the host's computer -> this computer -------------------------------------
_df = None
try:                                        # files that ship with Blender live in each computer's own install
    _df = bpy.utils.system_resource('DATAFILES')
    if OPTS.get("datafiles") and _df and split_any(_df) != split_any(OPTS["datafiles"]):
        MAPS.append([OPTS["datafiles"], _df])
except Exception:
    pass


def _swap(owner, attr):
    fp = getattr(owner, attr, None)
    if not fp or not isinstance(fp, str) or fp.startswith("//") or fp.startswith("<"):
        return 0
    new = map_path(fp, MAPS)
    if new == fp:
        return 0
    try:
        setattr(owner, attr, new)
        return 1
    except Exception as e:
        log("couldn't convert %s: %s" % (fp, e))
        return 0


def remap():
    n = 0
    if not MAPS:
        return 0
    for coll in ("images", "libraries", "cache_files", "sounds", "movieclips", "volumes", "fonts"):
        for idb in list(getattr(bpy.data, coll, [])):
            fp = getattr(idb, "filepath", "") or ""
            if not fp or fp.startswith("//") or fp.startswith("<") or getattr(idb, "packed_file", None):
                continue
            new = map_path(fp, MAPS)
            if new != fp and os.path.isdir(os.path.dirname(new)):
                try:
                    idb.filepath = new
                    if coll == "libraries":
                        idb.reload()
                    n += 1
                except Exception as e:
                    log("couldn't convert %s: %s" % (fp, e))
    for ob in bpy.data.objects:
        for md in getattr(ob, "modifiers", []):
            n += _swap(md, "bake_directory")
            if md.type == "FLUID" and getattr(md, "domain_settings", None):
                n += _swap(md.domain_settings, "cache_directory")
        try:                                        # FLIP Fluids cache folder
            n += _swap(ob.flip_fluid.domain.cache, "cache_directory")
        except Exception:
            pass
    trees = [g for g in bpy.data.node_groups if g.bl_idname == "CompositorNodeTree"]
    for s in bpy.data.scenes:
        try:
            if getattr(s, "node_tree", None):
                trees.append(s.node_tree)
        except Exception:
            pass
    for tree in trees:                              # compositor File Output nodes
        for node in tree.nodes:
            if node.bl_idname == "CompositorNodeOutputFile":
                n += _swap(node, "directory") + _swap(node, "base_path")
    return n


AWAY = bool(cfg.get("away"))    # this computer renders into its own copy of the folder, which a sync program carries
LIBMAPS = []    # [the host's folder, this computer's folder] of copied files: files next to them are found too


def host_forms(p, raw=None, lib=None):
    """The ways the host may know a path that this computer sees as p (raw: as the scene has it; lib: the
    library it comes from)."""
    out = []

    def to_host(q):
        for maps in ([[lp, hp] for hp, lp in LIBMAPS], [[lp, hp] for hp, lp in MAPS]):
            r = map_path(q, maps)
            if r != q:
                return r
        return q
    if raw and raw.startswith("//"):
        # A path relative to a file: the host worked it out from where ITS copy of that file is, which can lead
        # somewhere else entirely than from this computer's copy.
        try:
            base = bpy.data.filepath if lib is None else os.path.normpath(bpy.path.abspath(lib.filepath))
            out.append(norm_join(strip_tail(to_host(base), 1), raw[2:]))
        except Exception:
            pass
    for q in (to_host(p), p):
        if q not in out:
            out.append(q)
    return out


def cache_dir(host_dir):
    """Where this computer keeps the files it was sent from one folder of the host. Files of one folder stay
    together (frames of a sequence, tiles of an image), and the folder is remembered so that a copied .blend
    library finds the files it uses by a path relative to itself."""
    d = os.path.join(STORE, hashlib.sha1(file_key(host_dir).encode("utf-8")).hexdigest()[:12], "_", "_", "_", "_")
    if not any(lp == d for _hp, lp in LIBMAPS):
        for k in range(5):
            hp, lp = strip_tail(host_dir, k), strip_tail(d, k)
            if not hp or not split_any(hp)[1]:
                break
            LIBMAPS.append([hp, lp])
    return d


def local_name(hp):
    name = (split_any(hp)[1] or ["file"])[-1]
    if os.name == "nt":
        name = re.sub(r'[:*?"|]', "_", name)
    return name


def fetch_one(hp):
    """Get one file from the host. Returns where it is kept on this computer, or None."""
    try:
        r = rpc(host_addr(), {"op": "file", "path": hp}, timeout=10)
    except OSError:
        return None
    if not r.get("ok"):
        log("the host didn't send %s: %s" % (local_name(hp), r.get("error")))
        return None
    size, mtime = int(r["size"]), float(r["mtime"])
    stamp = r.get("stamp")
    name = local_name(hp)
    d = cache_dir(strip_tail(hp, 1))
    dst, lock = os.path.join(d, name), os.path.join(d, name + ".lock")
    tmp, meta = dst + ".part", dst + ".part.json"
    identity = {"path": hp, "size": size, "mtime": mtime, "stamp": stamp}

    def good():
        return os.path.isfile(dst) and os.path.getsize(dst) == size and abs(os.path.getmtime(dst) - mtime) < 0.000001
    mine = False
    stop, heartbeat = threading.Event(), None

    def keep_lock():
        while not stop.wait(10):
            try:
                os.utime(lock, None)
            except OSError:
                return

    try:
        os.makedirs(d, exist_ok=True)
        while not good():                             # one card at a time gets a file; the others wait for it
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                mine = True
                heartbeat = threading.Thread(target=keep_lock, daemon=True)
                heartbeat.start()
                break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lock) > 90:
                        os.remove(lock)
                        continue
                except OSError:
                    continue
                time.sleep(0.5)
        if good():
            os.utime(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(d)))), None)   # still in use
            return dst
        if size < 0 or size > FILE_MAX:
            log("not copying %s from the host: invalid or oversized file" % name)
            return None
        try:
            with open(meta, encoding="utf-8") as f:
                resume = json.load(f) == identity and os.path.getsize(tmp) <= size
        except (OSError, ValueError):
            resume = False
        got = os.path.getsize(tmp) if resume else 0
        if shutil.disk_usage(d).free < size - got + (2 << 30):
            log("not copying %s from the host: no room for it in this computer's temporary folder" % name)
            return None
        if not resume:
            with open(tmp, "wb"):
                pass
            with open(meta, "w", encoding="utf-8") as f:
                json.dump(identity, f)
        chunk = min(FILE_CHUNK, 64 << 10)
        with open(tmp, "ab") as f:
            while got < size:
                want = min(chunk, size - got)
                for attempt in range(4):
                    t0 = time.monotonic()
                    try:
                        r = rpc(host_addr(), {"op": "file", "path": hp, "off": got,
                                              "n": want, "stamp": stamp}, timeout=60)
                        break
                    except OSError:
                        if attempt == 3:
                            raise
                        os.utime(lock, None)
                        time.sleep(2 ** attempt)
                        want = max(1, min(size - got, max(16 << 10, want // 2)))
                data = base64.b64decode(r.get("data") or "", validate=True) if r.get("ok") else b""
                if not data or len(data) > want or int(r.get("size", -1)) != size \
                        or r.get("mtime") != mtime or (stamp is not None and r.get("stamp") != stamp):
                    # A new version must never be appended to the old version's partial copy.
                    try:
                        os.remove(meta)
                    except OSError:
                        pass
                    raise OSError(r.get("error") or "the file changed or the copy stopped")
                f.write(data)
                f.flush()
                got += len(data)
                os.utime(lock, None)
                elapsed = max(0.001, time.monotonic() - t0)
                chunk = min(FILE_CHUNK, max(16 << 10, min(want * 2, int(len(data) * 5 / elapsed))))
                put(state="copying %s: %.1f / %.1f MB" % (name, got / (1 << 20), size / (1 << 20)))
        # Revalidate even a complete partial file left behind by an interrupted publish.
        r = rpc(host_addr(), {"op": "file", "path": hp}, timeout=10)
        if not r.get("ok") or r.get("size") != size or r.get("mtime") != mtime \
                or (stamp is not None and r.get("stamp") != stamp):
            os.remove(meta)
            raise OSError("the file changed during the copy")
        os.utime(tmp, (mtime, mtime))
        os.replace(tmp, dst)
        COPIED[0] += 1
        COPIED[1] += size
        return dst
    except (OSError, ValueError) as e:
        log("couldn't copy %s from the host: %s" % (name, e))
        return None
    finally:
        stop.set()
        if heartbeat is not None:
            heartbeat.join()
        # Keep partial bytes and their identity across retries and worker restarts.
        for q in ([lock] if mine else []) + ([meta] if mine and good() else []):
            try:
                os.remove(q)
            except OSError:
                pass


def fetch(p, raw=None, lib=None):
    """Get a file only the host has, with the files that belong to it (the frames of a sequence, the tiles of
    an image). Returns the path to use on this computer, or None."""
    for hp in host_forms(p, raw, lib):
        try:
            r = rpc(host_addr(), {"op": "file", "path": hp, "members": True}, timeout=15)
        except OSError:
            return None
        if not r.get("ok"):
            continue
        got = [fetch_one(m) for m in r["members"]]
        if not any(got):
            log("couldn't copy %s from the host" % local_name(hp))
            return None
        if not all(got):
            log("%d of the %d files of %s couldn't be copied" % (len(got) - len([g for g in got if g]), len(got), local_name(hp)))
        return os.path.join(cache_dir(strip_tail(hp, 1)), local_name(hp))
    return None


STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(cfg["status"])))), "files")
COPIED = [0, 0]


def find_missing():
    """Files this computer can't open: copy them from the host (if the host allows it), or use the one that
    ships with this computer's Blender. Every kind of file Blender knows the scene uses, and again after a
    copied library has been loaded, for the files that library uses. Returns (copied, taken from Blender)."""
    walk = None if os.environ.get("MULTIGPU_OLD_FILE_LIST") else getattr(bpy.data, "file_path_foreach", None)
    try:                                              # copies nothing has used for a month
        for n in os.listdir(STORE):
            q = os.path.join(STORE, n)
            if time.time() - os.path.getmtime(q) > 30 * 86400:
                shutil.rmtree(q, ignore_errors=True)
    except OSError:
        pass
    tried, count = {}, [0, 0]

    def better(p, raw=None, lib=None):
        """A path that works on this computer for p, or None."""
        if p not in tried:
            new = fetch(p, raw, lib) if OPTS.get("copy") else None
            if new:
                count[0] += 1
            else:
                new = bundled_here(p, _df)
                count[1] += bool(new)
            tried[p] = new
        return tried[p]

    for _round in range(8):
        changed = []
        if walk is not None:
            def visit(idb, path, meta):
                kind = type(idb).__name__
                if not path or kind in ("Sound", "Text") or getattr(meta, "is_readonly", False):
                    return None
                if (kind == "Scene" and not OPTS.get("sequencer")) or (kind == "Image" and not idb.users):
                    return None
                lib = getattr(idb, "library", None)
                p = os.path.normpath(bpy.path.abspath(path, library=lib))
                if path_exists(p) or os.path.isdir(p):
                    return None
                new = better(p, path, lib)
                if new:
                    changed.append((kind, idb.name, lib.filepath if lib else None))
                    if os.environ.get("MULTIGPU_DEBUG"):
                        log("round %d: %s %s: %s -> %s" % (_round, kind, idb.name, p, new))
                return new
            walk(visit, flags=set(WALK))
        else:                                         # an older Blender: the kinds of file that matter most
            for coll in ("images", "libraries", "cache_files", "volumes"):
                for idb in list(getattr(bpy.data, coll, [])):
                    fp = getattr(idb, "filepath", "") or ""
                    if not fp or getattr(idb, "packed_file", None):
                        continue
                    if coll == "images" and (idb.source not in ("FILE", "MOVIE") or idb.users == 0):
                        continue
                    lib = getattr(idb, "library", None)
                    p = os.path.normpath(bpy.path.abspath(fp, library=lib))
                    if os.path.exists(p):
                        continue
                    new = better(p, fp, lib)
                    if new:
                        idb.filepath = new
                        changed.append((type(idb).__name__, idb.name, lib.filepath if lib else None))
        if not changed:
            break
        # a library that was just found brings its own files: load it, then look again
        for kind, name, lib in changed:
            if kind == "Library":
                try:
                    bpy.data.libraries[name].reload()
                except Exception as e:
                    log("couldn't load the copied %s: %s" % (name, e))
        for kind, name, lib in changed:
            if kind == "Image":
                try:
                    img = bpy.data.images.get((name, lib))
                    if img is not None:
                        img.reload()
                except Exception:
                    pass
    if COPIED[0]:
        log("copied %d file%s from the host (%.1f MB)" % (COPIED[0], "" if COPIED[0] == 1 else "s", COPIED[1] / 1048576.0))
    return count[0], count[1]


def missing_files():
    """Files this computer can't open although the host can."""
    host = set(OPTS.get("host_missing") or [])
    miss = []
    rows = scene_paths(bpy, sequencer=bool(OPTS.get("sequencer")))
    if rows is not None:
        miss = [os.path.basename(p) for _k, _n, _l, p, _e in rows if not path_exists(p) and not os.path.isdir(p)]
        return sorted(set(m for m in miss if m.lower() not in host))
    for img in bpy.data.images:
        if img.packed_file or img.source not in ("FILE", "MOVIE") or not img.filepath or img.users == 0:
            continue
        p = bpy.path.abspath(img.filepath, library=img.library)
        if not os.path.exists(p):
            miss.append(os.path.basename(p))
    for coll in ("libraries", "cache_files", "volumes"):
        for idb in getattr(bpy.data, coll, []):
            if not idb.filepath or getattr(idb, "packed_file", None):
                continue
            p = bpy.path.abspath(idb.filepath, library=getattr(idb, "library", None))
            if not os.path.exists(p):
                miss.append(os.path.basename(p))
    return sorted(set(m for m in miss if m.lower() not in host))


def check_needs():
    """Add-ons the scene can't be rendered correctly without (found by the host)."""
    have = set(k.split(".")[-1] for k in bpy.context.preferences.addons.keys())
    lacking = [n for n in OPTS.get("needs") or [] if n not in have]
    if lacking:
        return ("the %s add-on isn't enabled on this computer, and this scene needs it to render correctly"
                % ", ".join(lacking))
    for ob in bpy.data.objects:                     # FLIP Fluids: no cache folder = no fluid in the render
        try:
            if ob.flip_fluid.object_type != "TYPE_DOMAIN":
                continue
            cache = ob.flip_fluid.domain.cache
            try:
                d = cache.get_cache_abspath()
            except Exception:
                d = bpy.path.abspath(cache.cache_directory)
            if d and not os.path.isdir(d):
                return "the FLIP Fluids cache folder isn't reachable from this computer: %s" % d
        except Exception:
            pass
    return None


nmap = remap()
try:
    ncopy, nown = find_missing()
except Exception:
    ncopy = nown = 0
    log("looking for missing files failed:\n" + traceback.format_exc())
OUT = cfg["output"]
if not OUT.startswith("//"):
    OUT = map_path(OUT, MAPS)
rd.filepath = OUT
miss = missing_files()
warn = ("can't find %d file%s the host has: %s" % (len(miss), "" if len(miss) == 1 else "s", ", ".join(miss[:3]) + (" ..." if len(miss) > 3 else ""))) if miss else None
if TILE:
    warn = "; ".join(x for x in (warn, "rendering in %d px tiles after running out of GPU memory" % TILE) if x)
log("%s with %s, persistent data %s, output %s%s%s%s" % (on[0], cfg["backend_label"], "on" if rd.use_persistent_data else "off",
    OUT, ", %d path%s converted" % (nmap, "" if nmap == 1 else "s") if nmap else "",
    ", %d file%s copied from the host" % (ncopy, "" if ncopy == 1 else "s") if ncopy else "", ", " + warn if warn else ""))


# ---- rendering one frame -------------------------------------------------------------
def rotate_log():
    """Blender prints a lot while rendering: start the log file over when it gets big."""
    try:
        p = cfg.get("log")
        if p and os.path.getsize(p) > cfg.get("log_max", 32 * 1024 * 1024):
            sys.stdout.flush()
            sys.stderr.flush()
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            os.close(fd)
            log("log started over (it had grown large)")
    except (OSError, ValueError):
        pass


def log_error():
    """The last error Blender printed while rendering the current frame."""
    try:
        with open(cfg["log"], "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            f.seek(max(min(ME["log_pos"], end), end - 30000, 0))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except (OSError, KeyError):
        return None
    best = None
    for line in lines:
        low = line.lower()
        if "multigpu" in low or line.startswith("Traceback") or line.startswith("  "):
            continue
        if "out of memory" in low or "out of gpu memory" in low:
            return "out of GPU memory"
        if "error" in low or "failed" in low or "illegal address" in low or " in cuda " in low or " in optix " in low:
            best = line.strip()
    return best[-160:] if best else None


def explain(e):
    msg = (str(e).strip().splitlines() or ["unknown error"])[-1].replace("Error: ", "")
    extra = log_error()
    if extra and extra not in msg:
        msg = "%s (%s)" % (msg, extra) if ("didn't save" in msg or "cancelled" in msg) else msg
        if extra == "out of GPU memory":
            msg = extra
    return msg[:300]


def prepare(f):
    """Go to frame f and point the output at this card's private folder next to the frames."""
    rotate_log()
    try:
        ME["log_pos"] = os.path.getsize(cfg["log"])
    except (OSError, KeyError):
        ME["log_pos"] = 0
    sc.frame_set(f)
    rd.filepath = OUT
    final = os.path.normpath(rd.frame_path(frame=f))
    d = os.path.dirname(final)
    os.makedirs(d, exist_ok=True)
    if AWAY:                                        # nothing half-done in a folder a sync program is watching
        tmp = os.path.join(STORE, "..", "partial", cfg["job"], TAG)
        tmp = os.path.normpath(tmp)
    else:
        tmp = os.path.join(d, ".multigpu-partial", TAG)
    os.makedirs(tmp, exist_ok=True)
    ME["tmps"].add(tmp)
    clear_tmp(tmp)
    base = os.path.basename(final)
    ext = rd.file_extension
    if rd.use_file_extension and ext and base.lower().endswith(ext.lower()):
        base = base[:-len(ext)]
    if OPTS.get("placeholders", True) and not AWAY and not os.path.exists(final):
        try:
            open(final, "xb").close()
        except OSError:
            pass
    rd.filepath = os.path.join(tmp, base)
    return final, tmp


def move_into_place(src, dst):
    try:
        os.replace(src, dst)
        return
    except OSError:
        pass
    # Another drive (or a NAS that can't rename from the worker folder): stage on the destination
    # drive, then replace atomically. Never delete the previous frame before its replacement is ready.
    # Each attempt owns its staging file, including when two cards race to finish a frame.
    prefix = ".syncthing.%s." if AWAY else ".multigpu-copy.%s."
    fd, part = tempfile.mkstemp(prefix=prefix % os.path.basename(dst), suffix=".tmp",
                                dir=os.path.dirname(dst) or ".")
    os.close(fd)
    try:
        shutil.copyfile(src, part)
        if os.path.getsize(part) != os.path.getsize(src):
            raise OSError("copying %s didn't finish" % os.path.basename(dst))
        os.replace(part, dst)
    finally:
        try:
            os.remove(part)
        except OSError:
            pass
    os.remove(src)


def collect(final, tmp):
    """Move the finished file(s) from the private folder into place, each in one step.
    Returns (total bytes, size of the frame's own file or None)."""
    rd.filepath = OUT
    names = sorted(os.listdir(tmp)) if os.path.isdir(tmp) else []
    if not names:
        raise RuntimeError("the render didn't save an image")
    d = os.path.dirname(final)
    total, own = 0, None
    for n in names:
        src = os.path.join(tmp, n)
        size = os.path.getsize(src)
        if size <= 0:
            raise RuntimeError("the render saved an empty file (is the disk full?)")
        try:
            move_into_place(src, os.path.join(d, n))
        except OSError as e:
            raise RuntimeError("couldn't save %s: %s" % (n, e.strerror or e))
        total += size
        if n == os.path.basename(final):
            own = size
    return total, own


def out_reachable():
    try:
        rd.filepath = OUT
        return os.path.isdir(os.path.dirname(os.path.normpath(rd.frame_path(frame=cfg.get("first") or 1))))
    except Exception:
        return True


def still_mine(f):
    """With its own copy of the folder, a card asks before saving: if the host gave the frame to another card
    meanwhile (the connection was down for long), saving it too would leave two different files for one frame."""
    t0 = time.time()
    while time.time() - t0 < 900 and not ME["exiting"]:
        try:
            r = call("beat", frame=f)
            return not (r.get("drop") or r.get("exit"))
        except OSError:
            put(state="waiting for host")
            time.sleep(5)
    return False


def report_for(f, t0, final=None, tmp=None, error=None):
    ME["n"] += 1
    rid = "%d-%d" % (os.getpid(), ME["n"])
    if error is None and AWAY and not still_mine(f):
        log("frame %d went to another card while the host couldn't be reached: not saved here" % f)
        clear_tmp(tmp)
        return None
    if error is None:
        try:
            total, own = collect(final, tmp)
            log("frame %d done in %.1fs" % (f, time.time() - t0))
            ME["warn"] = warn
            return {"frame": f, "ok": True, "secs": time.time() - t0, "bytes": total, "size": own, "rid": rid}
        except Exception as e:
            error = e
    msg = explain(error)
    log("frame %d failed: %s" % (f, msg))
    if "out of" in msg.lower() and "memory" in msg.lower():
        new = smaller_tile()
        if new:                                     # start over with smaller tiles; the frame goes back to the host
            log("out of GPU memory: restarting with %d px tiles" % new)
            ME["exiting"] = True
            put(state="oom", frame=None, tile=new, error=None)
            try:
                call("bye", timeout=3.0, why="oom", tile=new)
            except OSError:
                pass
            sys.stdout.flush()
            os._exit(8)
        msg = "out of GPU memory, even in %s px tiles" % (tile_now() or "small")
    env = not out_reachable()
    if env:                                         # e.g. the NAS dropped out: wait for it instead of failing frames
        ME["env_wait"] = True
        ME["warn"] = "can't reach the output folder; waiting for it to come back"
    return {"frame": f, "ok": False, "error": msg, "rid": rid, "env": env}


def beat_loop():
    while not ME["exiting"]:
        time.sleep(5)
        try:
            r = call("beat", frame=ME["frame"])
        except OSError:
            continue
        if r.get("exit") and not ME["exiting"]:
            on_exit_reply(r)


def check_output():
    """Is this computer's output folder really the host's, and can we write there? None when all is well."""
    first = cfg.get("first")
    if first is None:
        return None
    rd.filepath = OUT
    d = os.path.dirname(os.path.normpath(rd.frame_path(frame=first)))
    if AWAY:                                        # this computer's own copy of the folder: only writing matters
        try:
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, ".syncthing.multigpu-write-test-%s.tmp" % TAG)
            with open(p, "w") as f:
                f.write("x")
            os.remove(p)
        except OSError as e:
            return "it can't write to %s: %s" % (d, e.strerror or e)
        return None
    marker = os.path.join(d, ".multigpu-lock.json")
    t0 = time.time()
    while True:
        try:
            with open(marker, encoding="utf-8") as f:
                if json.load(f).get("job") == JOB:
                    break
        except (OSError, ValueError):
            pass
        if time.time() - t0 > cfg.get("marker_wait", 45):
            return ("its output folder (%s) isn't the one the host renders into. Keep the .blend and the output "
                    "inside the shared NAS folder" % d)
        time.sleep(3)
    try:
        tmp = os.path.join(d, ".multigpu-partial", TAG)
        os.makedirs(tmp, exist_ok=True)
        ME["tmps"].add(tmp)
        p = os.path.join(tmp, ".write-test")
        with open(p, "w") as f:
            f.write("x")
        os.remove(p)
    except OSError as e:
        return "it can't write to %s: %s" % (d, e.strerror or e)
    return None


# ---- check in with the host ------------------------------------------------------------
r = ask("hello", warn=warn)
if r.get("exit"):
    on_exit_reply(r)
bad = check_needs() or check_output()
if bad:
    leave("error", 3, error=bad)
put(state="ready", error=None, warn=warn)
ME["warn"] = warn
threading.Thread(target=beat_loop, daemon=True).start()


def run_background():
    report = None
    while True:
        if ME.pop("env_wait", None):
            put(state="waiting for the output folder", frame=None)
            time.sleep(15)
        r = ask("next", report=report, warn=ME["warn"])
        report = None
        if r.get("exit"):
            on_exit_reply(r)
        if "wait" in r:
            put(state="idle", frame=None)
            time.sleep(min(10.0, float(r["wait"])))
            continue
        f = r.get("frame")
        if f is None:
            time.sleep(2)
            continue
        t0 = time.time()
        ME["frame"] = f
        put(state="rendering", frame=f)
        try:
            final, tmp = prepare(f)
            if cfg.get("test_oom_above") and (tile_now() or 99999) > cfg["test_oom_above"]:
                raise RuntimeError("System is out of GPU memory (test)")
            if cfg.get("test_sleep"):
                time.sleep(cfg["test_sleep"])
            try:
                res = bpy.ops.render.render(write_still=True, scene=sc.name)
            finally:
                rd.filepath = OUT
            if "CANCELLED" in res:
                raise RuntimeError("the render was cancelled")
            report = report_for(f, t0, final, tmp)
        except Exception as e:
            traceback.print_exc()
            report = report_for(f, t0, error=e)
        ME["frame"] = None
        put(state="ready", frame=None)


# ---- Show Render: a normal Blender window that shows each frame rendering live -----------
U = {"phase": "ask", "report": None, "frame": None, "final": None, "tmp": None, "cancel": False,
     "t0": 0.0, "started": False, "fail_t": None}
TITLE = "MultiGPU  |  %s  (%s)" % (cfg["label"], cfg["backend_label"])


def _ui_step():
    try:
        if U["phase"] == "render":
            if bpy.app.is_job_running("RENDER"):
                U["started"] = True
                return 0.5
            done_files = U["tmp"] and os.path.isdir(U["tmp"]) and os.listdir(U["tmp"])
            if not U["started"] and not done_files and not U["cancel"] and time.time() - U["t0"] < 30:
                return 0.5
            f = U["frame"]
            rd.filepath = OUT
            if U["cancel"]:
                # Blender also "cancels" a render that failed (out of GPU memory, a driver error ...). So this is
                # a failed frame, never taken as the user stopping the card: closing its window does that.
                why = log_error()
                U["report"] = report_for(f, U["t0"], error=RuntimeError(
                    "the render was cancelled: " + (why or "Blender didn't say why (Esc in its window? see the card's log)")))
            else:
                U["report"] = report_for(f, U["t0"], U["final"], U["tmp"])
            ME["frame"] = None
            U["phase"] = "ask"
        if ME.pop("env_wait", None):
            put(state="waiting for the output folder", frame=None)
            return 15.0
        try:
            r = call("next", report=U["report"], warn=ME["warn"])
        except OSError as e:
            U["fail_t"] = U["fail_t"] or time.time()
            if time.time() - U["fail_t"] > 600:
                leave("error", 4, error="lost the host for 10 minutes (%s)" % e, tell=False)
            put(state="waiting for host", error=str(e))
            return 3.0
        U["fail_t"] = None
        U["report"] = None
        if r.get("exit"):
            on_exit_reply(r)
        if "wait" in r:
            put(state="idle", frame=None)
            return float(r["wait"])
        f = r.get("frame")
        if f is None:
            return 2.0
        U.update(frame=f, t0=time.time(), cancel=False, started=False)
        ME["frame"] = f
        put(state="rendering", frame=f)
        try:
            U["final"], U["tmp"] = prepare(f)
            if cfg.get("test_oom_above") and (tile_now() or 99999) > cfg["test_oom_above"]:
                raise RuntimeError("System is out of GPU memory (test)")
            win = bpy.context.window_manager.windows[0]
            with bpy.context.temp_override(window=win, screen=win.screen):
                bpy.ops.render.render("INVOKE_DEFAULT", write_still=True, scene=sc.name)
        except Exception as e:
            rd.filepath = OUT
            U["report"] = report_for(f, U["t0"], error=e)
            ME["frame"] = None
            return 1.0
        U["phase"] = "render"
        return 1.0
    except Exception as e:
        traceback.print_exc()
        leave("error", 3, error="card window: %s" % e)


def _cancel(*args):
    U["cancel"] = True


def _window():
    try:
        place_own(cfg.get("rect"), TITLE)
    except Exception as e:
        log("couldn't arrange this window: %s" % e)
    return None


def _title():
    try:
        place_own(None, TITLE)
    except Exception:
        return None
    return 3.0


def _atexit():
    if not ME["exiting"]:
        put(state="stopped", frame=None)
        try:
            call("bye", timeout=2.0, why="closed")
        except OSError:
            pass


if cfg.get("ui"):
    P = bpy.context.preferences
    P.use_preferences_save = False
    try:
        P.view.render_display_type = "SCREEN"
    except Exception:
        pass
    for scr in bpy.data.screens:
        for a in scr.areas:
            for sp in a.spaces:
                if sp.type == "VIEW_3D":
                    sp.shading.type = "SOLID"
    bpy.app.handlers.render_cancel.append(_cancel)
    atexit.register(_atexit)
    if cfg.get("rect"):
        for delay in (1.0, 3.0, 7.0):
            bpy.app.timers.register(_window, first_interval=delay)
    bpy.app.timers.register(_title, first_interval=2.0)
    bpy.app.timers.register(_ui_step, first_interval=1.5)
else:
    run_background()
'''


# ======================================================================================
# This computer's side: starts the host process, launches and watches the cards
# ======================================================================================
_COORD_INFO = os.path.join(_ROOT, "host.json")
_COORD_STATE = os.path.join(_ROOT, "host-state.json")
_COORD_LOG = os.path.join(_ROOT, "host.log")
_AGENT_FILE = os.path.join(_ROOT, "cards.json")
_OWNER_FILE = os.path.join(_ROOT, "agent.json")
_ADDR_FILE = os.path.join(_ROOT, "host-addr.json")
# Taking over as host: seconds without the host / without its marker being refreshed, the extra wait per
# place in the queue when several computers could take over, and how long a half-done takeover may sit
_FO = {"watch": 15.0, "after": 60.0, "stale": 50.0, "step": 20.0, "claim": 120.0, "holdoff": 20, "back": 15.0}

_A = {"addr": None, "snap": None, "reply": None, "err": None, "procs": {}, "seq": {}, "resolve": {},
      "layout": {}, "msg": None, "coord_t": 0.0, "coord_err": None, "thread": None, "quit": threading.Event(),
      "sync": None, "use_v": {}, "drawn": None, "last_draw": 0.0, "running_job": None, "ip": (0.0, None), "cid": None,
      "owner": None, "verified": {}, "refused": None, "pre_id": None, "pre_result": None, "saving": False,
      "unsynced_send": False, "token_unsaved": False, "warn_lines": [], "quit_t": 0.0,
      "host_ok": (None, 0.0), "rep": None, "job_maps": None, "fo": {}, "fo_in": None, "fo_action": None,
      "fo_thread": None, "takeover": None, "boot_job": None, "boot_done": False, "addr_note": None,
      "fo_over": None}


def _prefs(context=None):
    a = (context or bpy.context).preferences.addons.get(__name__)
    return a.preferences if a else None


def _mark_dirty():
    """Settings changed from code aren't saved with the preferences unless Blender is told they changed."""
    try:
        bpy.context.preferences.is_dirty = True
    except Exception:
        pass


def _read_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(p, data):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(p + ".tmp", p)
    except OSError:
        pass


def _fmt(sec):
    sec = int(max(0, sec or 0))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    if h:
        return "%dh %02dm" % (h, m)
    if m:
        return "%dm %02ds" % (m, s)
    return "%ds" % s


def _clock(t):
    return time.strftime("%I:%M %p", time.localtime(t)).lstrip("0")


def _computer_id():
    if _A["cid"]:
        return _A["cid"]
    if _TEST.get("computer"):
        _A["cid"] = _TEST["computer"]
        return _A["cid"]
    p = os.path.join(bpy.utils.user_resource('CONFIG'), "multigpu_id.txt")
    try:
        with open(p, encoding="utf-8") as f:
            cid = f.read().strip()
    except OSError:
        cid = ""
    if not cid:
        cid = "%s-%s" % (socket.gethostname().split(".")[0].lower() or "pc", uuid.uuid4().hex[:4])
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(cid)
        except OSError:
            pass
    _A["cid"] = cid
    return cid


def _computer_name():
    return _TEST.get("name") or socket.gethostname().split(".")[0] or "this computer"


def _card_id(key):
    return "%s-%s" % (_computer_id(), hashlib.sha1(key.encode("utf-8")).hexdigest()[:6])


def _pid_running(pid):
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = ctypes.c_void_p
        h = k.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
        k.CloseHandle(ctypes.c_void_p(h))
        return bool(ok) and code.value == 259
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    try:                                            # a finished child we started is a zombie until reaped
        with open("/proc/%d/stat" % int(pid)) as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _alive(pid, popen=None, pstart=None):
    """Is the process we started still running? pstart (its start time) guards against the pid having been
    reused by some other program, e.g. after a restart of the computer."""
    if not pid:
        return False
    if popen is not None:
        return popen.poll() is None
    if not _pid_running(pid):
        return False
    if pstart is None:
        return True
    key = "%s/%s" % (pid, pstart)
    seen = _A["verified"].get(key)
    if seen and time.time() - seen[0] < 10:
        return seen[1]
    cur = proc_start(pid)
    ok = cur is None or str(cur) == str(pstart)
    _A["verified"] = {key: (time.time(), ok)} if len(_A["verified"]) > 50 else dict(_A["verified"], **{key: (time.time(), ok)})
    return ok


def _kill(pid, hard=False):
    if not pid:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        import signal
        try:
            os.kill(int(pid), signal.SIGKILL if hard else signal.SIGTERM)
        except OSError:
            pass


def _popen(cmd, log, env=None):
    lf = open(log, "w")
    try:
        if os.name == "nt":
            return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                start_new_session=True)
    finally:
        lf.close()


def _owner(take=False):
    """Only one Blender on a computer launches and watches its cards; any other Blender open at the same
    time just shows what's going on. The one where Render is pressed takes over."""
    me = os.getpid()
    cur = _read_json(_OWNER_FILE) or {}
    if cur.get("pid") == me:
        return True
    if not take and cur.get("pid") and _alive(cur["pid"], pstart=cur.get("pstart")):
        return False
    _write_json(_OWNER_FILE, {"pid": me, "pstart": proc_start(me), "t": time.time()})
    return (_read_json(_OWNER_FILE) or {}).get("pid") == me


def _route_ip(target):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 1))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _addresses():
    """This computer's address on the local network, and on Tailscale if that's running."""
    t, val = _A["ip"]
    if val and time.time() - t < 30:
        return val
    lan = _route_ip("8.8.8.8") or _route_ip("10.254.254.254") or "127.0.0.1"
    ts = _route_ip("100.100.100.100")
    try:
        a, b = [int(x) for x in (ts or "0.0").split(".")[:2]]
    except ValueError:
        a = b = 0
    val = {"lan": lan, "tailscale": ts if (a == 100 and 64 <= b <= 127 and ts != lan) else None}
    _A["ip"] = (time.time(), val)
    return val


def _lan_ip():
    return _addresses()["lan"]


def _join_addr(prefs):
    a = prefs.host_address.strip().replace("http://", "").rstrip("/")
    if not a:
        return None
    if a.count(":") == 1:
        return a
    return "%s:%d" % (a, prefs.port)


def _net_error(addr, e):
    s = str(e)
    if isinstance(e, ConnectionRefusedError) or "refused" in s.lower():
        return "Nothing answers on %s. Is MultiGPU set to Host on that computer?" % addr
    if isinstance(e, socket.timeout) or "timed out" in s.lower():
        return ("No answer from %s. Check the address, and that the port is allowed through the host's "
                "firewall." % addr)
    if isinstance(e, socket.gaierror):
        return "Can't find a computer called %s. Check the name or use its address." % addr.rsplit(":", 1)[0]
    return "Can't reach %s: %s" % (addr, s)


def _is_host_mode(prefs):
    return prefs.mode in ("SOLO", "HOST")


def _same_file(a, b):
    if not a or not b:
        return False
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def _origin_here(job):
    """Was this render started on this computer? Its paths are then this computer's own. (The host can be
    another computer by now: one that took over when this one stopped answering.)"""
    return (job.get("origin") or job.get("host")) == _computer_id()


def _origin_job():
    """The render that's running, if it was started on this computer (whichever computer hosts it now)."""
    job = (_A.get("reply") or {}).get("job")
    return job if job and job.get("state") == "running" and _origin_here(job) else None


def _away_now():
    """This computer renders into its own copy of the folder, not the host's: it must never take over as host
    (a short break in the connection would start a second render of everything)."""
    job = (_A.get("reply") or {}).get("job") or {}
    return bool((_A["resolve"].get(job.get("id")) or {}).get("away"))


def _blend_here(job):
    """This computer's path to the .blend being rendered, if it knows it."""
    if _origin_here(job):
        return (job.get("blend") or {}).get("path")
    res = _A["resolve"].get(job.get("id")) or {}
    return res.get("path") if res.get("ok") else None


def _host_exe():
    """The program the host process runs in. On Windows that's blender.exe itself: the firewall rule people
    already have for Blender then covers it, where Blender's python.exe is often blocked."""
    if os.name == "nt" and bpy.app.binary_path and os.path.exists(bpy.app.binary_path):
        return bpy.app.binary_path
    if sys.executable and os.path.exists(sys.executable):
        return sys.executable
    return bpy.app.binary_path


def _fw_check():
    """Windows: is the program that runs the host blocked by a firewall rule? Runs in a thread."""
    py = _host_exe().replace("'", "''")
    ps = ("$r = Get-NetFirewallApplicationFilter -Program '%s' -ErrorAction SilentlyContinue | Get-NetFirewallRule | "
          "Where-Object { $_.Enabled -eq 'True' -and $_.Direction -eq 'Inbound' }; "
          "@{block = @($r | Where-Object Action -eq 'Block').Count; allow = @($r | Where-Object Action -eq 'Allow').Count}"
          " | ConvertTo-Json -Compress" % py)

    def run():
        try:
            p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                               text=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _A["fw"] = json.loads(p.stdout.strip() or "{}")
        except (OSError, ValueError, subprocess.SubprocessError):
            _A["fw"] = None
    _A["fw_t"] = time.time()
    threading.Thread(target=run, daemon=True).start()


def _fw_fix(port):
    py = _host_exe().replace("'", "''")
    return ("Get-NetFirewallApplicationFilter -Program '%s' | Get-NetFirewallRule | Where-Object Action -eq 'Block' | "
            "Remove-NetFirewallRule; if (-not (Get-NetFirewallRule -DisplayName 'MultiGPU host' -ErrorAction "
            "SilentlyContinue)) { New-NetFirewallRule -DisplayName 'MultiGPU host' -Direction Inbound -Protocol TCP "
            "-LocalPort %d -Action Allow -Profile Any }" % (py, port))


# ---- the host process ------------------------------------------------------------------
def _coord_info():
    info = _read_json(_COORD_INFO)
    if info and not info.get("error") and _alive(info.get("pid"), pstart=info.get("pstart")):
        return info
    return None


def _start_coord(prefs, resume=False):
    os.makedirs(_ROOT, exist_ok=True)
    script = os.path.join(_ROOT, "host.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(SHARED + "\n" + COORD)
    try:
        os.remove(_COORD_INFO)
    except OSError:
        pass
    lan = prefs.mode == "HOST"
    opts = {"bind": "0.0.0.0" if lan else "127.0.0.1", "port": prefs.port if lan else 0, "state": _COORD_STATE,
            "info": _COORD_INFO, "version": VERSION, "resume": bool(resume)}
    a = _addresses()
    addrs = [a["lan"], a.get("tailscale")]
    tk = _A.get("takeover")
    if resume and tk and time.time() - tk["t"] < 120:   # this computer is taking a render over from its host
        opts.update(takeover=True, holdoff=_fo("holdoff"))
        addrs.insert(0, tk.get("ip"))
    opts["addrs"] = [x for i, x in enumerate(addrs) if x and x not in addrs[:i]]
    opts.update(_TEST.get("host_opts") or {})
    exe = _host_exe()
    if exe == bpy.app.binary_path:
        cmd = [exe, "-b", "--factory-startup", "--python", script]
    else:
        cmd = [exe, "-I", script]
    _popen(cmd + ["--", json.dumps(opts)], _COORD_LOG)
    _A["coord_t"] = time.time()
    _A["coord_err"] = None
    _A["refused"] = None


def _ensure_coord(prefs, start=False, wait=0.0, owner=True):
    """Address of this computer's host process, starting it if asked. None while it isn't up."""
    want_bind = "0.0.0.0" if prefs.mode == "HOST" else "127.0.0.1"
    info = _coord_info()
    if not owner:                                   # a second Blender only looks at what's there
        return "127.0.0.1:%d" % info["port"] if info else None
    rendering = bool(_A.get("running_job"))
    t = time.time()
    if info and _A.get("refused") and t - _A["refused"] > 6 and t - _A["coord_t"] > 10:
        try:                                        # nothing listens there: that isn't our host process (anymore)
            os.remove(_COORD_INFO)
        except OSError:
            pass
        _A["refused"] = None
        info = None
    wrong = info and (info.get("bind") != want_bind or info.get("version") != VERSION or
                      (want_bind == "0.0.0.0" and info.get("port") != prefs.port))
    if wrong and not rendering and t - _A["quit_t"] > 5:
        _A["quit_t"] = t                            # Host <-> This Computer, a new port, or a new version
        try:
            r = rpc("127.0.0.1:%d" % info["port"], {"op": "quit"}, timeout=2)
        except OSError:
            r = {}
        if not r.get("error"):                      # (it refuses while it's rendering: then we keep using it)
            info = None
            t0 = time.time()
            while _coord_info() and time.time() - t0 < 3:
                time.sleep(0.1)
    if not info:
        bad = _read_json(_COORD_INFO)
        if bad and bad.get("error"):
            _A["coord_err"] = bad["error"]
        recently = time.time() - _A["coord_t"] < 8
        if (start or rendering) and not recently:
            _start_coord(prefs, resume=rendering)
        if wait:
            t0 = time.time()
            while time.time() - t0 < wait:
                info = _coord_info()
                if info:
                    break
                bad = _read_json(_COORD_INFO)
                if bad and bad.get("error"):
                    _A["coord_err"] = bad["error"]
                    return None
                time.sleep(0.1)
        if not info:
            if time.time() - _A["coord_t"] > 8 and _A["coord_t"] and not _A["coord_err"]:
                _A["coord_err"] = "The host process didn't start. See %s" % _COORD_LOG
            return None
    _A["coord_err"] = None
    return "127.0.0.1:%d" % info["port"]


def _retire_coord():
    """This computer no longer hosts: close its idle host process (it refuses while it's rendering)."""
    t = time.time()
    if t - _A["quit_t"] < 20:
        return
    info = _coord_info()
    if info:
        _A["quit_t"] = t
        try:
            rpc("127.0.0.1:%d" % info["port"], {"op": "quit"}, timeout=2)
        except OSError:
            pass


# ---- talking to the host (in a background thread so the UI never waits on the network) --
def _net_loop():
    while not _A["quit"].is_set():
        addr, snap = _A.get("addr"), _A.get("snap")
        if addr and snap:
            try:
                comp = dict(snap["computer"])
                procs = {}
                for cid, pr in (comp.get("procs") or {}).items():
                    p = _A["procs"].get(cid)       # Blender's own window may be busy: check the cards from here
                    if pr.get("alive") and p is not None and not _alive(p.get("pid"), p.get("popen"), p.get("pstart")):
                        pr = dict(pr, alive=False)
                    procs[cid] = pr
                comp["procs"] = procs
                comp["pre"] = _A.get("pre_result")
                r = rpc(addr, {"op": "hello", "computer": comp}, timeout=6.0)
                if r.get("error"):
                    _A["err"] = r["error"]
                else:
                    r["_t"] = time.time()
                    r["_addr"] = addr
                    _A["reply"] = r
                    _A["err"] = None
                    _A["host_ok"] = (addr, time.time())
                    _fetch_replica(addr, comp, r.get("job"))
                _A["refused"] = None
            except OSError as e:
                _A["err"] = _net_error(addr, e)
                if addr.startswith("127.0.0.1:") and isinstance(e, ConnectionRefusedError):
                    _A["refused"] = _A.get("refused") or time.time()
            except Exception:
                traceback.print_exc()
        _A["quit"].wait(1.0)


def _fetch_replica(addr, comp, job):
    """A computer that has joined keeps a copy of the host's render, so it can carry it on as host itself
    if the host stops answering. Runs in the network thread."""
    if not comp.get("failover") or not job or job.get("state") != "running":
        _A["rep"] = None
        return
    old = _A.get("rep")
    if old and old.get("addr") == addr and old.get("job") == job["id"] and time.time() - old["got"] < 5:
        return
    jm = _A.get("job_maps")
    if not jm or jm[0] != job["id"]:
        return                                       # (not yet known where the render's folders are here)
    try:
        x = rpc(addr, {"op": "replica"}, timeout=15.0)
    except OSError:
        return
    if x.get("blob") and x.get("job") == job["id"]:
        _A["rep"] = {"addr": addr, "got": time.time(), "job": x["job"], "term": x.get("term", 0), "blob": x["blob"],
                     "maps": [list(m) for m in jm[1]], "data": None}


def _snapshot(prefs, owner=True):
    base = {"id": _computer_id(), "name": _computer_name(), "os": sys.platform, "blender": bpy.app.version_string,
            "addon": VERSION, "protocol": PROTOCOL, "is_host": _is_host_mode(prefs)}
    if not owner:
        return {"op": "hello", "computer": dict(base, viewer=True)}
    cards = []
    for g in prefs.gpus:
        ids = json.loads(g.ids or "{}")
        cid = _card_id(g.key)
        cards.append({"id": cid, "label": g.name, "backend": g.backend,
                      "use": bool(g.use and g.backend in ids), "use_v": _A["use_v"].get(cid, 0)})
    procs = {}
    for cid, p in _A["procs"].items():
        procs[cid] = {"alive": not p.get("dead"), "job": p.get("job"), "sync": p.get("sync"),
                      "seq": _A["seq"].get(cid, 0), "why": p.get("why"), "error": p.get("error"), "tile": p.get("tile")}
    job = (_A.get("reply") or {}).get("job")
    res = None
    a = _addresses()
    base.update(port=prefs.port, addrs=[x for x in (a["lan"], a.get("tailscale")) if x],
                failover=bool(prefs.mode == "JOIN" and prefs.joined and prefs.auto_takeover and not _away_now()))
    if job:
        if _origin_here(job):
            res = {"job": job["id"], "sync": job["sync"], "ok": True}
        else:
            r = _A["resolve"].get(job["id"]) or {}
            res = {"job": job["id"], "sync": r.get("sync"), "ok": bool(r.get("ok")), "busy": bool(r.get("busy")),
                   "error": r.get("error"), "host_prefix": r.get("hp"), "local_prefix": r.get("lp"),
                   "away": bool(r.get("ok") and r.get("away"))}
    return {"op": "hello", "computer": dict(base, cards=cards, procs=procs, resolve=res)}


def _fresh_reply():
    r = _A.get("reply")
    if r and r.get("_addr") == _A.get("addr") and time.time() - r["_t"] < 10:
        return r
    return None


def _status():
    r = _A.get("reply")
    if r and r.get("_addr") == _A.get("addr") and time.time() - r["_t"] < 30:
        return r.get("status")
    return None


# ---- when the host stops answering: another computer carries the render on as host ------------
def _fo(k):
    return (_TEST.get("fo") or {}).get(k, _FO[k])


def _reach(addrs, port, job):
    """Which of a host's addresses answers for this render? None if none does."""
    for a in addrs or []:
        addr = "%s:%s" % (a, port)
        try:
            r = rpc(addr, {"op": "ping"}, timeout=3.0)
        except (OSError, ValueError):
            continue
        if r.get("ok") and r.get("job") == job and r.get("running"):
            return addr
    return None


def _fo_loop():
    while not _A["quit"].is_set():
        try:
            _fo_step()
        except Exception:
            traceback.print_exc()
        _A["quit"].wait(2.0)


def _fo_step():
    """Runs in its own thread (it reads the NAS and the network, which can both be slow)."""
    inp, st = _A.get("fo_in"), _A["fo"]
    if not inp or not inp["owner"] or _A.get("fo_action"):
        return
    me = inp["cid"]
    lockname = ".multigpu-lock.json"
    # This computer was the host, and another one has taken its render over: join that one
    mv = inp.get("moved")
    if mv and inp["mode"] == "HOST":
        addr = _reach(mv.get("addrs"), mv.get("port"), mv.get("job"))
        if addr:
            st["msg"] = None
            _A["fo_action"] = {"kind": "join", "addr": addr, "name": mv.get("name"), "job": mv.get("job")}
        else:
            st["msg"] = "%s took over as host, but this computer can't reach it. Check its firewall" % mv.get("name")
        return
    boot = _A.get("boot_job")                        # (the same, found out after this computer was restarted)
    if boot and inp["mode"] == "HOST":
        if time.time() - boot.setdefault("t0", time.time()) > 180:
            _A["boot_job"] = None
            return
        lock = _read_json(os.path.join(boot["outdir"], lockname))
        if lock and lock.get("job") == boot["id"] and lock.get("host") != me and (lock.get("term") or 0) > boot["term"]:
            addr = _reach(lock.get("addrs"), lock.get("port"), boot["id"])
            if addr:
                _A["boot_job"] = None
                _A["fo_action"] = {"kind": "join", "addr": addr, "name": lock.get("host_name"), "job": boot["id"]}
        elif lock is not None or os.path.isdir(boot["outdir"]):
            _A["boot_job"] = None                    # nobody took it over
        return
    if inp["mode"] != "JOIN" or not inp["joined"] or not inp["addr"]:
        st.pop("watch", None)
        st["msg"] = None
        return
    # This computer has joined a host. Is the host still there?
    ok_addr, ok_t = _A["host_ok"]
    rep = _A.get("rep")
    down = time.time() - ok_t
    if not rep or rep.get("addr") != inp["addr"] or ok_addr != inp["addr"] or down < _fo("watch"):
        st.pop("watch", None)
        st.pop("claim_seen", None)
        st["msg"] = None
        return
    if rep.get("data") is None:
        rep["data"] = json.loads(zlib.decompress(base64.b64decode(rep["blob"])).decode("utf-8"))
    data = rep["data"]
    j = data["job"]
    host = j.get("host_name") or "The host"
    outdir = map_path(j["outdir"], rep.get("maps") or [])
    lockp = os.path.join(outdir, lockname)
    lock = _read_json(lockp)
    if lock is None or lock.get("job") != j["id"]:
        st.pop("watch", None)
        st["msg"] = "%s isn't answering%s" % (host, "" if lock is not None or os.path.isdir(outdir) else
                                              ", and this computer can't reach the output folder")
        return
    term = j.get("term", 0)
    if lock.get("host") != j["host"] and (lock.get("term") or 0) > term:      # another computer took over
        if lock.get("host") == me:
            return
        addr = _reach(lock.get("addrs"), lock.get("port"), j["id"])
        if addr:
            st["msg"] = None
            _A["fo_action"] = {"kind": "join", "addr": addr, "name": lock.get("host_name"), "job": j["id"]}
        else:
            st["msg"] = ("%s took over as host, but this computer can't reach it. Check its firewall"
                         % (lock.get("host_name") or "Another computer"))
        return
    if not inp["auto"]:
        st["msg"] = "%s isn't answering. Waiting for it to come back" % host
        return
    # The marker in the output folder is still the old host's. A live host refreshes it every few seconds
    try:
        mt = os.path.getmtime(lockp)
    except OSError:
        mt = None
    key = [lock.get("n"), lock.get("t"), mt]
    mono = time.monotonic()
    w = st.get("watch")
    if not w or w["job"] != j["id"]:
        w = st["watch"] = {"job": j["id"], "key": key, "t": mono, "moved": False}
    elif w["key"] != key:
        w.update(key=key, t=mono, moved=True)
    quiet = mono - w["t"]
    if w["moved"] and quiet < _fo("stale"):
        st["msg"] = "%s is still rendering, but this computer can't reach it" % host
        return
    cards = data.get("cards") or {}
    ready = sorted((c for c in (data.get("computers") or {}).values()
                    if c.get("failover") and c.get("id") != j["host"] and data["t"] - c.get("seen", 0) < 90),
                   key=lambda c: (-sum(1 for k in c.get("cards") or [] if (cards.get(k) or {}).get("use")), c["id"]))
    ids = [c["id"] for c in ready]
    wait = (ids.index(me) if me in ids else len(ids)) * _fo("step")      # the computer with the most cards goes first
    left = max(_fo("after") + wait - down, _fo("stale") + wait - quiet)
    if left > 0:
        st["msg"] = "%s isn't answering. This computer takes over as host in %d s" % (host, int(left) + 1)
        return
    claim = os.path.join(outdir, ".multigpu-takeover-%d.json" % (term + 1))
    mine = False
    try:                                             # only one computer can create this file
        with open(claim, "x", encoding="utf-8") as f:
            json.dump({"host": me, "name": inp["name"], "t": time.time()}, f)
        mine = True
    except FileExistsError:
        c = _read_json(claim) or {}
        if c.get("host") == me:
            mine = mono - st.get("promoted", -1e9) > 60
        else:
            if mono - st.setdefault("claim_seen", mono) > _fo("claim"):    # it never finished taking over
                st.pop("claim_seen", None)
                try:
                    os.remove(claim)
                except OSError:
                    pass
            st["msg"] = "%s isn't answering. %s is taking over as host" % (host, c.get("name") or "Another computer")
    except OSError as e:
        st["msg"] = "%s isn't answering, and this computer can't write to the output folder (%s)" % (host, e.strerror or e)
    if not mine:
        return
    again = _read_json(lockp) or {}
    if again.get("host") != j["host"] or again.get("n") != lock.get("n"):   # the host is back, or someone was faster
        try:
            os.remove(claim)
        except OSError:
            pass
        st.pop("watch", None)
        return
    st["promoted"] = mono
    st["msg"] = "%s isn't answering. This computer is taking over as host" % host
    _promote(inp, rep, data, outdir)


def _promote(inp, rep, data, outdir):
    """Turn the copy of the host's render into this computer's own host state, in this computer's paths."""
    j, me, maps = data["job"], inp["cid"], rep.get("maps") or []
    old, old_name = j["host"], j.get("host_name") or "The host"
    paths = [map_path(p, maps) for p in j["paths"]]
    if not paths or not _same_file(os.path.dirname(paths[0]), outdir):
        raise OSError("can't take over: the output folder isn't known on this computer")
    skew = rep["got"] - data["t"]                    # this computer's clock minus the old host's
    j["origin"] = j.get("origin") or old             # the .blend's paths stay those of where it was started
    j.update(paths=paths, outdir=outdir, host=me, host_name=inp["name"], term=j.get("term", 0) + 1, sync_pending=False)
    j.pop("moved", None)
    for k in ("started", "synced_at"):
        if j.get(k):
            j[k] += skew
    for fs in j["fs"].values():
        for k in ("t", "free_t"):
            if fs.get(k):
                fs[k] += skew
    for c in data["cards"].values():
        if c.get("frame_t"):
            c["frame_t"] += skew
        if c.get("computer") == old:                 # its cards are gone with it: their frames go back in the queue
            c.update(frame=None, frame_t=None, backup=False)
            if c.get("phase") in ("loading", "ready", "rendering", "idle", "reloading", "restarting"):
                c["phase"] = "lost"
    for c in data["computers"].values():
        c["host"] = c.get("id") == me
    events = [[t + skew, text] for t, text in data.get("events") or []]
    events.append([time.time(), "%s stopped answering. %s took over as host" % (old_name, inp["name"])])
    os.makedirs(_ROOT, exist_ok=True)
    with open(_COORD_STATE + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"job": j, "cards": data["cards"], "computers": data["computers"], "events": events}, f)
    os.replace(_COORD_STATE + ".tmp", _COORD_STATE)
    _A["fo_action"] = {"kind": "promote", "job": j["id"], "old": old_name, "addr": rep["addr"],
                       "ip": (data["computers"].get(me) or {}).get("ip")}


def _fo_apply(prefs):
    """Back in Blender's own thread: switch this computer over."""
    act = _A.get("fo_action")
    if not act:
        return
    _A["fo_action"] = None
    if act["kind"] == "promote" and (prefs.mode != "JOIN" or _join_addr(prefs) != act["addr"]):
        return                                       # (you changed the connection in the meantime)
    if not prefs.fo_return:                          # remember what to go back to when this render is over
        prefs.fo_return = json.dumps({"mode": prefs.mode, "addr": prefs.host_address, "job": act.get("job")})
    if act["kind"] == "promote":
        prefs.mode = "HOST"
        _A["running_job"] = act["job"]
        _A["takeover"] = {"t": time.time(), "ip": act.get("ip")}
        _A["coord_t"] = 0.0
        _A["msg"] = "%s stopped answering, so this computer took over as host" % act["old"]
    else:
        name = act.get("name") or act["addr"]
        prefs.mode = "JOIN"
        prefs.host_address = act["addr"]
        prefs.joined = True
        if not any(h.address.strip().lower() == act["addr"].lower() for h in prefs.hosts):
            h = prefs.hosts.add()
            h.address, h.name = act["addr"], name
        _A["msg"] = "%s took over as host. This computer now helps it" % name
    _mark_dirty()


def _fo_back(prefs):
    """After a take-over, once that render is over: each computer goes back to what it was before
    (the computer you started it on is the host again)."""
    if not prefs.fo_return:
        return
    try:
        d = json.loads(prefs.fo_return)
    except ValueError:
        d = {}
    r = _fresh_reply()
    if not d or not r or _A.get("fo_action") or _A.get("running_job"):
        _A["fo_over"] = None
        return
    job = r.get("job") or {}
    if job.get("state") == "running":                # a newer render: leave things as they are
        prefs.fo_return = ""
        _mark_dirty()
        return
    if job.get("id") != d.get("job") and job:
        prefs.fo_return = ""
        _mark_dirty()
        return
    t = time.time()
    _A["fo_over"] = _A.get("fo_over") or t
    if prefs.mode == "HOST" and t - _A["fo_over"] < _fo("back"):
        return                                       # (the others first get to see that the render is over)
    _A["fo_over"] = None
    prefs.fo_return = ""
    if d.get("mode") == "JOIN" and d.get("addr"):
        prefs.mode, prefs.host_address, prefs.joined = "JOIN", d["addr"], True
        name = next((h.name for h in prefs.hosts if h.address.strip().lower() == d["addr"].strip().lower()), d["addr"])
        _A["msg"] = "The render is done: this computer is back to helping %s" % name
    elif d.get("mode") in ("HOST", "SOLO"):
        prefs.mode = d["mode"]
        _A["msg"] = "The render is done: this computer is the host again"
    _mark_dirty()


def _fo_tick(prefs, owner, addr):
    """What the take-over thread needs to know, gathered where Blender's data may be read."""
    if owner and not _A["boot_done"]:
        _A["boot_done"] = True
        if prefs.mode == "HOST" and not _coord_info():   # was this computer hosting a render when it went down?
            j = (_read_json(_COORD_STATE) or {}).get("job")
            if j and j.get("state") == "running" and j.get("host") == _computer_id():
                _A["boot_job"] = {"id": j["id"], "term": j.get("term", 0), "outdir": j["outdir"]}
    job = (_fresh_reply() or {}).get("job") or {}
    moved = dict(job["moved"], job=job["id"]) if job.get("state") == "moved" and job.get("moved") else None
    _A["fo_in"] = {"mode": prefs.mode, "joined": bool(prefs.joined), "owner": bool(owner), "addr": addr,
                   "auto": bool(prefs.auto_takeover and not _away_now()), "cid": _computer_id(),
                   "name": _computer_name(), "moved": moved}
    job_id = _A.get("running_job")
    if owner and addr and job_id and _A["addr_note"] != (job_id, addr):
        _write_json(_ADDR_FILE, {"job": job_id, "addr": addr, "t": time.time()})   # the cards read where the host is
        _A["addr_note"] = (job_id, addr)


# ---- a computer with its own copy of the project folder ----------------------------------
def _frames_here(pattern, ranges, maps):
    """Of the render's frames, the ones this computer already has in its own copy of the output folder."""
    if not pattern or not ranges:
        return []
    d = map_path(pattern["dir"], maps)
    try:
        have = set(e.name for e in os.scandir(d) if e.is_file() and e.stat().st_size > 0)
    except OSError:
        return []
    return [f for f in from_ranges(ranges) if pattern_name(pattern, f) in have]


def _own_folder(out):
    """This computer's own copy of the output folder can be written to: None, or what's wrong."""
    try:
        os.makedirs(out, exist_ok=True)
        p = os.path.join(out, ".syncthing.multigpu-write-test-%s.tmp" % uuid.uuid4().hex[:6])
        with open(p, "w") as f:
            f.write("x")
        os.remove(p)
    except OSError as e:
        return "it can't write to its own copy of the output folder, %s (%s)" % (out, e.strerror or e)
    return None


def _frame_pattern(frames, paths):
    """How the render names its frames, so another computer can tell which ones it already has."""
    if not frames or min(frames) < 0:
        return None
    base, f = os.path.basename(paths[0]), frames[0]
    for pad in range(12, len(str(f)) - 1, -1):
        num = str(f).zfill(pad)
        i = base.rfind(num)
        if i >= 0 and not (i > 0 and base[i - 1].isdigit()):
            pat = {"dir": os.path.dirname(paths[0]), "head": base[:i], "tail": base[i + len(num):], "pad": pad}
            if all(os.path.basename(paths[k]) == pattern_name(pat, frames[k]) and os.path.dirname(paths[k]) == pat["dir"]
                   for k in (0, len(frames) // 2, len(frames) - 1)):
                return pat
    return None


# ---- finding the .blend on a joined computer -------------------------------------------
def _resolve_away(job, res, addr):
    """Does this computer see the host's output folder itself, or does it have its own copy that a sync
    program keeps in step? The host drops a small file into its folder: if it shows up here within seconds,
    it is one folder. With its own copy, tell the host which frames are here already."""
    if not res.get("hp") or not res.get("lp") or not addr:
        return                                       # the very same path as the host's: one folder
    maps = [[res["hp"], res["lp"]]]
    out = map_path(job["outdir"], maps)
    try:
        r = rpc(addr, {"op": "probe", "job": job["id"]}, timeout=15.0)
    except OSError:
        return
    if not r.get("ok"):
        return
    t0 = time.time()
    while time.time() - t0 < 8:
        if os.path.isfile(os.path.join(out, r["name"])):
            return
        time.sleep(0.5)
    err = _own_folder(out)
    if err:
        res.update(ok=False, error=err)
        return
    res["away"] = True
    if not (job.get("opts") or {}).get("overwrite"):
        have = _frames_here(job.get("pattern"), job.get("ranges"), maps)
        if have:
            try:
                rpc(addr, {"op": "have", "job": job["id"], "computer": _computer_id(), "frames": have}, timeout=30.0)
            except OSError:
                res.update(ok=False, error="couldn't tell the host which frames are already here")


def _resolve_thread(job, extra, key, prev, addr=None):
    path, size = job["blend"]["path"], job["blend"].get("size")
    res = {"busy": False, "key": key, "sync": job["sync"], "ok": False}
    t0 = time.time()
    while True:
        try:
            if prev.get("path") and os.path.isfile(prev["path"]) and \
                    (not size or os.path.getsize(prev["path"]) == size):
                res.update(ok=True, path=prev["path"], hp=prev.get("hp"), lp=prev.get("lp"), error=None)
            else:
                lp, hp, lpre = find_blend(path, size, extra)
                res.update(ok=True, path=lp, hp=hp, lp=lpre, error=None)
            break
        except LookupError as e:                    # just saved on the host: wait for the NAS to catch up
            res["error"] = str(e)
            if "isn't the same version" not in str(e) or time.time() - t0 > 60:
                break
        except Exception as e:
            res["error"] = str(e)
            break
        time.sleep(2.0)
    if res.get("ok"):
        try:
            _resolve_away(job, res, addr)
        except Exception:
            traceback.print_exc()
    res["t"] = time.time()
    _A["resolve"][job["id"]] = res


def _local_blend(prefs, job):
    """This computer's path to the job's .blend, or None while it's being looked for."""
    if _origin_here(job):
        return job["blend"]["path"]
    key = [job["id"], job["sync"], job["blend"].get("size")]
    res = _A["resolve"].get(job["id"]) or {}
    if res.get("key") == key:
        if res.get("ok"):
            return res["path"]
        if res.get("busy") or time.time() - res.get("t", 0) < 15:
            return None                              # still looking, or failed a moment ago
    prev = res if res.get("ok") or res.get("prev") is None else res.get("prev")
    prev = {k: prev.get(k) for k in ("path", "hp", "lp")} if prev else {}
    _A["resolve"][job["id"]] = {"busy": True, "key": key, "sync": job["sync"], "prev": prev}
    addr = _A.get("addr") if _is_host_mode(prefs) else _join_addr(prefs)
    threading.Thread(target=_resolve_thread, args=(job, prefs.shared_roots, key, prev, addr), daemon=True).start()
    return None


# ---- the check before a render: can this computer do its part? -----------------------------
def _pre_thread(pre, extra, datafiles, addons):
    res = {"id": pre["id"], "ok": False}
    try:
        lp, hp, lpre = find_blend(pre["blend"]["path"], pre["blend"].get("size"), extra)
        maps = [[hp, lpre]] if hp and lpre else []
        if pre.get("datafiles") and datafiles and split_any(datafiles) != split_any(pre["datafiles"]):
            maps.append([pre["datafiles"], datafiles])
        miss = [f for f in pre.get("files") or [] if not os.path.exists(map_path(f, maps))]
        sizes = pre.get("copy") or {}
        sent = [f for f in miss if f in sizes]                       # the host sends these before the render
        miss = [f for f in miss if f not in sizes and not bundled_here(f, datafiles)]
        if sent:
            need = sum(sizes[f] for f in sent)
            try:
                os.makedirs(_ROOT, exist_ok=True)
                free = shutil.disk_usage(_ROOT).free
            except OSError:
                free = None
            if free is not None and free < need + (2 << 30):
                res["copy_error"] = "its temporary folder has no room for the %s it would be sent" % _size_text(need)
                miss = sent + miss
            else:
                res["copy"] = [len(sent), need, [(split_any(f)[1] or [f])[-1] for f in sent[:9]]]
        res["missing"] = [(split_any(f)[1] or [f])[-1] for f in miss[:9]]
        res["nmissing"] = len(miss)
        out = map_path(pre["outdir"], maps)
        t0 = time.time()
        while not os.path.isfile(os.path.join(out, pre["probe"])) and time.time() - t0 < 6:
            time.sleep(1.0)                         # network shares can take a moment to show a new file
        if not os.path.isfile(os.path.join(out, pre["probe"])):
            # not the host's folder itself: this computer's own copy, which a sync program keeps in step
            err = _own_folder(out) if maps else "its output folder (%s) isn't this computer's" % out
            if err:
                res["out_error"] = err
            else:
                res["away"] = True
                if not pre.get("overwrite"):
                    res["have"] = to_ranges(_frames_here(pre.get("pattern"), pre.get("ranges"), maps))
        else:
            try:
                p = os.path.join(out, ".multigpu-write-test-%s" % uuid.uuid4().hex[:6])
                with open(p, "w") as f:
                    f.write("x")
                os.remove(p)
            except OSError as e:
                res["out_error"] = "it can't write to %s (%s)" % (out, e.strerror or e)
        res["needs"] = [n for n in pre.get("needs") or [] if n not in addons]
        res["ok"] = True
    except LookupError as e:
        res["error"] = str(e)
    except Exception as e:
        res["error"] = "the check failed: %s" % e
    _A["pre_result"] = res


def _pre_tick(prefs):
    r = _fresh_reply()
    pre = (r or {}).get("pre")
    if not pre or pre.get("id") == _A["pre_id"] or pre.get("host") == _computer_id():
        return
    _A["pre_id"] = pre["id"]
    _A["pre_result"] = None
    try:
        datafiles = bpy.utils.system_resource('DATAFILES')
    except Exception:
        datafiles = None
    addons = set(k.split(".")[-1] for k in bpy.context.preferences.addons.keys())
    threading.Thread(target=_pre_thread, args=(pre, prefs.shared_roots, datafiles, addons), daemon=True).start()


# ---- launching and watching this computer's cards --------------------------------------
def _save_procs():
    _write_json(_AGENT_FILE, {"procs": {k: {x: y for x, y in p.items() if x != "popen"} for k, p in _A["procs"].items()},
                              "seq": _A["seq"]})


def _load_procs():
    data = _read_json(_AGENT_FILE) or {}
    _A["seq"].update(data.get("seq") or {})
    for cid, p in (data.get("procs") or {}).items():
        if cid not in _A["procs"] and not p.get("dead") and _alive(p.get("pid"), pstart=p.get("pstart")):
            _A["procs"][cid] = p


def _layout(prefs, job, order):
    """Grid cells for this job: [main window] + one per card, on the monitor Blender is on."""
    key = (job["id"], tuple(order))
    if _A["layout"].get("key") == key:
        return _A["layout"]
    lay = {"key": key, "rects": {}, "screen_h": None}
    if prefs.show_windows and prefs.arrange_windows and sys.platform != "darwin":
        try:
            area, sh, found = screen_info(os.getpid())
            cells = grid_cells(area, len(order) + 1)
            lay["screen_h"] = sh
            lay["rects"] = {cid: cells[i + 1] for i, cid in enumerate(order)}
            if found:
                if not _A.get("win_saved"):
                    _A["win_saved"] = save_own()          # put back when the render ends
                place_own(cells[0])
                _A["msg"] = None
            else:
                _A["msg"] = ("Wayland doesn't let MultiGPU move this Blender window (the render windows are still "
                             "tiled). To include it, start Blender with: WAYLAND_DISPLAY= blender")
        except Exception as e:
            _A["msg"] = "Couldn't arrange the windows: %s" % e
    _A["layout"] = lay
    return lay


def _launch(prefs, g, cid, job, blend, maps, order, addr):
    d = os.path.join(_ROOT, "cards", job["id"])
    os.makedirs(d, exist_ok=True)
    worker = os.path.join(d, "worker.py")
    if not os.path.exists(worker):
        with open(worker, "w", encoding="utf-8") as f:
            f.write(SHARED + "\n" + WORKER)
    ids = json.loads(g.ids or "{}")
    show = bool(prefs.show_windows)
    lay = _layout(prefs, job, order) if show else {"rects": {}}
    rect = lay["rects"].get(cid)
    n = _A["seq"].get(cid, 0)
    status = os.path.join(d, "%s.status.json" % cid)
    logp = os.path.join(d, "%s.%d.log" % (cid, n))
    cfg = {"addr": addr, "card": cid, "computer": _computer_id(), "label": g.name, "backend": g.backend,
           "backend_label": _LABEL.get(g.backend, g.backend), "device_id": ids.get(g.backend, ""),
           "other_ids": [v for o in prefs.gpus if o.key != g.key for v in json.loads(o.ids or "{}").values()],
           "job": job["id"], "sync": job["sync"], "token": job["blend"].get("token"), "scene": job["scene"],
           "output": job["output"], "opts": job["opts"], "maps": maps, "ui": show, "rect": rect,
           "status": status, "log": logp, "version": VERSION, "first": job.get("first"),
           "lock": os.path.join(_ROOT, "locks", "%s.lock" % cid), "addr_file": _ADDR_FILE,
           "away": bool((_A["resolve"].get(job["id"]) or {}).get("away")) and not _origin_here(job)}
    prev = _A["procs"].get(cid) or {}
    if prev.get("job") == job["id"] and prev.get("tile"):
        cfg["tile"] = prev["tile"]                  # this card ran out of GPU memory earlier in this render
    cfg.update(_TEST.get("cfg") or {})
    cfgp = os.path.join(d, "%s.cfg.json" % cid)
    with open(cfgp, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    try:
        os.remove(status)
    except OSError:
        pass
    exe = list(_TEST["cmd"]) if _TEST.get("cmd") else [bpy.app.binary_path]
    if show:
        cmd = exe + ["--no-window-focus"]
        if rect and lay.get("screen_h"):
            x, y, w, h = rect
            cmd += ["--window-geometry", str(x), str(lay["screen_h"] - (y + h)), str(w), str(h)]
    else:
        cmd = exe + ["-b"]
    cmd += ["-y" if job["opts"].get("autoexec", True) else "-Y", blend, "-S", job["scene"],
            "--python-exit-code", "3", "--python", worker, "--", cfgp]
    env = dict(os.environ, MULTIGPU_WORKER="1")
    if show and prefs.arrange_windows and sys.platform.startswith("linux") and env.get("DISPLAY"):
        env["WAYLAND_DISPLAY"] = ""           # X11 windows can be placed; Wayland ones can't
    p = _popen(cmd, logp, env)
    old = _A["procs"].get(cid) or {}
    same = old.get("job") == job["id"]
    _A["procs"][cid] = {"pid": p.pid, "pstart": proc_start(p.pid), "popen": p, "job": job["id"], "sync": job["sync"],
                        "status": status, "log": logp, "launched": time.time(),
                        "crashes": old.get("crashes", []) if same else [], "stale": old.get("stale", 0) if same else 0,
                        "dups": old.get("dups", 0) if same else 0, "gave_up": old.get("gave_up") if same else None,
                        "tile": cfg.get("tile")}
    _save_procs()


def _card_state(p):
    st = _read_json(p["status"]) or {}
    return st.get("state")


def _on_exit(cid, p):
    code = p["popen"].poll() if p.get("popen") else None
    st = _read_json(p["status"]) or {}
    state = st.get("state")
    if p.get("relaunch"):
        why = "reload"
    elif code == 5 or state == "reload":
        why = "reload"
    elif code == 6 or state == "stale":
        why = "stale"
    elif code == 7 or state == "duplicate":
        why = "duplicate"
    elif code == 8 or state == "oom":
        why = "oom"
    elif p.get("killed"):
        why = "killed"
    elif state in ("done", "stopped", "over"):
        why = state
    elif state == "error" or code in (3, 4):
        why = "error"
    else:
        why = "crash"
    t = time.time()
    p.update(dead=True, why=why, error=st.get("error"), ended=t, code=code)
    if why == "crash" and _log_says_oom(p.get("log")):   # some drivers take Blender down with them
        new = (p.get("tile") or 2048) // 2
        if new >= 256:
            why = "oom"
            st = dict(st, tile=new)
        else:
            why = "error"
            p["error"] = "out of GPU memory, even in small tiles"
        p["why"] = why
    if why == "oom":
        p["tile"] = st.get("tile") or max(256, (p.get("tile") or 2048) // 2)
        p["retry_at"] = t
    elif why == "reload":
        p["retry_at"] = t
    elif why == "stale":
        p["stale"] = p.get("stale", 0) + 1
        p["retry_at"] = t + 3
        if p["stale"] > 20:
            p.update(why="error", error="this computer keeps seeing a different version of the .blend than the host saved",
                     gave_up=p.get("epoch"))
    elif why == "duplicate":                        # another Blender already renders as this card
        p["dups"] = p.get("dups", 0) + 1
        p["retry_at"] = t + 20
        if p["dups"] > 6:
            p.update(why="error", error="another Blender on this computer is still running as this card. Close "
                     "leftover Blender windows, then press play here", gave_up=p.get("epoch"))
    elif why == "crash":
        p["crashes"] = [c for c in p.get("crashes", []) if t - c < 600] + [t]
        p["retry_at"] = t + 5
        if len(p["crashes"]) >= 3:
            p.update(why="gave_up", gave_up=p.get("epoch"))
    elif why == "error":
        p["gave_up"] = p.get("epoch")
    if p["why"] not in ("killed", "done", "stopped", "over", "duplicate"):
        _A["seq"][cid] = _A["seq"].get(cid, 0) + 1
    _save_procs()


def _log_says_oom(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            low = f.read().decode("utf-8", "replace").lower()
        return "out of memory" in low or "out of gpu memory" in low
    except (OSError, TypeError):
        return False


def _track():
    """Follow the render (every Blender on the computer does this, also one that only looks)."""
    r = _fresh_reply()
    if not r:
        return None
    job = r.get("job")
    was = _A.get("running_job")
    _A["running_job"] = job["id"] if job and job["state"] == "running" else None
    if was and not _A["running_job"] and _A.get("win_saved"):
        try:                                           # the render is over: Blender's window goes back
            restore_own(_A["win_saved"])
        except Exception as e:
            print("MultiGPU: couldn't restore the window:", e)
        _A["win_saved"] = None
    return r


def _reconcile(prefs, r):
    t = time.time()
    for cid, p in list(_A["procs"].items()):
        if p.get("dead"):
            continue
        if not _alive(p["pid"], p.get("popen"), p.get("pstart")):
            _on_exit(cid, p)
        elif p.get("killed") and t - p["killed"] > 5:
            _kill(p["pid"], hard=True)
    if not r:
        return
    changed = False
    for g in prefs.gpus:                               # cards ticked / unticked from the host's panel
        use, v = (r.get("uses") or {}).get(_card_id(g.key), (None, 0))
        if use is not None and v > _A["use_v"].get(_card_id(g.key), 0):
            if g.use != use:
                g.use = use
                changed = True
            _A["use_v"][_card_id(g.key)] = v
    if changed:
        _mark_dirty()
    job = r.get("job")
    kill = set(r.get("kill") or [])
    for cid, p in _A["procs"].items():
        if p.get("dead") or p.get("killed"):
            continue
        if cid in kill or not job or p["job"] != job["id"] or job["state"] != "running":
            _kill(p["pid"])
            p["killed"] = t
    if not job or job["state"] != "running":
        return
    # Sync: a card still loading the old file is restarted now instead of after its first frame
    if job.get("sync_pending"):
        for cid, p in _A["procs"].items():
            if not p.get("dead") and not p.get("killed") and _card_state(p) in (None, "starting"):
                _kill(p["pid"])
                p["killed"] = t
                p["relaunch"] = True
        return
    if job.get("unsynced"):
        return        # the file on disk is newer than what's being rendered: nothing may load it until Sync Changes
    blend = _local_blend(prefs, job)
    if not blend:
        return
    addr = _A["addr"] if _is_host_mode(prefs) else _join_addr(prefs)
    maps = [list(m) for m in r.get("maps") or []]
    res = _A["resolve"].get(job["id"]) or {}
    if not _origin_here(job) and res.get("hp") and res.get("lp"):
        maps.insert(0, [res["hp"], res["lp"]])       # host's NAS path -> ours (known before the host echoes it)
    _A["job_maps"] = (job["id"], maps)
    want = r.get("want") or {}
    gpus = {_card_id(g.key): g for g in prefs.gpus}
    order = [_card_id(g.key) for g in prefs.gpus if _card_id(g.key) in want]
    for cid in order:
        p = _A["procs"].get(cid)
        epoch = want[cid]
        if p and not p.get("dead"):
            continue
        if p and p.get("job") == job["id"]:
            if p.get("gave_up") is not None and p.get("gave_up") == epoch:
                continue
            if p.get("why") in ("done", "stopped", "over", "killed") and p.get("epoch") == epoch and \
                    not p.get("relaunch") and t - p.get("ended", 0) < 10:
                continue
            if p.get("retry_at", 0) > t and p.get("epoch") == epoch:
                continue
        try:
            _launch(prefs, gpus[cid], cid, job, blend, maps, order, addr)
            _A["procs"][cid]["epoch"] = epoch
        except OSError as e:
            _A["msg"] = "Couldn't start Blender for %s: %s" % (gpus[cid].name, e)


def _sync_tick():
    """Save once, then retry notifying the host off the UI thread until it acknowledges."""
    s = _A.get("sync")
    if not s:
        return
    t = time.time()
    if t < s["next"]:
        return
    if s.get("payload") is not None:
        reply = s.pop("reply", None)
        if reply is not None:
            r, error = reply
            s.pop("notifying", None)
            if error is not None:
                s["next"] = t + 2
                _A["msg"] = "Changes saved. Waiting to reach the host; retrying automatically: %s" % error
                return
            _A["sync"] = None
            _A["msg"] = ("Saved, but couldn't sync: %s" % r["error"]) if r.get("error") else \
                "Changes saved. Each card picks them up after its current frame"
            return
        if not s.get("notifying"):
            s["notifying"] = True

            def notify():
                try:
                    s["reply"] = (rpc(s["addr"], s["payload"]), None)
                except OSError as e:
                    s["reply"] = (None, str(e))

            threading.Thread(target=notify, daemon=True).start()
        return
    if not _same_file(bpy.data.filepath, s["blend"]):
        err, give_up = "this Blender now has a different file open than the one being rendered", True
    else:
        err, give_up = _save_now(s["token"], s["scene"]), t - s["t0"] > 40
    if err is None:
        more = {}
        p = _prefs()
        if p is not None and p.copy_missing:
            try:
                sets = {}
                more = {"files": sorted(_sendable(_external_files(sets)[0])), "sets": sets}
            except Exception:
                traceback.print_exc()
        s["payload"] = dict({"op": "sync_end", "token": s["token"], "size": os.path.getsize(s["blend"]),
                             "job": s["job"], "computer": _computer_id()}, **more)
        _A["msg"] = "Changes saved. Notifying the host..."
    elif give_up:
        _A["sync"] = None
        try:
            rpc(s["addr"], {"op": "sync_abort", "computer": _computer_id()})
        except OSError:
            pass
        _A["msg"] = "Couldn't sync: %s" % err
    else:
        s["next"] = t + 1.5


def _save_now(token, scene_name=None):
    """Save the .blend with a new version token in the scene being rendered. Returns an error or None."""
    sc = bpy.data.scenes.get(scene_name) if scene_name else bpy.context.scene
    if sc is None:
        return "the scene %r isn't in this file anymore" % scene_name
    old = sc.get(TOKEN)
    sc[TOKEN] = token
    if _TEST.get("nosave"):
        return None
    _A["saving"] = True
    try:
        wm = bpy.context.window_manager
        win = wm.windows[0] if wm and wm.windows else None
        if win is not None:
            with bpy.context.temp_override(window=win):
                result = bpy.ops.wm.save_mainfile()
        else:
            result = bpy.ops.wm.save_mainfile()
        if 'FINISHED' not in result:
            raise RuntimeError("Blender cancelled saving the file")
        _A["token_unsaved"] = False
        return None
    except RuntimeError as e:
        if old is not None:
            sc[TOKEN] = old
        else:
            del sc[TOKEN]
        return str(e).strip().splitlines()[-1] if str(e).strip() else "unknown error"
    finally:
        _A["saving"] = False


@persistent
def _on_save_pre(*args):
    """Ctrl+S while rendering: the file on disk stops being what the cards render. Give it a new version
    token, so no card can load it by accident before Sync Changes is pressed."""
    try:
        if _A.get("saving"):
            return
        job = (_A.get("reply") or {}).get("job")
        mine = _A.get("host_job")                    # (a render started a moment ago isn't in the reply yet)
        if mine and _A.get("running_job") == mine["id"] and (not job or job.get("id") != mine["id"]):
            job = mine
        if not job or job.get("state") != "running" or (job.get("host") != _computer_id() and not _origin_here(job)):
            return
        target = args[0] if args and isinstance(args[0], str) else bpy.data.filepath
        here = _blend_here(job)
        if not _same_file(bpy.data.filepath, here) or not _same_file(target, here):
            return
        sc = bpy.data.scenes.get(job["scene"])
        if sc is None:
            return
        sc[TOKEN] = uuid.uuid4().hex
        _A["token_unsaved"] = True
        _A["save_seen"] = True
    except Exception:
        traceback.print_exc()


@persistent
def _on_save_post(*args):
    if _A.pop("save_seen", None):
        _A["token_unsaved"] = False
        _A["unsynced_send"] = True


def _redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for a in win.screen.areas:
            if a.type in {'PROPERTIES', 'VIEW_3D', 'PREFERENCES'}:
                a.tag_redraw()


def _agent_tick():
    try:
        prefs = _prefs()
        if prefs is None:
            return 2.0
        owner = _owner()
        if owner and not _A["owner"]:
            _load_procs()                               # (also picks cards back up after Blender was restarted)
        elif not owner and _A["owner"]:
            _A["procs"] = {}                            # another Blender on this computer took over
        _A["owner"] = owner
        _fo_apply(prefs)
        if owner and not _A["boot_done"]:
            _fo_tick(prefs, owner, _A["addr"])          # (before a new host process replaces the saved state)
        if _is_host_mode(prefs):
            addr = _ensure_coord(prefs, start=prefs.mode == "HOST", owner=owner)
        else:
            if owner:
                _retire_coord()
            addr = _join_addr(prefs) if prefs.mode == "JOIN" and prefs.joined else None
        if addr != _A["addr"]:
            _A["reply"] = None
            _A["err"] = None
        _A["addr"] = addr
        _A["snap"] = _snapshot(prefs, owner) if addr else None
        if os.name == "nt" and prefs.mode == "HOST" and \
                time.time() - _A.get("fw_t", 0) > (60 if (_A.get("fw") or {}).get("block") else 600):
            _fw_check()
        r = _track()
        _fo_tick(prefs, owner, addr)
        if owner:
            _fo_back(prefs)
        if owner:
            _reconcile(prefs, r)
            _pre_tick(prefs)
        _sync_tick()
        if _A.get("unsynced_send") and addr and (_is_host_mode(prefs) or _origin_job()):
            try:
                rpc(addr, {"op": "unsynced", "computer": _computer_id()}, timeout=2)
                _A["unsynced_send"] = False
            except OSError:
                pass
        r = _A.get("reply")
        stamp = (id(r), _A.get("err"), _A.get("msg"), owner)
        if stamp != _A["drawn"] or (_A["running_job"] and time.time() - _A["last_draw"] > 1.0):
            _A["drawn"] = stamp
            _A["last_draw"] = time.time()
            _redraw()
    except Exception:
        traceback.print_exc()
    return 0.5


# ======================================================================================
# GPU detection and preferences
# ======================================================================================
def _backend_items(self, context):
    key = self.backends or ""
    items = _ENUM_CACHE.get(key)
    if items is None:
        items = [(b, _LABEL.get(b, b), "Render this card with " + _LABEL.get(b, b)) for b in key.split(",") if b]
        items = items or [("NONE", "None", "No supported backend")]
        _ENUM_CACHE[key] = items
    return items


def _cards_stale(prefs):
    """True when the saved list of cards no longer matches what Cycles sees (a card was added, removed,
    moved to another slot, or stopped working). Cheap: Cycles keeps its own list."""
    try:
        import _cycles
        seen = set()
        for t in _BACKENDS:
            for d in _cycles.available_devices(t):
                if len(d) >= 3 and d[1] == t:
                    seen.add(d[2])
    except Exception:
        return False
    saved = set(v for g in prefs.gpus if g.key != "CPU" and not g.key.startswith("CPU-")
                for v in json.loads(g.ids or "{}").values())
    return seen != saved


def refresh_cards(prefs):
    """Look for the cards again if they changed since the list was saved. Never while this computer renders:
    its running cards are known to the host by their place in the list."""
    if not prefs.gpus:
        detect_gpus(prefs)
        return True
    if _TEST.get("cpu") or any(not p.get("dead") for p in _A["procs"].values()):
        return False
    if _cards_stale(prefs):
        before = [(g.key, g.name) for g in prefs.gpus]
        detect_gpus(prefs)
        after = [(g.key, g.name) for g in prefs.gpus]
        if before and before != after:
            gone = [n for k, n in before if k not in dict(after)]
            new = [n for k, n in after if k not in dict(before)]
            print("MultiGPU: the graphics cards changed. Gone or moved: %s. Found: %s"
                  % (", ".join(gone) or "none", ", ".join(new) or "none"))
        return True
    return False


def detect_gpus(prefs):
    """Fill prefs.gpus with every graphics card Cycles can see, one row per physical card, plus the CPU
    (unticked unless there's no usable graphics card, as on some ARM computers)."""
    try:
        import _cycles
    except ImportError:
        return "Cycles isn't enabled"
    found = {}
    for t in _BACKENDS:
        try:
            devs = _cycles.available_devices(t)
        except Exception:
            continue
        for d in devs:
            if len(d) < 3 or d[1] != t:
                continue
            did = d[2]
            key = did[:-len("_OptiX")] if did.endswith("_OptiX") else did   # OptiX and CUDA = same card
            e = found.setdefault(key, {"name": d[0], "ids": {}})
            e["ids"][t] = did
    has_gpu = any(not any(s in e["name"] for s in _INTEGRATED) for e in found.values())
    found["CPU"] = {"name": "CPU (%d threads)" % (os.cpu_count() or 1), "ids": {"CPU": "CPU"}}
    if _TEST.get("cpu"):
        found = {"CPU-%d" % i: {"name": "Test CPU %d" % (i + 1), "ids": {"CPU": "CPU"}} for i in range(_TEST["cpu"])}
    old = {g.key: (g.use, g.backend) for g in prefs.gpus}
    try:
        for k, v in json.loads(prefs.gpu_memory or "{}").items():   # cards that weren't there last time
            old.setdefault(k, tuple(v))
    except (ValueError, TypeError):
        pass
    # A card gets another address when a card is added, removed or stops working. It is still the same
    # card: its tick and backend go with it.
    spare = {}
    for k in old:
        if k not in found:
            spare.setdefault(device_base(k), []).append(k)
    for key in found:
        if key not in old and spare.get(device_base(key)):
            old[key] = old.pop(spare[device_base(key)].pop(0))
    prefs.gpu_memory = json.dumps({k: list(v) for k, v in old.items() if k not in found})   # gone for now
    prefs.gpus.clear()
    names = [e["name"] for e in found.values()]
    seen = {}
    for key, e in found.items():
        nm = e["name"]
        short = nm.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        if names.count(nm) > 1:
            seen[nm] = seen.get(nm, 0) + 1
            short += " #%d" % seen[nm]
        g = prefs.gpus.add()
        g.key, g.name, g.ids = key, short, json.dumps(e["ids"])
        g.backends = ",".join(t for t in _BACKENDS + ("CPU",) if t in e["ids"])
        if key in old:
            g.use = old[key][0]
            if old[key][1] in e["ids"]:
                g.backend = old[key][1]
        elif key == "CPU":
            g.use = not has_gpu
        else:
            g.use = not any(s in nm for s in _INTEGRATED)
    _mark_dirty()
    return None


class MULTIGPU_Device(bpy.types.PropertyGroup):
    key: StringProperty()
    ids: StringProperty()
    backends: StringProperty()
    use: BoolProperty(name="Use", default=True, description="Render on this card")
    backend: EnumProperty(name="Backend", items=_backend_items,
                          description="Which Cycles backend this card renders with")


class MULTIGPU_Host(bpy.types.PropertyGroup):
    name: StringProperty(name="Name", description="Your own name for this host")
    address: StringProperty(name="Address")


class MULTIGPU_Prefs(bpy.types.AddonPreferences):
    bl_idname = __name__
    gpus: CollectionProperty(type=MULTIGPU_Device)
    hosts: CollectionProperty(type=MULTIGPU_Host)
    gpu_memory: StringProperty(default="", options={'HIDDEN'},
                               description="Ticks of cards that aren't connected right now")
    mode: EnumProperty(
        name="Network", default="SOLO",
        items=[("SOLO", "This Computer", "Render on this computer's cards only"),
               ("HOST", "Host", "Render here and let other computers on your network join in. They type in "
                                "this computer's address"),
               ("JOIN", "Join", "Help another computer's render: type in its address")])
    port: IntProperty(name="Port", default=DEFAULT_PORT, min=1024, max=65535,
                      description="Network port the host listens on. Must be the same on every computer and "
                                  "allowed through the host's firewall")
    host_address: StringProperty(name="Host", default="",
                                 description="The host computer's address or name, e.g. 192.168.1.20, a Tailscale "
                                             "address (100.x.x.x) or a Tailscale machine name. Shown on the host")
    joined: BoolProperty(name="Joined", default=False)
    fo_return: StringProperty(default="", options={'HIDDEN'},
                              description="What this computer was before it took over as host or followed a new host")
    auto_takeover: BoolProperty(
        name="Take Over if the Host Stops", default=True,
        description="If the host stops answering for a minute in the middle of a render (it crashed or lost its "
                    "network), this computer carries the render on as the new host. Frames that are being "
                    "rendered are kept, and the old host joins back in as a helper when it returns")
    shared_roots: StringProperty(
        name="Project Folder on This Computer", default="",
        description="Where this computer has the project: the NAS share (e.g. B:\\ or /mnt/nas), or its own copy "
                    "of it that a sync program such as Syncthing keeps in step (e.g. D:\\Sync). Only needed when "
                    "the host's .blend isn't found by itself, or to work from this computer's own copy instead of "
                    "a far-away original. Separate several with ;")
    show_windows: BoolProperty(
        name="Show Render", default=True,
        description="Open a Blender window for each card that shows its frame rendering live. They close by "
                    "themselves when done. Off: the cards render hidden")
    arrange_windows: BoolProperty(
        name="Arrange Windows", default=True,
        description="Tile this Blender window and the card windows in a grid on the monitor Blender is on")
    persistent_data: BoolProperty(
        name="Persistent Data", default=True,
        description="Keep the scene in memory between frames on every card, so it isn't rebuilt each frame")
    save_first: BoolProperty(
        name="Save First", default=True,
        description="Save the .blend before rendering. The cards render the saved file, so unsaved changes "
                    "wouldn't be in the render")
    placeholders: BoolProperty(
        name="Placeholders", default=True,
        description="Put a 0 KB file in place of each frame while it renders, so you can see which frames are "
                    "taken. Frames are handed out by the host either way, so none is ever rendered twice")
    smart_finish: BoolProperty(
        name="Smart Finish", default=True,
        description="Give the last frames to the cards that finish them soonest. A slow card doesn't start a "
                    "frame that the faster cards would have finished before it, and a fast card with nothing "
                    "left to do also renders a frame a slow card still needs long for (the first one done counts)")
    copy_missing: BoolProperty(
        name="Copy Missing Files", default=True,
        description="When another computer can't find a file this scene uses (a texture, an image sequence, a "
                    "linked .blend and the files it uses, a font, a volume ...), send it there from this "
                    "computer before that computer renders. The copy is kept in its temporary folder. "
                    "Files over 4 GB are never sent, and more than 2 GB for one computer is asked about first")
    overwrite: BoolProperty(
        name="Overwrite", default=False,
        description="Render frames that already exist again. Off: they're skipped, so Render carries on where "
                    "it stopped")
    show_advanced: BoolProperty(name="Advanced", default=False, description="Show the advanced settings")
    set_format: BoolProperty(
        name="Set Output Format", default=True,
        description="Render with the format below (it's also written into the scene's Output settings). "
                    "Off: use whatever format the scene has")
    out_format: EnumProperty(
        name="File Format", default="OPEN_EXR",
        items=[("OPEN_EXR", "OpenEXR (.exr)", "High dynamic range, best for compositing"),
               ("OPEN_EXR_MULTILAYER", "OpenEXR MultiLayer (.exr)", "Every view layer and render pass in one file"),
               ("PNG", "PNG (.png)", "Lossless 8 or 16 bit"),
               ("JPEG", "JPEG (.jpg)", "Small lossy files, no transparency")])
    out_color: EnumProperty(
        name="Color", default="RGB",
        items=[("BW", "BW", "Greyscale"), ("RGB", "RGB", "Color"), ("RGBA", "RGBA", "Color and transparency")])
    exr_depth: EnumProperty(name="Color Depth", default="32",
                            items=[("16", "Float (Half)", "16-bit float"), ("32", "Float (Full)", "32-bit float")])
    exr_codec: EnumProperty(
        name="Codec", default="DWAA",
        items=[("NONE", "None", ""), ("ZIP", "ZIP (lossless)", ""), ("PIZ", "PIZ (lossless)", ""),
               ("DWAA", "DWAA (lossy)", ""), ("DWAB", "DWAB (lossy)", ""), ("ZIPS", "ZIPS (lossless)", ""),
               ("RLE", "RLE (lossless)", ""), ("PXR24", "Pxr24 (lossy)", ""), ("B44", "B44 (lossy)", ""),
               ("B44A", "B44A (lossy)", "")])
    exr_quality: IntProperty(name="Quality", default=90, min=0, max=100, subtype='PERCENTAGE',
                             description="Quality of the lossy DWAA / DWAB codecs")
    png_depth: EnumProperty(name="Color Depth", default="8", items=[("8", "8", "8 bits per channel"),
                                                                   ("16", "16", "16 bits per channel")])
    png_compression: IntProperty(name="Compression", default=15, min=0, max=100, subtype='PERCENTAGE',
                                 description="Higher is smaller but slower to save. PNG is lossless either way")
    jpeg_quality: IntProperty(name="Quality", default=90, min=0, max=100, subtype='PERCENTAGE')

    def draw(self, context):
        draw_ui(self.layout, context, prefs_only=True)


# ======================================================================================
# Before a render: gather everything, and look for trouble
# ======================================================================================
def _format(prefs):
    if not prefs.set_format:
        return None
    return {"format": prefs.out_format, "color": prefs.out_color, "exr_depth": prefs.exr_depth,
            "exr_codec": prefs.exr_codec, "exr_quality": prefs.exr_quality, "png_depth": prefs.png_depth,
            "png_compression": prefs.png_compression, "jpeg_quality": prefs.jpeg_quality}


def _format_text(prefs):
    if prefs.out_format in ("OPEN_EXR", "OPEN_EXR_MULTILAYER"):
        s = "OpenEXR MultiLayer" if prefs.out_format.endswith("LAYER") else "OpenEXR, %s" % prefs.out_color
        s += ", %s float, %s" % ("full" if prefs.exr_depth == "32" else "half", prefs.exr_codec)
        return s + (" %d%%" % prefs.exr_quality if prefs.exr_codec in ("DWAA", "DWAB") else "")
    if prefs.out_format == "PNG":
        return "PNG, %s, %s bit" % (prefs.out_color, prefs.png_depth)
    return "JPEG, %s, quality %d%%" % ("RGB" if prefs.out_color == "RGBA" else prefs.out_color, prefs.jpeg_quality)


def _addr_or_none():
    return _A.get("addr")


def _job_frames(sc):
    rd = sc.render
    frames = list(range(sc.frame_start, sc.frame_end + 1, max(1, sc.frame_step)))
    view = None
    try:                                            # stereo saved as one file per eye: look for the first eye's file
        if rd.use_multiview and rd.image_settings.views_format == 'INDIVIDUAL' and \
                rd.image_settings.file_format != 'OPEN_EXR_MULTILAYER':
            view = next((v.name for v in rd.views if v.use), None)
    except Exception:
        view = None
    paths = []
    for f in frames:
        p = rd.frame_path(frame=f, view=view) if view else rd.frame_path(frame=f)
        paths.append(os.path.normpath(bpy.path.abspath(p)))
    return frames, paths


def _external_files(sets=None):
    """(files outside the .blend that this computer can open, names of the ones it can't). Everything Blender
    knows the scene uses, each frame of a sequence and each tile of a tiled image on its own.
    sets, if given, is filled with {path as the scene has it: [the files that belong to it]}."""
    sc = bpy.context.scene
    seqr = bool(sc and sc.render.use_sequencer and _sequencer_strips(sc))
    rows = scene_paths(bpy, expand=True, sequencer=seqr)
    if rows is None:
        return _external_files_old()
    have, missing, seen = [], set(), set()
    groups = {}
    for kind, name, lib, p, one_of in rows:
        if one_of:
            groups.setdefault((kind, name, lib), []).append(p)
        k = file_key(p)
        if k in seen or os.path.isdir(p):
            continue
        seen.add(k)
        if any(t in p for t in TOKENS):               # a tiled image that couldn't be listed tile by tile
            if not path_exists(p):
                missing.add(os.path.basename(p))
        elif os.path.isfile(p):
            have.append(p)
        else:
            missing.add(os.path.basename(p) or p)
    if sets is not None and groups:
        for kind, name, lib, p, _one in scene_paths(bpy, sequencer=seqr) or []:
            if (kind, name, lib) in groups:
                sets[p] = groups[(kind, name, lib)]
    return sorted(have), sorted(missing)


def _external_files_old():
    """The same for a Blender too old to list them itself: textures, movies, libraries, caches, volumes."""
    have, missing = set(), set()

    def add(fp, lib):
        try:
            p = os.path.normpath(bpy.path.abspath(fp, library=lib))
        except Exception:
            return
        (have if os.path.exists(p) else missing).add(p)
    for img in bpy.data.images:
        if img.packed_file or img.source not in ("FILE", "MOVIE") or not img.filepath or img.users == 0:
            continue
        add(img.filepath, img.library)
    for coll in ("libraries", "cache_files", "volumes"):
        for idb in getattr(bpy.data, coll, []):
            if idb.filepath and not getattr(idb, "packed_file", None):
                add(idb.filepath, getattr(idb, "library", None))
    return sorted(have), sorted(set(os.path.basename(m) for m in missing))


def _sendable(files):
    """Of the files this scene uses, the ones that may be sent to a computer that can't find them:
    {path: size}. Nothing over 4 GB."""
    out = {}
    for f in files:
        try:
            n = os.path.getsize(f)
        except OSError:
            continue
        if os.path.isfile(f) and n <= FILE_MAX:
            out[f] = n
    return out


COPY_ASK = 2 << 30             # sending more than this to one computer is asked about first


def _size_text(n):
    return "%.1f GB" % (n / float(1 << 30)) if n >= 1 << 30 else "%.0f MB" % max(1.0, n / float(1 << 20))


def _needs():
    """Add-ons this scene can't be rendered correctly without. Today: FLIP Fluids, when the scene uses it."""
    out = []
    try:
        for ob in bpy.data.objects:
            ff = getattr(ob, "flip_fluid", None)
            if ff is None or getattr(ff, "object_type", "TYPE_NONE") == "TYPE_NONE":
                continue
            mod = type(ff).__module__ or ""
            keys = [k for k in bpy.context.preferences.addons.keys() if mod == k or mod.startswith(k + ".")]
            out.append((keys[0] if keys else "flip_fluids_addon").split(".")[-1])
            break
    except Exception:
        pass
    return out


def _datafiles():
    try:
        return bpy.utils.system_resource('DATAFILES')
    except Exception:
        return None


def _basic_checks(context, prefs):
    """Things that make rendering impossible. Returns a message or None."""
    if prefs.mode == "JOIN":
        return "This computer joins another one: start the render there"
    if _A.get("running_job"):
        return "Already rendering"
    refresh_cards(prefs)
    sc, rd = context.scene, context.scene.render
    if not bpy.data.filepath:
        return "Save the .blend file first"
    if sc.render.engine != 'CYCLES':
        return "MultiGPU renders with Cycles only"
    if not rd.filepath:
        return "Set an output path"
    apply_format(rd.image_settings, _format(prefs))
    if rd.is_movie_format:
        return "A movie file can't be split between cards. Render to EXR or PNG frames"
    if sc.camera is None:
        return "The scene has no camera"
    if not any(vl.use for vl in sc.view_layers):
        return "No view layer is enabled for rendering"
    if sc.frame_end < sc.frame_start:
        return "The frame range is empty"
    local = sum(1 for g in prefs.gpus if g.use)
    st = _status() or {}
    remote = sum(len([c for c in comp["cards"] if c.get("use")]) for comp in st.get("computers", [])
                 if comp["online"] and comp["id"] != _computer_id())
    if local + remote == 0:
        return "Tick at least one card"
    return None


def _names(names, total=None, per=3, rows=3):
    """Lines listing file names, a few per line."""
    out = [", ".join(names[i:i + per]) for i in range(0, min(len(names), per * rows), per)]
    more = (total or len(names)) - min(len(names), per * rows)
    if more > 0 and out:
        out[-1] += "  ... and %d more" % more
    return [("BLANK1", "      " + line[:110]) for line in out]


def _precheck(context, prefs):
    """Look for anything that could spoil the render, on this computer and on every one that joined.
    Returns [(icon, text)]; empty when all is well."""
    sc, rd = context.scene, context.scene.render
    lines = []
    files, host_missing = _external_files()
    if host_missing:
        n = len(host_missing)
        lines.append(('ERROR', "This computer can't find %d file%s the scene uses:" % (n, "" if n == 1 else "s")))
        lines += _names(host_missing)
    frames, paths = _job_frames(sc)
    outdir = os.path.dirname(paths[0])
    if prefs.overwrite:
        n = 0
        for d in set(os.path.dirname(p) for p in paths):
            try:
                have = set(e.name for e in os.scandir(d) if e.is_file() and e.stat().st_size > 0)
            except OSError:
                have = set()
            n += sum(1 for p in paths if os.path.dirname(p) == d and os.path.basename(p) in have)
        if n:
            lines.append(('ERROR', "Overwrite is on: %d frame%s already rendered will be rendered again"
                          % (n, " that is" if n == 1 else "s that are")))
    if rd.use_sequencer and _sequencer_strips(sc):
        lines.append(('ERROR', "Sequencer is on and has %d strip%s: the frames will show those strips, not the "
                      "3D scene (Advanced > Post Processing)" % (_sequencer_strips(sc), "" if _sequencer_strips(sc) == 1 else "s")))
    try:
        if rd.film_transparent and rd.image_settings.color_mode != 'RGBA' and \
                rd.image_settings.file_format != 'OPEN_EXR_MULTILAYER':
            lines.append(('ERROR', "Transparent film is on, but the format saves %s: the transparency is lost"
                          % rd.image_settings.color_mode))
    except Exception:
        pass
    st = _status() or {}
    me = _computer_id()
    others = [c for c in st.get("computers", []) if c["online"] and not c.get("host") and c["id"] != me
              and any(x.get("use") for x in c["cards"])]
    addr = _A.get("addr")
    if prefs.mode != "HOST" or not others or not addr:
        return lines
    pid = uuid.uuid4().hex[:10]
    probe = ".multigpu-check-%s" % pid
    results = {}
    try:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, probe), "w") as f:
            f.write("x")
        rpc(addr, {"op": "precheck", "pre": {
            "id": pid, "host": me, "blend": {"path": bpy.data.filepath, "size": os.path.getsize(bpy.data.filepath)},
            "files": files[:FILES_MAX], "outdir": outdir, "probe": probe, "needs": _needs(),
            "pattern": _frame_pattern(frames, paths), "ranges": to_ranges(frames), "overwrite": bool(prefs.overwrite),
            "datafiles": _datafiles(), "copy": _sendable(files[:FILES_MAX]) if prefs.copy_missing else None}},
            timeout=4)
        t0 = time.time()
        while time.time() - t0 < _TEST.get("pre_wait", 14):     # the other computers answer within a few seconds
            time.sleep(0.3)
            now = (_A.get("reply") or {}).get("status") or {}
            for c in now.get("computers", []):
                if (c.get("pre") or {}).get("id") == pid:
                    results[c["id"]] = c["pre"]
            if all(c["id"] in results for c in others):
                break
    except OSError as e:
        lines.append(('ERROR', "Couldn't check the other computers: %s" % (getattr(e, "strerror", None) or e)))
    finally:
        try:
            os.remove(os.path.join(outdir, probe))
        except OSError:
            pass
    mine = bpy.app.version_string
    have = {}
    _A["pre_have"] = (time.time(), have)       # frames other computers already hold in their own copy of the folder
    for c in others:
        name, r = c["name"], results.get(c["id"])
        if c.get("blender") and c["blender"] != mine:
            lines.append(('ERROR', "%s has Blender %s, this computer %s: its frames may look different"
                          % (name, c["blender"], mine)))
        if r is None:
            lines.append(('INFO', "%s didn't answer the check in time. It may still join in" % name))
        elif not r.get("ok"):
            lines.append(('CANCEL', "%s can't take part: %s" % (name, (r.get("error") or "unknown reason")[:150])))
        else:
            if r.get("out_error"):
                lines.append(('CANCEL', "%s can't take part: %s" % (name, r["out_error"][:150])))
            if r.get("away"):
                nh = len(from_ranges(r.get("have")))
                have[c["id"]] = r.get("have") or []
                lines.append(('IMPORT', "%s renders into its own copy of the folder%s"
                              % (name, " and already has %d frame%s there" % (nh, "" if nh == 1 else "s") if nh else "")))
            if r.get("needs"):
                lines.append(('CANCEL', "%s can't take part: the %s add-on isn't enabled there"
                              % (name, ", ".join(r["needs"]))))
            if r.get("copy"):
                n, size, names = r["copy"]
                big = size > COPY_ASK
                lines.append(('INFO' if big else 'IMPORT', "%s gets %d file%s from this computer first (%s%s):"
                              % (name, n, "" if n == 1 else "s", _size_text(size),
                                 ", this will take a while" if big else "")))
                lines += _names(names, total=n)
            if r.get("nmissing"):
                n = r["nmissing"]
                lines.append(('ERROR', "%s can't find %d file%s that this computer has%s:"
                              % (name, n, "" if n == 1 else "s", " (%s)" % r["copy_error"] if r.get("copy_error") else "")))
                lines += _names(r.get("missing") or [], total=n)
    return lines


# ======================================================================================
# Operators
# ======================================================================================
class MULTIGPU_OT_detect(bpy.types.Operator):
    bl_idname = "multigpu.detect"
    bl_label = "Find GPUs"
    bl_description = "Look for graphics cards again"

    def execute(self, context):
        prefs = _prefs(context)
        err = detect_gpus(prefs)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        n = len(prefs.gpus)
        self.report({'INFO'}, "Found %d device%s" % (n, "" if n == 1 else "s"))
        return {'FINISHED'}


class MULTIGPU_OT_render(bpy.types.Operator):
    bl_idname = "multigpu.render"
    bl_label = "Render Animation"
    bl_description = ("Render the animation on every ticked card of this computer and of every computer that "
                      "joined. Anything that could spoil the render is shown first, for you to confirm")

    def invoke(self, context, event):
        prefs = _prefs(context)
        err = _basic_checks(context, prefs)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        try:
            context.window.cursor_set('WAIT')
        except Exception:
            pass
        try:
            lines = _precheck(context, prefs)
        except Exception as e:
            traceback.print_exc()
            lines = [('ERROR', "The check before rendering failed: %s" % e)]
        finally:
            try:
                context.window.cursor_set('DEFAULT')
            except Exception:
                pass
        _A["warn_lines"] = lines
        if not any(icon not in ('IMPORT', 'BLANK1') for icon, _t in lines):
            return self.execute(context)               # nothing wrong (files to copy over aren't a problem)
        wm = context.window_manager
        try:
            return wm.invoke_props_dialog(self, width=660, title="MultiGPU: check before rendering",
                                          confirm_text="Render Anyway")
        except TypeError:
            return wm.invoke_props_dialog(self, width=660)

    def draw(self, context):
        col = self.layout.column(align=True)
        for icon, text in _A.get("warn_lines") or []:
            col.label(text=text, icon=icon)
        col.separator()
        col.label(text="Render anyway?")

    def execute(self, context):
        prefs = _prefs(context)
        err = _basic_checks(context, prefs)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        sc, rd = context.scene, context.scene.render
        if not _A.get("owner"):                        # this Blender now looks after this computer's cards
            _owner(take=True)
            _A["owner"] = True
            _load_procs()
        addr = _ensure_coord(prefs, start=True, wait=8.0)
        if not addr:
            self.report({'ERROR'}, _A.get("coord_err") or "The host process didn't start. See %s" % _COORD_LOG)
            return {'CANCELLED'}
        _A["addr"] = addr
        for p in _A["procs"].values():                 # leftovers from an earlier render
            if not p.get("dead") and not p.get("killed"):
                _kill(p["pid"])
                p["killed"] = time.time()
        if not sc.get(TOKEN) or _A.get("token_unsaved") or (prefs.save_first and bpy.data.is_dirty):
            err = None
            for i in range(3):
                err = _save_now(uuid.uuid4().hex, sc.name)
                if err is None:
                    break
                time.sleep(1.0)
            if err:
                self.report({'ERROR'}, "Couldn't save the file: %s" % err)
                return {'CANCELLED'}
        frames, paths = _job_frames(sc)
        sets = {}
        files, host_missing = _external_files(sets)
        spec = {"host": _computer_id(), "host_name": _computer_name(), "scene": sc.name, "output": rd.filepath,
                "frames": frames, "paths": paths,
                "blend": {"path": bpy.data.filepath, "name": bpy.path.basename(bpy.data.filepath),
                          "size": os.path.getsize(bpy.data.filepath), "token": sc.get(TOKEN)},
                "opts": {"placeholders": bool(prefs.placeholders), "overwrite": bool(prefs.overwrite),
                         "persistent": bool(prefs.persistent_data), "autoexec": not bpy.app.autoexec_fail,
                         "format": _format(prefs), "ext": bool(rd.use_file_extension),
                         "compositing": bool(rd.use_compositing), "sequencer": bool(rd.use_sequencer),
                         "host_missing": [m.lower() for m in host_missing], "needs": _needs(),
                         "datafiles": _datafiles(), "copy": bool(prefs.copy_missing),
                         "smart_finish": bool(prefs.smart_finish)}}
        if prefs.copy_missing:
            spec["files"] = sorted(_sendable(files))
            spec["sets"] = sets
        spec["pattern"] = _frame_pattern(frames, paths)
        ph = _A.get("pre_have")
        if ph and time.time() - ph[0] < 300 and not prefs.overwrite:
            spec["have"] = ph[1]
        _A["pre_have"] = None
        try:
            r = rpc(addr, {"op": "start", "job": spec}, timeout=120)
        except OSError as e:
            self.report({'ERROR'}, "Couldn't reach the host process: %s" % e)
            return {'CANCELLED'}
        if r.get("error"):
            self.report({'ERROR'}, r["error"])
            return {'CANCELLED'}
        _A["layout"] = {}
        _A["msg"] = None
        _A["dismissed"] = None
        _A["unsynced_send"] = False
        n_ex = r.get("existing", 0)
        if n_ex >= len(frames):
            self.report({'INFO'}, "All %d frames are already rendered. Turn on Overwrite to render them again"
                        % len(frames))
            return {'FINISHED'}
        _A["running_job"] = r.get("job")
        _A["host_job"] = {"id": r.get("job"), "state": "running", "host": spec["host"], "origin": spec["host"],
                          "blend": spec["blend"], "scene": sc.name}
        try:                                           # old logs of earlier renders
            base = os.path.join(_ROOT, "cards")
            for d in os.listdir(base):
                p = os.path.join(base, d)
                if d != r.get("job") and time.time() - os.path.getmtime(p) > 3 * 86400:
                    shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass
        self.report({'INFO'}, "Rendering %d frames%s" % (len(frames) - n_ex, " (%d already done)" % n_ex if n_ex else ""))
        return {'FINISHED'}


class MULTIGPU_OT_sync(bpy.types.Operator):
    bl_idname = "multigpu.sync"
    bl_label = "Sync Changes"
    bl_description = ("Save the file and send your changes to every card. Each card finishes the frame it's on, "
                      "then reloads the file and carries on. Frames already done keep the old version")

    @classmethod
    def poll(cls, context):
        p = _prefs(context)
        return bool(p and (_is_host_mode(p) or _origin_job()) and _A.get("running_job") and not _A.get("sync")
                    and bpy.data.filepath)

    def execute(self, context):
        addr = _addr_or_none()
        job = ((_A.get("reply") or {}).get("job")) or {}
        blend = _blend_here(job) if job else None
        if (job.get("host") != _computer_id() and not _origin_here(job)) or not _same_file(bpy.data.filepath, blend):
            self.report({'ERROR'}, "This Blender doesn't have the file that's being rendered open (%s)"
                        % ((job.get("blend") or {}).get("name") or "?"))
            return {'CANCELLED'}
        if bpy.data.scenes.get(job.get("scene") or "") is None:
            self.report({'ERROR'}, "The scene being rendered (%s) isn't in this file anymore" % job.get("scene"))
            return {'CANCELLED'}
        try:
            r = rpc(addr, {"op": "sync_begin", "computer": _computer_id()})
        except OSError as e:
            self.report({'ERROR'}, "Couldn't reach the host process: %s" % e)
            return {'CANCELLED'}
        if r.get("error"):
            self.report({'ERROR'}, r["error"])
            return {'CANCELLED'}
        t = time.time()
        for p in _A["procs"].values():                  # cards still reading the old file hold it open
            if not p.get("dead") and not p.get("killed") and _card_state(p) in (None, "starting"):
                _kill(p["pid"])
                p["killed"] = t
                p["relaunch"] = True
        _A["sync"] = {"t0": t, "next": t, "token": uuid.uuid4().hex, "scene": job["scene"], "blend": blend,
                      "addr": addr, "job": job["id"]}
        _A["unsynced_send"] = False
        _A["msg"] = "Waiting for the cards to let go of the file..."
        _sync_tick()
        if _A.get("sync"):
            self.report({'INFO'}, _A.get("msg") or "Waiting for the cards to let go of the file...")
        else:
            self.report({'INFO'}, _A.get("msg") or "Changes saved")
        return {'FINISHED'}


class MULTIGPU_OT_card_stop(bpy.types.Operator):
    bl_idname = "multigpu.card_stop"
    bl_label = "Stop This Card"
    bl_description = "Stop just this card, e.g. to use the GPU for something else. Its frame goes to another card"
    card: StringProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        try:
            rpc(_addr_or_none(), {"op": "card_stop", "card": self.card})
        except OSError as e:
            self.report({'ERROR'}, "Couldn't reach the host: %s" % e)
            return {'CANCELLED'}
        p = _A["procs"].get(self.card)
        if p and not p.get("dead"):
            _kill(p["pid"])
            p["killed"] = time.time()
        return {'FINISHED'}


class MULTIGPU_OT_card_use(bpy.types.Operator):
    bl_idname = "multigpu.card_use"
    bl_label = "Use This Card"
    bl_description = "Tick or untick this card on the other computer. Its own panel follows"
    card: StringProperty()
    use: BoolProperty()

    def execute(self, context):
        try:
            r = rpc(_addr_or_none(), {"op": "card_use", "card": self.card, "use": self.use})
        except OSError as e:
            self.report({'ERROR'}, "Couldn't reach the host: %s" % e)
            return {'CANCELLED'}
        if r.get("error"):
            self.report({'ERROR'}, r["error"])
            return {'CANCELLED'}
        for comp in ((_A.get("reply") or {}).get("status") or {}).get("computers", []):   # show it right away
            for c in comp["cards"]:
                if c["id"] == self.card:
                    c["use"] = self.use
        _redraw()
        return {'FINISHED'}


class MULTIGPU_OT_card_resume(bpy.types.Operator):
    bl_idname = "multigpu.card_resume"
    bl_label = "Resume This Card"
    bl_description = "Start this card again. It picks up frames that haven't been rendered yet"
    card: StringProperty()

    def execute(self, context):
        try:
            rpc(_addr_or_none(), {"op": "card_resume", "card": self.card})
        except OSError as e:
            self.report({'ERROR'}, "Couldn't reach the host: %s" % e)
            return {'CANCELLED'}
        return {'FINISHED'}


class MULTIGPU_OT_stop(bpy.types.Operator):
    bl_idname = "multigpu.stop"
    bl_label = "Stop All Cards"
    bl_description = ("Stop every card on every computer. Saved frames are kept and unfinished ones removed, so "
                      "Render carries on where it stopped")

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        try:
            rpc(_addr_or_none(), {"op": "stop", "by": _computer_name()})
        except OSError as e:
            self.report({'ERROR'}, "Couldn't reach the host: %s" % e)
        t = time.time()
        for p in _A["procs"].values():
            if not p.get("dead"):
                _kill(p["pid"])
                p["killed"] = t
        self.report({'INFO'}, "Stopped")
        return {'FINISHED'}


class MULTIGPU_OT_join(bpy.types.Operator):
    bl_idname = "multigpu.join"
    bl_label = "Join"
    bl_description = ("Connect to the host and help with its renders, also when one is already running. "
                      "The host is remembered in the list below")
    leave: BoolProperty(default=False)
    index: IntProperty(default=-1)

    def execute(self, context):
        prefs = _prefs(context)
        if self.leave:
            prefs.joined = False
            _mark_dirty()
            t = time.time()
            for p in _A["procs"].values():
                if not p.get("dead"):
                    _kill(p["pid"])
                    p["killed"] = t
            return {'FINISHED'}
        if 0 <= self.index < len(prefs.hosts):
            prefs.host_address = prefs.hosts[self.index].address
        addr = _join_addr(prefs)
        if not addr:
            self.report({'ERROR'}, "Type the host's address first")
            return {'CANCELLED'}
        try:
            r = rpc(addr, {"op": "ping"}, timeout=6.0)
        except (OSError, ValueError) as e:
            self.report({'ERROR'}, _net_error(addr, e))
            return {'CANCELLED'}
        if r.get("protocol") != PROTOCOL or r.get("version") != VERSION:
            self.report({'ERROR'}, "The host runs MultiGPU %s, this computer %s. Install the same version on both"
                        % (r.get("version"), VERSION))
            return {'CANCELLED'}
        typed = prefs.host_address.strip()
        saved = next((h for h in prefs.hosts if h.address.strip().lower() == typed.lower()), None)
        if saved is None:
            saved = prefs.hosts.add()
            saved.address = typed
            saved.name = r.get("name") or typed
        prefs.joined = True
        refresh_cards(prefs)
        _mark_dirty()
        self.report({'INFO'}, "Joined %s%s" % (saved.name, ". It's rendering: this computer joins in" if r.get("running") else ""))
        return {'FINISHED'}


class MULTIGPU_OT_host_remove(bpy.types.Operator):
    bl_idname = "multigpu.host_remove"
    bl_label = "Forget This Host"
    bl_description = "Remove this host from the saved list"
    index: IntProperty()

    def execute(self, context):
        prefs = _prefs(context)
        if 0 <= self.index < len(prefs.hosts):
            prefs.hosts.remove(self.index)
            _mark_dirty()
        return {'FINISHED'}


class MULTIGPU_OT_folder(bpy.types.Operator):
    bl_idname = "multigpu.open_folder"
    bl_label = "Open Output Folder"
    bl_description = "Open the folder the frames are saved to"

    def execute(self, context):
        r = _A.get("reply") or {}
        job = r.get("job")
        if job:
            maps = [list(m) for m in r.get("maps") or []]
            res = _A["resolve"].get(job["id"]) or {}
            if res.get("hp") and res.get("lp"):
                maps.insert(0, [res["hp"], res["lp"]])
            d = map_path(job["outdir"], maps)
        else:
            sc = context.scene
            d = os.path.dirname(bpy.path.abspath(sc.render.frame_path(frame=sc.frame_start)))
        if not os.path.isdir(d):
            self.report({'WARNING'}, "The output folder doesn't exist yet")
            return {'CANCELLED'}
        bpy.ops.wm.path_open(filepath=d)
        return {'FINISHED'}


def _short_dir(sc, keep=3):
    """The folder frames are saved in, by its last few folder names."""
    try:
        d = os.path.dirname(bpy.path.abspath(sc.render.frame_path(frame=sc.frame_start)))
    except Exception:
        return "?"
    _, parts = split_any(d)
    sep = "\\" if os.name == "nt" else "/"
    return (("..." + sep) if len(parts) > keep else "") + sep.join(parts[-keep:]) + sep


class MULTIGPU_OT_relative(bpy.types.Operator):
    bl_idname = "multigpu.relative"
    bl_label = "Follow the .blend"
    bl_description = ("Nothing moves: frames still go to the same folder. The output is just written down as "
                      "'this many folders from the .blend' (it then starts with //) instead of the full drive path, "
                      "so it stays right if the project folder is moved or opened through another drive letter")

    def execute(self, context):
        rd = context.scene.render
        try:
            rel = bpy.path.relpath(rd.filepath)
        except ValueError:
            self.report({'ERROR'}, "The output is on a different drive than the .blend, so it can't follow it")
            return {'CANCELLED'}
        keep = rd.filepath.endswith(("/", "\\"))
        rd.filepath = rel + ("/" if keep and not rel.endswith(("/", "\\")) else "")
        self.report({'INFO'}, "Same folder, now found from the .blend's folder: %s" % rd.filepath)
        return {'FINISHED'}


class MULTIGPU_OT_firewall(bpy.types.Operator):
    bl_idname = "multigpu.firewall"
    bl_label = "Copy Firewall Fix"
    bl_description = ("Copy a PowerShell command that removes the firewall rules blocking Blender and allows the "
                      "MultiGPU port. Paste it into PowerShell opened with Run as administrator")

    def execute(self, context):
        context.window_manager.clipboard = _fw_fix(_prefs(context).port)
        _A["fw_t"] = time.time() - 570                  # look again in 30 seconds
        self.report({'INFO'}, "Copied. Paste it into PowerShell opened with 'Run as administrator'")
        return {'FINISHED'}


class MULTIGPU_OT_clear(bpy.types.Operator):
    bl_idname = "multigpu.clear"
    bl_label = "Dismiss"
    bl_description = "Hide the result of the last render"

    def execute(self, context):
        st = _status() or {}
        if st.get("job"):
            _A["dismissed"] = st["job"]["id"]
        _A["msg"] = None
        return {'FINISHED'}


# ======================================================================================
# UI
# ======================================================================================
_PHASE = {"loading": "loading scene...", "ready": "getting ready...", "idle": "waiting (no frames left)",
          "reloading": "reloading your changes...", "restarting": "closed unexpectedly, restarting...",
          "stopped": "stopped by you", "lost": "no contact", "done": "finished", "off": "-"}


def _card_text(c):
    ph = c["phase"]
    done = "  |  %d done" % c["done"] if c.get("done") else ""
    if ph == "rendering" and c.get("frame") is not None:
        s = "frame %d%s  |  %s" % (c["frame"], " (backup)" if c.get("backup") else "", _fmt(c.get("secs")))
        if c.get("slow"):
            return s + ", unusually long" + done
        if c.get("avg"):
            s += " (avg %s)" % _fmt(c["avg"])
        return s + done
    if ph == "error":
        return "error: %s" % (c.get("err") or "see log")
    if ph == "stopped" and c.get("stop_why"):
        return "stopped: its window was closed" if c["stop_why"] == "closed" else "stopped: render cancelled in its window"
    if ph == "idle" and c.get("held"):
        return "waiting (faster cards finish the rest)" + done
    return _PHASE.get(ph, ph) + done


def _green_icon():
    """Blender can't colour panel text (only red, for alerts), so the finish time gets a green marker."""
    if "green" not in _ENUM_CACHE:
        try:
            have = bpy.types.UILayout.bl_rna.functions['label'].parameters['icon'].enum_items
            _ENUM_CACHE["green"] = next((n for n in ('COLORSET_03_VEC', 'COLLECTION_COLOR_04', 'STRIP_COLOR_04')
                                         if n in have), 'TIME')
        except Exception:
            _ENUM_CACHE["green"] = 'TIME'
    return _ENUM_CACHE["green"]


def _alert(layout, text, icon='ERROR'):
    r = layout.row()
    r.alert = True
    r.label(text=text, icon=icon)


def draw_ui(layout, context, prefs_only=False):
    prefs = _prefs(context)
    if prefs is None:
        return
    st = _status()
    job = (st or {}).get("job")
    running = bool(job and job["state"] == "running")
    viewer = _A.get("owner") is False
    col = layout.column()

    box = col.box()
    head = box.row()
    head.label(text="Cards on this computer")
    head.operator("multigpu.detect", text="", icon='FILE_REFRESH', emboss=False)
    cards = box.column(align=True)
    cards.enabled = not running and not viewer
    if not prefs.gpus:
        cards.label(text="Press refresh to find your cards", icon='INFO')
    for g in prefs.gpus:
        r = cards.split(factor=0.6, align=True)
        r.prop(g, "use", text=g.name)
        s = r.row(align=True)
        s.enabled = g.use
        s.prop(g, "backend", text="")
    if viewer:
        box.label(text="Another Blender on this computer runs the cards. Press Render here to take over", icon='INFO')

    # ---- network ------------------------------------------------------------------
    box = col.box()
    r = box.row()
    r.enabled = not running
    r.prop(prefs, "mode", expand=True)
    if prefs.mode == "HOST":
        info = box.column(align=True)
        a = _addresses()
        port = "" if prefs.port == DEFAULT_PORT else ":%d" % prefs.port
        info.label(text="Others join with:  %s%s" % (a["lan"], port), icon='WORLD')
        if a.get("tailscale"):
            info.label(text="or over Tailscale:  %s%s" % (a["tailscale"], port), icon='BLANK1')
        if (_A.get("fw") or {}).get("block"):
            w = info.column(align=True)
            w.alert = True
            w.label(text="Windows Firewall blocks Blender: other computers can't connect", icon='ERROR')
            w.operator("multigpu.firewall", text="Copy Fix (paste into PowerShell as administrator)", icon='COPYDOWN')
        others = [c for c in (st or {}).get("computers", []) if not c.get("host")]
        if not running:
            if not others:
                info.label(text="No other computer has joined yet")
            for c in others:
                info.label(text="%s  |  %s" % (c["name"], "connected" if c["online"] else "offline"),
                           icon='DESKTOP' if c["online"] else 'ERROR')
                for x in c["cards"]:
                    rr = info.row(align=True)
                    rr.enabled = c["online"]
                    op = rr.operator("multigpu.card_use", text="%s (%s)" % (x["label"], _LABEL.get(x["backend"], x["backend"])),
                                     icon='CHECKBOX_HLT' if x.get("use") else 'CHECKBOX_DEHLT', emboss=False)
                    op.card, op.use = x["id"], not x.get("use")
        if prefs_only:
            box.prop(prefs, "port")
    elif prefs.mode == "JOIN":
        if prefs.joined:
            typed = prefs.host_address.strip().lower()
            name = next((h.name for h in prefs.hosts if h.address.strip().lower() == typed), prefs.host_address)
            rr = box.row()
            rr.label(text="Host: %s" % name, icon='LINKED')
            rr.operator("multigpu.join", text="Leave", icon='UNLINKED').leave = True
            box.prop(prefs, "auto_takeover")
            if _A["fo"].get("msg"):
                _alert(box, _A["fo"]["msg"])
            elif _A.get("err"):
                _alert(box, _A["err"])
            elif st is None:
                box.label(text="Connecting...", icon='TIME')
            elif not running:
                host = next((c["name"] for c in st.get("computers", []) if c.get("host")), "the host")
                box.label(text="Connected to %s. Renders start there" % host, icon='CHECKMARK')
        else:
            rr = box.row(align=True)
            rr.prop(prefs, "host_address", text="Host")
            op = rr.operator("multigpu.join", text="Join", icon='LINKED')
            op.leave, op.index = False, -1
            if len(prefs.hosts):
                lst = box.column(align=True)
                lst.label(text="Saved hosts (the name is yours to change)")
                for i, h in enumerate(prefs.hosts):
                    rr = lst.row(align=True)
                    rr.prop(h, "name", text="")
                    rr.label(text=h.address)
                    op = rr.operator("multigpu.join", text="Join", icon='LINKED')
                    op.leave, op.index = False, i
                    rr.operator("multigpu.host_remove", text="", icon='X').index = i
        folder = box.column(align=True)
        folder.enabled = not running
        folder.prop(prefs, "shared_roots")
        folder.label(text="Slow connection? Use a local synced project folder", icon='FILE_FOLDER')
        if prefs_only:
            box.prop(prefs, "port")
    if _is_host_mode(prefs) and _A["fo"].get("msg"):
        _alert(box, _A["fo"]["msg"])
    if _is_host_mode(prefs) and _A.get("coord_err"):
        _alert(box, _A["coord_err"])
    if _is_host_mode(prefs) and _A.get("err") and _A.get("addr"):
        _alert(box, _A["err"])

    # ---- options --------------------------------------------------------------------
    box = col.box()
    opts = box.column()
    opts.enabled = not running
    r = opts.row()
    r.prop(prefs, "show_windows")
    s = r.row()
    s.enabled = prefs.show_windows
    s.prop(prefs, "arrange_windows")
    if _is_host_mode(prefs):
        _draw_advanced(box, prefs, running, getattr(context, "scene", None))
    if prefs_only:
        if prefs.mode != "JOIN":
            box.prop(prefs, "shared_roots")
        return

    sc, rd = context.scene, context.scene.render
    if not running and _is_host_mode(prefs):
        notes = box.column(align=True)
        notes.scale_y = 0.8
        if prefs.set_format:
            notes.label(text="Saves %s" % _format_text(prefs), icon='FILE_IMAGE')
        elif rd.is_movie_format:
            notes.label(text="A movie can't be split between cards: render EXR/PNG frames", icon='ERROR')
        if sc.render.engine != 'CYCLES':
            notes.label(text="Only Cycles can be put on a chosen card", icon='ERROR')
        if not bpy.data.filepath:
            notes.label(text="Save the .blend file first", icon='ERROR')
        else:
            notes.prop(rd, "filepath", text="Frame output")
            notes.label(text="Frames save in  %s" % _short_dir(sc), icon='FILE_FOLDER')
            if prefs.mode == "HOST" and rd.filepath and not rd.filepath.startswith("//"):
                rr = notes.row()
                rr.label(text="Set as a full drive path. The other computers still find it", icon='BLANK1')
                rr.operator("multigpu.relative", text="Follow the .blend")

    if _A.get("msg"):
        m = col.row()
        m.scale_y = 0.8
        m.label(text=_A["msg"], icon='INFO')
    col.separator()

    # ---- progress -------------------------------------------------------------------
    if running:
        total = max(1, job["total"])
        done = job["done"]
        text = "%d / %d frames  (%d%%)" % (done, job["total"], 100 * done // total)
        if hasattr(col, "progress"):
            col.progress(factor=done / total, type='BAR', text=text)
        else:
            col.label(text=text)
        info = col.row()
        info.scale_y = 0.8
        if (_A.get("sync") or {}).get("payload") is not None:
            info.label(text="Changes saved; waiting for the host...", icon='FILE_REFRESH')
        elif job.get("sync_pending") or _A.get("sync"):
            info.label(text="Saving your changes...", icon='FILE_REFRESH')
        elif job.get("eta"):
            info.label(text="%s left  |  done around %s" % (_fmt(job["eta"]), _clock(time.time() + job["eta"])),
                       icon=_green_icon())
        else:
            info.label(text="Working out the finish time...", icon='TIME')
        if job.get("unsynced"):
            w = col.column(align=True)
            w.alert = True
            w.label(text="You saved the file. The cards keep rendering the earlier version", icon='ERROR')
            w.label(text="until you press Sync Changes (a card that restarts waits for that).", icon='BLANK1')
        if job.get("nas_down"):
            _alert(col, "This computer can't reach the output folder. Waiting for it to come back")
        if job.get("stalled") and not job.get("unsynced"):
            _alert(col, "No card is rendering right now. See the messages on the cards below")
        _draw_cards(col, st, prefs, running=True)
        if job.get("nfailed"):
            _alert(col, "%d frame%s failed: %s" % (job["nfailed"], "" if job["nfailed"] == 1 else "s",
                                                   ", ".join(str(f) for f in job["failed"][:8])))
        _draw_events(col, st)
        if job.get("arriving"):
            a = col.row()
            a.scale_y = 0.8
            a.label(text="%d frame%s rendered elsewhere, still on the way to this folder"
                    % (job["arriving"], "" if job["arriving"] == 1 else "s"), icon='IMPORT')
        if job.get("synced_at"):
            s = col.row()
            s.scale_y = 0.8
            s.label(text="Changes synced at %s" % _clock(job["synced_at"]), icon='FILE_REFRESH')
        r = col.row(align=True)
        r.scale_y = 1.6
        if _is_host_mode(prefs) or _origin_job():
            r.operator("multigpu.sync", text="SYNC CHANGES", icon='FILE_REFRESH')
        r.operator("multigpu.stop", text="Stop All", icon='CANCEL')
        r.operator("multigpu.open_folder", text="", icon='FILE_FOLDER')
        return

    if _is_host_mode(prefs):
        n = sum(1 for g in prefs.gpus if g.use)
        n += sum(len([x for x in c["cards"] if x.get("use")]) for c in (st or {}).get("computers", [])
                 if c["online"] and not c.get("host") and c["id"] != _computer_id())
        r = col.row(align=True)
        r.scale_y = 2.2
        r.enabled = n > 0 and bool(bpy.data.filepath)
        r.operator("multigpu.render", text="Render on %d card%s" % (n, "" if n == 1 else "s"), icon='RENDER_ANIMATION')
        r.operator("multigpu.open_folder", text="", icon='FILE_FOLDER')
    if job and _A.get("dismissed") != job["id"]:
        res = col.box().row()
        took = _fmt((job.get("ended") or time.time()) - job["started"])
        if job["state"] == "finished" and not job.get("nfailed"):
            res.label(text="Finished: %d frames in %s%s" % (job["total"], took, ", %d still on the way here"
                                                           % job["arriving"] if job.get("arriving") else ""),
                      icon='CHECKMARK')
        elif job["state"] == "finished":
            res.alert = True
            res.label(text="Done, but %d frame%s failed: %s" % (job["nfailed"], "" if job["nfailed"] == 1 else "s",
                                                               job.get("fail_err") or "see the card logs"), icon='ERROR')
        elif job["state"] == "moved":
            res.label(text="Another computer took over this render as host", icon='INFO')
        else:
            res.label(text="Stopped at %d / %d. Press Render to carry on" % (job["done"], job["total"]), icon='INFO')
        res.operator("multigpu.clear", text="", icon='X', emboss=False)


def _sequencer_strips(sc):
    try:
        ed = sc.sequence_editor
        if ed is None:
            return 0
        strips = getattr(ed, "strips_all", None)
        if strips is None:
            strips = getattr(ed, "sequences_all", ())
        return len(strips)
    except Exception:
        return 0


def _compositor_set(sc):
    try:
        if hasattr(sc, "compositing_node_group"):
            return sc.compositing_node_group is not None
        return bool(sc.use_nodes and sc.node_tree)
    except Exception:
        return False


def _draw_advanced(box, prefs, running, sc=None):
    head = box.row()
    head.prop(prefs, "show_advanced", text="Advanced", emboss=False,
              icon='TRIA_DOWN' if prefs.show_advanced else 'TRIA_RIGHT')
    if not prefs.show_advanced:
        return
    adv = box.column()
    adv.enabled = not running
    r = adv.row()
    r.prop(prefs, "overwrite")
    r.prop(prefs, "placeholders")
    r = adv.row()
    r.prop(prefs, "persistent_data")
    r.prop(prefs, "save_first")
    adv.prop(prefs, "copy_missing")
    adv.prop(prefs, "smart_finish")
    if sc is not None:                                # Blender's own Post Processing settings of this scene
        adv.separator()
        adv.label(text="Post Processing (all cards use it)")
        r = adv.row()
        r.prop(sc.render, "use_compositing", text="Compositing")
        r.prop(sc.render, "use_sequencer", text="Sequencer")
        notes = adv.column(align=True)
        notes.scale_y = 0.8
        if sc.render.use_compositing and not _compositor_set(sc):
            notes.label(text="No compositor nodes: no effect", icon='INFO')
        if sc.render.use_sequencer and _sequencer_strips(sc):
            notes.label(text="Sequencer strips replace the 3D scene", icon='ERROR')
    adv.separator()
    adv.prop(prefs, "set_format")
    f = adv.column()
    f.enabled = prefs.set_format
    f.use_property_split = True
    f.use_property_decorate = False
    f.prop(prefs, "out_format")
    exr = prefs.out_format in ("OPEN_EXR", "OPEN_EXR_MULTILAYER")
    if prefs.out_format != "OPEN_EXR_MULTILAYER":
        r = f.row(align=True)
        r.prop(prefs, "out_color", expand=True)
    if exr:
        r = f.row(align=True)
        r.prop(prefs, "exr_depth", expand=True)
        f.prop(prefs, "exr_codec")
        if prefs.exr_codec in ("DWAA", "DWAB"):
            f.prop(prefs, "exr_quality", slider=True)
    elif prefs.out_format == "PNG":
        r = f.row(align=True)
        r.prop(prefs, "png_depth", expand=True)
        f.prop(prefs, "png_compression", slider=True)
    else:
        if prefs.out_color == "RGBA":
            f.label(text="JPEG has no transparency: saved as RGB", icon='INFO')
        f.prop(prefs, "jpeg_quality", slider=True)


def _draw_cards(col, st, prefs, running):
    wb = col.box().column(align=True)
    host_blender = next((c.get("blender") for c in st["computers"] if c.get("host")), None)
    for comp in st["computers"]:
        cards = [c for c in comp["cards"] if c.get("use") or c.get("phase") in ("rendering", "loading", "ready")]
        if not cards and not comp.get("resolve_error"):
            continue
        head = wb.row()
        name = comp["name"] + ("  (host)" if comp.get("host") else "") + ("  (own copy of the folder)" if comp.get("away") else "")
        if not comp["online"]:
            name += "  -  offline"
        head.label(text=name, icon='DESKTOP')
        if comp.get("blocked"):
            _alert(wb, "Stopped: " + comp["blocked"])
        if comp.get("resolve_error"):
            _alert(wb, "Can't render: " + comp["resolve_error"])
        elif comp.get("resolving"):
            wb.label(text="Looking for the .blend on this computer...", icon='VIEWZOOM')
        if host_blender and comp.get("blender") and comp["blender"] != host_blender:
            wb.label(text="Blender %s here, %s on the host: renders may differ" % (comp["blender"], host_blender),
                     icon='ERROR')
        for c in cards:
            row = wb.row(align=True)
            r = row.split(factor=0.38)
            r.label(text="%s (%s)" % (c["label"], _LABEL.get(c["backend"], c["backend"])))
            t = r.row()
            t.alert = c["phase"] in ("error", "lost") or bool(c.get("slow"))
            t.label(text=_card_text(c))
            if c.get("stopped") or c["phase"] == "error":
                row.operator("multigpu.card_resume", text="", icon='PLAY', emboss=False).card = c["id"]
            elif c["phase"] not in ("done",):
                row.operator("multigpu.card_stop", text="", icon='PAUSE', emboss=False).card = c["id"]
            if c.get("warn"):
                w = wb.row()
                w.scale_y = 0.8
                w.label(text="    " + c["warn"], icon='ERROR')


def _draw_events(col, st):
    ev = (st or {}).get("events") or []
    if not ev:
        return
    b = col.column(align=True)
    b.scale_y = 0.75
    for t, text in ev[-3:]:
        b.label(text="%s  %s" % (_clock(t), text))


class MULTIGPU_PT_output(bpy.types.Panel):
    bl_label = "MultiGPU"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "output"

    def draw(self, context):
        draw_ui(self.layout, context)


class MULTIGPU_PT_sidebar(bpy.types.Panel):
    bl_label = "MultiGPU"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MultiGPU"

    def draw(self, context):
        draw_ui(self.layout, context)


# ======================================================================================
# Register
# ======================================================================================
@persistent
def _on_load(*args):
    if not bpy.app.timers.is_registered(_startup):
        bpy.app.timers.register(_startup, first_interval=1.0)


def _startup():
    p = _prefs()
    if p is not None and not any(g.key == "CPU" or g.key.startswith("CPU-") for g in p.gpus):
        detect_gpus(p)                               # first run (or the first with a CPU row)
    elif p is not None:
        refresh_cards(p)                             # a card was added, removed or moved since last time
    return None


def start_agent():
    """Begin talking to the host and watching this computer's cards (also used by tests)."""
    os.makedirs(_ROOT, exist_ok=True)
    _A["quit"].clear()
    if not (_A["thread"] and _A["thread"].is_alive()):
        _A["thread"] = threading.Thread(target=_net_loop, daemon=True)
        _A["thread"].start()
    if not (_A["fo_thread"] and _A["fo_thread"].is_alive()):
        _A["fo_thread"] = threading.Thread(target=_fo_loop, daemon=True)
        _A["fo_thread"].start()


classes = (MULTIGPU_Device, MULTIGPU_Host, MULTIGPU_Prefs, MULTIGPU_OT_detect, MULTIGPU_OT_render, MULTIGPU_OT_sync,
           MULTIGPU_OT_card_stop, MULTIGPU_OT_card_use, MULTIGPU_OT_card_resume, MULTIGPU_OT_stop, MULTIGPU_OT_join,
           MULTIGPU_OT_host_remove, MULTIGPU_OT_folder, MULTIGPU_OT_relative, MULTIGPU_OT_firewall, MULTIGPU_OT_clear,
           MULTIGPU_PT_output, MULTIGPU_PT_sidebar)
_HANDLERS = (("load_post", _on_load), ("save_pre", _on_save_pre), ("save_post", _on_save_post))


_LEGACY = "multigpu"                                # the module name of the old single-file install
_REG = {"on": False}


def _legacy_on():
    """True in the drag-in (extension) install while the old single-file add-on is still enabled."""
    try:
        return __name__ != _LEGACY and _LEGACY in bpy.context.preferences.addons
    except Exception:
        return False


def _prefs_dump(p):
    out = {}
    for pr in p.bl_rna.properties:
        k = pr.identifier
        if k in ("rna_type", "bl_idname") or pr.is_readonly and pr.type != 'COLLECTION':
            continue
        try:
            if pr.type == 'COLLECTION':
                out[k] = [dict((q.identifier, getattr(it, q.identifier)) for q in it.bl_rna.properties
                               if q.identifier != "rna_type") for it in getattr(p, k)]
            elif pr.type in ('BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM'):
                out[k] = getattr(p, k)
        except Exception:
            pass
    return out


def _prefs_load(p, data):
    for k, v in data.items():
        try:
            if isinstance(v, list):
                col = getattr(p, k)
                col.clear()
                for row in v:
                    it = col.add()
                    for a, b in row.items():
                        if a != "rna_type":
                            try:
                                setattr(it, a, b)
                            except Exception:
                                pass
            else:
                setattr(p, k, v)
        except Exception:
            pass


def _replace_legacy():
    """The old single-file MultiGPU is still enabled: keep its settings, switch it off and take its place."""
    import addon_utils
    if _REG["on"]:
        return None
    if _legacy_on():
        old = sys.modules.get(_LEGACY)
        try:
            st = old._status() if old else None
            if ((st or {}).get("job") or {}).get("state") == "running":
                return 10.0                          # never swap in the middle of a render
        except Exception:
            pass
        data = {}
        try:
            data = _prefs_dump(bpy.context.preferences.addons[_LEGACY].preferences)
        except Exception:
            pass
        try:
            addon_utils.disable(_LEGACY, default_set=True)
        except Exception:
            traceback.print_exc()
        if _legacy_on():
            return 10.0
        _register()
        p = _prefs()
        if p is not None and data:
            _prefs_load(p, data)
        _mark_dirty()
        print("MultiGPU: replaced the old multigpu.py install and kept its settings")
        return None
    _register()
    return None


def _register():
    if _REG["on"]:
        return
    for c in classes:
        bpy.utils.register_class(c)
    _REG["on"] = True
    if os.environ.get("MULTIGPU_WORKER"):          # this Blender is one of the cards: stay passive
        return
    for name, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        if fn not in lst:
            lst.append(fn)
    bpy.app.timers.register(_startup, first_interval=1.0)
    start_agent()
    bpy.app.timers.register(_agent_tick, first_interval=1.0, persistent=True)


def register():
    if _legacy_on():                               # two copies can't both be active
        if not (bpy.app.background or os.environ.get("MULTIGPU_WORKER")):
            bpy.app.timers.register(_replace_legacy, first_interval=0.5, persistent=True)
        return
    _register()


def unregister():
    if bpy.app.timers.is_registered(_replace_legacy):
        bpy.app.timers.unregister(_replace_legacy)
    if not _REG["on"]:
        return
    _REG["on"] = False
    _A["quit"].set()
    for f in (_agent_tick, _startup):
        if bpy.app.timers.is_registered(f):
            bpy.app.timers.unregister(f)
    for name, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        if fn in lst:
            lst.remove(fn)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
