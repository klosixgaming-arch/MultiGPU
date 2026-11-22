"""Build the drag-in package:  python build.py /path/to/blender

Writes dist/multigpu-<version>.zip and dist/index.json (the list Blender reads for installs and
updates), and puts the new version's link into README.md.
"""
import json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "https://raw.githubusercontent.com/klosixgaming-arch/MultiGPU/main/dist/"


def main():
    blender = sys.argv[1:] or ["blender"]
    src = open(os.path.join(HERE, "multigpu.py"), encoding="utf-8").read()
    version = re.search(r'^VERSION = "([^"]+)"', src, re.M).group(1)
    manifest = open(os.path.join(HERE, "blender_manifest.toml"), encoding="utf-8").read()
    if 'version = "%s"' % version not in manifest:
        sys.exit("blender_manifest.toml doesn't say version %s" % version)
    vmin = re.search(r'^blender_version_min = "([^"]+)"', manifest, re.M).group(1)
    dist = os.path.join(HERE, "dist")
    shutil.rmtree(dist, ignore_errors=True)
    os.makedirs(dist)
    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, "multigpu")
        os.makedirs(pkg)
        shutil.copy(os.path.join(HERE, "multigpu.py"), os.path.join(pkg, "__init__.py"))
        shutil.copy(os.path.join(HERE, "blender_manifest.toml"), pkg)
        shutil.copy(os.path.join(HERE, "LICENSE"), pkg)
        ext = blender + ["--command", "extension"]
        subprocess.check_call(ext + ["validate", pkg])
        subprocess.check_call(ext + ["build", "--source-dir", pkg, "--output-dir", dist])
    subprocess.check_call(ext + ["server-generate", "--repo-dir", dist])
    index = os.path.join(dist, "index.json")
    data = json.load(open(index, encoding="utf-8"))
    data.pop("assetlib_auth_method", None)         # newer-Blender extra; the default means the same
    open(index, "w", encoding="utf-8", newline="\n").write(json.dumps(data, indent=2) + "\n")
    name = "multigpu-%s.zip" % version
    assert os.path.isfile(os.path.join(dist, name)), name
    link = "%s%s?repository=.%%2Findex.json&blender_version_min=%s" % (RAW, name, vmin)
    readme = os.path.join(HERE, "README.md")
    text = open(readme, encoding="utf-8").read()
    text, n = re.subn(re.escape(RAW) + r"multigpu-[^)\s\"]+", link, text)
    open(readme, "w", encoding="utf-8", newline="\n").write(text)
    print("built", name, "- README links updated:", n)


if __name__ == "__main__":
    main()
