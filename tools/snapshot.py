"""Keep a run's result for the project website: the photo and the 3D scene.

Runs are git-ignored and get rebuilt, so a result worth showing is copied into
docs/showcase/<slug>/ and listed in docs/showcase/index.json, which the page at
docs/index.html reads. Each entry records the commit that made it, so a picture
on the site can always be traced back to its code. Snapshotting the same slug
again replaces it.

Usage: uv run python tools/snapshot.py runs/<run> --title "Kitchen" [--slug kitchen]
"""

import argparse
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

from PIL import Image

from aomma.dashboard import scene_glb

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "docs" / "showcase"


def commit() -> str:
    """The short hash of HEAD, marked dirty when the working tree has uncommitted changes to tracked files."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return f"{head}-dirty" if dirty else head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--slug", help="folder name on the site; the run's image name by default")
    args = parser.parse_args()

    run = args.run.resolve()
    for path in (run / "input.jpg", run / "scene.usda"):
        if not path.exists():
            raise SystemExit(f"{path} is missing: snapshot a finished run")
    # The dashboard's export, so the site shows exactly what its 3D view does.
    glb = scene_glb(run)

    slug = args.slug or run.name.split("-", 2)[-1].lower()
    out = SHOWCASE / slug
    out.mkdir(parents=True, exist_ok=True)
    Image.open(run / "input.jpg").convert("RGB").save(out / "photo.jpg", quality=88)
    shutil.copyfile(glb, out / "scene.glb")

    index_path = SHOWCASE / "index.json"
    entries = json.loads(index_path.read_text()) if index_path.exists() else []
    entry = {"slug": slug, "title": args.title, "date": date.today().isoformat(),
             "commit": commit(), "run": run.name}
    entries = [e for e in entries if e["slug"] != slug] + [entry]
    index_path.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {out.relative_to(ROOT)} ({entry['commit']})")


if __name__ == "__main__":
    main()
