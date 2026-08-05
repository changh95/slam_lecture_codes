#!/usr/bin/env python3
"""
Download Humanoid Everyday task recordings to ~/data/humanoid_everyday/
with tqdm progress bars, then unzip and keep the archive.

Humanoid Everyday is a Unitree G1 / H1 teleoperation dataset: egocentric
RealSense D435 colour + depth at 30 Hz, a Livox MID360 scan, the full joint
state, IMU and legged odometry, for 260 everyday tasks in 7 categories. The
whole thing is ~500 GB, so this script downloads task by task from the upstream
Dropbox links, which live in a public spreadsheet.

Reference: https://github.com/physical-superintelligence-lab/Humanoid-Everyday
Paper:     https://arxiv.org/abs/2510.08807

Default download is the two tasks the nvblox demo is verified on, both from the
loco_manipulation category (the one where the robot actually walks):

    walk_towards_chair_and_rotate_the_chair              ~483 MB
    walk_towards_outside_chair_and_pull_it_out           ~347 MB

Examples:
    python3 download_humanoid_everyday.py
    python3 download_humanoid_everyday.py --list
    python3 download_humanoid_everyday.py --category loco_manipulation
    python3 download_humanoid_everyday.py --tasks walk_towards_elevator_and_push_button
"""

import argparse
import csv
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm -q")
    from tqdm import tqdm


# The upstream README points at this sheet for the per-task download links.
SHEET_ID = "158Wzf8Xywky3aHJSCfp3OZxf4bkhzAJdcG94eHf8gVc"
SHEET_GID = "1307250382"
SHEET_CSV = (f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
             f"/export?format=csv&gid={SHEET_GID}")

DEST_DIR = Path.home() / "data" / "humanoid_everyday"

# Verified with nvblox/. Both are loco_manipulation, i.e. the robot walks up to
# the object before manipulating it.
DEFAULT_TASKS = [
    "walk_towards_chair_and_rotate_the_chair",
    "walk_towards_outside_chair_and_pull_it_out",
]


class TqdmDownloadHook:
    """Hook class for urllib.request.urlretrieve with tqdm progress."""

    def __init__(self, filename: str):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  > {self.filename}",
                ncols=100,
                miniters=1,
                mininterval=0.5,
                position=0,
                leave=True,
            )
        if total_size > 0:
            self.pbar.update(min(block_size, total_size - self.pbar.n))
        else:
            self.pbar.update(block_size)

    def close(self):
        if self.pbar:
            self.pbar.close()


def fetch_task_table() -> list:
    """[{'index','category','task','robot','description','url','missing'}, ...]"""
    with urllib.request.urlopen(SHEET_CSV) as fh:
        text = fh.read().decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    tasks = []
    for row in rows:
        # Data rows start with an integer index and carry a task name and a link.
        if len(row) < 8 or not row[0].strip().isdigit():
            continue
        tasks.append({
            "index": int(row[0]),
            "category": row[1].strip(),
            "task": row[2].strip(),
            "robot": row[3].strip(),
            "description": row[4].strip(),
            "url": row[5].strip(),
            "missing": row[7].strip(),
        })
    if not tasks:
        sys.exit("could not parse the task spreadsheet; has its layout changed?")
    return tasks


def download_file(url: str, dest: Path, name: str) -> Path:
    """Download with a tqdm progress bar. Skips if the file already exists."""
    filepath = dest / f"{name}.zip"
    if filepath.exists() and filepath.stat().st_size > 0:
        print(f"  = {filepath.name} already downloaded "
              f"({filepath.stat().st_size / 1e6:.0f} MB)")
        return filepath
    # Dropbox share links serve an HTML preview unless dl=1 is set.
    direct = url.replace("dl=0", "dl=1")
    if "dl=" not in direct:
        direct += ("&" if "?" in direct else "?") + "dl=1"
    hook = TqdmDownloadHook(filepath.name)
    try:
        urllib.request.urlretrieve(direct, filepath, reporthook=hook)
    finally:
        hook.close()
    return filepath


def extract(zip_path: Path, dest: Path, name: str):
    """Unzip into <dest>/<task>/, flattening a redundant top-level directory.

    Some archives contain episode_N/ at the root and others wrap them in another
    copy of the task name; normalise both to <task>/episode_N/.
    """
    target = dest / name
    if any(target.glob("episode_*/robot_data.jsonl")):
        print(f"  = {name}/ already extracted")
        return
    target.mkdir(parents=True, exist_ok=True)
    print(f"  > unzipping {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    nested = target / name
    if nested.is_dir():
        for child in nested.iterdir():
            child.rename(target / child.name)
        nested.rmdir()
    episodes = sorted(p.name for p in target.glob("episode_*"))
    print(f"    {len(episodes)} episodes: {', '.join(episodes)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="print every task with its category and exit")
    ap.add_argument("--tasks", nargs="+", help="task names to download")
    ap.add_argument("--category", help="download a whole category, "
                                      "e.g. loco_manipulation")
    ap.add_argument("--all", action="store_true",
                    help="download every task (~500 GB)")
    ap.add_argument("--dest", type=Path, default=DEST_DIR)
    ap.add_argument("--keep-zip", action="store_true", default=True,
                    help="keep the archives (default; they are the only copy "
                         "if a link rots)")
    args = ap.parse_args()

    print("Fetching the Humanoid Everyday task list...")
    table = fetch_task_table()
    print(f"  {len(table)} tasks listed\n")

    if args.list:
        by_category = {}
        for t in table:
            by_category.setdefault(t["category"], []).append(t)
        for category in sorted(by_category):
            rows = by_category[category]
            print(f"{category}  ({len(rows)} tasks)")
            for t in rows:
                flag = f"  [{t['missing']}]" if t["missing"] else ""
                print(f"    {t['robot']:3s} {t['task']}{flag}")
            print()
        return

    if args.all:
        wanted = [t["task"] for t in table]
    elif args.category:
        wanted = [t["task"] for t in table if t["category"] == args.category]
        if not wanted:
            sys.exit(f"no tasks in category '{args.category}'; "
                     f"try --list to see them")
    elif args.tasks:
        wanted = args.tasks
    else:
        wanted = DEFAULT_TASKS

    index = {t["task"]: t for t in table}
    missing = [name for name in wanted if name not in index]
    if missing:
        sys.exit(f"unknown task(s): {', '.join(missing)}\nuse --list to see them")

    zips = args.dest / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(wanted)} task(s) into {args.dest}\n")
    for name in wanted:
        t = index[name]
        note = f"  ({t['missing']})" if t["missing"] else ""
        print(f"[{t['category']} / {t['robot']}] {name}{note}")
        zip_path = download_file(t["url"], zips, name)
        extract(zip_path, args.dest, name)
        if not args.keep_zip:
            zip_path.unlink()
        print()

    print(f"Done. Data in {args.dest}")
    print("Each episode holds color/frame_NNNNNN.jpg, depth/frame_NNNNNN.npy.lzma")
    print("(raw uint16 millimetres, no npy header), lidar/*.pcd and "
          "robot_data.jsonl.")


if __name__ == "__main__":
    main()
