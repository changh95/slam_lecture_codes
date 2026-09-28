#!/usr/bin/env python3
"""
Download ICL-NUIM RGB-D sequences (TUM/Freiburg PNG format) to ~/data/icl_nuim/.

Each archive is extracted into its OWN folder, named after the archive:

  ~/data/icl_nuim/living_room_traj2_frei_png/{rgb/, depth/, associations.txt, livingRoom2.gt.freiburg}

Every archive holds the same rgb/ and depth/ names, so extracting them all into one
directory (what this script used to do) silently mixes the sequences together.

Reference: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html

Usage:
    python3 download_icl_nuim.py                             # living_room_traj2_frei_png (ConceptFusion default)
    python3 download_icl_nuim.py --list
    python3 download_icl_nuim.py traj0_frei_png living_room_traj0_frei_png
    python3 download_icl_nuim.py --all
    python3 download_icl_nuim.py --keep-archives ...          # keep the .tar.gz after extracting
"""

import argparse
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm


BASE_URL = "http://www.doc.ic.ac.uk/~ahanda"

# sequence name -> (description, archive size from the server's Content-Length, 2026-09)
SEQUENCES = {
    "living_room_traj0_frei_png":  ("living room, trajectory 0", "0.71 GB"),
    "living_room_traj1_frei_png":  ("living room, trajectory 1", "0.47 GB"),
    "living_room_traj2_frei_png":  ("living room, trajectory 2 (ConceptFusion default)", "0.43 GB"),
    "living_room_traj3_frei_png":  ("living room, trajectory 3", "0.56 GB"),
    "living_room_traj0n_frei_png": ("living room, trajectory 0, with noise", "0.94 GB"),
    "living_room_traj1n_frei_png": ("living room, trajectory 1, with noise", "0.61 GB"),
    "living_room_traj2n_frei_png": ("living room, trajectory 2, with noise", "0.56 GB"),
    "living_room_traj3n_frei_png": ("living room, trajectory 3, with noise", "0.76 GB"),
    "traj0_frei_png":              ("office room, trajectory 0", "0.50 GB"),
    "traj1_frei_png":              ("office room, trajectory 1", "0.24 GB"),
    "traj2_frei_png":              ("office room, trajectory 2", "0.30 GB"),
    "traj3_frei_png":              ("office room, trajectory 3", "0.36 GB"),
    "traj0n_frei_png":             ("office room, trajectory 0, with noise", "0.93 GB"),
    "traj1n_frei_png":             ("office room, trajectory 1, with noise", "0.54 GB"),
    "traj2n_frei_png":             ("office room, trajectory 2, with noise", "0.54 GB"),
    "traj3n_frei_png":             ("office room, trajectory 3, with noise", "0.75 GB"),
}

DEFAULT_SEQUENCE = "living_room_traj2_frei_png"
DEST_DIR = Path.home() / "data" / "icl_nuim"


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
                desc=f"  ↓ {self.filename}",
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


def download_file(url: str, filepath: Path, retries: int = 3) -> Path:
    """Download to <file>.part and rename on success, so an interrupted
    download is never mistaken for a finished archive."""
    if filepath.exists():
        print(f"  ⏭  {filepath.name} already downloaded, skipping.")
        return filepath

    part = filepath.with_name(filepath.name + ".part")
    for attempt in range(1, retries + 1):
        hook = TqdmDownloadHook(filepath.name)
        try:
            urllib.request.urlretrieve(url, str(part), reporthook=hook)
            if not tarfile.is_tarfile(part):
                raise IOError("server returned something that is not a tar archive")
            part.rename(filepath)
            return filepath
        except Exception as e:
            print(f"  ⚠  attempt {attempt}/{retries} failed: {e}")
            part.unlink(missing_ok=True)
            # 4xx other than 429 is permanent: don't retry it.
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries:
                raise
            time.sleep(10 * attempt)
        finally:
            hook.close()
    return filepath


def extract_tgz(tgz_path: Path, dest: Path):
    """Extract a .tar.gz into dest/ with a tqdm progress bar."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc=f"  📦 Extracting {tgz_path.name}", ncols=100):
            tf.extract(member, dest, filter="data")


def is_complete(seq_dir: Path) -> bool:
    """A sequence folder only ever appears by an atomic rename after a full
    extraction (see extract_sequence), but check the contents too."""
    rgb, depth = seq_dir / "rgb", seq_dir / "depth"
    if not ((seq_dir / "associations.txt").is_file() and rgb.is_dir() and depth.is_dir()):
        return False
    n_rgb = sum(1 for _ in rgb.iterdir())
    return n_rgb > 0 and n_rgb == sum(1 for _ in depth.iterdir())


def extract_sequence(archive: Path, seq_dir: Path):
    """Extract into a hidden staging folder, check it, then rename it into place,
    so a killed extraction never leaves a half-filled <sequence>/ behind."""
    staging = seq_dir.with_name(f".{seq_dir.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)  # leftover from a killed run
    extract_tgz(archive, staging)
    if not is_complete(staging):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("archive extracted, but rgb/, depth/ or associations.txt is missing or unequal")
    shutil.rmtree(seq_dir, ignore_errors=True)  # an incomplete folder from an older script version
    staging.rename(seq_dir)


def print_list(dest: Path):
    print("ICL-NUIM sequences (TUM/Freiburg PNG format):\n")
    for name, (desc, size) in SEQUENCES.items():
        mark = "*" if name == DEFAULT_SEQUENCE else " "
        state = "on disk" if is_complete(dest / name) else ""
        print(f"  {mark} {name:30s} ~{size:7s} {desc:50s} {state}")
    print(f"\n  * = default.  Destination: {dest}/<sequence>/")


def main():
    ap = argparse.ArgumentParser(
        description="Download ICL-NUIM sequences, one folder per sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sequences:\n" + "\n".join(f"  {n}" for n in SEQUENCES),
    )
    ap.add_argument("sequences", nargs="*", default=[DEFAULT_SEQUENCE],
                    help=f"sequence name(s) (default: {DEFAULT_SEQUENCE})")
    ap.add_argument("--all", action="store_true", help="download all 16 sequences")
    ap.add_argument("--list", action="store_true", help="list sequences and exit")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep each .tar.gz after it extracts cleanly")
    ap.add_argument("--dest", type=Path, default=DEST_DIR,
                    help=f"destination directory (default: {DEST_DIR})")
    args = ap.parse_args()

    if args.list:
        print_list(args.dest)
        return

    sequences = list(SEQUENCES) if args.all else args.sequences
    unknown = [s for s in sequences if s not in SEQUENCES]
    if unknown:
        print(f"  Unknown sequence(s): {', '.join(unknown)}  (see --list)")
        sys.exit(1)

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ICL-NUIM Dataset Downloader")
    print("=" * 60)
    print(f"\n  Destination: {dest}")
    print(f"  Sequences:   {', '.join(sequences)}\n")

    failed = []
    for seq in sequences:
        seq_dir = dest / seq
        if is_complete(seq_dir):
            print(f"  ⏭  {seq}/ already extracted, skipping.")
            continue

        archive = dest / f"{seq}.tar.gz"
        try:
            download_file(f"{BASE_URL}/{seq}.tar.gz", archive)
            extract_sequence(archive, seq_dir)
        except (tarfile.TarError, EOFError) as e:
            # A corrupt archive is kept so it can be inspected; delete it by hand to re-download.
            print(f"  ❌ {archive.name} is corrupted ({e}); kept for inspection.")
            failed.append(seq)
            continue
        except Exception as e:
            print(f"  ❌ {seq}: {e}")
            failed.append(seq)
            continue

        print(f"  ✅ {seq} -> {seq_dir}")
        if not args.keep_archives:
            archive.unlink()
            print(f"  🗑  Removed {archive.name}")

    print("\n" + "=" * 60)
    if failed:
        print(f"  ⚠  Failed: {', '.join(failed)}")
    else:
        print("  ✅ Done!")
    print(f"     {dest}")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
