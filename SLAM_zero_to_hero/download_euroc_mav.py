#!/usr/bin/env python3
"""
Download EuRoC MAV sequences (ASL format) to ~/data/euroc_mav/<SEQUENCE>/mav0/.

ETH publishes EuRoC as three group archives (machine_hall, vicon_room1, vicon_room2).
Each holds one folder per sequence with a nested <SEQUENCE>.zip (ASL format) and a
<SEQUENCE>.bag (ROS). This script downloads the group archive that a sequence
lives in and extracts ONLY that sequence's ASL zip to

  ~/data/euroc_mav/MH_01_easy/mav0/{cam0,cam1,imu0,leica0,state_groundtruth_estimate0}

which is the path basalt and orb_slam2 mount. Pass --with-bag to also keep the .bag.

Safety:
- Downloads go to <file>.part and are renamed only when complete, so a partial
  download is never mistaken for a finished archive.
- A group archive is deleted only after every requested sequence in it extracted
  cleanly (and never with --keep-archives). A corrupt archive is kept.

Reference: https://projects.asl.ethz.ch/datasets/euroc-mav/
Dataset:   https://doi.org/10.3929/ethz-b-000690084

Usage:
    python3 download_euroc_mav.py                        # MH_01_easy (basalt default)
    python3 download_euroc_mav.py --list
    python3 download_euroc_mav.py V1_01_easy MH_03_medium
    python3 download_euroc_mav.py --all
    python3 download_euroc_mav.py --with-bag --keep-archives MH_01_easy
"""

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm


BASE_URL = "https://www.research-collection.ethz.ch/bitstreams"

# group archive -> bitstream id
GROUPS = {
    "machine_hall": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
    "vicon_room1":  "02ecda9a-298f-498b-970c-b7c44334d880",
    "vicon_room2":  "ea12bc01-3677-4b4c-853d-87c7870b8c44",
}

# sequence -> group archive it is stored in
SEQUENCES = {
    "MH_01_easy":      "machine_hall",
    "MH_02_easy":      "machine_hall",
    "MH_03_medium":    "machine_hall",
    "MH_04_difficult": "machine_hall",
    "MH_05_difficult": "machine_hall",
    "V1_01_easy":      "vicon_room1",
    "V1_02_medium":    "vicon_room1",
    "V1_03_difficult": "vicon_room1",
    "V2_01_easy":      "vicon_room2",
    "V2_02_medium":    "vicon_room2",
    "V2_03_difficult": "vicon_room2",
}

DEFAULT_SEQUENCE = "MH_01_easy"
DEST_DIR = Path.home() / "data" / "euroc_mav"


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


def download_file(url: str, filepath: Path, retries: int = 5) -> Path:
    """Download to <file>.part, rename when complete. The ETH server rate-limits
    (HTTP 429), so failures back off and retry."""
    if filepath.exists():
        print(f"  ⏭  {filepath.name} already downloaded, skipping.")
        return filepath

    part = filepath.with_name(filepath.name + ".part")
    for attempt in range(1, retries + 1):
        hook = TqdmDownloadHook(filepath.name)
        try:
            urllib.request.urlretrieve(url, str(part), reporthook=hook)
            if not zipfile.is_zipfile(part):
                raise IOError("server returned something that is not a zip (rate limit page?)")
            part.rename(filepath)
            return filepath
        except Exception as e:
            part.unlink(missing_ok=True)
            print(f"  ⚠  attempt {attempt}/{retries} failed: {e}")
            http = e.code if isinstance(e, urllib.error.HTTPError) else None
            # 4xx other than 429 is permanent: don't retry it.
            if http is not None and 400 <= http < 500 and http != 429:
                raise
            wait = 60 * attempt if http == 429 else 10 * attempt
            if attempt == retries:
                raise
            print(f"     retrying in {wait} s ...")
            time.sleep(wait)
        finally:
            hook.close()
    return filepath


def find_member(names: list, suffix: str):
    hits = [n for n in names if n.endswith(suffix) and "__MACOSX" not in n]
    return min(hits, key=len) if hits else None


def extract_zip(zf: zipfile.ZipFile, dest: Path, desc: str):
    members = [m for m in zf.namelist() if "__MACOSX" not in m and not m.endswith(".DS_Store")]
    for member in tqdm(members, desc=f"  📦 {desc}", ncols=100):
        zf.extract(member, dest)


def extract_sequence(group_zip: Path, seq: str, dest: Path, with_bag: bool):
    """Pull <seq>.zip out of the group archive and unzip it to dest/<seq>/ (giving
    dest/<seq>/mav0). Optionally copy <seq>.bag next to it."""
    seq_dir = dest / seq
    with zipfile.ZipFile(group_zip) as gz:
        names = gz.namelist()
        inner = find_member(names, f"{seq}/{seq}.zip") or find_member(names, f"{seq}.zip")
        if inner is None:
            raise FileNotFoundError(f"{seq}.zip not found inside {group_zip.name}")

        # Fixed-name staging folder: a leftover from a killed run is swept here,
        # instead of piling up as random tmpXXXX folders.
        tmp = dest / f".{seq}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            nested = tmp / f"{seq}.zip"
            with gz.open(inner) as src, open(nested, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 << 20)
            with zipfile.ZipFile(nested) as sz:
                staging = tmp / "x"
                extract_zip(sz, staging, f"Extracting {seq}")
            mav0 = next(staging.rglob("mav0"), None)
            if mav0 is None or not is_complete(mav0.parent):
                raise FileNotFoundError(f"no complete mav0/ inside {seq}.zip")
            seq_dir.mkdir(parents=True, exist_ok=True)
            if (seq_dir / "mav0").exists():
                # Only reached when the existing mav0 is incomplete (main() skips complete
                # ones); replace it rather than let shutil.move nest mav0/mav0.
                print(f"  ⚠  replacing incomplete {seq_dir / 'mav0'}")
                shutil.rmtree(seq_dir / "mav0")
            mav0.rename(seq_dir / "mav0")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if with_bag:
            bag = find_member(names, f"{seq}/{seq}.bag") or find_member(names, f"{seq}.bag")
            if bag is None:
                print(f"  ⚠  {seq}.bag not found inside {group_zip.name}")
            else:
                with gz.open(bag) as src, open(seq_dir / f"{seq}.bag", "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 << 20)


def is_complete(seq_dir: Path) -> bool:
    mav0 = seq_dir / "mav0"
    return all((mav0 / c / "data").is_dir() for c in ("cam0", "cam1")) and (mav0 / "imu0" / "data.csv").is_file()


def print_list(dest: Path):
    print("EuRoC MAV sequences (ASL format):\n")
    for seq, group in SEQUENCES.items():
        mark = "*" if seq == DEFAULT_SEQUENCE else " "
        state = "on disk" if is_complete(dest / seq) else ""
        print(f"  {mark} {seq:16s} in {group}.zip   {state}")
    print(f"\n  * = default.  Destination: {dest}/<SEQUENCE>/mav0/")
    print("  One group archive is downloaded per group needed (several GB each).")


def main():
    ap = argparse.ArgumentParser(
        description="Download EuRoC MAV sequences to ~/data/euroc_mav/<SEQUENCE>/mav0/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sequences:\n" + "\n".join(f"  {s}  ({g})" for s, g in SEQUENCES.items()),
    )
    ap.add_argument("sequences", nargs="*", default=[DEFAULT_SEQUENCE],
                    help=f"sequence name(s) (default: {DEFAULT_SEQUENCE})")
    ap.add_argument("--all", action="store_true", help="all 11 sequences")
    ap.add_argument("--list", action="store_true", help="list sequences and exit")
    ap.add_argument("--with-bag", action="store_true", help="also extract <SEQUENCE>.bag")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep the group archive(s) after extracting")
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
    print("  EuRoC MAV Dataset Downloader")
    print("=" * 60)
    print(f"\n  Destination: {dest}")
    print(f"  Sequences:   {', '.join(sequences)}\n")

    todo = {}
    for seq in sequences:
        if is_complete(dest / seq) and not args.with_bag:
            print(f"  ⏭  {seq}/mav0 already extracted, skipping.")
            continue
        todo.setdefault(SEQUENCES[seq], []).append(seq)

    failed = []
    for group, seqs in todo.items():
        archive = dest / f"{group}.zip"
        try:
            download_file(f"{BASE_URL}/{GROUPS[group]}/download", archive)
        except Exception as e:
            print(f"  ❌ Could not download {archive.name}: {e}")
            failed += seqs
            continue

        group_ok = True
        for seq in seqs:
            try:
                if not is_complete(dest / seq):
                    extract_sequence(archive, seq, dest, args.with_bag)
                elif args.with_bag:  # mav0 already there; only the bag is missing
                    with zipfile.ZipFile(archive) as gz:
                        bag = find_member(gz.namelist(), f"{seq}/{seq}.bag")
                        if bag:
                            with gz.open(bag) as src, open(dest / seq / f"{seq}.bag", "wb") as dst:
                                shutil.copyfileobj(src, dst, length=16 << 20)
                if not is_complete(dest / seq):
                    raise RuntimeError("mav0/cam0 or mav0/imu0 missing after extraction")
                print(f"  ✅ {seq} -> {dest / seq / 'mav0'}")
            except zipfile.BadZipFile as e:
                print(f"  ❌ {archive.name} is corrupted ({e}); kept for inspection.")
                failed.append(seq)
                group_ok = False
            except Exception as e:
                print(f"  ❌ {seq}: {e}")
                failed.append(seq)
                group_ok = False

        if group_ok and not args.keep_archives:
            archive.unlink()
            print(f"  🗑  Removed {archive.name}")
        elif not group_ok:
            print(f"  ⚠  Kept {archive.name} because an extraction failed.")

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
