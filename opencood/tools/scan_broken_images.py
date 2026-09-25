#!/usr/bin/env python3
"""Scan AirV2X dataset for corrupted camera/depth PNG files."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import List, Optional, Set, Tuple

from PIL import Image
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from opencood.utils.airv2x_utils import DRONE_FILES, RSU_FILES, VEHICLE_FILES


def collect_image_paths(root_dir: str) -> List[str]:
    """Collect camera/depth image paths referenced by the dataset layout."""
    paths: List[str] = []
    target_names = set()
    for files in (RSU_FILES, VEHICLE_FILES, DRONE_FILES):
        for name in files:
            if name.endswith(".png") and ("camera" in name or "depth" in name):
                target_names.add(name)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in target_names:
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def quick_png_suspect(path: str) -> Optional[str]:
    """Fast header/tail check; returns error message if obviously bad."""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return "empty file (0 bytes)"
        with open(path, "rb") as f:
            header = f.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                return "invalid PNG header"
            f.seek(max(0, size - 12))
            tail = f.read()
            if b"IEND" not in tail:
                return "truncated PNG (missing IEND chunk)"
    except OSError as exc:
        return str(exc)
    return None


def check_image(path: str) -> Tuple[str, Optional[str]]:
    """Return (path, error_message). error_message is None if image is OK."""
    try:
        if not os.path.isfile(path):
            return path, "file not found"
        quick_err = quick_png_suspect(path)
        if quick_err and quick_err != "invalid PNG header":
            # Still run PIL for header issues; truncated files are definite failures.
            if "truncated" in quick_err or "empty" in quick_err:
                return path, quick_err
        with Image.open(path) as img:
            img.load()
        return path, None
    except OSError as exc:
        return path, str(exc)
    except Exception as exc:  # noqa: BLE001
        return path, f"{type(exc).__name__}: {exc}"


def load_done_paths(checkpoint_file: str) -> Set[str]:
    """Load already-checked paths from checkpoint file."""
    done: Set[str] = set()
    if not os.path.isfile(checkpoint_file):
        return done
    with open(checkpoint_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                done.add(line.split("\t", 1)[0])
    return done


def scan_directory(
    root_dir: str,
    workers: int,
    output_file: str,
    checkpoint_file: str,
    timeout_sec: float,
    start_from: int,
) -> int:
    """Scan one dataset split and write broken image list incrementally."""
    print(f"\n=== Scanning: {root_dir} ===")
    if not os.path.isdir(root_dir):
        print(f"[skip] directory does not exist: {root_dir}")
        return 0

    all_paths = collect_image_paths(root_dir)
    done_paths = load_done_paths(checkpoint_file)
    paths = all_paths[start_from:]
    if done_paths:
        paths = [p for p in paths if p not in done_paths]
    print(
        f"Total images: {len(all_paths)}, start_from={start_from}, "
        f"remaining={len(paths)}, workers={workers}, timeout={timeout_sec}s"
    )

    broken_count = 0
    checked = 0
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    write_header = not os.path.isfile(output_file) or os.path.getsize(output_file) == 0

    with open(output_file, "a", encoding="utf-8") as out_f, open(
        checkpoint_file, "a", encoding="utf-8"
    ) as ckpt_f:
        if write_header:
            out_f.write(f"# root_dir: {root_dir}\n")
            out_f.write(f"# total_images: {len(all_paths)}\n\n")
            ckpt_f.write(f"# root_dir: {root_dir}\n")

        if workers <= 1:
            iterator = tqdm(paths, desc="Checking", unit="img")
            for path in iterator:
                _, err = check_image(path)
                checked += 1
                ckpt_f.write(f"{path}\tok\n")
                if err:
                    broken_count += 1
                    out_f.write(f"{path}\t{err}\n")
                    out_f.flush()
                    print(f"\nBROKEN [{broken_count}]: {path}\n  -> {err}")
                if checked % 500 == 0:
                    ckpt_f.flush()
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(check_image, p): p for p in paths}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Checking"):
                    path = futures[future]
                    try:
                        _, err = future.result(timeout=timeout_sec)
                    except FuturesTimeoutError:
                        err = f"timeout after {timeout_sec}s (possible slow/bad disk I/O)"
                    except Exception as exc:  # noqa: BLE001
                        err = f"worker error: {exc}"

                    checked += 1
                    ckpt_f.write(f"{path}\t{'broken' if err else 'ok'}\n")
                    if err:
                        broken_count += 1
                        out_f.write(f"{path}\t{err}\n")
                        out_f.flush()
                        print(f"\nBROKEN [{broken_count}]: {path}\n  -> {err}")
                    if checked % 500 == 0:
                        ckpt_f.flush()

    print(f"Checked: {checked}, broken: {broken_count}")
    print(f"Report: {output_file}")
    print(f"Checkpoint: {checkpoint_file}")
    return broken_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan AirV2X images for corruption")
    parser.add_argument(
        "--train-dir",
        default="/home/dell/suyi/AirV2X-Perception/train/train",
    )
    parser.add_argument(
        "--val-dir",
        default="/home/dell/suyi/AirV2X-Perception/val/val",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="/home/dell/suyi/AirV2X-Perception_copy/opencood/logs",
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    total_broken = 0

    if not args.val_only:
        total_broken += scan_directory(
            args.train_dir,
            args.workers,
            os.path.join(args.output_dir, "broken_images_train.txt"),
            os.path.join(args.output_dir, "scan_checkpoint_train.txt"),
            args.timeout,
            args.start_from,
        )
    if not args.train_only:
        total_broken += scan_directory(
            args.val_dir,
            args.workers,
            os.path.join(args.output_dir, "broken_images_val.txt"),
            os.path.join(args.output_dir, "scan_checkpoint_val.txt"),
            args.timeout,
            0,
        )

    print(f"\n=== Done. Total broken images: {total_broken} ===")
    if total_broken:
        sys.exit(1)


if __name__ == "__main__":
    main()
