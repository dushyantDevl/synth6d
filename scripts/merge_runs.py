import os
import shutil
import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.utils import get_logger

logger = get_logger(__name__)


def merge_bop(runs_dir: str, merged_bop: str) -> int:
    """
    Merge BOP train_pbr folders from all runs into one folder with sequential naming.
    Each run's scenes are renumbered to avoid conflicts:
        run1/000000 -> merged/000000
        run2/000000 -> merged/000200  (offset by run1 scene count)
        run3/000000 -> merged/000500  (offset by run1+run2 counts)
    """
    os.makedirs(merged_bop, exist_ok=True)
    global_idx = 0

    run_folders = sorted([
        d for d in os.listdir(runs_dir)
        if d.startswith("run") and os.path.isdir(os.path.join(runs_dir, d))
    ])

    for run_folder in run_folders:
        train_pbr = os.path.join(runs_dir, run_folder, "bop", "train_pbr")
        if not os.path.exists(train_pbr):
            logger.warning(f"No train_pbr found in {run_folder}, skipping")
            continue

        scene_folders = sorted(os.listdir(train_pbr))
        logger.info(f"{run_folder}: {len(scene_folders)} BOP scenes")

        for scene_folder in scene_folders:
            src = os.path.join(train_pbr, scene_folder)
            dst = os.path.join(merged_bop, f"{global_idx:06d}")
            shutil.copytree(src, dst)
            global_idx += 1

    return global_idx


def merge_hdf5(runs_dir: str, merged_hdf5: str) -> int:
    """
    Merge HDF5 files from all runs with sequential numbering.
    run1/0.hdf5 stays as 0.hdf5
    run2/0.hdf5 becomes N.hdf5 where N = total scenes from previous runs
    """
    os.makedirs(merged_hdf5, exist_ok=True)
    global_idx = 0

    run_folders = sorted([
        d for d in os.listdir(runs_dir)
        if d.startswith("run") and os.path.isdir(os.path.join(runs_dir, d))
    ])

    for run_folder in run_folders:
        hdf5_src = os.path.join(runs_dir, run_folder, "hdf5")
        if not os.path.exists(hdf5_src):
            logger.warning(f"No hdf5 folder found in {run_folder}, skipping")
            continue

        hdf5_files = sorted(
            [f for f in os.listdir(hdf5_src) if f.endswith(".hdf5")],
            key=lambda x: int(x.replace(".hdf5", ""))
        )
        logger.info(f"{run_folder}: {len(hdf5_files)} HDF5 files")

        for fname in hdf5_files:
            src = os.path.join(hdf5_src, fname)
            dst = os.path.join(merged_hdf5, f"{global_idx}.hdf5")
            shutil.copy(src, dst)
            global_idx += 1

    return global_idx


def main():
    parser = argparse.ArgumentParser(description="Merge multiple Kaggle run outputs")
    parser.add_argument("--runs_dir",   default="output",
                        help="Directory containing run1, run2, ... folders")
    parser.add_argument("--output_dir", default="output/merged",
                        help="Where to write merged output")
    args = parser.parse_args()

    merged_bop  = os.path.join(args.output_dir, "bop", "train_pbr")
    merged_hdf5 = os.path.join(args.output_dir, "hdf5")

    # Clear existing merged output to avoid stale data
    if os.path.exists(args.output_dir):
        logger.info(f"Clearing existing merged output at {args.output_dir}")
        shutil.rmtree(args.output_dir)

    logger.info(f"Merging runs from: {args.runs_dir}")
    logger.info(f"Output to: {args.output_dir}")

    total_bop  = merge_bop(args.runs_dir, merged_bop)
    total_hdf5 = merge_hdf5(args.runs_dir, merged_hdf5)

    logger.info(f"Merge complete!")
    logger.info(f"Total BOP scenes: {total_bop}")
    logger.info(f"Total HDF5 files: {total_hdf5}")

    if total_bop != total_hdf5:
        logger.warning(
            f"BOP and HDF5 counts differ ({total_bop} vs {total_hdf5}). "
            f"Some scenes may have had BOP write failures (expected on Windows)."
        )


if __name__ == "__main__":
    main()
