import os
import json
import argparse
import random
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not installed, image checks disabled. Install: pip install opencv-python")

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import h5py
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logger.warning("matplotlib or h5py not installed, plots disabled.")


def validate_bop_structure(bop_dir: str) -> dict:
    train_dir = os.path.join(bop_dir, "train_pbr")
    if not os.path.exists(train_dir):
        logger.error(f"train_pbr not found in {bop_dir}")
        return {}

    scene_folders = sorted(os.listdir(train_dir))
    logger.info(f"Found {len(scene_folders)} scene folders (chunks) in {bop_dir}")

    results = {
        "total_scenes":      0,   # total frames across all chunks
        "valid_scenes":      0,
        "missing_rgb":       0,
        "missing_depth":     0,
        "missing_gt":        0,
        "missing_camera":    0,
        "black_images":      0,
        "invalid_poses":     0,
        "total_objects":     0,
        "objects_per_scene": [],
        "depth_means":       [],
        "issues":            []
    }

    for scene_folder in scene_folders:
        scene_path  = os.path.join(train_dir, scene_folder)
        chunk_issues = []

        rgb_dir     = os.path.join(scene_path, "rgb")
        depth_dir   = os.path.join(scene_path, "depth")
        gt_path     = os.path.join(scene_path, "scene_gt.json")
        camera_path = os.path.join(scene_path, "scene_camera.json")

        if not os.path.exists(rgb_dir):
            results["missing_rgb"] += 1
            chunk_issues.append("missing rgb/")
        if not os.path.exists(depth_dir):
            results["missing_depth"] += 1
            chunk_issues.append("missing depth/")
        if not os.path.exists(gt_path):
            results["missing_gt"] += 1
            chunk_issues.append("missing scene_gt.json")
        if not os.path.exists(camera_path):
            results["missing_camera"] += 1
            chunk_issues.append("missing scene_camera.json")

        if chunk_issues:
            results["issues"].append(f"{scene_folder}: {', '.join(chunk_issues)}")
            continue

        try:
            with open(gt_path) as f:
                gt = json.load(f)

            # Each key in scene_gt.json is one frame (one render)
            num_frames = len(gt)
            results["total_scenes"] += num_frames
            logger.info(f"{scene_folder}: {num_frames} frames")

            for frame_id, objects in gt.items():
                # Objects per frame
                results["objects_per_scene"].append(len(objects))
                results["total_objects"] += len(objects)

                # Validate rotation matrices for this frame by checking if its determinant is 1 or not
                for obj in objects:
                    R   = np.array(obj["cam_R_m2c"]).reshape(3, 3)
                    det = np.linalg.det(R)
                    if abs(det - 1.0) > 0.01:
                        results["invalid_poses"] += 1
                        chunk_issues.append(
                            f"frame {frame_id}: invalid rotation (det={det:.3f})"
                        )

            results["valid_scenes"] += num_frames

        except Exception as e:
            chunk_issues.append(f"scene_gt.json parse error: {e}")

        try:
            with open(camera_path) as f:
                cam = json.load(f)
            for frame_id, frame_data in cam.items():
                if "cam_K" not in frame_data:
                    chunk_issues.append(f"frame {frame_id}: missing cam_K")
                elif len(frame_data["cam_K"]) != 9:
                    chunk_issues.append(f"frame {frame_id}: cam_K wrong length")
        except Exception as e:
            chunk_issues.append(f"scene_camera.json parse error: {e}")

        # Check one RGB image per chunk
        if CV2_AVAILABLE and os.path.exists(rgb_dir):
            for img_file in os.listdir(rgb_dir)[:1]:
                img = cv2.imread(os.path.join(rgb_dir, img_file))
                if img is None:
                    chunk_issues.append(f"corrupted image: {img_file}")
                elif img.max() < 5:
                    results["black_images"] += 1
                    chunk_issues.append(f"black image: {img_file}")

        # Check one depth image per chunk
        if CV2_AVAILABLE and os.path.exists(depth_dir):
            for depth_file in os.listdir(depth_dir)[:1]:
                depth = cv2.imread(
                    os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED
                )
                if depth is not None and depth.max() > 0:
                    results["depth_means"].append(float(depth[depth > 0].mean()))

        if chunk_issues:
            results["issues"].extend(
                [f"{scene_folder}: {i}" for i in chunk_issues]
            )

    return results


def print_validation_summary(results: dict) -> None:
    """Print a readable validation summary to console."""
    if not results:
        return

    total = max(results["total_scenes"], 1)
    print("=" * 50)
    logger.info("DATASET VALIDATION SUMMARY")
    print("=" * 50)
    logger.info(f"Total scenes:    {results['total_scenes']}")
    logger.info(f"Valid scenes:    {results['valid_scenes']} ({100 * results['valid_scenes'] / total:.1f}%)")
    logger.info(f"Missing rgb/:    {results['missing_rgb']}")
    logger.info(f"Missing depth/:  {results['missing_depth']}")
    logger.info(f"Missing gt:      {results['missing_gt']}")
    logger.info(f"Missing camera:  {results['missing_camera']}")
    logger.info(f"Black images:    {results['black_images']}")
    logger.info(f"Invalid poses:   {results['invalid_poses']}")

    if results["objects_per_scene"]:
        ops = results["objects_per_scene"]
        logger.info(f"Objects/scene:   min={min(ops)}, max={max(ops)}, mean={np.mean(ops):.1f}")
        logger.info(f"Total annotations: {results['total_objects']}")

    if results["depth_means"]:
        dm = results["depth_means"]
        logger.info(f"Depth (mm):      min={min(dm):.0f}, max={max(dm):.0f}, mean={np.mean(dm):.0f}")

    if results["issues"]:
        logger.warning(f"Issues found: {len(results['issues'])}")
        for issue in results["issues"][:10]:
            logger.warning(f"  {issue}")
        if len(results["issues"]) > 10:
            logger.warning(f"  ... and {len(results['issues']) - 10} more")
    else:
        logger.info("No issues found!")


def plot_dataset_statistics(results: dict,
                            save_path: str = "docs/sample_outputs/dataset_stats.png") -> None:
    """
    Plot 3 dataset statistics charts: objects per scene, depth distribution, valid/invalid pie.
    Saves to file for README and Twitter use.
    """
    if not MPL_AVAILABLE or not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#111111")
    title_kw = dict(color="white", fontsize=12, fontweight="bold")
    tick_kw  = dict(colors="white")

    if results["objects_per_scene"]:
        axes[0].hist(results["objects_per_scene"], bins=10, color="#3498db", edgecolor="#111111")
        axes[0].set_title("Objects per Scene", **title_kw)
        axes[0].set_xlabel("Object Count", color="white")
        axes[0].set_ylabel("Number of Scenes", color="white")
        axes[0].tick_params(**tick_kw)
        axes[0].set_facecolor("#1a1a1a")

    if results["depth_means"]:
        axes[1].hist(results["depth_means"], bins=20, color="#2ecc71", edgecolor="#111111")
        axes[1].set_title("Mean Depth Distribution (mm)", **title_kw)
        axes[1].set_xlabel("Depth (mm)", color="white")
        axes[1].set_ylabel("Number of Scenes", color="white")
        axes[1].tick_params(**tick_kw)
        axes[1].set_facecolor("#1a1a1a")

    valid   = results["valid_scenes"]
    invalid = results["total_scenes"] - valid
    slices  = [valid, invalid] if invalid > 0 else [valid]
    labels  = ["Valid", "Invalid"] if invalid > 0 else ["All Valid"]
    colors  = ["#2ecc71", "#e74c3c"] if invalid > 0 else ["#2ecc71"]
    axes[2].pie(slices, labels=labels, colors=colors,
                autopct="%1.1f%%", textprops={"color": "white"})
    axes[2].set_title("Scene Quality", **title_kw)
    axes[2].set_facecolor("#1a1a1a")

    fig.suptitle("synth6d Dataset Statistics", color="white", fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    logger.info(f"Stats plot saved to: {save_path}")
    plt.show()


def generate_sample_grid(bop_dir: str, hdf5_dir: str = None,
                         num_samples: int = 16,
                         save_path: str = "docs/sample_outputs/randomization_grid.png") -> None:
    """
    Generate a grid showing RGB, depth (clipped 0-2m), normals and class segmentation
    for a random selection of scenes. If hdf5_dir is provided, loads all modalities
    from HDF5 files. Otherwise shows RGB only from BOP folder.
    Saves grid to file for Twitter Post 2.
    """
    if not MPL_AVAILABLE:
        return

    if hdf5_dir and os.path.exists(hdf5_dir):
        _generate_full_grid(hdf5_dir, num_samples, save_path)
    else:
        _generate_rgb_grid(bop_dir, num_samples, save_path)


def _generate_full_grid(hdf5_dir: str, num_samples: int, save_path: str) -> None:
    """
    Generate a grid showing all 5 modalities from HDF5 files.
    Each row = one scene: RGB | Depth | Normals | Class Seg | Instance Seg
    Depth is clipped to 0-2m for visibility (raw HDF5 values go to 5m+).
    """
    hdf5_files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5")])
    if not hdf5_files:
        logger.warning(f"No HDF5 files found in {hdf5_dir}")
        return

    samples = random.sample(hdf5_files, min(num_samples, len(hdf5_files)))
    rows    = len(samples)
    cols    = 5
    titles  = ["RGB", "Depth (0-2m)", "Normals", "Class Seg", "Instance Seg"]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.patch.set_facecolor("#111111")

    if rows == 1:
        axes = [axes]

    for row_idx, fname in enumerate(samples):
        fpath = os.path.join(hdf5_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                colors    = np.array(f["colors"])    if "colors"          in f else None
                depth     = np.array(f["depth"])     if "depth"           in f else None
                normals   = np.array(f["normals"])   if "normals"         in f else None
                class_seg = np.array(f["class_segmaps"])    if "class_segmaps"    in f else None
                inst_seg  = np.array(f["instance_segmaps"]) if "instance_segmaps" in f else None

            images = [
                colors,
                np.clip(depth, 0, 2.0) if depth is not None else None,
                np.clip((normals + 1.0) / 2.0, 0, 1) if normals is not None else None,
                class_seg,
                inst_seg,
            ]
            cmaps = [None, "plasma", None, "tab10", "tab20"]

            for col_idx, (img, cmap) in enumerate(zip(images, cmaps)):
                ax = axes[row_idx][col_idx]
                if img is not None:
                    ax.imshow(img, cmap=cmap)
                else:
                    ax.set_facecolor("#1a1a1a")
                if row_idx == 0:
                    ax.set_title(titles[col_idx], color="white", fontsize=10, fontweight="bold")
                ax.axis("off")

        except Exception as e:
            logger.warning(f"Could not load {fname}: {e}")

    fig.suptitle("synth6d: Domain Randomization across Dataset",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    logger.info(f"Full modality grid saved to: {save_path}")
    plt.show()


def _generate_rgb_grid(bop_dir: str, num_samples: int, save_path: str) -> None:
    """
    Generate a simple grid of RGB images from the BOP folder.
    Used when HDF5 files are not available locally.
    """
    train_dir = os.path.join(bop_dir, "train_pbr")
    if not os.path.exists(train_dir):
        logger.error(f"train_pbr not found in {bop_dir}")
        return

    rgb_paths = []
    for scene_folder in os.listdir(train_dir):
        rgb_dir = os.path.join(train_dir, scene_folder, "rgb")
        if os.path.exists(rgb_dir):
            imgs = sorted(os.listdir(rgb_dir))
            if imgs:
                rgb_paths.append(os.path.join(rgb_dir, imgs[0]))

    if not rgb_paths:
        logger.error("No RGB images found")
        return

    samples = random.sample(rgb_paths, min(num_samples, len(rgb_paths)))
    cols    = 4
    rows    = max(1, len(samples) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
    fig.patch.set_facecolor("#111111")

    for ax, img_path in zip(axes.flat, samples):
        ax.imshow(mpimg.imread(img_path))
        ax.axis("off")

    fig.suptitle("synth6d: Domain Randomization across Dataset",
                 color="white", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    logger.info(f"RGB grid saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate synth6d BOP dataset")
    parser.add_argument("--bop_dir",     default="output/merged/bop",
                        help="Path to BOP output directory")
    parser.add_argument("--hdf5_dir",    default=None,
                        help="Path to HDF5 output directory (enables full modality grid)")
    parser.add_argument("--visualize",   action="store_true",
                        help="Generate sample grid visualization")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of scenes in grid (default 8)")
    parser.add_argument("--stats_only",  action="store_true",
                        help="Only plot statistics, skip structure validation")
    args = parser.parse_args()

    if not args.stats_only:
        results = validate_bop_structure(args.bop_dir)
        print_validation_summary(results)
        plot_dataset_statistics(results)

    if args.visualize:
        generate_sample_grid(
            bop_dir=args.bop_dir,
            hdf5_dir=args.hdf5_dir,
            num_samples=args.num_samples
        )