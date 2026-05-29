import os
import shutil
import logging
import yaml

def get_logger(name: str) -> logging.Logger:
    """Central logger used by all modules."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(name)


def load_config(config_path="configs/scene_config.yaml") -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    get_logger(__name__).info(f"Config loaded: {config_path}")
    return cfg


def setup_output_dirs(cfg: dict) -> None:
    """Clean and recreate output directories before generation."""
    for key in ["output_hdf5_dir", "output_bop_dir"]:
        d = cfg["paths"][key]
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        get_logger(__name__).info(f"Output dir ready: {d}")


def get_obj_path(obj_name: str, objects_dir: str) -> str:
    """
    Return path to textured .obj for a YCB object.
    Prefers google_16k (complete mesh) over poisson (may have holes).
    """
    google_path  = os.path.join(objects_dir, obj_name, "google_16k", "textured.obj")
    poisson_path = os.path.join(objects_dir, obj_name, "poisson",    "textured.obj")
 
    if os.path.exists(google_path):
        return google_path
    elif os.path.exists(poisson_path):
        return poisson_path
    raise FileNotFoundError(
        f"No .obj found for '{obj_name}'.\n"
        f"Checked:\n  {google_path}\n  {poisson_path}"
    )


def build_category_id_map(object_pool: list) -> dict:
    """
    Map object names to integer category IDs 
    (1-indexed, 0 = background in segmentation maps.
    """
    return {name: idx + 1 for idx, name in enumerate(object_pool)}


def cleanup_objects(objects: list) -> None:
    """
    Delete scene objects after each render.
    Accepts list of MeshObjects or (MeshObject, name) tuples.
    Without this, objects accumulate across scenes.
    """
    for item in objects:
        obj = item[0] if isinstance(item, tuple) else item
        obj.delete()