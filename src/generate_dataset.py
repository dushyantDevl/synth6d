import blenderproc as bproc
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.utils import (
    load_config,
    setup_output_dirs,
    build_category_id_map,
    cleanup_objects,
    get_logger
)
from src.scene_builder import (
    setup_renderer,
    build_room,
    load_scene_objects,
    setup_object_physics,
    run_physics,
    add_basic_light,
    sample_camera_pose
)

logger = get_logger(__name__)

def main():
    cfg = load_config("configs/scene_config.yaml")

    num_scenes  = cfg["dataset"]["num_scenes"]
    hdf5_dir    = cfg["paths"]["output_hdf5_dir"]
    object_pool = cfg["objects"]["pool"]

    logger.info(f"synth6d: generating {num_scenes} scenes")

    bproc.init()
    setup_output_dirs(cfg)
    setup_renderer_done = False  # renderer enabled after first object load (normals fix)
    category_id_map = build_category_id_map(object_pool)

    for scene_idx in range(num_scenes):
        logger.info(f"Scene {scene_idx + 1}/{num_scenes}")

        bproc.utility.reset_keyframes()  # must be first, clears previous scene keyframes

        room_objs = build_room(cfg)
        loaded_objects = load_scene_objects(cfg, category_id_map)

        if not loaded_objects:
            logger.warning("No objects loaded, skipping scene")
            cleanup_objects(room_objs)
            continue

        setup_object_physics(loaded_objects, cfg)
        run_physics(cfg)
        add_basic_light(cfg)

        cam2world = sample_camera_pose(loaded_objects, cfg)
        bproc.camera.add_camera_pose(cam2world)

        # Enable renderer after objects exist, calling before causes blank normals
        if not setup_renderer_done:
            setup_renderer(cfg)
            setup_renderer_done = True

        data     = bproc.renderer.render()
        seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

        # Build output dict explicitly to control exactly which keys get saved
        output = {
            "colors":               data["colors"],
            "depth":                data["depth"],
            "normals":              data["normals"],
        }

        # Only add segmaps if they exist
        if "instance_segmaps" in seg_data:
            output["instance_segmaps"] = seg_data["instance_segmaps"]
        if "class_segmaps" in seg_data:
            output["class_segmaps"] = seg_data["class_segmaps"]

        # Skip scene if segmaps missing (bad camera angle / empty view)
        if "class_segmaps" not in output:
            logger.warning(f"Scene {scene_idx} missing segmaps, skipping")
            cleanup_objects(loaded_objects)
            cleanup_objects(room_objs)
            continue
 
        bproc.writer.write_hdf5(
            output_dir_path=hdf5_dir,
            output_data_dict=output,
            append_to_existing_output=True
        )

        cleanup_objects(loaded_objects)
        cleanup_objects(room_objs)
        logger.info(f"Scene {scene_idx + 1} saved to {hdf5_dir}")
 
    logger.info(f"Done! {num_scenes} scenes saved to {hdf5_dir}")


if __name__ == "__main__":
    main()