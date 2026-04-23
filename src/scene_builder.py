import blenderproc as bproc
import numpy as np
import random
import os
import bpy

from src.utils import get_logger, get_obj_path

logger = get_logger(__name__)


def setup_renderer(cfg: dict) -> None:
    """
    Configure camera intrinsics and renderer settings.
    Called once per session after objects are loaded (calling before causes blank normals).
    Camera K matrix stays fixed so BOP scene_camera.json is consistent across scenes.
    """
    r = cfg["render"]

    bproc.camera.set_intrinsics_from_blender_params(
        lens=r["focal_length_mm"],
        image_width=r["image_width"],
        image_height=r["image_height"],
        lens_unit="MILLIMETERS"
    )
    bproc.renderer.set_render_devices(use_only_cpu=r["use_only_cpu"])
    bproc.renderer.set_noise_threshold(r["noise_threshold"])
    bproc.renderer.set_max_amount_of_samples(r["max_samples"])
    bproc.renderer.set_denoiser(r["denoiser"])

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()

    logger.info(f"Renderer: {r['image_width']}x{r['image_height']} (GPU: {not r['use_only_cpu']})")


def _apply_polyhaven_material(obj, texture_folder: str, material_name: str, append: bool = False):
    """
    Apply a Poly Haven PBR material (diffuse + roughness maps).
    append=True adds as a second material slot instead of replacing.
    Used internally by build_room().
    """
    files = os.listdir(texture_folder)
    diff_file  = next((f for f in files if "diff"  in f.lower()), None)
    rough_file = next((f for f in files if "rough" in f.lower()), None)

    mat = bproc.material.create(material_name)

    if diff_file:
        img_diff = bpy.data.images.load(
            filepath=os.path.abspath(os.path.join(texture_folder, diff_file))
        )
        mat.set_principled_shader_value("Base Color", img_diff)

    if rough_file:
        img_rough  = bpy.data.images.load(
            filepath=os.path.abspath(os.path.join(texture_folder, rough_file))
        )
        img_rough.colorspace_settings.name = "Non-Color" # roughness maps must be non-color
        mat.set_principled_shader_value("Roughness", img_rough)

    if append:
        obj.blender_obj.data.materials.append(mat.blender_obj)
    else:
        obj.replace_materials(mat)
    
    return mat


def build_room(cfg: dict) -> list:
    """
    Build a simple lab room: wood table + inverted cube acting as walls/floor/ceiling.
    Wall and floor get different PBR materials via polygon normal detection.
    All objects returned for cleanup at end of scene.
    """
    room_objs = []

    # Table, a flat cube, objects land on its top face (Z=0)
    table = bproc.object.create_primitive("CUBE")
    table.set_scale([0.8, 0.6, 0.04])
    table.set_location([0, 0, -0.04])
    _apply_polyhaven_material(
        obj=table,
        texture_folder="assets/textures/wood_cabinet_worn_long_1k.gltf/textures",
        material_name="wood_table"
    )
    table.enable_rigidbody(active=False, collision_shape="BOX")
    room_objs.append(table)

    # Room, a large cube with flipped normals so inside faces are visible to camera
    room = bproc.object.create_primitive("CUBE")
    room.set_scale([3, 3, 2])
    room.set_location([0, 0, 1.5])

    # Flip normals so inside faces are visible
    bpy.ops.object.select_all(action='DESELECT')
    room.blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = room.blender_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Slot 0: walls, Slot 1: floor
    _apply_polyhaven_material(
        obj=room,
        texture_folder="assets/textures/concrete_wall_006_1k.gltf/textures",
        material_name="wall_mat",
        append=False
    )
    
    _apply_polyhaven_material(
        obj=room,
        texture_folder="assets/textures/concrete_wall_001_1k.gltf/textures",
        material_name="floor_mat",
        append=True
    )

    # Assign floor material to horizontal faces only
    for poly in room.blender_obj.data.polygons:
        poly.material_index = 1 if abs(poly.normal.z) > 0.9 else 0
 
    room_objs.append(room)
    return room_objs


def load_scene_objects(cfg: dict, category_id_map: dict) -> list:
    """
    Randomly pick 5-10 objects from pool, load into scene.
    Returns list of (MeshObject, obj_name) tuples.
    """
    ds_cfg      = cfg["dataset"]
    objects_dir = cfg["paths"]["objects_dir"]
    pool        = cfg["objects"]["pool"]

    num_objects = random.randint(ds_cfg["objects_per_scene_min"], ds_cfg["objects_per_scene_max"])
    selected_names = random.sample(pool, num_objects)

    loaded = []
    for name in selected_names:
        try:
            path = os.path.abspath(get_obj_path(name, objects_dir))
            obj = bproc.loader.load_obj(path)[0]
            obj.set_cp("category_id", category_id_map[name])
            obj.set_cp("bop_dataset_name", name)
            loaded.append((obj, name))
            logger.info(f"Loaded asset: {name}")

        except FileNotFoundError as e:
            logger.warning(f"Skipping {name}: {e}")

    logger.info(f"Scene objects: {len(loaded)}")
    return loaded


def setup_object_physics(loaded_objects: list, cfg: dict) -> None:
    """
    Place objects above table with mixed orientations, then enable rigid body
    60% upright, only Z rotation randomized (natural standing poses)
    40% fallen, fully random rotation (tumbles on impact)
    This mix creates realistic occlusions and varied pose annotations.
    """
    p = cfg["object_placement"]
    for obj, _ in loaded_objects:
        x = np.random.uniform(p["drop_x_min"], p["drop_x_max"])
        y = np.random.uniform(p["drop_y_min"], p["drop_y_max"])
        z = np.random.uniform(p["drop_z_min"], p["drop_z_max"])
        obj.set_location([x, y, z])

        if random.random() < p["upright_probability"]:
            obj.set_rotation_euler([0, 0, np.random.uniform(0, 2 * np.pi)])
        else:
            obj.set_rotation_euler(np.random.uniform([0,0,0], [2*np.pi, 2*np.pi, 2*np.pi]))

        obj.enable_rigidbody(
            active=True,
            collision_shape="CONVEX_HULL",
            mass=1.0,
            friction=0.5
        )


def run_physics(cfg: dict) -> None: 
    """
    Drop objects onto table. 
    Simulation runs then poses are frozen for rendering.
    """
    p = cfg["physics"]
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=p["min_simulation_time"],
        max_simulation_time=p["max_simulation_time"],
        check_object_interval=p["check_object_interval"]
    )


def add_basic_light(cfg):
    """
    Fixed point light
    """
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([1, -1, 3])
    light.set_energy(500)
    return light


def sample_camera_pose(loaded_objects: list, cfg: dict):
    """
    Sample camera on a sphere around scene POI with height constraints.
    Falls back to a computed overhead position if sampling fails.
    """
    cam_cfg  = cfg["camera"]
    poi      = bproc.object.compute_poi([obj for obj, _ in loaded_objects])

    for _ in range(cam_cfg["max_sampling_attempts"]):
        location = bproc.sampler.sphere(
            center=poi,
            radius=np.random.uniform(cam_cfg["dist_min"], cam_cfg["dist_max"]),
            mode="SURFACE"
        )

        # Camera must be above table and not too high
        if not (cam_cfg["height_min"] <= location[2] <= cam_cfg["height_max"]):
            continue

        rotation = bproc.camera.rotation_from_forward_vec(
            poi - location,
            inplane_rot=np.random.uniform(
                -cam_cfg["inplane_rot_range"],
                 cam_cfg["inplane_rot_range"]
            )
        )
        return bproc.math.build_transformation_mat(location, rotation)

    # Fallback
    logger.warning("Camera sampling failed, using computed fallback")
    
    # Put camera directly above POI at safe height
    fallback_loc = np.array([poi[0], poi[1] - 0.5, poi[2] + 0.8])
    fallback_rot  = bproc.camera.rotation_from_forward_vec(poi - fallback_loc)
    return bproc.math.build_transformation_mat(fallback_loc, fallback_rot)