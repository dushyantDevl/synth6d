import blenderproc as bproc
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def randomize_lighting(cfg: dict):
    """
    Create a key light and fill light with randomized properties per scene.
    Key light is bright and directional. Fill light is dimmer and placed on
    the opposite side to soften harsh shadows from the key light.
    """
    light_cfg = cfg["lighting"]

    # Key light: primary illumination, randomized position and energy
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location([
        np.random.uniform(*light_cfg["x_range"]),
        np.random.uniform(*light_cfg["y_range"]),
        np.random.uniform(*light_cfg["z_range"]),
    ])
    light.set_energy(np.random.uniform(light_cfg["energy_min"], light_cfg["energy_max"]))

    light_size = np.random.uniform(0.5,2.0)
    light.set_scale([light_size, light_size, 1.0])
    # Slight warm/cool color variation, values stay close to 1.0 for realism
    light.set_color([
        np.random.uniform(0.90, 1.00),
        np.random.uniform(0.90, 1.00),
        np.random.uniform(0.85, 1.00), # slight blue shift for cool light
    ])

    return light


def randomize_table_material(table) -> None:
    """
    Perturb table surface finish per scene.
    Simulates variation between different wood conditions (new, worn, polished).
    """
    mats = table.get_materials()

    for mat in mats:
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.7, 0.95))

        # Low specular range (0.0-0.1) prevents table from becoming a bright reflector.
        mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.0, 0.1))
        

def perturb_object_materials(loaded_objects: list, cfg: dict):
    """
    Perturb material properties per object within realistic bounds.
    Only adjusts roughness and specular — never replaces original YCB textures.
    Simulates wear, handling, and manufacturing variation across scenes.
    Metallic objects additionally get specular variation for sheen differences.
    """
    perturb_cfg = cfg.get("material_perturbation", {})
    if not perturb_cfg.get("enabled", True):
        return

    # Roughness ranges based on real material properties of each object type
    roughness_ranges = {
        "002_master_chef_can":  (0.3, 0.6),   # metal, varies with wear
        "003_cracker_box":      (0.7, 0.95),  # cardboard, always rough
        "004_sugar_box":        (0.6, 0.90),  # cardboard
        "006_mustard_bottle":   (0.4, 0.7),   # plastic
        "017_orange":           (0.5, 0.80),  # organic skin
        "013_apple":            (0.3, 0.6),   # waxy skin, can be shiny
        "021_bleach_cleanser":  (0.4, 0.7),   # plastic bottle
        "024_bowl":             (0.3, 0.6),   # ceramic, matte to glossy
        "025_mug":              (0.3, 0.65),  # ceramic
        "035_power_drill":      (0.4, 0.75),  # composite plastic and metal
        "051_large_clamp":      (0.3, 0.6),   # metal tool
    }

    metallic_objects = {"002_master_chef_can", "051_large_clamp"}

    for obj, name in loaded_objects:
        rough_min, rough_max = roughness_ranges.get(name, (0.3, 0.7))

        for mat in obj.get_materials():
            mat.set_principled_shader_value(
                "Roughness",
                np.random.uniform(rough_min, rough_max)
            )

            # Metallic objects get specular variation to simulate surface finish differences
            if name in metallic_objects:
                mat.set_principled_shader_value(
                    "Specular IOR Level",
                    np.random.uniform(0.1, 0.4)
                )