# synth6d

Physically-based synthetic data generation pipeline for 6DoF object pose estimation.
Built with **BlenderProc** + **Blender Cycles**. Generates fully annotated training data
with zero real images and zero manual labeling.

![Pipeline Overview](docs/sample_outputs/pipeline.png)

---

## Overview

Each scene produces 5 synchronized outputs:
RGB render, surface normals, depth map, instance segmentation and class segmentation.

Objects are placed using Blender's rigid body physics simulation — they fall and
settle naturally, creating realistic occlusion and pose variety without manual placement.
Every scene independently randomizes lighting, wall/floor/table textures, object
material properties and camera position.

**Stack:** BlenderProc · Blender Cycles · YCB benchmark objects · Poly Haven PBR textures

---

## Results

| Output | Details |
|--------|---------|
| Scenes | 2000 synthetic scenes, 100% valid |
| Objects | 10 YCB benchmark objects (google_16k meshes), 6-10 per scene |
| Annotations | RGB + depth + normals + instance/class segmentation + 6DoF poses |
| Format | HDF5 + BOP |
| Randomization | Lighting, wall/floor/table textures, material perturbation, camera |

![Dataset Statistics](docs/sample_outputs/dataset_stats.png)

![Domain Randomization Grid](docs/sample_outputs/randomization_grid.png)

*Training metrics (ADD score on real YCB images) coming after GDR-Net training. See roadmap below.*

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download YCB objects
Go to [ycb-benchmarks.s3-website-us-east-1.amazonaws.com](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/)
and download the **16k laser scan** for each object listed in `configs/scene_config.yaml`.

Extract into `assets/objects/` with this structure:
```
assets/objects/
    006_mustard_bottle/
        google_16k/
            textured.obj
            textured.mtl
            texture_map.png
    025_mug/
    ...
```

### 3. Download textures
Download PBR textures from [Poly Haven](https://polyhaven.com/textures) at 1K resolution
and extract into `assets/textures/`. See `configs/scene_config.yaml` for the full list of
wall, floor and table texture pools currently used.

### 4. Run
```bash
blenderproc run src/generate_dataset.py
```

Edit `configs/scene_config.yaml` to change number of scenes, objects, render quality
and all other parameters. Keep `num_scenes: 3-5` for local testing.

For large runs (1000+ scenes) use Kaggle T4 GPU — see Kaggle Setup below.

### 5. Visualize output
```bash
blenderproc vis hdf5 output/hdf5/0.hdf5
```

Or use the visualization script for the full 5-panel pipeline image:
```bash
python scripts/visualize.py output/hdf5/0.hdf5
python scripts/visualize.py output/hdf5/4.hdf5 --out docs/sample_outputs/scene4.png
```

### 6. Validate dataset
```bash
python src/validator.py --bop_dir output/bop --hdf5_dir output/hdf5 --visualize
```

### 7. Merge multiple Kaggle runs
If you generate in batches across multiple sessions:
```bash
python scripts/merge_runs.py --runs_dir output --output_dir output/merged
```

---

## Kaggle Setup

For generating 1000+ scenes, run on Kaggle T4 GPU:

1. Upload your project code, YCB objects and textures as separate Kaggle datasets
2. In the notebook, copy code and symlink assets then run:
```python
import os
os.chdir("/kaggle/working/synth6d")
os.environ["MPLBACKEND"] = "Agg"
!blenderproc run /kaggle/working/synth6d/src/generate_dataset.py
```
3. Update `configs/scene_config.yaml` for Kaggle: `use_only_cpu: false`, `num_scenes: 500`, `noise_threshold: 0.02`

---

## Project Structure

```
synth6d/
├── configs/
│   └── scene_config.yaml       # all tunable parameters
├── src/
│   ├── generate_dataset.py     # main generation loop
│   ├── scene_builder.py        # room, objects, camera, physics
│   ├── randomizer.py           # lighting, texture, material randomization
│   ├── utils.py                # config loading, path helpers, logging
│   └── validator.py            # dataset quality checks and statistics
├── scripts/
│   ├── visualize.py            # pipeline visualization grid
│   └── merge_runs.py           # merge multiple Kaggle run outputs
├── assets/
│   ├── objects/                # YCB .obj files (gitignored)
│   └── textures/               # Poly Haven textures (gitignored)
├── output/                     # generated dataset (gitignored)
└── docs/
    └── sample_outputs/         # sample renders for README
```

---

## Domain Randomization

Each scene independently randomizes:

| Property | Details |
|----------|---------|
| Lighting | Area light, random position, energy (5-25W), size and color temperature |
| Wall textures | 5 textures: brick, factory brick, concrete, stone, rock wall |
| Floor textures | 5 textures: asphalt, linoleum, slate, tiles, worn tile |
| Table textures | 5 textures: dark wood, black planks, worn wood, fine grain, wood table |
| Object materials | Roughness perturbed per object type, specular on metallic objects |
| Camera | Sphere sampling around scene POI, 0.6-1.5m distance, 0.3-0.8m height |
| Object placement | 60% upright, 40% random rotation; physics handles settling |

---

## Implementation Notes

A few non-obvious things worth documenting for anyone building something similar:

**Depth visualization requires clipping.**
BlenderProc depth maps store raw metric values in meters. The full range (0-5m+)
compresses nearby objects into a narrow band. Clipping to 0-2m gives useful contrast
for tabletop scenes. The raw values are preserved in the HDF5 file — clipping is
only applied at visualization time.

**Surface normals must be enabled after objects are loaded.**
Calling `enable_normals_output()` before any mesh exists in the scene results in
blank normal maps (all 0.5 values). The fix is to call renderer setup after the
first scene's objects and room are created. A `setup_renderer_done` flag ensures
it only runs once per session.

**Area lights instead of point lights in enclosed rooms.**
Point lights in a closed room cause severe overexposure because light bounces off
walls repeatedly. Area lights spread illumination over a surface and produce
natural soft shadows. Energy range 5-25W with random size (0.5-2.0m) gives
realistic variation between small lamp and large ceiling panel.

**YCB poisson meshes are incomplete.**
The `poisson/` folder in the YCB processed download contains meshes reconstructed
from RGB-D scans which often have holes. Use the `google_16k/` meshes from the
separate "16k laser scan" download — scanner-clean and complete. The pipeline
checks for google_16k first and falls back to poisson.

**BOP writer on Windows.**
The BOP writer has path separator issues on Windows and fails silently for most
scenes. HDF5 output works correctly on all platforms. Run BOP generation on
Kaggle (Linux) for full BOP output.

---

## Roadmap

- [x] Project structure and config
- [x] Basic scene rendering: RGB, depth, normals, segmentation
- [x] Domain randomization: lighting, textures, material perturbation
- [x] BOP format output with 6DoF pose annotations
- [x] 2000 scene dataset generated (Kaggle T4 GPU)
- [x] Dataset validation and statistics
- [ ] GDR-Net training on synthetic data
- [ ] Evaluation on real YCB-Video images (ADD score)
- [ ] Webcam demo: real-time 6DoF pose overlay

---

## Open Questions / Suggestions Welcome

This is a learning project for synthetic data engineering.
If you work in this space and have thoughts on any of the following, I'd appreciate input:

- Is a simple tabletop YCB scene the right complexity for demonstrating a 6DoF pipeline,
  or would a more varied scene (bin picking, shelf arrangement) be more relevant?
- Any BlenderProc patterns that would make the randomization more effective for sim-to-real?
- Recommended pose estimators to train on this kind of data beyond GDR-Net?

---

## References

- [BlenderProc](https://github.com/DLR-RM/BlenderProc) - Denninger et al., RSS 2020
- [YCB Object and Model Set](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/) - Calli et al., ICRA 2015
- [BOP Benchmark](https://bop.felk.cvut.cz/) - Hodaň et al., ECCV 2018
- [Poly Haven](https://polyhaven.com/) - CC0 PBR assets
