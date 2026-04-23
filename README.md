# synth6d

Physically-based synthetic data generation pipeline for 6DoF object pose estimation.
Built with **BlenderProc** + **Blender Cycles**. Generates fully annotated training data
with zero real images and zero manual labeling.

![Pipeline Overview](docs/sample_outputs/pipeline.png)

---

## Overview

Each scene produces 5 synchronized outputs:
* **RGB Render:** Photorealistic output from the Cycles engine.
* **Surface Normals:** Camera-space XYZ vectors.
* **Depth Map:** Metric depth in meters (raw HDF5 data).
* **Segmentation:** Instance and Class-level masks.


Objects are placed using Blender's rigid body physics simulation, they fall and
settle naturally, creating realistic occlusion and pose variety without manual placement.

**Stack:** BlenderProc · Blender Cycles · YCB benchmark objects · Poly Haven PBR textures

---

## Results

| Output | Details |
|--------|---------|
| Objects | 10 YCB benchmark objects (google_16k meshes) |
| Annotations | RGB + depth + normals + instance/class segmentation |
| Format | HDF5 (BOP format coming in Commit 3) |
| Scene | Physics-based placement, PBR room materials |

*Training metrics (ADD score on real YCB images) coming after GDR-Net training. See roadmap below.*


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
and extract into `assets/textures/`. Currently used:
- `wood_cabinet_worn_long` : table surface
- `concrete_wall_006` : room walls
- `concrete_wall_001` : room floor

### 4. Run
```bash
blenderproc run src/generate_dataset.py
```

Edit `configs/scene_config.yaml` to change number of scenes, objects, render quality,
and all other parameters. Keep `num_scenes: 3-5` for local testing.

### 5. Visualize output
```bash
blenderproc vis hdf5 output/hdf5/0.hdf5
```

Or use the visualization script for the full 5-panel pipeline image:
```bash
python scripts/visualize.py output/hdf5/0.hdf5
python scripts/visualize.py output/hdf5/4.hdf5 --out docs/sample_outputs/scene4.png
```


## Project Structure

```
synth6d/
├── configs/
│   └── scene_config.yaml       # all tunable parameters
├── src/
│   ├── generate_dataset.py     # main generation loop
│   ├── scene_builder.py        # room, objects, camera, physics
│   └── utils.py                # config loading, path helpers, logging
├── scripts/
│   └── visualize.py            # pipeline visualization grid
├── assets/
│   ├── objects/                # YCB .obj files (gitignored)
│   └── textures/               # Poly Haven textures (gitignored)
├── output/                     # generated dataset (gitignored)
└── docs/
    └── sample_outputs/         # sample renders for README
```

---

## Implementation Notes

A few non-obvious things worth documenting for anyone building something similar:
- **Depth visualization requires clipping.**
BlenderProc depth maps store raw metric values in meters. The full range (0-5m+)
compresses nearby objects into a narrow band. Clipping to 0-2m gives useful contrast
for tabletop scenes. The raw values are preserved in the HDF5 file, clipping is only applied at visualization time.

- **Surface normals must be enabled after objects are loaded.**
Calling `enable_normals_output()` before any mesh exists in the scene results in
blank normal maps (all 0.5 values). The fix is to call renderer setup after the
first scene's objects and room are created. A `setup_renderer_done` flag ensures
it only runs once per session.

- **YCB poisson meshes are incomplete.**
The `poisson/` folder in the YCB processed download contains meshes reconstructed from RGB-D scans which often have holes on the bottom/back. Use the `google_16k/` meshes from the separate "16k laser scan" download, these are scanner-clean and complete. The pipeline checks for google_16k first, falls back to poisson.

- **Poly Haven zip structure.**
Downloaded texture zips extract with a `textures/` subfolder containing the actual
map files. Point `_apply_polyhaven_material()` at the `textures/` subfolder,
not the zip root.

---

## Roadmap

- [x] Project structure and config
- [x] Basic scene rendering: RGB, depth, normals, segmentation
- [ ] Domain randomization: lighting, textures, HDRI backgrounds
- [ ] BOP format output for pose annotations
- [ ] 1000+ scene dataset generation (Kaggle T4 GPU)
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
