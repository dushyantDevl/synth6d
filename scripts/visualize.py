import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument("hdf5", help="path to .hdf5 file, e.g. output/hdf5/0.hdf5")
parser.add_argument("--out", default="docs/sample_outputs/pipeline.png", help="output image path")
args = parser.parse_args()

with h5py.File(args.hdf5) as f:
    colors       = np.array(f["colors"])
    depth        = np.array(f["depth"])
    normals      = np.array(f["normals"])
    instance_seg = np.array(f["instance_segmaps"])
    class_seg    = np.array(f["class_segmaps"])

depth_vis   = np.clip(depth, 0, 2.0)
normals_vis = np.clip((normals + 1.0) / 2.0, 0, 1)

fig = plt.figure(figsize=(30, 16), facecolor="#111111")
gs  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.15, wspace=0.05)

title_kw = dict(color="white", fontsize=14, fontweight="bold", pad=10)

ax_rgb = fig.add_subplot(gs[0, 0:2])
ax_rgb.imshow(colors)
ax_rgb.set_title("RGB Render", **title_kw)
ax_rgb.axis("off")

ax_norm = fig.add_subplot(gs[0, 2:4])
ax_norm.imshow(normals_vis)
ax_norm.set_title("Surface Normals", **title_kw)
ax_norm.axis("off")

ax_depth = fig.add_subplot(gs[0, 4:6])
ax_depth.imshow(depth_vis, cmap="plasma")
ax_depth.set_title("Depth (0-2m)", **title_kw)
ax_depth.axis("off")

ax_cls = fig.add_subplot(gs[1, 1:3])
ax_cls.imshow(class_seg, cmap="tab10")
ax_cls.set_title("Class Segmentation", **title_kw)
ax_cls.axis("off")

ax_inst = fig.add_subplot(gs[1, 3:5])
ax_inst.imshow(instance_seg, cmap="tab20")
ax_inst.set_title("Instance Segmentation", **title_kw)
ax_inst.axis("off")

fig.suptitle("synth6d: Synthetic 6DoF Pose Dataset Pipeline",
             color="white", fontsize=18, fontweight="bold", y=1.01)

plt.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="#111111")
print(f"Saved to {args.out}")
plt.show()