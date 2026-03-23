"""
Visualize TrixiParticles VTU output files without a GUI.
Usage: python visualize_vtu.py --folder /path/to/vtu/files --prefix cylinder_drop

MANUAL OVERRIDES (edit below or pass as CLI args):
  --x_min, --x_max       : X axis limits
  --y_min, --y_max       : Y axis limits
  --z_min, --z_max       : Z axis limits
  --c_min, --c_max       : Colorbar limits
  --frame_start          : First frame index to render (0-based)
  --frame_end            : Last frame index to render (-1 = all)
  --frame_step           : Render every Nth frame (1 = all, 2 = every other, etc.)
  --elev                 : 3D view elevation angle (degrees)
  --azim                 : 3D view azimuth angle (degrees)
  --color_field          : Field to color by: auto, z_position, pressure, velocity, von_mises_stress
"""

import meshio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import argparse

# ==========================================================================================
# *** MANUAL OVERRIDES — edit these directly if you don't want to use CLI args ***
# Set to None to auto-compute from data
# ==========================================================================================

MANUAL = dict(
    x_min        = -0.02,   # e.g. -0.01
    x_max        = 0.02,   # e.g.  0.01
    y_min        = -0.02,
    y_max        = 0.02,
    z_min        = -0.01,
    z_max        = 0.01,
    c_min        = None,   # colorbar min
    c_max        = None,   # colorbar max
    frame_start  = 0,      # first frame to render (0-based index)
    frame_end    = -1,     # last frame to render (-1 = all)
    frame_step   = 1,      # render every Nth frame
    elev         = 25,     # 3D elevation angle
    azim         = 45,     # 3D azimuth angle
    color_field  = 'auto', # 'auto', 'z_position', 'pressure', 'velocity', 'von_mises_stress'
    marker_size_structure = 3,
    marker_size_boundary  = 1,
    dpi          = 130,
)

# ==========================================================================================
# CLI arguments (override MANUAL dict above)
# ==========================================================================================

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
parser.add_argument('--folder',       type=str,   default='.')
parser.add_argument('--prefix',       type=str,   default='cylinder_drop')
parser.add_argument('--x_min',        type=float, default=None)
parser.add_argument('--x_max',        type=float, default=None)
parser.add_argument('--y_min',        type=float, default=None)
parser.add_argument('--y_max',        type=float, default=None)
parser.add_argument('--z_min',        type=float, default=None)
parser.add_argument('--z_max',        type=float, default=None)
parser.add_argument('--c_min',        type=float, default=None)
parser.add_argument('--c_max',        type=float, default=None)
parser.add_argument('--frame_start',  type=int,   default=None)
parser.add_argument('--frame_end',    type=int,   default=None)
parser.add_argument('--frame_step',   type=int,   default=None)
parser.add_argument('--elev',         type=float, default=None)
parser.add_argument('--azim',         type=float, default=None)
parser.add_argument('--color_field',  type=str,   default=None,
                    choices=['auto','z_position','pressure','velocity','von_mises_stress'])
parser.add_argument('--dpi',          type=int,   default=None)
args = parser.parse_args()

# CLI args override MANUAL dict
for key in MANUAL:
    cli_val = getattr(args, key, None)
    if cli_val is not None:
        MANUAL[key] = cli_val

folder  = args.folder
prefix  = args.prefix
out_dir = os.path.join(folder, 'frames')
os.makedirs(out_dir, exist_ok=True)

# ==========================================================================================
# Find files
# ==========================================================================================

structure_files = sorted(glob.glob(os.path.join(folder, f'{prefix}_structure_*.vtu')))
boundary_files  = sorted(glob.glob(os.path.join(folder, f'{prefix}_boundary_*.vtu')))

if not structure_files:
    print(f"No structure VTU files found in {folder} with prefix '{prefix}'")
    print("Available VTU files:")
    for f in glob.glob(os.path.join(folder, '*.vtu')):
        print(f"  {f}")
    exit(1)

print(f"Found {len(structure_files)} structure files, {len(boundary_files)} boundary files")

# Apply frame selection
frame_start = MANUAL['frame_start']
frame_end   = MANUAL['frame_end'] if MANUAL['frame_end'] >= 0 else len(structure_files)
frame_step  = MANUAL['frame_step']

structure_files = structure_files[frame_start:frame_end:frame_step]
print(f"Rendering {len(structure_files)} frames "
      f"(start={frame_start}, end={frame_end}, step={frame_step})")

def get_index(filepath):
    parts = os.path.basename(filepath).replace('.vtu', '').split('_')
    try:
        return int(parts[-1])
    except ValueError:
        return 0

# ==========================================================================================
# Load boundary once
# ==========================================================================================

boundary_pts = None
if boundary_files:
    boundary_pts = meshio.read(boundary_files[0]).points

# ==========================================================================================
# Auto-compute bounds if not manually set
# ==========================================================================================

need_auto = any(MANUAL[k] is None for k in ['x_min','x_max','y_min','y_max','z_min','z_max'])
need_color_auto = MANUAL['c_min'] is None or MANUAL['c_max'] is None
need_field_auto = MANUAL['color_field'] == 'auto'

x_min_d, x_max_d = np.inf, -np.inf
y_min_d, y_max_d = np.inf, -np.inf
z_min_d, z_max_d = np.inf, -np.inf
c_min_d, c_max_d = np.inf, -np.inf
detected_field    = None
color_label       = 'Z position (m)'

if need_auto or need_color_auto or need_field_auto:
    print("Scanning data for auto bounds...")

    if boundary_pts is not None:
        x_min_d = min(x_min_d, boundary_pts[:, 0].min())
        x_max_d = max(x_max_d, boundary_pts[:, 0].max())
        y_min_d = min(y_min_d, boundary_pts[:, 1].min())
        y_max_d = max(y_max_d, boundary_pts[:, 1].max())
        z_min_d = min(z_min_d, boundary_pts[:, 2].min())
        z_max_d = max(z_max_d, boundary_pts[:, 2].max())

    for f in structure_files:
        mesh = meshio.read(f)
        pts  = mesh.points
        x_min_d = min(x_min_d, pts[:, 0].min()); x_max_d = max(x_max_d, pts[:, 0].max())
        y_min_d = min(y_min_d, pts[:, 1].min()); y_max_d = max(y_max_d, pts[:, 1].max())
        z_min_d = min(z_min_d, pts[:, 2].min()); z_max_d = max(z_max_d, pts[:, 2].max())

        if detected_field is None and need_field_auto:
            if 'von_mises_stress' in mesh.point_data:
                detected_field = 'von_mises_stress'; color_label = 'Von Mises Stress (Pa)'
            elif 'pressure' in mesh.point_data:
                detected_field = 'pressure';         color_label = 'Pressure (Pa)'
            elif 'velocity' in mesh.point_data:
                detected_field = 'velocity';         color_label = 'Velocity magnitude (m/s)'
            else:
                detected_field = 'z_position';       color_label = 'Z position (m)'

        if need_color_auto:
            cf = detected_field or MANUAL['color_field']
            if cf == 'velocity' and 'velocity' in mesh.point_data:
                cdata = np.linalg.norm(mesh.point_data['velocity'], axis=1)
            elif cf and cf != 'z_position' and cf in mesh.point_data:
                cdata = mesh.point_data[cf]
            else:
                cdata = pts[:, 2]
            c_min_d = min(c_min_d, cdata.min())
            c_max_d = max(c_max_d, cdata.max())

# Resolve final bounds — manual overrides auto
def resolve(manual_val, auto_val):
    return manual_val if manual_val is not None else auto_val

x_min_f = resolve(MANUAL['x_min'], x_min_d)
x_max_f = resolve(MANUAL['x_max'], x_max_d)
y_min_f = resolve(MANUAL['y_min'], y_min_d)
y_max_f = resolve(MANUAL['y_max'], y_max_d)
z_min_f = resolve(MANUAL['z_min'], z_min_d)
z_max_f = resolve(MANUAL['z_max'], z_max_d)
c_min_f = resolve(MANUAL['c_min'], c_min_d)
c_max_f = resolve(MANUAL['c_max'], c_max_d)

color_field = MANUAL['color_field'] if MANUAL['color_field'] != 'auto' else (detected_field or 'z_position')

# Equal aspect ratio — pad all axes to same range
max_range = max(x_max_f - x_min_f, y_max_f - y_min_f, z_max_f - z_min_f)
pad = max_range * 0.06
x_mid, y_mid, z_mid = (x_max_f+x_min_f)/2, (y_max_f+y_min_f)/2, (z_max_f+z_min_f)/2
x_lo, x_hi = x_mid - max_range/2 - pad, x_mid + max_range/2 + pad
y_lo, y_hi = y_mid - max_range/2 - pad, y_mid + max_range/2 + pad
z_lo, z_hi = z_mid - max_range/2 - pad, z_mid + max_range/2 + pad

# Allow manual override of final limits too
if MANUAL['x_min'] is not None: x_lo = MANUAL['x_min']
if MANUAL['x_max'] is not None: x_hi = MANUAL['x_max']
if MANUAL['y_min'] is not None: y_lo = MANUAL['y_min']
if MANUAL['y_max'] is not None: y_hi = MANUAL['y_max']
if MANUAL['z_min'] is not None: z_lo = MANUAL['z_min']
if MANUAL['z_max'] is not None: z_hi = MANUAL['z_max']

print(f"Axis limits — X:[{x_lo:.4f},{x_hi:.4f}]  Y:[{y_lo:.4f},{y_hi:.4f}]  Z:[{z_lo:.4f},{z_hi:.4f}]")
print(f"Color: {color_field}  range [{c_min_f:.3e}, {c_max_f:.3e}]")

# Color label map
color_label_map = {
    'von_mises_stress': 'Von Mises Stress (Pa)',
    'pressure':         'Pressure (Pa)',
    'velocity':         'Velocity magnitude (m/s)',
    'z_position':       'Z position (m)',
}
color_label = color_label_map.get(color_field, color_field)

# ==========================================================================================
# Render frames
# ==========================================================================================

print(f"\nRendering {len(structure_files)} frames...")

for i, struct_file in enumerate(structure_files):
    mesh = meshio.read(struct_file)
    pts  = mesh.points

    if color_field == 'velocity' and 'velocity' in mesh.point_data:
        cdata = np.linalg.norm(mesh.point_data['velocity'], axis=1)
    elif color_field and color_field != 'z_position' and color_field in mesh.point_data:
        cdata = mesh.point_data[color_field]
    else:
        cdata = pts[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    if boundary_pts is not None:
        ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], boundary_pts[:, 2],
                   c='#999999', s=MANUAL['marker_size_boundary'],
                   alpha=0.2, label='Floor', rasterized=True)

    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=cdata, cmap='plasma',
                    vmin=c_min_f, vmax=c_max_f,
                    s=MANUAL['marker_size_structure'],
                    alpha=0.9, label='Cylinder', rasterized=True)

    cbar = plt.colorbar(sc, ax=ax, label=color_label, shrink=0.5, pad=0.08)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X (m)', fontsize=9, labelpad=8)
    ax.set_ylabel('Y (m)', fontsize=9, labelpad=8)
    ax.set_zlabel('Z (m)', fontsize=9, labelpad=8)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.zaxis.set_tick_params(labelsize=7)

    idx = get_index(struct_file)
    ax.set_title(f'Cylinder Drop  |  Step {idx:05d}  ({i+1}/{len(structure_files)})',
                 fontsize=10, pad=10)
    ax.legend(loc='upper left', markerscale=4, fontsize=8, framealpha=0.7)
    ax.view_init(elev=MANUAL['elev'], azim=MANUAL['azim'])

    frame_path = os.path.join(out_dir, f'frame_{idx:05d}.png')
    plt.savefig(frame_path, dpi=MANUAL['dpi'], bbox_inches='tight')
    plt.close(fig)

    if i % 5 == 0:
        print(f"  [{i+1:4d}/{len(structure_files)}] {os.path.basename(frame_path)}")

print(f"\nAll frames saved to: {out_dir}/")

# ==========================================================================================
# Center-of-mass trajectory
# ==========================================================================================

print("Plotting center-of-mass trajectory...")

steps = []
x_cms, y_cms, z_cms = [], [], []

all_structure = sorted(glob.glob(os.path.join(folder, f'{prefix}_structure_*.vtu')))
for struct_file in all_structure:
    mesh = meshio.read(struct_file)
    pts  = mesh.points
    x_cms.append(pts[:, 0].mean())
    y_cms.append(pts[:, 1].mean())
    z_cms.append(pts[:, 2].mean())
    steps.append(get_index(struct_file))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax_, data, color, label in zip(axes,
                                    [x_cms, y_cms, z_cms],
                                    ['tab:red', 'tab:green', 'tab:blue'],
                                    ['X (m)', 'Y (m)', 'Z (m) — indentation']):
    ax_.plot(steps, data, color=color, linewidth=1.5)
    ax_.set_xlabel('Output step', fontsize=9)
    ax_.set_ylabel(label, fontsize=9)
    ax_.set_title(f'CoM {label[0]}', fontsize=10)
    ax_.grid(True, alpha=0.3)
    ax_.tick_params(labelsize=8)

plt.suptitle('Cylinder Center of Mass Trajectory', fontsize=12)
plt.tight_layout()
com_path = os.path.join(folder, 'center_of_mass.png')
plt.savefig(com_path, dpi=150)
plt.close()
print(f"Saved: {com_path}")

# ==========================================================================================
# Animation
# ==========================================================================================

ffmpeg_cmd = (
    f'ffmpeg -y -framerate 10 '
    f'-pattern_type glob -i "{out_dir}/frame_*.png" '
    f'-c:v libx264 -pix_fmt yuv420p '
    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
    f'{folder}/cylinder_drop_animation.mp4'
)

print("\nAttempting to create animation with ffmpeg...")
ret = os.system(ffmpeg_cmd)
if ret == 0:
    print(f"Animation saved: {folder}/cylinder_drop_animation.mp4")
else:
    print("ffmpeg not available. Install with: sudo apt-get install ffmpeg")
    print(f"Manual command:\n  {ffmpeg_cmd}")

print("\nDone!")