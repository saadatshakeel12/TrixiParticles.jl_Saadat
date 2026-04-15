"""
Energy conservation validation for TLSPH cylinder drop simulation.
Plots KE, SE, PE, and total energy over time.

PE reference = floor surface (physically meaningful: PE=0 when cylinder rests on floor)
ΔE = step-to-step energy change (not cumulative from E0)
Time axis shifted so t=0 = first contact.

Usage:
    cd ~/SPH_Code/TrixiParticles.jl_Saadat/out
    python energy_validation.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import meshio
import glob
import os
import re
from collections import defaultdict

# ==========================================================================================
# Parameters
# ==========================================================================================
rho_mat = 1500.0   # kg/m³
E       = 1e5      # Pa (match the current implicit stamping setup)
nu      = 0.3
g_acc   = 9.81     # m/s²
ps      = 0.002   # m
dt_out  = 0.002    # s
prefix  = "molding_cfrp_3d_thermo_implicit_krylov_mod2"
folder  = "."

G      = E / (2 * (1 + nu))
K_bulk = E / (3 * (1 - 2*nu))
V_part = ps**3
m_part = rho_mat * V_part

# ==========================================================================================
# Discover files — structure_1 only, sorted by frame number
# ==========================================================================================
def extract_structure_info(fpath):
    match = re.search(r'_structure_(\d+)_(\d+)\.vtu$', fpath)
    if not match:
        return None
    sid = int(match.group(1))
    frame = int(match.group(2))
    return sid, frame

all_structure_files = glob.glob(os.path.join(folder, f'{prefix}_structure_*_*.vtu'))
files_by_sid = defaultdict(list)
for fpath in all_structure_files:
    info = extract_structure_info(fpath)
    if info is None:
        continue
    sid, _ = info
    files_by_sid[sid].append(fpath)

if not files_by_sid:
    print(f"No structure VTU files found with prefix '{prefix}' in {folder}")
    exit(1)

for sid in files_by_sid:
    files_by_sid[sid].sort(key=lambda p: extract_structure_info(p)[1])

structure_ids = sorted(files_by_sid.keys())
print(f"Found structure IDs: {structure_ids}")
for sid in structure_ids:
    print(f"  structure_{sid}: {len(files_by_sid[sid])} files")

# Detect cylinder as the structure with the smallest particle count.
counts = {}
z_mean0 = {}
for sid in structure_ids:
    m0 = meshio.read(files_by_sid[sid][0])
    counts[sid] = len(m0.points)
    z_mean0[sid] = float(m0.points[:, 2].mean())

cylinder_sid = min(structure_ids, key=lambda sid: counts[sid])

other_sids = [sid for sid in structure_ids if sid != cylinder_sid]
if len(other_sids) >= 2:
    floor_sid = min(other_sids, key=lambda sid: z_mean0[sid])
    mold_sid = max(other_sids, key=lambda sid: z_mean0[sid])
else:
    floor_sid = None
    mold_sid = None

struct_files = files_by_sid[cylinder_sid]
print(f"Using structure_{cylinder_sid} as cylinder (N={counts[cylinder_sid]})")
if floor_sid is not None and mold_sid is not None:
    print(f"Using structure_{floor_sid} as floor and structure_{mold_sid} as mold")
print(f"First cylinder file: {os.path.basename(struct_files[0])}")
print(f"Last  cylinder file: {os.path.basename(struct_files[-1])}")

# ==========================================================================================
# Reference state
# ==========================================================================================
m0    = meshio.read(struct_files[0])
pts0  = m0.points
N     = len(pts0)
M_tot = N * m_part

# Prefer particle spacing stored in VTU output so validation matches simulation settings.
if 'particle_spacing' in m0.point_data:
    ps_data = np.asarray(m0.point_data['particle_spacing']).reshape(-1)
    if ps_data.size > 0 and np.isfinite(ps_data[0]) and ps_data[0] > 0:
        ps = float(ps_data[0])
        V_part = ps**3
        m_part = rho_mat * V_part
        M_tot = N * m_part

print(f"\nN particles = {N}")
print(f"m_particle  = {m_part:.4e} kg")
print(f"M_total     = {M_tot*1e3:.4f} g")

# ==========================================================================================
# Floor z-position
# ==========================================================================================
if floor_sid is not None:
    floor0 = meshio.read(files_by_sid[floor_sid][0])
    z_floor = floor0.points[:, 2].max()
    print(f"Floor z     = {z_floor*1e3:.4f} mm  (from structure_{floor_sid})")
else:
    z_floor = pts0[:, 2].min() - 2.0 * ps
    print(f"Floor z     = {z_floor*1e3:.4f} mm  (estimated)")

# ==========================================================================================
# Per-timestep extraction
# ==========================================================================================
n_files      = len(struct_files)
times        = np.zeros(n_files)
KE           = np.zeros(n_files)
SE           = np.zeros(n_files)
PE           = np.zeros(n_files)
z_cyl_bottom = np.zeros(n_files)
z_cyl_top    = np.zeros(n_files)
z_cm_arr     = np.zeros(n_files)
T_mean       = np.zeros(n_files)
T_max        = np.zeros(n_files)
T_min        = np.zeros(n_files)
T_top_band   = np.zeros(n_files)

for i, fpath in enumerate(struct_files):
    m   = meshio.read(fpath)
    pts = m.points
    vel = m.point_data['velocity']
    frame = extract_structure_info(fpath)[1]

    z_cm_arr[i]     = pts[:, 2].mean()
    z_cyl_bottom[i] = pts[:, 2].min()
    z_cyl_top[i]    = pts[:, 2].max()
    times[i]        = frame * dt_out

    temp = m.point_data.get('temperature', None)
    if temp is not None:
        temp = np.asarray(temp).flatten()
        T_mean[i] = temp.mean()
        T_max[i] = temp.max()
        T_min[i] = temp.min()
        # Average temperature in the top 1*ps layer (closest to mold).
        top_mask = pts[:, 2] >= (z_cyl_top[i] - ps)
        T_top_band[i] = temp[top_mask].mean() if np.any(top_mask) else T_mean[i]
    else:
        T_mean[i] = np.nan
        T_max[i] = np.nan
        T_min[i] = np.nan
        T_top_band[i] = np.nan

    # Kinetic energy
    KE[i] = 0.5 * np.sum(m_part * np.sum(vel**2, axis=1))

    # Strain energy
    if 'von_mises_stress' in m.point_data and 'sigma_11' in m.point_data:
        vm      = m.point_data['von_mises_stress'].flatten()
        s11     = m.point_data['sigma_11'].flatten()
        s22     = m.point_data['sigma_22'].flatten()
        s33     = m.point_data['sigma_33'].flatten()
        sigma_h = (s11 + s22 + s33) / 3.0
        u_dev   = vm**2 / (6 * G)
        u_vol   = sigma_h**2 / (2 * K_bulk)
        SE[i]   = np.sum((u_dev + u_vol) * V_part)

    # PE relative to floor surface — positive above floor, zero at floor level
    # Using CoM height above floor
    PE[i] = M_tot * g_acc * (z_cm_arr[i] - z_floor)

# ==========================================================================================
# Contact detection
# ==========================================================================================
contact_tol = 0.05 * ps
contact_idx = None
mold_contact_idx = None
mold_min_gap = np.inf
mold_min_gap_idx = None
if floor_sid is not None:
    floor_files = files_by_sid[floor_sid]
    floor_files_by_frame = {extract_structure_info(f)[1]: f for f in floor_files}
    mold_files_by_frame = {}
    if mold_sid is not None:
        mold_files = files_by_sid[mold_sid]
        mold_files_by_frame = {extract_structure_info(f)[1]: f for f in mold_files}

    for i, fpath in enumerate(struct_files):
        frame = extract_structure_info(fpath)[1]

        if frame not in floor_files_by_frame:
            continue

        floor_mesh = meshio.read(floor_files_by_frame[frame])
        z_floor_frame = floor_mesh.points[:, 2].max()
        gap = z_cyl_bottom[i] - z_floor_frame
        if gap <= contact_tol:
            contact_idx = i

        if mold_sid is not None and frame in mold_files_by_frame:
            mold_mesh = meshio.read(mold_files_by_frame[frame])
            z_mold_frame = mold_mesh.points[:, 2].min()
            gap_mold = z_mold_frame - z_cyl_top[i]
            if gap_mold < mold_min_gap:
                mold_min_gap = gap_mold
                mold_min_gap_idx = i
            if mold_contact_idx is None and 0.0 <= gap_mold <= 0.25 * ps:
                mold_contact_idx = i

        if contact_idx is not None and (mold_sid is None or mold_contact_idx is not None):
            break
else:
    # Fallback if floor structure is unavailable.
    for i in range(n_files):
        if z_cyl_bottom[i] <= (z_floor + contact_tol):
            contact_idx = i
            break

if contact_idx is None:
    contact_idx = 0
    print(f"\nNo contact detected in available frames (tol={contact_tol*1e3:.4f} mm).")

times_contact = times - times[contact_idx]

print(f"\nContact at frame {contact_idx}, t={times[contact_idx]:.4f} s")
print(f"Cylinder bottom : {z_cyl_bottom[contact_idx]*1e3:.4f} mm")
print(f"Floor z         : {z_floor*1e3:.4f} mm")
print(f"PE at t=0       : {PE[0]*1e3:.4f} mJ  (should be positive — cylinder above floor)")
print(f"SE range        : min={SE.min()*1e3:.4e} mJ, max={SE.max()*1e3:.4e} mJ")

if mold_contact_idx is None:
    print(f"Mold contact    : not detected in available frames (thermal band=0..{0.25*ps*1e3:.4f} mm)")
    if mold_min_gap_idx is not None:
        print(f"Mold min gap    : {mold_min_gap*1e3:.4f} mm at frame {mold_min_gap_idx}"
              f" (t={times[mold_min_gap_idx]:.4f} s)")
else:
    print(f"Mold contact    : frame {mold_contact_idx}, t={times[mold_contact_idx]:.4f} s")
    if np.isfinite(T_mean[mold_contact_idx]):
        print(f"T@mold_contact  : mean={T_mean[mold_contact_idx]:.3f} K, "
              f"top_band={T_top_band[mold_contact_idx]:.3f} K, max={T_max[mold_contact_idx]:.3f} K")

if np.isfinite(T_mean[0]):
    print(f"Temperature span: mean {T_mean[0]:.3f} -> {T_mean[-1]:.3f} K, "
          f"top_band {T_top_band[0]:.3f} -> {T_top_band[-1]:.3f} K")

# ==========================================================================================
# Total energy
# ==========================================================================================
E_total = KE + SE + PE

# ==========================================================================================
# ΔE = step-to-step fractional change (not cumulative)
# ΔE[i] = (E[i] - E[i-1]) / |E[i-1]| * 100
# ΔE[0] = 0 by definition
# ==========================================================================================
dE_pct = np.zeros(n_files)
for i in range(1, n_files):
    if abs(E_total[i-1]) > 1e-15:
        dE_pct[i] = (E_total[i] - E_total[i-1]) / abs(E_total[i-1]) * 100
    else:
        dE_pct[i] = 0.0

# ==========================================================================================
# Print summary
# ==========================================================================================
print(f"\n{'t(s)':>8} {'t_c(s)':>8} {'KE(mJ)':>10} {'SE(mJ)':>10} "
      f"{'PE(mJ)':>10} {'Total(mJ)':>12} {'ΔE_step(%)':>12}")
print("-" * 82)
for i in range(n_files):
    print(f"{times[i]:>8.4f} {times_contact[i]:>8.4f} {KE[i]*1e3:>10.4f} "
        f"{SE[i]*1e3:>10.3e} {PE[i]*1e3:>10.4f} "
          f"{E_total[i]*1e3:>12.4f} {dE_pct[i]:>12.2f}")

print(f"\nContact time    : {times[contact_idx]:.4f} s")
print(f"E at t=0        : {E_total[0]*1e3:.6f} mJ")
print(f"E at contact    : {E_total[contact_idx]*1e3:.6f} mJ")
print(f"E at end        : {E_total[-1]*1e3:.6f} mJ")
print(f"Total ΔE (0→end): {(E_total[-1]-E_total[0])/abs(E_total[0])*100:.2f} %")

# ==========================================================================================
# Plots
# ==========================================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def shade(ax):
    ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.8, label='Contact')
    if times_contact[0] < 0:
        ax.axvspan(times_contact[0], 0, alpha=0.07, color='gray', label='Freefall')

# Plot 1 — Energy components
ax = axes[0]
ax.plot(times_contact, KE*1e3,      'r-',  lw=2, label='KE')
ax.plot(times_contact, SE*1e3,      'b-',  lw=2, label='SE')
ax.plot(times_contact, PE*1e3,      'g--', lw=2, label='PE (ref=floor)')
ax.plot(times_contact, E_total*1e3, 'k:',  lw=2, label='Total')
shade(ax)
ax.set_xlabel('Time since contact (s)')
ax.set_ylabel('Energy (mJ)')
ax.set_title('Energy vs Time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2 — Step-to-step energy change
ax = axes[1]
ax.plot(times_contact, dE_pct, color='purple', lw=2, label='ΔE step-to-step (%)')
ax.axhline(0,  color='black',  lw=1, ls='--')
ax.axhline(1,  color='orange', lw=1, ls=':', label='±1% band')
ax.axhline(-1, color='orange', lw=1, ls=':')
shade(ax)
ax.set_xlabel('Time since contact (s)')
ax.set_ylabel('Step energy change (%)')
ax.set_title('Step-to-Step Energy Change')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3 — KE ↔ SE post-contact
ax   = axes[2]
post = times_contact >= 0
if post.any():
    ax.plot(times_contact[post], KE[post]*1e3, 'r-', lw=2, label='KE')
    ax.plot(times_contact[post], SE[post]*1e3, 'b-', lw=2, label='SE')
ax.axvline(0, color='black', lw=1.5, ls='--', label='Contact')
ax.set_xlabel('Time since contact (s)')
ax.set_ylabel('Energy (mJ)')
ax.set_title('KE ↔ SE Exchange (Impact)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Energy Conservation — E={E:.2e} Pa  ν={nu}  ρ={rho_mat} kg/m³  '
    f'N={N} particles  M={M_tot*1e3:.3f}g  '
    f'[contact t={times[contact_idx]:.4f}s  PE ref=floor]',
    fontsize=10)
plt.tight_layout()

outpath = os.path.join(folder, 'energy_validation.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
plt.close()
