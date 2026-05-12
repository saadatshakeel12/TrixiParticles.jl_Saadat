#!/usr/bin/env python3
"""
Compare isothermal SPH compression simulation output against:
  1. Analytical J2 plasticity reference (same constitutive parameters)
  2. Typical experimental PP homopolymer data at 23°C from literature

Reads VTU files from out/ folder (cylinder = structure_1).
Produces:
  - stress_strain_comparison.png  (main comparison: σ₃₃ and VM vs references)
  - stress_strain_interior.png    (spatial averaging study: all / mid50 / mid30)

Particles are filtered by spatial region to account for the stress wave
propagation lag in 3D SPH:
  - "all":   all unclamped cylinder particles
  - "mid50": middle 50% of current z-height  (25th–75th percentile)
  - "mid30": middle 30% of current z-height  (35th–65th percentile)
  - "<90%":  exclude particles above 90% of max VM (contact outliers)
"""

import vtk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob, re

# =============================================================================
# 1. Read simulation VTU data
# =============================================================================
out_dir = "out"
pattern = os.path.join(out_dir, "stamping_isothermal_elastoplastic_0.0006_structure_1_*.vtu")
files = glob.glob(pattern)

if not files:
    raise FileNotFoundError(f"No VTU files matching {pattern}")

def iter_num(f):
    m = re.search(r'_1_(\d+)\.vtu$', f)
    return int(m.group(1)) if m else -1

# Use only files from the latest run (most recent iter 0 marks the cutoff)
files_with_mtime = [(f, os.path.getmtime(f)) for f in files]
files_with_mtime.sort(key=lambda x: x[1])
iter0_files = [f for f, _ in files_with_mtime if iter_num(f) == 0]
if iter0_files:
    cutoff_time = os.path.getmtime(iter0_files[-1]) - 10  # 10s tolerance
    files = [f for f, mt in files_with_mtime if mt >= cutoff_time]

files = sorted(files, key=iter_num)
print(f"Found {len(files)} cylinder VTU files (iters {iter_num(files[0])}..{iter_num(files[-1])})")

# Reference: initial height from iter 0
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(files[0])
reader.Update()
grid0 = reader.GetOutput()
pts0 = np.array([grid0.GetPoint(i) for i in range(grid0.GetNumberOfPoints())])
L0 = pts0[:, 2].max() - pts0[:, 2].min()
print(f"Initial cylinder height L0 = {L0*1e3:.3f} mm")

# Extract per-timestep data with spatial filtering
data = {k: [] for k in [
    'time', 'eng_strain',
    's33_all', 's33_mid50', 's33_mid30', 's33_filt90',
    'vm_all', 'vm_filt90',
]}

for f in files:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(f)
    reader.Update()
    grid = reader.GetOutput()
    pd = grid.GetPointData()

    t = grid.GetFieldData().GetArray("time").GetValue(0)
    pts = np.array([grid.GetPoint(i) for i in range(grid.GetNumberOfPoints())])
    z = pts[:, 2]
    L_cur = z.max() - z.min()
    eng_strain = (L0 - L_cur) / L0

    # Read arrays
    vm_arr = pd.GetArray("von_mises_stress")
    vm = np.array([vm_arr.GetValue(i) for i in range(vm_arr.GetNumberOfTuples())]) if vm_arr else np.zeros(grid.GetNumberOfPoints())
    s33_arr = pd.GetArray("sigma_33")
    s33 = np.array([s33_arr.GetValue(i) for i in range(s33_arr.GetNumberOfTuples())]) if s33_arr else np.zeros(grid.GetNumberOfPoints())
    clamp_arr = pd.GetArray("is_clamped")
    unclamped = (np.array([clamp_arr.GetValue(i) for i in range(clamp_arr.GetNumberOfTuples())]) == 0) if clamp_arr else np.ones(grid.GetNumberOfPoints(), dtype=bool)

    n_unc = unclamped.sum()
    z_unc = z[unclamped]
    vm_unc = vm[unclamped]
    s33_unc = np.abs(s33[unclamped])

    if n_unc == 0:
        data['time'].append(t)
        data['eng_strain'].append(eng_strain)
        for k in list(data.keys())[2:]:
            data[k].append(0.0)
        continue

    z_min_c, z_max_c = z_unc.min(), z_unc.max()
    z_range = z_max_c - z_min_c if z_max_c > z_min_c else 1.0

    # Spatial masks (on unclamped subset)
    mid50 = (z_unc > z_min_c + 0.25 * z_range) & (z_unc < z_max_c - 0.25 * z_range)
    mid30 = (z_unc > z_min_c + 0.35 * z_range) & (z_unc < z_max_c - 0.35 * z_range)

    # Stress outlier mask: exclude particles above 90% of max VM
    # vm_max = vm_unc.max() if vm_unc.max() > 0 else 1.0
    # filt90 = vm_unc < 0.9 * vm_max
    # Top 10% of |σ₃₃|: particles at or above the 90th percentile of axial stress
    top10 = s33_unc >= np.percentile(s33_unc, 90) if len(s33_unc) > 0 else np.ones(len(s33_unc), dtype=bool)
    # top10 = s33_unc >= np.percentile(s33_unc, 95) if len(s33_unc) > 0 else np.ones(len(s33_unc), dtype=bool)

    data['time'].append(t)
    data['eng_strain'].append(eng_strain)
    data['s33_all'].append(np.mean(s33_unc))
    data['s33_mid50'].append(np.mean(s33_unc[mid50]) if mid50.sum() > 0 else 0.0)
    data['s33_mid30'].append(np.mean(s33_unc[mid30]) if mid30.sum() > 0 else 0.0)
    data['s33_filt90'].append(np.mean(s33_unc[top10]) if top10.sum() > 0 else 0.0)
    data['vm_all'].append(np.mean(vm_unc))
    data['vm_filt90'].append(np.mean(vm_unc[top10]) if top10.sum() > 0 else 0.0)

# Convert to sorted numpy arrays
for k in data:
    data[k] = np.array(data[k])
idx = np.argsort(data['time'])
for k in data:
    data[k] = data[k][idx]

eps = data['eng_strain']
print(f"Time range: {data['time'][0]:.6f} to {data['time'][-1]:.6f} s")
print(f"Strain range: {eps[0]*100:.2f}% to {eps[-1]*100:.2f}%")
print(f"Max |σ₃₃| (all): {data['s33_all'].max()/1e6:.1f} MPa")
print(f"Max VM (all): {data['vm_all'].max()/1e6:.1f} MPa")

# =============================================================================
# 2. Analytical J2 plasticity reference (uniaxial compression)
# =============================================================================
E = 1.5e9       # Pa  — same as simulation
nu = 0.42
sigma_y = 3.5e7 # Pa (35 MPa)
H_mod = 8.0e7   # Pa (80 MPa) linear isotropic hardening
eps_y = sigma_y / E  # ~0.0233

eps_ref = np.linspace(0, 0.55, 500)
sigma_ref = np.where(eps_ref < eps_y, E * eps_ref, sigma_y + H_mod * (eps_ref - eps_y))

# =============================================================================
# 3. Experimental PP homopolymer data at 23°C
# =============================================================================
# Representative data from literature (Jerabek, Major, Lang 2010; Arruda & Boyce 1993)
eps_exp = np.array([0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04,
                    0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                    0.40, 0.45, 0.50])
sig_exp = np.array([0.0, 7.5, 15.0, 22.0, 28.0, 32.0, 35.0, 37.0,
                    38.0, 38.5, 39.0, 41.0, 44.0, 48.0, 53.0, 59.0,
                    66.0, 74.0, 83.0]) * 1e6  # Pa

# =============================================================================
# 4. Plot 1: Main stress–strain comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Engineering stress = Cauchy stress / (1 - engineering_strain)
sigma_eng_top10 = data['s33_filt90'] / np.where(np.abs(1 - eps) > 1e-6, 1 - eps, 1.0)
sigma_ref_eng = sigma_ref / np.where(np.abs(1 - eps_ref) > 1e-6, 1 - eps_ref, 1.0)

ax1 = axes[0]
ax1.plot(eps * 100, sigma_eng_top10 / 1e6, 'b-o', ms=4, lw=2,
         label='SPH eng. stress (top 10% |σ₃₃|, corrected)')
ax1.plot(eps_ref * 100, sigma_ref_eng / 1e6, 'r--', lw=2.5,
         label=f'Analytical J2 eng. stress (E={E/1e9}GPa, σy={sigma_y/1e6}MPa, H={H_mod/1e6}MPa)')
ax1.plot(eps_exp * 100, sig_exp / 1e6, 'k-^', ms=5, lw=1.5,
         label='Experimental PP-H 23°C (literature)')
ax1.set_xlabel('Engineering Strain [%]', fontsize=13)
ax1.set_ylabel('Stress [MPa]', fontsize=13)
ax1.set_title('Stress–Strain: SPH vs Analytical vs Experimental', fontsize=14)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
strain_max = max(eps.max() * 100, eps_ref.max() * 100, eps_exp.max() * 100)
strain_lim = min(12, strain_max + 1) if strain_max < 15 else strain_max + 2
ax1.set_xlim(0, 30)
ax1.set_xticks(np.arange(0, 30 + 1, 2))
stress_max = max(80, sigma_eng_top10.max() / 1e6 + 5, sigma_ref_eng.max() / 1e6 + 5, sig_exp.max() / 1e6 + 5)
ax1.set_ylim(0, stress_max)
ax1.set_yticks(np.arange(0, stress_max + 1, 10))

# Time history
ax2a = axes[1]
ax2a.plot(data['time'] * 1e3, eps * 100, '-', color='tab:blue', lw=2,
          label='Engineering strain')
ax2a.set_xlabel('Time [ms]', fontsize=13)
ax2a.set_ylabel('Engineering Strain [%]', fontsize=13, color='tab:blue')
ax2a.tick_params(axis='y', labelcolor='tab:blue')
ax2a.grid(True, alpha=0.3)

ax2b = ax2a.twinx()
ax2b.plot(data['time'] * 1e3, sigma_eng_top10 / 1e6, '-', color='tab:red', lw=2,
          label='Eng. stress (top 10%, corrected)')
ax2b.set_ylabel('Stress [MPa]', fontsize=13, color='tab:red')
ax2b.tick_params(axis='y', labelcolor='tab:red')

axes[1].set_title('Time History', fontsize=14)
lines1 = ax2a.get_lines()
lines2 = ax2b.get_lines()
ax2a.legend(lines1 + lines2, [l.get_label() for l in lines1 + lines2],
            loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig("stress_strain_comparison_0.0007.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: stress_strain_comparison.png")

# =============================================================================
# 5. Plot 2: Spatial averaging study (interior particles)
# =============================================================================
fig2, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.plot(eps * 100, data['s33_all'] / 1e6, 'b-o', ms=4, lw=2,
        label='SPH |σ₃₃| all particles')
ax.plot(eps * 100, data['s33_mid50'] / 1e6, 'g-s', ms=4, lw=2,
        label='SPH |σ₃₃| middle 50% of z')
ax.plot(eps * 100, data['s33_mid30'] / 1e6, 'm-D', ms=4, lw=2,
        label='SPH |σ₃₃| middle 30% of z')
ax.plot(eps_ref * 100, sigma_ref / 1e6, 'r--', lw=2.5,
        label=f'Analytical J2 (σy={sigma_y/1e6}MPa, H={H_mod/1e6}MPa)')
ax.plot(eps_exp * 100, sig_exp / 1e6, 'k-^', ms=5, lw=1.5,
        label='Experimental PP-H 23°C')

ax.set_xlabel('Engineering Strain [%]', fontsize=14)
ax.set_ylabel('|σ₃₃| Stress [MPa]', fontsize=14)
ax.set_title('Axial Stress: Effect of Spatial Averaging Region', fontsize=15)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
strain_max2 = max(eps.max() * 100, eps_ref.max() * 100, eps_exp.max() * 100)
strain_lim2 = min(12, strain_max2 + 1) if strain_max2 < 15 else strain_max2 + 2
ax.set_xlim(0, strain_lim2)
ax.set_xticks(np.arange(0, strain_lim2 + 1, 2))
stress_max2 = max(55, data['s33_all'].max() / 1e6 + 5, sigma_ref.max() / 1e6 + 5, sig_exp.max() / 1e6 + 5)
ax.set_ylim(0, stress_max2)
ax.set_yticks(np.arange(0, stress_max2 + 1, 10))

plt.tight_layout()
plt.savefig("stress_strain_interior.png", dpi=150, bbox_inches='tight')
print(f"Saved: stress_strain_interior.png")

# =============================================================================
# 6. Numerical comparison table
# =============================================================================
print("\n" + "=" * 85)
print("NUMERICAL COMPARISON (interpolated at key strain levels)")
print("=" * 85)
print(f"{'Strain[%]':>10} {'s33_all':>9} {'s33_mid50':>10} {'s33_mid30':>10} "
      f"{'VM_filt':>9} {'Analytical':>11} {'Expt':>8}")
print("-" * 85)

check_strains = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
for es in check_strains:
    es_frac = es / 100.0
    if es_frac > eps.max():
        continue
    s33_a = np.interp(es_frac, eps, data['s33_all']) / 1e6
    s33_m50 = np.interp(es_frac, eps, data['s33_mid50']) / 1e6
    s33_m30 = np.interp(es_frac, eps, data['s33_mid30']) / 1e6
    vm_f = np.interp(es_frac, eps, data['vm_filt90']) / 1e6
    ana = np.interp(es_frac, eps_ref, sigma_ref) / 1e6
    exp = np.interp(es_frac, eps_exp, sig_exp) / 1e6
    print(f"{es:>10} {s33_a:>9.1f} {s33_m50:>10.1f} {s33_m30:>10.1f} "
          f"{vm_f:>9.1f} {ana:>11.1f} {exp:>8.1f}")

print("=" * 85)

# Per-iteration detail
print(f"\n{'It':>4} {'t[ms]':>8} {'e[%]':>7} {'s33_all':>9} {'s33_m50':>9} "
      f"{'s33_m30':>9} {'VM_filt':>9} {'Analyt':>8} {'Expt':>8}")
print("-" * 78)
for i in range(len(eps)):
    e = eps[i]
    if abs(e) < 0.003 and i > 5 and i < len(eps) - 5:
        continue
    ana = (E * e if e < eps_y else sigma_y + H_mod * (e - eps_y)) / 1e6 if e > 0 else 0.0
    exp = np.interp(e, eps_exp, sig_exp) / 1e6 if e >= 0 else 0.0
    print(f"{i:>4} {data['time'][i]*1e3:>8.1f} {e*100:>7.2f} "
          f"{data['s33_all'][i]/1e6:>9.2f} {data['s33_mid50'][i]/1e6:>9.2f} "
          f"{data['s33_mid30'][i]/1e6:>9.2f} {data['vm_filt90'][i]/1e6:>9.2f} "
          f"{ana:>8.2f} {exp:>8.1f}")
print("=" * 78)


# Engineering stress is already computed above as sigma_eng_top10 and sigma_ref_eng

# =============================================================================
# 8. Mesh convergence: compare multiple particle spacings
# =============================================================================

def read_sph_stress_strain(spacing_label):
    """Read VTU files for a given spacing label and return (eng_strain, s33_filt90) arrays."""
    pat = os.path.join(out_dir, f"stamping_isothermal_elastoplastic_{spacing_label}_structure_1_*.vtu")
    flist = glob.glob(pat)
    if not flist:
        return None, None

    def _iter(f):
        m = re.search(r'_1_(\d+)\.vtu$', f)
        return int(m.group(1)) if m else -1

    # Keep only files from the latest run
    flist_mt = [(f, os.path.getmtime(f)) for f in flist]
    flist_mt.sort(key=lambda x: x[1])
    iter0 = [f for f, _ in flist_mt if _iter(f) == 0]
    if iter0:
        cutoff = os.path.getmtime(iter0[-1]) - 10
        flist = [f for f, mt in flist_mt if mt >= cutoff]
    flist = sorted(flist, key=_iter)

    # Initial height
    rd = vtk.vtkXMLUnstructuredGridReader()
    rd.SetFileName(flist[0]); rd.Update()
    g0 = rd.GetOutput()
    p0 = np.array([g0.GetPoint(i) for i in range(g0.GetNumberOfPoints())])
    h0 = p0[:, 2].max() - p0[:, 2].min()

    strains, stresses = [], []
    for f in flist:
        rd = vtk.vtkXMLUnstructuredGridReader()
        rd.SetFileName(f); rd.Update()
        g = rd.GetOutput(); pd = g.GetPointData()
        pts = np.array([g.GetPoint(i) for i in range(g.GetNumberOfPoints())])
        z = pts[:, 2]
        h_cur = z.max() - z.min()
        e = (h0 - h_cur) / h0

        vm_a = pd.GetArray("von_mises_stress")
        vm = np.array([vm_a.GetValue(i) for i in range(vm_a.GetNumberOfTuples())]) if vm_a else np.zeros(g.GetNumberOfPoints())
        s33_a = pd.GetArray("sigma_33")
        s33 = np.array([s33_a.GetValue(i) for i in range(s33_a.GetNumberOfTuples())]) if s33_a else np.zeros(g.GetNumberOfPoints())
        cl_a = pd.GetArray("is_clamped")
        unc = (np.array([cl_a.GetValue(i) for i in range(cl_a.GetNumberOfTuples())]) == 0) if cl_a else np.ones(g.GetNumberOfPoints(), dtype=bool)

        if unc.sum() == 0:
            strains.append(e); stresses.append(0.0)
            continue

        vm_unc = vm[unc]
        s33_unc = np.abs(s33[unc])
        # Top 10% of |σ₃₃|
        top10 = s33_unc >= np.percentile(s33_unc, 90) if len(s33_unc) > 0 else np.ones(len(s33_unc), dtype=bool)
        sigma_top10_true = np.mean(s33_unc[top10]) if top10.sum() > 0 else np.mean(s33_unc)
        # Convert to engineering stress: sigma_eng = sigma_true / (1 - strain)
        sigma_top10_eng = sigma_top10_true / (1 - e) if e < 1.0 else sigma_top10_true
        strains.append(e)
        stresses.append(sigma_top10_eng)

    strains = np.array(strains); stresses = np.array(stresses)
    idx = np.argsort(strains)
    return strains[idx], stresses[idx]


# Define spacings to compare (label used in filename, display name)
spacing_configs = [
    ("0.0015",   "Δx = 1.5 mm"),
    ("0.001",    "Δx = 1.0 mm"),   
    ("0.0007_2", "Δx = 0.7 mm"),
    ("0.0006", "Δx = 0.6 mm"),
    #("0.0005",   "Δx = 0.5 mm"),
]

colors = ['tab:green', 'tab:blue', 'tab:purple', 'tab:cyan']
markers = ['s', 'D', 'o', '^']

fig_conv, ax_conv = plt.subplots(1, 1, figsize=(10, 7))

for i, (sp_label, sp_name) in enumerate(spacing_configs):
    e_sp, s_sp = read_sph_stress_strain(sp_label)
    if e_sp is None:
        print(f"  Skipping {sp_label}: no VTU files found")
        continue
    print(f"  {sp_label}: {len(e_sp)} frames, strain {e_sp[0]*100:.1f}–{e_sp[-1]*100:.1f}%")
    ax_conv.plot(e_sp * 100, s_sp / 1e6, f'-{markers[i]}', color=colors[i],
                 ms=4, lw=2, label=sp_name)

# Add analytical and experimental references
ax_conv.plot(eps_ref * 100, sigma_ref / 1e6, 'r--', lw=2.5,
             label=f'Analytical J2 (σy={sigma_y/1e6:.0f} MPa, H={H_mod/1e6:.0f} MPa)')
ax_conv.plot(eps_exp * 100, sig_exp / 1e6, 'k-^', ms=5, lw=1.5,
             label='Experimental PP-H 23°C')

ax_conv.set_xlabel('Engineering Strain [%]', fontsize=14)
ax_conv.set_ylabel('|σ₃₃| Stress [MPa]', fontsize=14)
ax_conv.set_title('Mesh Convergence: SPH Stress–Strain at Different Particle Spacings', fontsize=14)
ax_conv.legend(fontsize=10, loc='upper left')
ax_conv.grid(True, alpha=0.3)
ax_conv.set_xlim(0, 30)
ax_conv.set_xticks(np.arange(0, 31, 2))
ax_conv.set_ylim(0, stress_max)
ax_conv.set_yticks(np.arange(0, stress_max + 1, 10))

plt.tight_layout()
plt.savefig("mesh_convergence_stress_strain.png", dpi=150, bbox_inches='tight')
print(f"Saved: mesh_convergence_stress_strain.png")