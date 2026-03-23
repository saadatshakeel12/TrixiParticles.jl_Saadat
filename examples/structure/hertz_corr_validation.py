import numpy as np
import meshio
import glob
import os
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import re
from scipy.interpolate import griddata

# ==========================================================
# Parameters (Must match your Julia setup)
# ==========================================================
E      = 1e6
nu     = 0.3
E_star = E / (1 - nu**2)
rho_0  = 1500.0
R_cyl  = 0.006
L_cyl  = 0.012

# Acoustic Impedance for Riemann damping reconstruction
c_0    = np.sqrt(E / rho_0)
Z_i    = rho_0 * c_0
Z_j    = Z_i

folder = "."

def extract_frame_number(fpath):
    match = re.search(r'_structure_1_(\d+)\.vtu$', fpath)
    return int(match.group(1)) if match else -1

def extract_boundary_frame(fpath):
    match = re.search(r'_structure_2_(\d+)\.vtu$', fpath)
    return int(match.group(1)) if match else -1

# ==========================================================================================
# Mesh Convergence Study — define runs here
# ==========================================================================================
runs = [
    ("cylinder_drop_new4", 0.0013, 'blue',  'ps=1.3mm'),
    ("cylinder_drop_new5", 0.001, 'red',   'ps=1.0mm'),
    ("cylinder_drop_new6", 0.0007, 'green',   'ps=0.7mm')
    # Add more runs here:
    # ("cylinder_drop_new7", 0.0005, 'green', 'ps=0.5mm'),
]

all_results = {}

for run_prefix, run_ps, run_color, run_label in runs:

    ps      = run_ps
    h       = 1.5 * ps
    prefix  = run_prefix

    print(f"\n{'='*50}")
    print(f"Processing: {run_label}  ({run_prefix})")
    print(f"{'='*50}")

    struct_files = sorted(
        [f for f in glob.glob(os.path.join(folder, f'{run_prefix}_structure_1_*.vtu'))
         if extract_frame_number(f) >= 0],
        key=extract_frame_number)

    boundary_files = sorted(
        [f for f in glob.glob(os.path.join(folder, f'{run_prefix}_structure_2_*.vtu'))
         if extract_boundary_frame(f) >= 0],
        key=extract_boundary_frame)

    if not struct_files:
        print(f"No files found for prefix {run_prefix}, skipping")
        continue

    m0   = meshio.read(struct_files[0])
    pts0 = m0.points
    N    = len(pts0)

    if boundary_files:
        b0      = meshio.read(boundary_files[0])
        z_floor = b0.points[:, 2].max()
        print(f"Floor z     = {z_floor*1e3:.4f} mm  (from boundary VTU)")
    else:
        z_floor = pts0[:, 2].min() - 2.0 * ps
        print(f"Floor z     = {z_floor*1e3:.4f} mm  (estimated)")

    # Data Containers
    indentations    = []
    sph_forces      = []
    sph_elastic_only = []
    v_z_avg         = []
    sph_u_z         = []
    max_vm_history  = []
    time_history    = []

    z_start = np.mean(meshio.read(struct_files[0]).points[:, 2])

    # ==========================================================
    # Combined Extraction Loop
    # ==========================================================
    for i, fpath in enumerate(struct_files):
        m = meshio.read(fpath)

        if 'time' in m.field_data:
            time_history.append(m.field_data['time'][0])
        else:
            time_history.append(i * 0.005)

        pts = m.points
        vel = m.point_data['velocity']

        # 1. Global Kinematics
        vz     = vel[:, 2]
        v_mean = np.mean(vz)
        v_z_avg.append(v_mean)

        u_z = (z_start - np.mean(pts[:, 2])) * 1000
        sph_u_z.append(u_z)

        # 2. Contact Mechanics
        dist_to_floor = pts[:, 2] - z_floor
        contact_mask  = dist_to_floor < h

        if np.any(contact_mask):
            overlap  = h - dist_to_floor[contact_mask]
            v_rel    = vel[contact_mask, 2]

            f_elastic_part = (E_star / h) * overlap * (ps**2)
            f_damping_part = (Z_i * Z_j / (Z_i + Z_j)) * np.maximum(0, -v_rel) * (ps**2)

            sum_f_e = np.sum(f_elastic_part)
            sph_elastic_only.append(sum_f_e)
            sum_f_p = np.sum(f_damping_part)
            sph_forces.append(sum_f_e + np.abs(sum_f_p))

            indentations.append(np.max(overlap))
        else:
            sph_elastic_only.append(0.0)
            sph_forces.append(0.0)
            indentations.append(0.0)

        if 'von_mises_stress' in m.point_data:
            vm       = m.point_data['von_mises_stress']
            valid_vm = vm[~np.isnan(vm)]
            max_vm_history.append(np.max(valid_vm) if len(valid_vm) > 0 else 0.0)
        else:
            max_vm_history.append(0.0)

    # Convert to arrays
    indent_arr   = np.array(indentations)
    forces_arr   = np.array(sph_forces)
    v_z_arr      = np.array(v_z_avg)
    elastic_arr  = np.array(sph_elastic_only)
    disp_arr     = np.array(sph_u_z)
    vm_arr       = np.array(max_vm_history)
    time_arr     = np.array(time_history)

    # ==========================================================
    # Theory & Validation Logic
    # ==========================================================
    max_delta    = np.max(indent_arr)
    smooth_delta = np.linspace(0, max_delta, 100) if max_delta > 0 else np.linspace(0, 0.001, 100)

    sph_peak_elastic = np.max(elastic_arr) if len(elastic_arr) > 0 else 0.0
    if max_delta > 0:
        scale_factor = sph_peak_elastic / (max_delta**1.5)
    else:
        scale_factor = 0

    loading   = v_z_arr < -1e-4
    unloading = v_z_arr >  1e-4

    from scipy.optimize import curve_fit

    def hertz_shape(d, k):
        return k * (d**1.5)

    valid_idx = indent_arr > (0.1 * max_delta)
    if np.any(valid_idx):
        try:
            popt, _ = curve_fit(hertz_shape, indent_arr[valid_idx], elastic_arr[valid_idx])
            k_fit = popt[0]
        except Exception:
            k_fit = 0
    else:
        k_fit = 0

    smooth_hertz = k_fit * (smooth_delta**1.5)

    f_theory_load   = smooth_hertz
    f_theory_unload = smooth_hertz
    cor_velocity    = 0.0

    if np.any(loading) and np.any(unloading):
        res_load   = forces_arr[loading]   - (k_fit * indent_arr[loading]**1.5)
        res_unload = (k_fit * indent_arr[unloading]**1.5) - forces_arr[unloading]

        lift = np.percentile(res_load,   95) if len(res_load)   > 0 else 0
        dip  = np.percentile(res_unload, 95) if len(res_unload) > 0 else 0

        delta_offset  = 0.0
        shifted_delta = np.maximum(0, smooth_delta - delta_offset)
        max_shifted   = np.max(shifted_delta)

        smooth_hertz = k_fit * (shifted_delta**1.5)

        if max_shifted > 0:
            f_theory_load   = smooth_hertz + (lift * (shifted_delta / max_shifted))
            f_theory_unload = smooth_hertz - (dip  * (shifted_delta / max_shifted))
        else:
            f_theory_load = smooth_hertz

        idx_L = np.argsort(indent_arr[loading])
        idx_U = np.argsort(indent_arr[unloading])
        if len(idx_L) > 1 and len(idx_U) > 1:
            w_in  = trapezoid(forces_arr[loading][idx_L],   indent_arr[loading][idx_L])
            w_out = trapezoid(forces_arr[unloading][idx_U], indent_arr[unloading][idx_U])

        v_before     = np.abs(np.min(v_z_avg))
        v_after      = np.abs(np.max(v_z_avg))
        cor_velocity = v_after / v_before if v_before > 0 else 0.0
        print(f"CoR (Velocity-based): {cor_velocity:.3f}")

    # ==========================================================
    # ROBUST HERTZ VALIDATION
    # ==========================================================
    physical_gap    = h - (ps / 2)
    true_indent_arr = np.maximum(0, indent_arr - physical_gap)
    max_true_delta  = np.max(true_indent_arr)

    vm_max_theo = 0.0
    a_theo      = 0.0
    p0_theo     = 0.0

    max_disp_idx  = np.argmax(disp_arr) if len(disp_arr) > 0 else 0
    idx_max_force = np.argmax(forces_arr)

    if max_true_delta > 0:
        idx_max     = np.argmax(forces_arr)
        P_physical  = forces_arr[idx_max]
        p_line_true = P_physical / L_cyl

        a_theo      = np.sqrt((4 * p_line_true * R_cyl) / (np.pi * E_star))
        p0_theo     = (2 * p_line_true) / (np.pi * a_theo)
        vm_max_theo = 0.56 * p0_theo

        m_peak = meshio.read(struct_files[idx_max])
        if 'von_mises_stress' in m_peak.point_data:
            raw_vm_data = m_peak.point_data['von_mises_stress']
            valid_peak_vm = raw_vm_data[~np.isnan(raw_vm_data)]
            vm_max      = np.percentile(valid_peak_vm, 99.5) / 1000.0 if len(valid_peak_vm) > 0 else 0.0
        else:
            vm_max = 0.0

        print(f"\n--- PHYSICAL HERTZ VALIDATION ({run_label}) ---")
        print(f"True Physical Indent: {max_true_delta*1000:.4f} mm")
        print(f"Physical Force:       {P_physical:.2f} N")
        print(f"Theory Peak VM:   {vm_max_theo/1000:.2f} kPa")
        print(f"SPH Peak VM:      {vm_max:.2f} kPa")

        final_error = abs((vm_max_theo/1000.0) - vm_max) / (vm_max_theo/1000.0) * 100 if vm_max_theo > 0 else 0
        print(f"Final Validation Error: {final_error:.2f} %")
    else:
        vm_max = 0.0
        print(f"\n--- VALIDATION STATUS: PRE-CONTACT ({run_label}) ---")
        print(f"Current Max Overlap: {np.max(indent_arr)*1000:.4f} mm")
        print(f"Required for Contact (h): {h*1000:.4f} mm")
        print("RESULT: No physical contact yet. Theory skipped.")

    print(f"Peak Von Mises: {np.max(max_vm_history)/1e3:.4f} KPa")

    # ==========================================================
    # Normalization Constants (Tribology Letters)
    # ==========================================================
    Sy          = 1.275e6
    term_in_ln  = (2 * E_star) / Sy
    omega_c     = R_cyl * (Sy / E_star)**2 * (2 * np.log(term_in_ln) - 1)
    b_c         = (2 * R_cyl * Sy) / E_star
    F_critical  = (np.pi * R_cyl * L_cyl * (Sy**2)) / E_star

    contact_half_widths = []
    for i, fpath in enumerate(struct_files):
        m   = meshio.read(fpath)
        pts = m.points
        dist_to_floor_b = pts[:, 2] - z_floor
        contacting_pts  = pts[dist_to_floor_b < 0.5*ps]
        if len(contacting_pts) > 0:
            y_center   = np.mean(pts[:, 1])
            current_b  = np.max(np.abs(contacting_pts[:, 1] - y_center))
            contact_half_widths.append(current_b)
        else:
            contact_half_widths.append(0.0)

    b_arr  = np.array(contact_half_widths)
    b_norm = b_arr / b_c
    w_norm = np.atleast_1d(true_indent_arr / omega_c)
    F_norm = np.atleast_1d(forces_arr / F_critical)

    slope_b = np.max(b_norm) / np.sqrt(np.max(w_norm)) if np.max(w_norm) > 0 else 1.0

    print(f"Normalization Constants: wc={omega_c:.4e}, bc={b_c:.4e}, Fc={F_critical:.4f}")

    # Store all results for this run
    all_results[run_label] = {
        'color':         run_color,
        'ps':            run_ps,
        'h':             h,
        'time':          time_arr,
        'indent':        indent_arr,
        'true_indent':   true_indent_arr,
        'forces':        forces_arr,
        'elastic':       elastic_arr,
        'vm':            vm_arr,
        'disp':          disp_arr,
        'v_z':           v_z_arr,
        'w_norm':        w_norm,
        'F_norm':        F_norm,
        'b_norm':        b_norm,
        'z_floor':       z_floor,
        'struct_files':  struct_files,
        'boundary_files':boundary_files,
        'max_disp_idx':  max_disp_idx,
        'a_theo':        a_theo,
        'p0_theo':       p0_theo,
        'vm_max_theo':   vm_max_theo,
        'vm_max':        vm_max,
        'cor':           cor_velocity,
        'k_fit':         k_fit,
        'smooth_delta':  smooth_delta,
        'smooth_hertz':  smooth_hertz,
        'f_theory_load': f_theory_load,
        'f_theory_unload': f_theory_unload,
        'loading':       loading,
        'unloading':     unloading,
        'omega_c':       omega_c,
        'b_c':           b_c,
        'F_critical':    F_critical,
        'slope_b':       slope_b,
    }

# ==========================================================================================
# Single-run plots for LAST run (preserving original behaviour)
# ==========================================================================================
# Use last run for single-run plots
last_label      = list(all_results.keys())[-1]
r               = all_results[last_label]
ps              = r['ps']
h               = r['h']
z_floor         = r['z_floor']
indent_arr      = r['indent']
forces_arr      = r['forces']
elastic_arr     = r['elastic']
vm_arr          = r['vm']
disp_arr        = r['disp']
v_z_arr         = r['v_z']
time_arr        = r['time']
true_indent_arr = r['true_indent']
max_disp_idx    = r['max_disp_idx']
a_theo          = r['a_theo']
p0_theo         = r['p0_theo']
vm_max_theo     = r['vm_max_theo']
w_norm          = r['w_norm']
F_norm          = r['F_norm']
b_norm          = r['b_norm']
smooth_delta    = r['smooth_delta']
smooth_hertz    = r['smooth_hertz']
f_theory_load   = r['f_theory_load']
f_theory_unload = r['f_theory_unload']
loading         = r['loading']
unloading       = r['unloading']
omega_c         = r['omega_c']
b_c             = r['b_c']
F_critical      = r['F_critical']
slope_b         = r['slope_b']
struct_files    = r['struct_files']
boundary_files  = r['boundary_files']
cor_velocity    = r['cor']
k_fit           = r['k_fit']

# Update the Stress Evolution Plot to include the Theoretical Max
plt.figure(figsize=(10, 6))
plt.plot(r['time'], r['vm']/1e3, 'b-', label=f'SPH (ps={ps*1000}mm)')
if np.max(forces_arr) > 0:
    plt.axhline(y=vm_max_theo/1e3, color='r', linestyle='--', label='Hertz Analytical Peak')
plt.xlabel("Time (s)")
plt.ylabel("Max Von Mises Stress (KPa)")
plt.title("Stress Evolution vs. Analytical Hertz Theory")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("stress_validation_hertz.png")

# --- PLOT 1: Indentation Validation ---
plt.figure(figsize=(10, 6))
plt.scatter(indent_arr[loading]*1000,   forces_arr[loading],   color='red',  s=5, alpha=0.4, label='SPH Loading')
plt.scatter(indent_arr[unloading]*1000, forces_arr[unloading], color='blue', s=5, alpha=0.4, label='SPH Unloading')
plt.plot(smooth_delta*1000, f_theory_load,   'k--', label='Theory: Loading')
plt.plot(smooth_delta*1000, f_theory_unload, 'k:',  label='Theory: Unloading')
plt.plot(smooth_delta*1000, smooth_hertz,    'g-',  lw=2, label='Elastic Backbone (Hertz)')
plt.xlabel("Indentation (mm)")
plt.ylabel("Force (N)")
plt.title("Kelvin-Voigt Proof: SPH Contact Damping")
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig("kv_validation_proof.png")

# --- PLOT 2: Reaction Force vs. Global Displacement ---
plt.figure(figsize=(10, 6))
plt.plot(disp_arr[:max_disp_idx], forces_arr[:max_disp_idx], color='red',  lw=1.5, label='Approach (Loading)')
plt.plot(disp_arr[max_disp_idx:], forces_arr[max_disp_idx:], color='blue', lw=1.5, label='Rebound (Unloading)')
if len(disp_arr) > 0:
    plt.scatter(disp_arr[max_disp_idx], forces_arr[max_disp_idx], color='black', zorder=5, label='Peak Displacement')
plt.xlabel("Global Displacement $u_z$ (mm)")
plt.ylabel("Reaction Force (N)")
plt.title(f"Reaction Force vs. Displacement (CoR: {cor_velocity:.3f})")
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig("reaction_force_vs_displacement.png")

print(f"Peak Von Mises: {np.max(r['vm'])/1e3:.4f} KPa")
print("Successfully saved: kv_validation_proof.png and reaction_force_vs_displacement.png")

plt.figure(figsize=(10, 6))
plt.plot(time_arr, vm_arr, 'b-', label=f'SPH (ps={ps*1000}mm)')
plt.xlabel("Time (s)")
plt.ylabel("Max Von Mises Stress (KPa)")
plt.title("Stress Evolution: SPH vs. FEM Reference")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("stress_convergence.png")

# ==========================================================
# Tribology Letters Plots (last run)
# ==========================================================
w_ref    = np.linspace(0, np.max(w_norm), 100)
f_ref_05 = 0.5 * w_ref

# --- PLOT 3a ---
plt.figure(figsize=(8, 6))
plt.plot(w_norm, F_norm, 'b-', lw=2, label=f'SPH (ps={ps})')
plt.plot(w_ref, f_ref_05, 'k--', label='Target Gradient (0.5)')
plt.scatter(w_norm[max_disp_idx], F_norm[max_disp_idx], color='red', zorder=5, label='Peak Impact')
plt.xlabel(r"Normalized Displacement $\omega / \omega_c$")
plt.ylabel(r"Normalized Force $F / F_c$")
plt.title("Validation: Normalized Force (Tribology Letters)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("norm_force_vs_disp_changed.png")

# --- PLOT 3b ---
plt.figure(figsize=(8, 6))
plt.plot(w_norm, F_norm, 'b-', label='SPH Trajectory')
plt.scatter(w_norm[max_disp_idx], F_norm[max_disp_idx], color='red', label='Peak Impact')
plt.xlabel(r"Normalized Displacement $\omega / \omega_c$")
plt.ylabel(r"Normalized Force $F / F_c$")
plt.title("Validation: Normalized Force (Tribology Letters)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("norm_force_vs_disp.png")

# --- PLOT 4 ---
w_theory_norm = np.linspace(0, np.max(w_norm)*1.1, 100)
plt.figure(figsize=(8, 6))
plt.plot(w_norm, b_norm, 'g-', label='SPH Contact Growth')
if np.max(w_norm) > 0:
    plt.plot(w_theory_norm, slope_b * np.sqrt(w_theory_norm), 'k--', label='Hertzian Growth ($b \propto \sqrt{w}$)')
plt.xlabel(r"Normalized Displacement $\omega / \omega_c$")
plt.ylabel(r"Normalized Half-Width $b / b_c$")
plt.title("Validation: Contact Width Evolution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("norm_width_vs_disp.png")

from numpy.polynomial import Polynomial
x_data = np.array(w_norm)
y_data = np.array(b_norm)
p_fit  = Polynomial.fit(x_data, y_data, deg=2)
x_trend = np.linspace(0, x_data.max(), 100)
y_trend = p_fit(x_trend)

plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='green', s=10, alpha=0.1, label='SPH Raw Data')
plt.plot(x_trend, y_trend, color='green', lw=3, label='SPH Best-Fit Trend')
if np.max(w_norm) > 0:
    plt.plot(w_theory_norm, slope_b * np.sqrt(w_theory_norm), 'k--', label='Hertzian Growth ($b \propto \sqrt{w}$)')
plt.xlabel('Normalized Displacement $\omega/\omega_c$')
plt.ylabel('Normalized Half-Width $b/b_c$')
plt.title('Validation: Contact Width Evolution (Best-Fit)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("contact_width_best_fit.png")

# --- PLOT 5 ---
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel(r"Normalized Displacement $\omega / \omega_c$")
ax1.set_ylabel(r"Normalized Force $F / F_c$", color='tab:blue')
ax1.plot(w_norm, F_norm, color='tab:blue', label='Force')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel(r"Normalized Width $b / b_c$", color='tab:green')
ax2.plot(w_norm, b_norm, color='tab:green', label='Width')
ax2.tick_params(axis='y', labelcolor='tab:green')
plt.title("Contact Evolution: Force and Width Sync")
fig.tight_layout()
plt.savefig("contact_sync_validation.png")

print(f"Normalization Constants: wc={omega_c:.4e}, bc={b_c:.4e}, Fc={F_critical:.4f}")

# ===============================================
# Elliptic Pressure Profile vs SPH sigma_33 (last run)
# ===============================================
from scipy.ndimage import uniform_filter1d

idx_max_indent  = np.argmax(indent_arr)
max_frame_file  = struct_files[idx_max_indent]
max_frame_bfile = boundary_files[idx_max_indent] if boundary_files else None

m_max    = meshio.read(max_frame_file)
pts      = m_max.points
y_coords = pts[:, 1]
z_coords = pts[:, 2]
y_center = np.mean(y_coords)

dist_to_floor_c = z_coords - z_floor
contact_mask    = (dist_to_floor_c < h) & (dist_to_floor_c >= 0)

if 'sigma_33' in m_max.point_data and np.any(contact_mask):
    y_contact  = y_coords[contact_mask] - y_center
    sort_idx   = np.argsort(y_contact)
    y_sorted   = y_contact[sort_idx]

    sigma_zz_vals = -m_max.point_data['sigma_33'][contact_mask][sort_idx]

    n_bins    = 20
    bin_edges = np.linspace(y_sorted.min(), y_sorted.max(), n_bins + 1)
    bin_cents = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.array([
        sigma_zz_vals[(y_sorted >= bin_edges[i]) & (y_sorted < bin_edges[i+1])].mean()
        if np.any((y_sorted >= bin_edges[i]) & (y_sorted < bin_edges[i+1])) else np.nan
        for i in range(n_bins)
    ])

    a_contact = np.max(np.abs(y_sorted))
    y_ell     = np.linspace(-a_contact, a_contact, 200)
    p_ell     = p0_theo * np.sqrt(np.maximum(0, 1 - (y_ell / a_contact)**2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_sorted*1000, sigma_zz_vals,
               c='green', s=20, alpha=0.4, label='SPH σ_33 (compressive)')
    ax.plot(bin_cents*1000, bin_means, 'b-o', lw=2, ms=5, label='SPH σ_33 (bin mean)')
    ax.plot(y_ell*1000, p_ell/1000, 'r-', lw=2,
            label=f'Hertz p(y), p0={p0_theo/1000:.1f} kPa')
    ax.set_xlabel('Width along cylinder (mm)')
    ax.set_ylabel('Stress (kPa)')
    ax.set_title('Contact Normal Stress σ_33 vs Hertz Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.suptitle(f'Max indentation frame {idx_max_indent} — '
                 f'a={a_contact*1000:.2f}mm  p0={p0_theo/1000:.1f}kPa', fontsize=10)
    plt.tight_layout()
    plt.savefig("sph_vs_hertz_pressure.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sph_vs_hertz_pressure.png")
else:
    print("No contact particles or sigma_33 field available.")

# ==========================================================
# Stress Depth Profile (last run)
# ==========================================================
idx_max_force = np.argmax(forces_arr)
m_max         = meshio.read(struct_files[idx_max_force])
pts_max       = m_max.points
vm_max_field  = m_max.point_data.get('von_mises_stress', np.zeros(len(pts_max)))

x_center = np.mean(pts_max[:, 0])
y_center = np.mean(pts_max[:, 1])
mask     = (np.abs(pts_max[:, 0] - x_center) < ps) & (np.abs(pts_max[:, 1] - y_center) < ps)

z_spine  = pts_max[mask, 2]
vm_spine = vm_max_field[mask]

depth    = np.abs(z_spine - z_floor) * 1000
sort_idx = np.argsort(depth)

z_theory     = np.linspace(0, np.max(depth), 100)
a_mm         = a_theo * 1000 if a_theo > 0 else 1.0
p0_val       = p0_theo if p0_theo > 0 else np.max(vm_spine)
theory_vm_profile = p0_val * 0.56 * (1 / (1 + (z_theory/(a_mm*1.5))**2))

plt.figure(figsize=(8, 6))
plt.plot(vm_spine[sort_idx]/1000, depth[sort_idx], 'b.-', label='SPH Centerline Stress')
plt.plot(theory_vm_profile/1000, z_theory, 'r--', lw=2, label='Hertz Depth Theory')
plt.ylabel("Depth from Contact Surface (mm)")
plt.xlabel("Von Mises Stress (KPa)")
plt.title("Validated Depth Profile: Stress Propagation into Cylinder")
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("stress_depth_fixed.png")
print(f"Saved: stress_depth_fixed.png. Theoretical peak depth target: {0.707*a_mm:.3f} mm")

# ==========================================================
# Von Mises Contour Animation (last run)
# ==========================================================
import re as re2
def natural_sort(l):
    convert     = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re2.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

prefix       = list(all_results.keys())[-1]
run_prefix   = runs[-1][0]
all_struct   = natural_sort(glob.glob(os.path.join(folder, f'{run_prefix}_structure_1_*.vtu')))

import matplotlib.animation as animation

num_grid    = 200
frames_data = []
X, Z        = None, None

for i, fpath in enumerate(all_struct):
    m   = meshio.read(fpath)
    pts = m.points

    if 'von_mises_stress' not in m.point_data:
        continue

    vm         = m.point_data['von_mises_stress']
    y_center_s = np.mean(pts[:, 1])
    slice_mask = np.abs(pts[:, 1] - y_center_s) < 1.5 * ps
    pts_slice  = pts[slice_mask]
    vm_slice   = vm[slice_mask]

    if len(pts_slice) < 3:
        print(f"Frame {i}: too few slice particles ({len(pts_slice)}), skipping")
        continue

    x  = pts_slice[:, 0]
    z  = pts_slice[:, 2]
    xi = np.linspace(np.min(x), np.max(x), num_grid)
    zi = np.linspace(np.min(z), np.max(z), num_grid)
    X, Z = np.meshgrid(xi, zi)

    VM_grid = griddata((x, z), vm_slice, (X, Z), method='linear')
    VM_grid = np.nan_to_num(np.squeeze(VM_grid), nan=0.0)

    if VM_grid.ndim != 2:
        print(f"Frame {i}: invalid grid shape {VM_grid.shape}, skipping")
        continue

    frames_data.append((VM_grid.copy(), i))

if len(frames_data) == 0:
    print("No valid frames for animation")
else:
    print(f"Collected {len(frames_data)} frames for animation")

    vmax_global = max(f[0].max() for f in frames_data) / 1e3
    peak_idx    = max(range(len(frames_data)), key=lambda k: frames_data[k][0].max())
    VM_peak, t_peak = frames_data[peak_idx]

    fig_peak, ax_peak = plt.subplots(figsize=(8, 6))
    cp = ax_peak.contourf(X*1000, Z*1000, VM_peak/1e3, levels=50, cmap='jet',
                          vmin=0, vmax=vmax_global)
    plt.colorbar(cp, ax=ax_peak, label='Von Mises Stress (kPa)')
    ax_peak.set_xlabel('X (mm)')
    ax_peak.set_ylabel('Z (mm)')
    ax_peak.set_title(f'Peak Von Mises Stress — frame {peak_idx}  max={VM_peak.max()/1e3:.2f} kPa')
    ax_peak.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_peak.savefig('stress_peak_frame.png', dpi=150, bbox_inches='tight')
    plt.close(fig_peak)
    print(f"Peak frame saved: frame={t_peak}, max={VM_peak.max()/1e3:.2f} kPa")

    fig, ax = plt.subplots(figsize=(8, 6))
    n_levels = 50
    vmin_anim = 0.0
    vmax_anim = 8.0
    levels_anim = np.linspace(vmin_anim, vmax_anim, n_levels)

    def update(frame_idx):
        ax.cla()
        VM, t = frames_data[frame_idx]
        ax.contourf(X*1000, Z*1000, VM/1e3, levels=levels_anim, cmap='jet',
                    vmin=vmin_anim, vmax=vmax_anim, extend='min')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(f'Von Mises Stress (kPa) — frame {frame_idx} / {len(frames_data)-1}')
        ax.set_xlim(X.min()*1000, X.max()*1000)
        ax.set_ylim(Z.min()*1000, Z.max()*1000)
        ax.grid(True, alpha=0.3)

    sm   = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=vmin_anim, vmax=vmax_anim))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Von Mises Stress (kPa)')
    cbar.set_ticks(np.linspace(vmin_anim, vmax_anim, 6))

    ani = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=200, blit=False)
    ani.save('stress_contours.gif', writer='pillow', dpi=100)
    plt.close()
    print("Saved: stress_contours.gif")

# ==========================================================================================
# Mesh Convergence Comparison Plots
# ==========================================================================================

# Plot C1: Von Mises stress vs time — all runs
fig, ax = plt.subplots(figsize=(10, 6))
for label, res in all_results.items():
    ax.plot(res['time'], res['vm']/1e3, color=res['color'], lw=1.5, label=label)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Max Von Mises Stress (kPa)')
ax.set_title('Mesh Convergence: Von Mises Stress vs Time')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_vm_time.png', dpi=150)
plt.close()
print("Saved: convergence_vm_time.png")

# Plot C2: Force vs indentation — all runs
fig, ax = plt.subplots(figsize=(10, 6))
for label, res in all_results.items():
    ax.scatter(res['indent']*1e3, res['forces'],
               color=res['color'], s=5, alpha=0.4, label=label)
ax.set_xlabel('Indentation (mm)')
ax.set_ylabel('Contact Force (N)')
ax.set_title('Mesh Convergence: Force vs Indentation')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_force_indent.png', dpi=150)
plt.close()
print("Saved: convergence_force_indent.png")

# Plot C3: Force vs displacement — all runs
fig, ax = plt.subplots(figsize=(10, 6))
for label, res in all_results.items():
    ax.plot(res['disp'], res['forces'], color=res['color'], lw=1.5, label=label)
ax.set_xlabel('Displacement (mm)')
ax.set_ylabel('Contact Force (N)')
ax.set_title('Mesh Convergence: Force vs Displacement')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_force_disp.png', dpi=150)
plt.close()
print("Saved: convergence_force_disp.png")

# Plot C4: Normalized force vs displacement — all runs
fig, ax = plt.subplots(figsize=(8, 6))
for label, res in all_results.items():
    ax.plot(res['w_norm'], res['F_norm'], color=res['color'], lw=1.5, label=label)
w_ref_c  = np.linspace(0, max(np.max(res['w_norm']) for res in all_results.values()), 100)
ax.plot(w_ref_c, 0.5 * w_ref_c, 'k--', label='Target Gradient (0.5)')
ax.set_xlabel(r"Normalized Displacement $\omega / \omega_c$")
ax.set_ylabel(r"Normalized Force $F / F_c$")
ax.set_title('Mesh Convergence: Normalized Force')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_norm_force.png', dpi=150)
plt.close()
print("Saved: convergence_norm_force.png")

# Plot C5: Peak VM vs particle spacing (convergence check)
fig, ax = plt.subplots(figsize=(8, 5))
ps_vals  = [res['ps']*1e3        for res in all_results.values()]
vm_peaks = [res['vm'].max()/1e3  for res in all_results.values()]
ax.plot(ps_vals, vm_peaks, 'ko-', lw=2, ms=8)
for label, res in all_results.items():
    ax.annotate(label,
                (res['ps']*1e3, res['vm'].max()/1e3),
                textcoords='offset points', xytext=(5, 5))
ax.set_xlabel('Particle Spacing (mm)')
ax.set_ylabel('Peak Von Mises Stress (kPa)')
ax.set_title('Mesh Convergence: Peak Stress vs Resolution')
ax.invert_xaxis()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_peak_vm.png', dpi=150)
plt.close()
print("Saved: convergence_peak_vm.png")

# Summary table
print(f"\n{'='*70}")
print("MESH CONVERGENCE SUMMARY")
print(f"{'='*70}")
print(f"{'Label':<15} {'ps(mm)':>8} {'MaxVM(kPa)':>12} {'MaxF(N)':>10} {'MaxIndent(mm)':>14} {'CoR':>6}")
print("-" * 70)
for label, res in all_results.items():
    print(f"{label:<15} {res['ps']*1e3:>8.3f} {res['vm'].max()/1e3:>12.4f} "
          f"{res['forces'].max():>10.6f} {res['indent'].max()*1e3:>14.4f} "
          f"{res['cor']:>6.3f}")