# ==========================================================================================
# Pseudo-2D Uniaxial Compression Test — Crossley Sandstone (TLSPH)
#
# Models a uniaxial compression test on a rectangular sandstone specimen
# as described in the Crossley sandstone benchmark [1]:
#
#   Specimen geometry : 82 mm wide × 140 mm tall (pseudo-2D: 1 particle thick in y)
#   Material          : Crossley sandstone
#       Bulk modulus  K  = 12.2 GPa
#       Shear modulus G  = 2.67 GPa
#       Density       ρ  = 2,300 kg/m³
#       → Young's modulus  E  = 9KG/(3K+G) ≈ 7.465 GPa
#       → Poisson ratio    ν  = (3K−2G)/(2(3K+G)) ≈ 0.398
#       → Sound speed      c  = 2,303 m/s (given)
#   Particle spacing  : 5 mm (coarse mesh for faster runs)
#                      → about 16 × 28 × 1 ≈ 448 specimen particles
#   Initial layout    : regular square grid
#
#   Boundary conditions:
#       Bottom : fixed rigid plate (clamped)
#       Top    : rigid piston descending at 1.5 mm/s
#
# The file uses the same pseudo-2D TLSPH framework and implicit stress-cache
# machinery as `stamping_cfk_clip_pseudo2d_implicit.jl`, stripped of all
# thermomechanical / composite-material logic.
#
# SOLVER NOTE:
#   The full 11,480-particle system is too large for implicit TRBDF2 with
#   dense LU (AutoFiniteDiff).  The file defaults to the explicit
#   SymplecticEuler integrator (explicit symplectic)
#   integrator, which works at the CFL time-step
#       dt ≈ 0.2 × Δx / c ≈ 4.34 × 10⁻⁷ s (for Δx = 5 mm).
#   For quasi-static studies run to large strain, either
#     (a) use an implicit solver with iterative GMRES + ILU (see commented
#         block at the bottom), or
#     (b) scale the piston velocity (v_piston) upward by 100–1000×; at
#         Mach ≪ 1 inertia effects are negligible so the stress field
#         remains quasi-static.
#
# [1] Crossley sandstone geometry and material taken from the reference study
#     on SPH uniaxial compression testing.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using LinearSolve
using RecursiveFactorization
using PointNeighbors
using Statistics
using LinearAlgebra
using StaticArrays
using SparseArrays
using Base.Threads
using Logging

include("../../src/schemes/structure/total_lagrangian_sph/fix_reimann.jl")

# Start loading at first platen contact for the Das-Cleary benchmark.
REIMANN_CONTACT_OVERLAP_TOLERANCE[] = 0.0

println("--- SANDSTONE UNIAXIAL COMPRESSION TEST (pseudo-2D) ---")
println("Threads available: ", nthreads())
println("--------------------------------------------------------")

# ==========================================================================================
# STEP 1: Material properties — Crossley sandstone
# ==========================================================================================

# Given moduli (from [1])
K_bulk   = 12.2e9   # Pa — bulk modulus
G_shear  = 2.67e9   # Pa — shear modulus
rho_specimen = 2300.0  # kg/m³

# Derived standard elastic constants
E_young    = 9.0 * K_bulk * G_shear / (3.0 * K_bulk + G_shear)
nu_poisson = (3.0 * K_bulk - 2.0 * G_shear) / (2.0 * (3.0 * K_bulk + G_shear))
c_sound    = 2303.0   # m/s — longitudinal wave speed, given in problem statement

println("Sandstone material properties:")
println("  K  = ", K_bulk  / 1e9,              " GPa")
println("  G  = ", G_shear / 1e9,              " GPa")
println("  E  = ", round(E_young    / 1e9; digits=4), " GPa")
println("  ν  = ", round(nu_poisson;         digits=4))
println("  ρ  = ", rho_specimen,               " kg/m³")
println("  c  = ", c_sound,                    " m/s")

# Yield stress and hardening for Crossley sandstone.
# The problem statement does not provide an explicit yield stress; the value
# below is representative of sandstone in uniaxial compression.  Adjust to
# match the target experiment.
yield_stress_specimen = 80.0e6   # Pa  — ≈ 80 MPa compressive yield stress
hardening_specimen    = 0.0      # perfectly plastic (brittle post-peak behaviour)

# Thermal parameters — held constant at room temperature throughout.
# TotalLagrangianSPHSystem requires these fields; they play no mechanical role here.
temp_room       = 293.0    # K
temp_melt_dummy = 5000.0   # K — far above any expected temperature (never reached)
cp_sandstone    = 800.0    # J/(kg·K) — specific heat capacity
k_sandstone     = 2.0      # W/(m·K) — thermal conductivity
h_contact_dummy = 0.0      # W/(m²·K) — no contact heat transfer

# ==========================================================================================
# STEP 2: Geometry and discretisation
# ==========================================================================================

particle_spacing = 0.0025  # 2.5 mm coarse screening mesh

# Specimen dimensions (Crossley sandstone benchmark [1])
specimen_width  = 0.082   # 82  mm
specimen_height = 0.140   # 140 mm

# Snap to whole-particle grid
n_specimen_x = round(Int, specimen_width  / particle_spacing)   # 82
n_specimen_y = 1                                                 # pseudo-2D slab
n_specimen_z = round(Int, specimen_height / particle_spacing)   # 140

specimen_width_discrete  = n_specimen_x * particle_spacing
specimen_height_discrete = n_specimen_z * particle_spacing

println("\nSpecimen discretisation:")
println("  nx × ny × nz = ", n_specimen_x, " × ", n_specimen_y, " × ", n_specimen_z,
        "  =  ", n_specimen_x * n_specimen_y * n_specimen_z, " particles")
println("  width  = ", specimen_width_discrete  * 1e3, " mm")
println("  height = ", specimen_height_discrete * 1e3, " mm")

# Place specimen: bottom face at z = 0, centred in x, single slab in y
specimen_origin = (-0.5 * specimen_width_discrete,
                   -0.5 * particle_spacing,
                   0.0)

# Smoothing kernel and length (same settings as stamping file)
factor           = 1.8
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = factor * particle_spacing

# Tool geometry: slightly wider than the specimen so the contact layer is
# fully enclosed on both sides
tool_layers   = 3
tool_width    = specimen_width_discrete + 4.0 * particle_spacing
tool_origin_x = specimen_origin[1] - 2.0 * particle_spacing

# ==========================================================================================
# STEP 3: Build particle arrays (specimen, floor, piston)
# ==========================================================================================

"""
Build a rectangular block of particles in the x–z plane (one particle thick in y).
All particles are given a uniform initial velocity velocity_z in the z-direction.
"""
function build_rect_particles(ps, x_min, x_max, y_center, z_min, z_max, density;
                               velocity_z=0.0)
    nx = round(Int, (x_max - x_min) / ps)
    nz = round(Int, (z_max - z_min) / ps)
    n  = nx * nz
    coords = Matrix{Float64}(undef, 3, n)
    vel    = zeros(3, n)
    mass   = fill(density * ps^3, n)
    dens   = fill(density,        n)
    p = 0
    for k in 1:nz, i in 1:nx
        p += 1
        coords[1, p] = x_min + (i - 0.5) * ps
        coords[2, p] = y_center
        coords[3, p] = z_min + (k - 0.5) * ps
        vel[3, p]    = velocity_z
    end
    return InitialCondition(; coordinates=coords, velocity=vel, mass=mass, density=dens)
end

# --- Specimen (deformable body) ---
specimen_ic = build_rect_particles(
    particle_spacing,
    specimen_origin[1], specimen_origin[1] + specimen_width_discrete,
    0.0,
    0.0, specimen_height_discrete,
    rho_specimen)

# --- Floor plate (rigid, clamped; sits directly below specimen bottom face) ---
z_floor_top = 0.0
z_floor_bot = z_floor_top - tool_layers * particle_spacing
floor_ic = build_rect_particles(
    particle_spacing,
    tool_origin_x, tool_origin_x + tool_width,
    0.0,
    z_floor_bot, z_floor_top,
    rho_specimen)

# --- Piston (rigid, moves downward onto the top of the specimen) ---
# Initial position: gap = 0 (punch starts flush against specimen surface).
initial_gap_piston = 0.0
z_piston_bot = specimen_height_discrete + initial_gap_piston
z_piston_top = z_piston_bot + tool_layers * particle_spacing
piston_ic = build_rect_particles(
    particle_spacing,
    tool_origin_x, tool_origin_x + tool_width,
    0.0,
    z_piston_bot, z_piston_top,
    rho_specimen)

println("\nParticle counts:")
println("  Specimen : ", nparticles(specimen_ic))
println("  Floor    : ", nparticles(floor_ic))
println("  Piston   : ", nparticles(piston_ic))

# ==========================================================================================
# STEP 4: Piston motion
# ==========================================================================================

# Physical test speed from the paper is 1.5 mm/s.
# For the short transient benchmark, start at full speed from t=0; a multi-ms
# velocity ramp suppresses the 100 us stresses by roughly two orders of magnitude.
v_piston_physical = -0.0015
piston_speedup    = 1.0   # physical speed: 1.5 mm/s
v_piston          = v_piston_physical * piston_speedup
t_ramp_piston = 0.0       # s   — benchmark uses immediate constant platen velocity

@inline function piston_displacement_at(t)
    if t_ramp_piston > 0.0 && t < t_ramp_piston
        0.5 * v_piston / t_ramp_piston * t^2
    else
        v_piston * t
    end
end

piston_motion = PrescribedMotion(
    (x, t) -> x + StaticArrays.SVector(0.0, 0.0, piston_displacement_at(t)),
    t -> true)

# ==========================================================================================
# STEP 5: Plastic-strain history state (specimen particles; no thermal softening)
# ==========================================================================================

n_specimen_particles = nparticles(specimen_ic)

# Plastic deformation gradient Fₚ and equivalent plastic strain α — both zero at t = 0
alpha_plastic_state = Ref(zeros(n_specimen_particles))

Fp_initial = zeros(3, 3, n_specimen_particles)
for i in 1:n_specimen_particles
    Fp_initial[:, :, i] .= Matrix{Float64}(I, 3, 3)
end
Fp_state = Ref(Fp_initial)

alpha_committed = Ref(zeros(n_specimen_particles))
Fp_committed    = Ref(copy(Fp_initial))

# Per-particle material arrays (constant — sandstone has no temperature dependence)
ys_particle_buf   = fill(yield_stress_specimen, n_specimen_particles)
hard_particle_buf = fill(hardening_specimen,    n_specimen_particles)
vis_particle_buf  = zeros(n_specimen_particles)   # viscosity not used (no viscoplastic flow)

# Solid fraction: always 1.0 (specimen never melts)
solid_fraction_particle_buf = ones(n_specimen_particles)

# ==========================================================================================
# STEP 6: Build the TLSPH systems
# ==========================================================================================

# Tool stiffness — use the same moduli as the specimen so the tool acts as a
# rigid boundary without introducing extreme stiffness contrasts.
E_tool = E_young

import PointNeighbors: DictionaryCellList

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=200)

floor_system = TotalLagrangianSPHSystem(floor_ic,
                                        smoothing_kernel, smoothing_length,
                                        E_tool, nu_poisson, 0.0,
                                        temp_room, temp_room,
                                        cp_sandstone, k_sandstone,
                                        temp_melt_dummy, h_contact_dummy,
                                        hardening_specimen, temp_melt_dummy,
                                        yield_stress_specimen;
                                        clamped_particles=collect(1:nparticles(floor_ic)),
                                        acceleration=(0.0, 0.0, 0.0))

piston_system = TotalLagrangianSPHSystem(piston_ic,
                                          smoothing_kernel, smoothing_length,
                                          E_tool, nu_poisson, 0.0,
                                          temp_room, temp_room,
                                          cp_sandstone, k_sandstone,
                                          temp_melt_dummy, h_contact_dummy,
                                          hardening_specimen, temp_melt_dummy,
                                          yield_stress_specimen;
                                          clamped_particles=collect(1:nparticles(piston_ic)),
                                          clamped_particles_motion=piston_motion,
                                          acceleration=(0.0, 0.0, 0.0),
                                          self_interaction_nhs=nhs_template)

specimen_system = TotalLagrangianSPHSystem(specimen_ic,
                                            smoothing_kernel, smoothing_length,
                                            E_young, nu_poisson, 0.0,
                                            temp_room, temp_room,
                                            cp_sandstone, k_sandstone,
                                            temp_melt_dummy, h_contact_dummy,
                                            hardening_specimen, temp_melt_dummy,
                                            yield_stress_specimen;
                                            acceleration=(0.0, 0.0, 0.0),
                                            viscosity=ArtificialViscosityMonaghan(alpha=2.0,
                                                                                   beta=4.0),
                                            tensile_stress=TensileArtificialStressMonaghan(psi=-0.1,
                                                                                            exponent=4),
                                            self_interaction_nhs=nhs_template)

# ==========================================================================================
# STEP 7: Semidiscretization
# ==========================================================================================

semi = Semidiscretization(specimen_system, floor_system, piston_system;
                          neighborhood_search=GridNeighborhoodSearch{3}(;
                              cell_list=DictionaryCellList{3}(),
                              search_radius=smoothing_length))

# Initialise temperature fields at room temperature throughout
semi.systems[1].temp .= temp_room
semi.systems[2].temp .= temp_room
semi.systems[3].temp .= temp_room

# Simulation time: 20 ms transient benchmark run.
t_total = 20e-3      # s  (20 milliseconds)
tspan   = (0.0, t_total)

ode_base = semidiscretize(semi, tspan)

# ==========================================================================================
# STEP 8: Pseudo-2D kernel correction matrix fix
#
# Identical to the fix in stamping_cfk_clip_pseudo2d_implicit.jl.
# The default 3D correction sees det(L) = 0 on a coplanar-y particle cloud
# and falls back to the identity, disabling kernel correction entirely.
# We invert only the x–z 2×2 block and set the y diagonal to 1.
# ==========================================================================================

println(">>> Applying pseudo-2D kernel correction matrix fix...")
TrixiParticles.foreach_system(semi) do system
    TrixiParticles.set_zero!(system.correction_matrix)
    initial_coords = TrixiParticles.initial_coordinates(system)

    TrixiParticles.foreach_point_neighbor(system, system, initial_coords, initial_coords,
                                          semi) do particle, neighbor, pos_diff, distance
        grad_kernel = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance, particle)
        iszero(grad_kernel) && return
        volume = system.mass[neighbor] / system.material_density[neighbor]
        result = volume * grad_kernel * pos_diff'
        for j in 1:3, i in 1:3
            system.correction_matrix[i, j, particle] -= result[i, j]
        end
    end

    for particle in TrixiParticles.eachparticle(system)
        L   = TrixiParticles.extract_smatrix(system.correction_matrix, system, particle)
        Lx  = L[1, 1]; Lxz = L[1, 3]
        Lzx = L[3, 1]; Lz  = L[3, 3]
        detL2d     = Lx * Lz - Lxz * Lzx
        L_inv      = Matrix(1.0I, 3, 3)
        detL2d_reg = sign(detL2d) * max(abs(detL2d), 1e-6)
        if abs(Lx) + abs(Lz) + abs(Lxz) + abs(Lzx) > 1e-9
            L_inv[1, 1] =  Lz  / detL2d_reg
            L_inv[1, 3] = -Lxz / detL2d_reg
            L_inv[3, 1] = -Lzx / detL2d_reg
            L_inv[3, 3] =  Lx  / detL2d_reg
        end
        for j in 1:3, i in 1:3
            system.correction_matrix[i, j, particle] = L_inv[i, j]
        end
    end
end
println(">>> Pseudo-2D kernel correction applied.")

# ==========================================================================================
# STEP 9: Constitutive model — elastoplastic sandstone (no thermal effects)
# ==========================================================================================

v_stress_buf        = zeros(3, 3, n_specimen_particles)
v_stress_elas_buf   = zeros(3, 3, n_specimen_particles)
# Kirchhoff stress tau stored directly (before L_corr pre-multiplication).
# Used as custom VTK fields tau_33 and tau_vm — bypasses the pk1_rho2 conversion chain.
tau_kirchhoff_buf   = zeros(3, 3, n_specimen_particles)

# dt state: used inside elastic_stress3d_trial_skip_liquid! to cap viscoplastic
# flow rate (only relevant when temp > 0.5*tmelt, which never occurs here).
trial_dt_state = Ref(1.0e-5)

nhs_updated_at_t = Ref(-Inf)

# Safety bounds for particle coordinate and velocity clamps
# IMPORTANT: Enable repair ONLY for truly catastrophic failures (NaN/Inf/negative J).
# Normal uniaxial compression causes J < 1.0 during volumetric contraction—that's OK!
# We only repair if det(F) becomes non-finite or negative (inverted deformation).
enable_emergency_repair_clamp = true
enable_repair_after_t = 0.0
x_safety_bound = 2.0 * tool_width  # Extremely loose x-bounds: allow large transient overlaps
z_safety_min   = z_floor_bot - 50.0 * particle_spacing
z_safety_max   = z_piston_top + 50.0 * particle_spacing
J_repair_min = 1e-10  # Only repair if J becomes negative or nearly singular
J_repair_max = 1e10   # Allow large deformations (both compression and extension)
diagnostics_interval = t_total <= 5e-3 ? 10e-6 : 100e-6
solution_save_interval = t_total <= 5e-3 ? 10e-6 : 100e-6
diagnostics_top_k = 8

# ----------------------------------------------------------------------------------
# Monitoring points A, B, C  (Das & Cleary Fig. 4)
#   Specimen is centred in x → symmetry axis at x = 0
#   h-h' plane at z = specimen_height_discrete / 2
#   Paper coords (from v-v' axis, from bottom): A=(0,70mm), B=(+21.5mm,103.5mm), C=(+21.5mm,70mm)
# ----------------------------------------------------------------------------------
monitor_targets = [
    (0.0,    0.5 * specimen_height_discrete,  "A"),   # centre
    (0.0215, 0.1035,                           "B"),   # off-axis upper
    (0.0215, 0.5 * specimen_height_discrete,  "C"),   # off-axis mid
]

# Find nearest specimen particle index for each target (using initial coordinates)
function find_nearest_particle(coords, tx, tz)
    best_idx  = 1
    best_dist = Inf
    for p in axes(coords, 2)
        dx = coords[1, p] - tx
        dz = coords[3, p] - tz
        d  = dx*dx + dz*dz
        if d < best_dist
            best_dist = d
            best_idx  = p
        end
    end
    return best_idx
end

monitor_indices = Dict{String,Int}()
for (tx, tz, label) in monitor_targets
    idx = find_nearest_particle(specimen_ic.coordinates, tx, tz)
    monitor_indices[label] = idx
    px = specimen_ic.coordinates[1, idx]
    pz = specimen_ic.coordinates[3, idx]
    println("  Monitor point ", label, ": target=(", round(tx*1e3,digits=2), ", ",
            round(tz*1e3,digits=2), ") mm  →  nearest particle ", idx,
            " at (", round(px*1e3,digits=2), ", ", round(pz*1e3,digits=2), ") mm")
end

# Output directory (same folder that SolutionSavingCallback writes VTK files to)
output_directory = "/home/sadaat/SPH_Code/TrixiParticles.jl_Saadat/out_sandstone_uniaxial"
mkpath(output_directory)

# Open CSV file for stress history at monitoring points
monitor_csv_path = joinpath(output_directory, "monitor_stress_history.csv")
monitor_csv_io   = open(monitor_csv_path, "w")
println(monitor_csv_io,
    "t_s,",
    "tau33_A_Pa,tauvm_A_Pa,",
    "tau33_B_Pa,tauvm_B_Pa,",
    "tau33_C_Pa,tauvm_C_Pa")
flush(monitor_csv_io)

@inline soft_limit(x, lim) = lim * tanh(x / max(lim, eps(Float64)))

# ------------------------------------------------------------------------------------
# Elastoplastic Kirchhoff-stress trial + radial-return (von Mises, rate-independent).
# Exactly the same algorithm as in the stamping file, with the viscoplastic branch
# disabled for sandstone (temp ≪ 0.5 * tmelt at all times).
# ------------------------------------------------------------------------------------
function elastic_stress3d_trial_skip_liquid!(system, ys, hard, vis, dt, _alpha, _Fp,
                                              semi_local;
                                              solid_fraction_buf,
                                              v_elas_buf=nothing,
                                              skip_solid_fraction=0.0)
    (; deformation_grad, young_modulus, poisson_ratio, temp, tmelt, hardening) = system

    n_particles = size(deformation_grad, 3)
    v_elas = v_elas_buf !== nothing ? v_elas_buf :
             zeros(eltype(young_modulus), 3, 3, n_particles)

    mu      = young_modulus / (2 + 2 * poisson_ratio)
    K_mod   = young_modulus / (3 - 6 * poisson_ratio)
    H_theta = 1.0 / (tmelt - system.temp_ref[1])

    Threads.@threads for particle in 1:n_particles
        if solid_fraction_buf[particle] <= skip_solid_fraction
            @inbounds v_elas[:, :, particle] .= 0.0
            continue
        end

        F        = TrixiParticles.deformation_gradient(system, particle)
        Fp_local = StaticArrays.SMatrix{3, 3}(@view _Fp[:, :, particle])
        L_corr   = @inbounds TrixiParticles.correction_matrix(system, particle)

        det_F = det(F)
        F_reg = if !isfinite(det_F) || abs(det_F) < 1e-10
            F + 1e-4 * one(StaticArrays.SMatrix{3, 3, eltype(F)})
        else
            F
        end

        d_fp = det(Fp_local)
        if isfinite(d_fp) && abs(d_fp) > 1e-14
            Fp_local = Fp_local / cbrt(d_fp)
        end

        Fp_inv = inv(Fp_local)
        Fe     = F_reg * Fp_inv
        J_e    = max(det(Fe), 1e-6)
        be     = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - 1 / 3 * tr(be_bar) * I
        tau    = K_mod * log(J_e) * I + mu * dev_be

        dev_tau = tau - 1 / 3 * tr(tau) * I
        yf = sqrt(1.5) * sqrt(sum(dev_tau .^ 2)) - (ys[particle] + hard[particle])

        if yf > 1e-6
            H_alpha_theta = hardening *
                            (1 - H_theta * (temp[particle] - system.temp_ref[1]))
            # Rate-independent plastic return (temp ≪ 0.5*tmelt for sandstone)
            if temp[particle] < 0.5 * tmelt
                delta_gamma = yf / (3 * mu + H_alpha_theta)
            else
                # Viscoplastic fallback — not reached for sandstone
                delta_gamma = yf * dt / max(vis[particle], 1e-8)
            end

            norm_dev = sqrt(sum(dev_tau .* dev_tau))
            if norm_dev > 1e-14 && isfinite(norm_dev)
                n_dir = dev_tau / norm_dev
                c     = delta_gamma * sqrt(1.5)
                A     = c * n_dir
                A2    = A * A
                sc    = abs(c) < 1e-14 ? one(c) : sinh(c) / c
                cc    = abs(c) < 1e-14 ? one(c) : (cosh(c) - 1) / (c * c)
                exp_A    = one(StaticArrays.SMatrix{3, 3, eltype(n_dir)}) + sc * A + cc * A2
                Fp_local = exp_A * Fp_local
                Fp_local = Fp_local / cbrt(det(Fp_local))

                Fp_inv = inv(Fp_local)
                Fe     = F_reg * Fp_inv
                J_e    = max(det(Fe), 1e-6)
                be     = Fe * Fe'
                be_bar = be / (J_e^(2 / 3))
                dev_be = be_bar - 1 / 3 * tr(be_bar) * I
                tau    = K_mod * log(J_e) * I + mu * dev_be
            end
        end

        FinvT = inv(F_reg)'
        v_elas[:, :, particle] .= (tau * FinvT) * L_corr
    end

    return v_elas
end

# ------------------------------------------------------------------------------------
# Update the implicit stress tensor cache used by system_interaction!.
# For sandstone: pure elastic-plastic, no viscous blending, no thermal field.
# ------------------------------------------------------------------------------------
function update_implicit_stress_cache!(semi_local, v_ode, t)
    spec_sys = semi_local.systems[1]

    elastic_stress3d_trial_skip_liquid!(
        spec_sys, ys_particle_buf, hard_particle_buf, vis_particle_buf,
        trial_dt_state[], alpha_committed[], Fp_committed[], semi_local;
        solid_fraction_buf=solid_fraction_particle_buf,
        skip_solid_fraction=0.0,
        v_elas_buf=v_stress_elas_buf)

    @inbounds for particle in TrixiParticles.each_integrated_particle(spec_sys)
        all_finite = true
        for j in 1:3, i in 1:3
            if !isfinite(v_stress_elas_buf[i, j, particle])
                all_finite = false
                break
            end
        end

        if all_finite
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] = v_stress_elas_buf[i, j, particle]
            end
        else
            # NaN guard: reset this particle's plastic state and zero stress
            Fp_committed[][1,1,particle]=1.0; Fp_committed[][1,2,particle]=0.0
            Fp_committed[][1,3,particle]=0.0; Fp_committed[][2,1,particle]=0.0
            Fp_committed[][2,2,particle]=1.0; Fp_committed[][2,3,particle]=0.0
            Fp_committed[][3,1,particle]=0.0; Fp_committed[][3,2,particle]=0.0
            Fp_committed[][3,3,particle]=1.0
            alpha_committed[][particle] = 0.0
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] = 0.0
            end
        end
    end

    TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(spec_sys), v_stress_buf)

    # Sync the true PK1 stress into pk1_rho2 for correct VTK export.
    #
    # The RHS force-assembly stress is:
    #     v_stress_buf = (tau * F^{-T}) * L_corr  =  P * L_corr
    # where P = tau * F^{-T} is the true first Piola-Kirchhoff stress and
    # L_corr = correction_matrix (the inverted kernel-gradient matrix).
    #
    # The VTK writer computes Cauchy stress as:
    #     sigma = (1/J) * pk1_rho2 * rho^2 * F'
    # so it expects pk1_rho2 = P / rho^2.
    #
    # Therefore we must undo the L_corr factor:
    #     P = v_stress_buf * inv(L_corr)
    #
    # This makes sigma_33 and von_mises_stress in ParaView show the true
    # spatially-varying Cauchy stress, not the L_corr-scrambled version.
    mu_mod = spec_sys.young_modulus / (2 + 2 * spec_sys.poisson_ratio)
    K_mod  = spec_sys.young_modulus / (3 - 6 * spec_sys.poisson_ratio)
    @inbounds for particle in TrixiParticles.each_integrated_particle(spec_sys)
        rho2    = spec_sys.material_density[particle]^2
        L_corr  = TrixiParticles.extract_smatrix(spec_sys.correction_matrix,
                                                  spec_sys, particle)
        # inv(L_corr) undoes the kernel-correction pre-multiplication.
        # For the pseudo-2D case L_corr has a trivial y-block so inv is cheap.
        L_inv_corr = inv(L_corr)
        stress  = StaticArrays.SMatrix{3,3}(@view v_stress_buf[:, :, particle])
        P       = stress * L_inv_corr          # true PK1 stress: tau * F^{-T}
        for j in 1:3, i in 1:3
            spec_sys.pk1_rho2[i, j, particle] = P[i, j] / rho2
        end

        # Recover Kirchhoff stress tau = P * F^T directly into tau_kirchhoff_buf.
        # This is what drives the forces and is what ParaView should show.
        F_sm  = TrixiParticles.extract_smatrix(spec_sys.deformation_grad, spec_sys, particle)
        tau_sm = P * F_sm'   # tau = (tau * F^{-T}) * F^T = tau
        for j in 1:3, i in 1:3
            tau_kirchhoff_buf[i, j, particle] = tau_sm[i, j]
        end
    end
end

# ------------------------------------------------------------------------------------
# Commit the plastic-strain history at the end of each accepted time step.
# ------------------------------------------------------------------------------------
function commit_plastic_history!(system, ys, hard, vis, dt, _alpha, _Fp)
    (; deformation_grad, young_modulus, poisson_ratio, temp, tmelt,
       hardening, temp_ref) = system

    mu      = young_modulus / (2 + 2 * poisson_ratio)
    K_mod   = young_modulus / (3 - 6 * poisson_ratio)
    H_theta = 1.0 / (tmelt - temp_ref[1])

    @inbounds for particle in TrixiParticles.eachparticle(system)
        F     = TrixiParticles.deformation_gradient(system, particle)
        Fp_sm = StaticArrays.SMatrix{3, 3}(@view _Fp[:, :, particle])
        det_F = det(F)
        F_reg = if !isfinite(det_F) || abs(det_F) < 1e-10
            F + 1e-4 * one(StaticArrays.SMatrix{3, 3, eltype(F)})
        else
            F
        end

        d_fp = det(Fp_sm)
        if isfinite(d_fp) && abs(d_fp) > 1e-14
            Fp_sm = Fp_sm / cbrt(d_fp)
        end

        Fp_inv = inv(Fp_sm)
        Fe     = F_reg * Fp_inv
        J_e    = max(det(Fe), 1e-6)
        be     = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - 1 / 3 * tr(be_bar) * I
        tau    = K_mod * log(J_e) * I + mu * dev_be

        dev_tau = tau - 1 / 3 * tr(tau) * I
        yf = sqrt(1.5) * sqrt(sum(dev_tau .^ 2)) - (ys[particle] + hard[particle])

        if yf > 1e-6
            H_alpha_theta = hardening *
                            (1 - H_theta * (temp[particle] - temp_ref[1]))
            delta_gamma = yf / (3 * mu + H_alpha_theta)
            norm_dev    = sqrt(sum(dev_tau .* dev_tau))
            if norm_dev > 1e-14 && isfinite(norm_dev)
                n_dir    = dev_tau / norm_dev
                c        = delta_gamma * sqrt(1.5)
                A        = c * n_dir; A2 = A * A
                sc       = abs(c) < 1e-14 ? one(c) : sinh(c) / c
                cc       = abs(c) < 1e-14 ? one(c) : (cosh(c) - 1) / (c * c)
                exp_A    = one(StaticArrays.SMatrix{3, 3, eltype(n_dir)}) + sc * A + cc * A2
                Fp_sm    = exp_A * Fp_sm
                Fp_sm    = Fp_sm / cbrt(det(Fp_sm))
                _alpha[particle] += delta_gamma
            end
        end

        for j in 1:3, i in 1:3
            _Fp[i, j, particle] = Fp_sm[i, j]
        end
    end
end

# ==========================================================================================
# STEP 10: Custom kick / drift functions (pseudo-2D, explicit symplectic)
# ==========================================================================================

function kick_sandstone!(dv_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.set_zero!(dv_ode)

    spec_sys = semi_local.systems[1]
    v_spec   = TrixiParticles.wrap_v(v_ode, spec_sys, semi_local)
    NDIMS_S  = TrixiParticles.ndims(spec_sys)

    # Pseudo-2D: zero out-of-plane velocity for all systems before computing forces
    TrixiParticles.foreach_system(semi_local) do system
        v_sys = TrixiParticles.wrap_v(v_ode, system, semi_local)
        @inbounds for particle in TrixiParticles.each_integrated_particle(system)
            v_sys[2, particle] = 0.0
        end
    end

    try
        # 1. Update current_coordinates from displacement field u
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        # 2. Rebuild neighborhood lists when positions have changed
        if t != nhs_updated_at_t[]
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end

        # Flag: only compute deformation gradient F for the specimen (not stress;
        # stress is provided by update_implicit_stress_cache! below).
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(spec_sys)

        # 3. Compute deformation gradient and other internal quantities
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        # 4. Pseudo-2D plane-strain fix for the deformation gradient.
        #    In a one-particle-thick slab pos_diff_current[2] = 0 for every
        #    neighbor pair, so the SPH sum gives F[2,:] = 0 → det(F) = 0.
        #    Enforce plane-strain by setting the y-row to [0,1,0] and zeroing
        #    all y off-diagonals; det(F) then equals the 2D Jacobian of the
        #    x–z sub-block, which is the correct volumetric measure.
        @inbounds for particle in TrixiParticles.each_integrated_particle(spec_sys)
            spec_sys.deformation_grad[2, 1, particle] = 0.0
            spec_sys.deformation_grad[2, 2, particle] = 1.0
            spec_sys.deformation_grad[2, 3, particle] = 0.0
            spec_sys.deformation_grad[1, 2, particle] = 0.0
            spec_sys.deformation_grad[3, 2, particle] = 0.0
        end

        # 5. Emergency repair: reset particles with severely distorted or
        #    non-finite Jacobians before the stress assembly.
        #    Skip this for the first 50 ms to allow contact forces to develop naturally.
        if enable_emergency_repair_clamp && t >= enable_repair_after_t
            u_spec = TrixiParticles.wrap_u(u_ode, spec_sys, semi_local)
            x0_spec = TrixiParticles.initial_coordinates(spec_sys)
            @inbounds for particle in TrixiParticles.each_integrated_particle(spec_sys)
                F11 = spec_sys.deformation_grad[1, 1, particle]
                F13 = spec_sys.deformation_grad[1, 3, particle]
                F22 = spec_sys.deformation_grad[2, 2, particle]
                F31 = spec_sys.deformation_grad[3, 1, particle]
                F33 = spec_sys.deformation_grad[3, 3, particle]
                J   = F22 * (F11 * F33 - F13 * F31)  # plane-strain Jacobian
                if !isfinite(J) || J <= J_repair_min || J >= J_repair_max
                    for ii in 1:3, jj in 1:3
                        spec_sys.deformation_grad[ii, jj, particle] = (ii == jj) ? 1.0 : 0.0
                    end
                    Fp_committed[][1,1,particle]=1.0; Fp_committed[][1,2,particle]=0.0
                    Fp_committed[][1,3,particle]=0.0; Fp_committed[][2,1,particle]=0.0
                    Fp_committed[][2,2,particle]=1.0; Fp_committed[][2,3,particle]=0.0
                    Fp_committed[][3,1,particle]=0.0; Fp_committed[][3,2,particle]=0.0
                    Fp_committed[][3,3,particle]=1.0
                    alpha_committed[][particle] = 0.0
                    # For invalid states, reset to the particle's own reference position.
                    # Clamping to global bounds can instantly create side columns.
                    x_reset = x0_spec[1, particle]
                    z_reset = x0_spec[3, particle]

                    u_spec[1, particle] = x_reset - x0_spec[1, particle]
                    u_spec[2, particle] = 0.0
                    u_spec[3, particle] = z_reset - x0_spec[3, particle]
                    spec_sys.current_coordinates[1, particle] = x_reset
                    spec_sys.current_coordinates[2, particle] = 0.0
                    spec_sys.current_coordinates[3, particle] = z_reset
                    v_spec[1, particle] = 0.0
                    v_spec[2, particle] = 0.0
                    v_spec[3, particle] = 0.0
                end
            end
        end

        # 6. Standard TLSPH pipeline steps
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_pressure!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_boundary_interpolation!(system, v, u, v_ode, u_ode,
                                                          semi_local, t)
        end
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_final!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        # 7. Compute elastic-plastic Kirchhoff stress and assemble internal forces
        update_implicit_stress_cache!(semi_local, v_ode, t)
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi_local, t)

    finally
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = nothing
        TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
    end

    # Pseudo-2D: zero out-of-plane accelerations
    TrixiParticles.foreach_system(semi_local) do system
        dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
        @inbounds for particle in TrixiParticles.each_integrated_particle(system)
            dv_sys[2, particle] = 0.0
        end
    end

    return dv_ode
end

function drift_sandstone!(du_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.drift!(du_ode, v_ode, u_ode, semi_local, t)

    # Pseudo-2D: suppress out-of-plane displacement drift
    TrixiParticles.foreach_system(semi_local) do system
        du_sys = TrixiParticles.wrap_u(du_ode, system, semi_local)
        @inbounds for particle in TrixiParticles.each_integrated_particle(system)
            du_sys[2, particle] = 0.0
        end
    end

    return du_ode
end

ode = DynamicalODEProblem(kick_sandstone!, drift_sandstone!,
                          Vector{Float64}(ode_base.u0.x[1]),
                          Vector{Float64}(ode_base.u0.x[2]),
                          tspan, semi)

# ==========================================================================================
# STEP 11: Callbacks
# ==========================================================================================

# Local PeriodicCallback helper (avoids DiffEqCallbacks.jl dependency)
function PeriodicCallback(affect!, dt_cb; save_positions=(false, false))
    next_t = Ref(dt_cb)
    condition(u, t, integrator) = t + 1.0e-14 >= next_t[]
    function wrapped_affect!(integrator)
        affect!(integrator)
        while next_t[] <= integrator.t + 1.0e-14
            next_t[] += dt_cb
        end
    end
    return DiscreteCallback(condition, wrapped_affect!; save_positions=save_positions)
end

function print_progress(integrator)
    spec_sys  = integrator.p.systems[1]
    z_max     = maximum(spec_sys.current_coordinates[3, :])
    z_min_sp  = minimum(spec_sys.current_coordinates[3, :])
    curr_h    = z_max - z_min_sp
    strain_pct = (specimen_height_discrete - curr_h) / specimen_height_discrete * 100.0
    disp_mm    = abs(piston_displacement_at(integrator.t)) * 1e3
    println("  t = ", lpad(round(integrator.t; digits=3), 6), " s  |",
            "  piston = ",   lpad(round(disp_mm;     digits=3), 7), " mm  |",
            "  ε_axial ≈ ",  lpad(round(strain_pct;  digits=3), 6), " %  |",
            "  iter = ",     integrator.iter)
    flush(stdout)
end

function print_particle_range_diagnostics(integrator)
    spec_sys = integrator.p.systems[1]
    floor_sys = integrator.p.systems[2]
    punch_sys = integrator.p.systems[3]
    v_wrap   = TrixiParticles.wrap_v(integrator.u.x[1], spec_sys, integrator.p)

    min_x = Inf; max_x = -Inf
    min_z = Inf; max_z = -Inf
    min_v = Inf; max_v = -Inf
    min_J = Inf; max_J = -Inf
    min_tau33 = Inf; max_tau33 = -Inf
    min_tauvm = Inf; max_tauvm = -Inf
    invalid_count = 0
    outlier_ids = Int[]

    @inbounds for particle in TrixiParticles.each_integrated_particle(spec_sys)
        x = spec_sys.current_coordinates[1, particle]
        z = spec_sys.current_coordinates[3, particle]
        vx = v_wrap[1, particle]
        vz = v_wrap[3, particle]
        vmag = sqrt(vx * vx + vz * vz)

        F11 = spec_sys.deformation_grad[1, 1, particle]
        F13 = spec_sys.deformation_grad[1, 3, particle]
        F22 = spec_sys.deformation_grad[2, 2, particle]
        F31 = spec_sys.deformation_grad[3, 1, particle]
        F33 = spec_sys.deformation_grad[3, 3, particle]
        J = F22 * (F11 * F33 - F13 * F31)

        # Convert PK1 → Cauchy stress: sigma = (1/J) * P * F'
        # This matches exactly what the VTK writer computes, so the terminal
        # diagnostics and ParaView colour maps are consistent.
        F_mat   = StaticArrays.SMatrix{3,3}(
                      spec_sys.deformation_grad[1,1,particle], spec_sys.deformation_grad[2,1,particle], spec_sys.deformation_grad[3,1,particle],
                      spec_sys.deformation_grad[1,2,particle], spec_sys.deformation_grad[2,2,particle], spec_sys.deformation_grad[3,2,particle],
                      spec_sys.deformation_grad[1,3,particle], spec_sys.deformation_grad[2,3,particle], spec_sys.deformation_grad[3,3,particle])
        rho2_d  = spec_sys.material_density[particle]^2
        P_mat   = StaticArrays.SMatrix{3,3}(@view spec_sys.pk1_rho2[:, :, particle]) * rho2_d
        J_d     = max(abs(J), 1e-14)
        cauchy  = (1.0 / J_d) * P_mat * F_mat'
        sigma33 = cauchy[3, 3]
        trace_c = cauchy[1,1] + cauchy[2,2] + cauchy[3,3]
        dev_c   = cauchy - (trace_c / 3.0) * one(StaticArrays.SMatrix{3,3,Float64})
        vm_stress = sqrt(1.5 * sum(dev_c .^ 2))

        min_x = min(min_x, x); max_x = max(max_x, x)
        min_z = min(min_z, z); max_z = max(max_z, z)
        min_v = min(min_v, vmag); max_v = max(max_v, vmag)
        min_J = min(min_J, J); max_J = max(max_J, J)

        # Kirchhoff stress directly from tau_kirchhoff_buf (no pk1_rho2 chain)
        tau33_p = tau_kirchhoff_buf[3, 3, particle]
        t11 = tau_kirchhoff_buf[1,1,particle]; t22 = tau_kirchhoff_buf[2,2,particle]
        t33 = tau_kirchhoff_buf[3,3,particle]
        t12 = tau_kirchhoff_buf[1,2,particle]; t13 = tau_kirchhoff_buf[1,3,particle]; t23 = tau_kirchhoff_buf[2,3,particle]
        tr3 = (t11 + t22 + t33) / 3.0
        tauvm_p = sqrt(1.5 * ((t11-tr3)^2 + (t22-tr3)^2 + (t33-tr3)^2 + 2*(t12^2 + t13^2 + t23^2)))
        min_tau33 = min(min_tau33, tau33_p); max_tau33 = max(max_tau33, tau33_p)
        min_tauvm = min(min_tauvm, tauvm_p); max_tauvm = max(max_tauvm, tauvm_p)

        invalid = !isfinite(x) || !isfinite(z) || !isfinite(vmag) || !isfinite(J) ||
                  abs(x) > x_safety_bound || z < z_safety_min || z > z_safety_max ||
                  J <= 0.2 || J >= 5.0
        if invalid
            invalid_count += 1
            if length(outlier_ids) < diagnostics_top_k
                push!(outlier_ids, particle)
            end
        end
    end

        # Ranges for rigid tools, useful to distinguish real specimen blow-up from visualization scale issues.
        floor_x_min = minimum(floor_sys.current_coordinates[1, :])
        floor_x_max = maximum(floor_sys.current_coordinates[1, :])
        floor_z_min = minimum(floor_sys.current_coordinates[3, :])
        floor_z_max = maximum(floor_sys.current_coordinates[3, :])

        punch_x_min = minimum(punch_sys.current_coordinates[1, :])
        punch_x_max = maximum(punch_sys.current_coordinates[1, :])
        punch_z_min = minimum(punch_sys.current_coordinates[3, :])
        punch_z_max = maximum(punch_sys.current_coordinates[3, :])

    # ---- Monitoring-point stress history (CSV) --------------------------------
    function tau_vm_scalar(buf, p)
        t11 = buf[1,1,p]; t22 = buf[2,2,p]; t33 = buf[3,3,p]
        t12 = buf[1,2,p]; t13 = buf[1,3,p]; t23 = buf[2,3,p]
        tr3 = (t11 + t22 + t33) / 3.0
        sqrt(max(0.0, 1.5*((t11-tr3)^2+(t22-tr3)^2+(t33-tr3)^2+2*(t12^2+t13^2+t23^2))))
    end
    pA = monitor_indices["A"]; pB = monitor_indices["B"]; pC = monitor_indices["C"]
    println(monitor_csv_io,
        integrator.t, ",",
        tau_kirchhoff_buf[3,3,pA], ",", tau_vm_scalar(tau_kirchhoff_buf, pA), ",",
        tau_kirchhoff_buf[3,3,pB], ",", tau_vm_scalar(tau_kirchhoff_buf, pB), ",",
        tau_kirchhoff_buf[3,3,pC], ",", tau_vm_scalar(tau_kirchhoff_buf, pC))
    flush(monitor_csv_io)

    println("  [diag] t=", round(integrator.t; digits=5),
            " | x=[", round(min_x; digits=5), ", ", round(max_x; digits=5), "]",
            " | z=[", round(min_z; digits=5), ", ", round(max_z; digits=5), "]",
            " | |v|=[", round(min_v; digits=5), ", ", round(max_v; digits=5), "]",
            " | J=[", round(min_J; digits=5), ", ", round(max_J; digits=5), "]",
            " | τ₃₃=[", round(min_tau33; digits=3), ", ", round(max_tau33; digits=3), "] Pa",
            " | τ_vm=[", round(min_tauvm; digits=3), ", ", round(max_tauvm; digits=3), "] Pa",
            " | invalid=", invalid_count)
        println("  [diag-tools] floor x=[", round(floor_x_min; digits=5), ", ",
            round(floor_x_max; digits=5), "]",
            " z=[", round(floor_z_min; digits=5), ", ", round(floor_z_max; digits=5), "]",
            " | punch x=[", round(punch_x_min; digits=5), ", ",
            round(punch_x_max; digits=5), "]",
            " z=[", round(punch_z_min; digits=5), ", ", round(punch_z_max; digits=5), "]")
    if !isempty(outlier_ids)
        println("  [diag] outlier particle ids (first ", diagnostics_top_k, "): ", outlier_ids)
    end
    flush(stdout)
end

# ------------------------------------------------------------------------------------
# Custom VTK quantities: Kirchhoff stress tau read directly from tau_kirchhoff_buf.
# These bypass the pk1_rho2 → Cauchy chain so ParaView shows the stress that
# actually drives the forces.  Returns nothing for non-specimen systems so the
# field is only written into the specimen VTK file.
# ------------------------------------------------------------------------------------
function tau_33_quantity(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    system === semi.systems[1] || return nothing
    return tau_kirchhoff_buf[3, 3, :]
end

function tau_vm_quantity(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    system === semi.systems[1] || return nothing
    n  = size(tau_kirchhoff_buf, 3)
    vm = zeros(Float64, n)
    @inbounds for p in 1:n
        t11 = tau_kirchhoff_buf[1,1,p]; t22 = tau_kirchhoff_buf[2,2,p]; t33 = tau_kirchhoff_buf[3,3,p]
        t12 = tau_kirchhoff_buf[1,2,p]; t13 = tau_kirchhoff_buf[1,3,p]; t23 = tau_kirchhoff_buf[2,3,p]
        tr3 = (t11 + t22 + t33) / 3.0
        s11 = t11 - tr3; s22 = t22 - tr3; s33 = t33 - tr3
        vm[p] = sqrt(1.5 * (s11^2 + s22^2 + s33^2 + 2*(t12^2 + t13^2 + t23^2)))
    end
    return vm
end

callbacks = CallbackSet(
    # Per-step: commit plastic history + emergency particle repair
    DiscreteCallback(
        (u, t, integrator) -> integrator.iter > 0 && mod(integrator.iter, 1) == 0,
        function(integrator)
            spec_sys = integrator.p.systems[1]
            v_wrap   = TrixiParticles.wrap_v(integrator.u.x[1], spec_sys, integrator.p)
            u_wrap   = TrixiParticles.wrap_u(integrator.u.x[2], spec_sys, integrator.p)
            x0_wrap  = TrixiParticles.initial_coordinates(spec_sys)

            # Commit plastic strain history at the end of each accepted step
            commit_plastic_history!(spec_sys,
                                    ys_particle_buf,
                                    hard_particle_buf,
                                    vis_particle_buf,
                                    integrator.dt,
                                    alpha_committed[],
                                    Fp_committed[])

            # Emergency repair of particle states
            # Skip for first 50 ms to allow contact development
            if enable_emergency_repair_clamp && integrator.t >= enable_repair_after_t
                @inbounds for particle in
                              TrixiParticles.each_integrated_particle(spec_sys)
                    x  = spec_sys.current_coordinates[1, particle]
                    z  = spec_sys.current_coordinates[3, particle]
                    vx = v_wrap[1, particle]; vz = v_wrap[3, particle]
                    F11 = spec_sys.deformation_grad[1, 1, particle]
                    F13 = spec_sys.deformation_grad[1, 3, particle]
                    F22 = spec_sys.deformation_grad[2, 2, particle]
                    F31 = spec_sys.deformation_grad[3, 1, particle]
                    F33 = spec_sys.deformation_grad[3, 3, particle]
                    J   = F22 * (F11 * F33 - F13 * F31)
                    invalid = !isfinite(x) || !isfinite(z) ||
                              !isfinite(vx) || !isfinite(vz) ||
                              x < -x_safety_bound || x > x_safety_bound ||
                              z < z_safety_min || z > z_safety_max ||
                              !isfinite(J) || J <= J_repair_min || J >= J_repair_max
                    if invalid
                        # Keep displacement and absolute coordinates consistent.
                        # Reset invalid particles to their own reference positions.
                        x_reset = x0_wrap[1, particle]
                        z_reset = x0_wrap[3, particle]

                        u_wrap[1, particle] = x_reset - x0_wrap[1, particle]
                        u_wrap[2, particle] = 0.0
                        u_wrap[3, particle] = z_reset - x0_wrap[3, particle]
                        spec_sys.current_coordinates[1, particle] = x_reset
                        spec_sys.current_coordinates[2, particle] = 0.0
                        spec_sys.current_coordinates[3, particle] = z_reset
                        v_wrap[1, particle] = 0.0
                        v_wrap[2, particle] = 0.0
                        v_wrap[3, particle] = 0.0
                        for ii in 1:3, jj in 1:3
                            spec_sys.deformation_grad[ii, jj, particle] =
                                (ii == jj) ? 1.0 : 0.0
                        end
                        Fp_committed[][1,1,particle]=1.0; Fp_committed[][1,2,particle]=0.0
                        Fp_committed[][1,3,particle]=0.0; Fp_committed[][2,1,particle]=0.0
                        Fp_committed[][2,2,particle]=1.0; Fp_committed[][2,3,particle]=0.0
                        Fp_committed[][3,1,particle]=0.0; Fp_committed[][3,2,particle]=0.0
                        Fp_committed[][3,3,particle]=1.0
                        alpha_committed[][particle] = 0.0
                    end
                end
            end
        end),

    PeriodicCallback(print_progress, 50e-6; save_positions=(false, false)),

    PeriodicCallback(print_particle_range_diagnostics,
                     diagnostics_interval;
                     save_positions=(false, false)),

    SolutionSavingCallback(dt=solution_save_interval,
                           prefix="sandstone_uniaxial_pseudo2d_solution",
                           max_coordinates=1.0e6,
                           tau_33=tau_33_quantity,
                           tau_vm=tau_vm_quantity),

    InfoCallback(interval=500))

# ==========================================================================================
# STEP 12: Solve
# ==========================================================================================

# CFL time step for explicit symplectic integration
# Eq. (14) suggests Δt = 0.5 h / c. For explicit TLSPH with contact and
# plasticity, use a stricter safety factor to avoid first-step instability.
dt_cfl = 0.05 * particle_spacing / c_sound

println("\n>>> Entering ODE solve")
println("    Solver          : SymplecticEuler (explicit symplectic)")
println("    tspan           : ", tspan, " s")
println("    dt_cfl          : ", round(dt_cfl; sigdigits=3), " s")
println("    Estimated steps : ", round(Int, t_total / dt_cfl))
println("    Piston velocity (physical) : ", abs(v_piston_physical) * 1e3, " mm/s")
println("    Piston speedup factor      : ", piston_speedup)
println("    Piston velocity (effective): ", abs(v_piston) * 1e3, " mm/s")
println("    Expected piston displacement at t_end : ",
        round(abs(v_piston) * t_total * 1e3; digits=2), " mm")
println("    Expected axial strain at t_end : ",
        round(abs(v_piston) * t_total / specimen_height_discrete * 100; digits=3), " %")
println()
println("    NOTE: To observe the yield point (ε ≈ ",
        round(yield_stress_specimen / E_young * 100; digits=3),
        " %), extend t_total to ≈ ",
        round(yield_stress_specimen / E_young * specimen_height_discrete / abs(v_piston);
              digits=1),
        " s or scale v_piston accordingly.")
flush(stdout)

# ---------- Primary solver: explicit symplectic ----------
sol = solve(ode, SymplecticEuler();
            callback=callbacks,
            dt=dt_cfl,
            save_everystep=false,
            maxiters=typemax(Int))

# ---------- Alternative: implicit TRBDF2 (uncomment for small-domain tests) ----------
# For the full 11,480-particle specimen the dense-LU Jacobian built by
# AutoFiniteDiff is not feasible (~46,000 DOF × 46,000 DOF matrix).
# Use this block only on reduced-size specimens (≲ 200 particles):
#
# using ADTypes
# linsolve_solver = RFLUFactorization()
# nlsolve = NLNewton(κ=1e-1, max_iter=20,
#                    fast_convergence_cutoff=0.9, always_new=false)
# sol = Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
#     solve(ode, TRBDF2(linsolve=linsolve_solver,
#                       autodiff=AutoFiniteDiff(),
#                       nlsolve=nlsolve);
#           callback=callbacks,
#           save_everystep=false,
#           abstol=1e-3, reltol=1e-3, dtmax=1e-3, maxiters=10_000_000)
# end

# ==========================================================================================
# STEP 13: Post-processing summary
# ==========================================================================================

println("\n=== SIMULATION COMPLETE ===")
println("Solver retcode    : ", sol.retcode)
println("Final time        : ", sol.t[end], " s")

spec_sys_final = sol.prob.p.systems[1]
z_max_f   = maximum(spec_sys_final.current_coordinates[3, :])
z_min_f   = minimum(spec_sys_final.current_coordinates[3, :])
h_final   = z_max_f - z_min_f
strain_f  = (specimen_height_discrete - h_final) / specimen_height_discrete * 100.0
disp_f    = abs(piston_displacement_at(sol.t[end])) * 1e3
alpha_max = maximum(alpha_committed[])
n_yielded = count(a -> a > 1e-6, alpha_committed[])

println("Piston displacement : ", round(disp_f;   digits=3), " mm")
println("Axial strain        : ", round(strain_f; digits=3), " %")
println("Max equiv. plastic strain α : ", round(alpha_max; digits=6))
println("Yielded particles   : ", n_yielded, " / ", n_specimen_particles,
        "  (", round(100.0 * n_yielded / n_specimen_particles; digits=1), " %)")
