# ==========================================================================================
# 3D Thermomechanical Forming of a CFK Clip-Style Part (Implicit)
#
# This variant follows the stamping_cfrp_3d_2_implicit workflow but replaces
# the simple flat mold setup with a more complex clip-style tooling concept:
# - Lower tool (female die): flat base + two side rails + center rib
# - Upper tool (male punch): flat holder + central punch + two shoulders
#
# The workpiece geometry is changed from a cylinder to a flat CFK raw charge
# (organosheet/UD-tape stack precursor), which is a common feedstock form for
# aerospace clip manufacturing.
# ==========================================================================================
using TrixiParticles
using OrdinaryDiffEq
using ADTypes
using LinearSolve
using RecursiveFactorization
using KrylovKit
using IncompleteLU
using PointNeighbors
using Plots
using Base.Threads
using Statistics  # for mean() function
using LinearAlgebra # for Identity matrix I
using SparseArrays
using Logging

println("--- SIMULATION STARTING ---")
println("Threads available: ", nthreads())
println("---------------------------")

# ==========================================================================================
# STEP 1: Build CFK raw-charge geometry (flat preform blank)
# ==========================================================================================

particle_spacing = 0.0025

# Mass Scaling Factor to artificially increase density for the explicit solver (~100x larger dt)
mass_scaling = 1.0

# Composite feedstock definition (glass-fibre reinforced thermoplastic)
matrix_density = 1180.0 * mass_scaling      # kg/m^3 (thermoplastic matrix)
glass_density = 2550.0 * mass_scaling       # kg/m^3 (E-glass)
matrix_E = 3.2e9             # Pa
glass_E = 72.0e9             # Pa
matrix_cp = 1800.0           # J/(kg.K)
glass_cp = 840.0             # J/(kg.K)
matrix_k = 0.22              # W/(m.K)
glass_k = 1.10               # W/(m.K)
matrix_tmelt = 430.0         # K (thermoplastic matrix melting range)
glass_tmelt = 1700.0         # K (glass does not melt in process window)
matrix_temp_liq = 390.0      # K
glass_temp_liq = 1500.0      # K
matrix_viscosity = 1.0e2     # Pa.s
glass_viscosity = 1.0e16     # Pa.s (effectively rigid in flow)
matrix_yield_stress = 9.0e7  # Pa
glass_yield_stress = 2.5e9   # Pa
matrix_hardening = 8.0e8     # Pa
glass_hardening = 8.0e9      # Pa
matrix_h_contact = 7.0e4
glass_h_contact = 2.0e4

# Flow-orientation kinetics parameters (Kinematic Draping / Organosheet)
fiber_aspect_ratio = 10000.0  # Approximates continuous/infinite filaments
xi_ft = (fiber_aspect_ratio^2 - 1.0) / (fiber_aspect_ratio^2 + 1.0)
ci_ft = 0.0                   # NO isotropic tumbling/collisions
kappa_rsc = 1.0               # NO strain reduction (1.0 = move exactly with the matrix)
ci_ard_parallel = 0.0         # NO anisotropic diffusion
ci_ard_perp = 0.0             # NO anisotropic diffusion
orientation_coupling_gain = 0.30

# Meso-architecture for the raw charge: alternating 0/90 plies with tow bands.
n_plies = 6
fiber_vf_in_tow = 0.58       # local Vf inside a glass-fibre tow
fiber_vf_resin_rich = 0.02   # local Vf in resin-rich regions between tows
tow_width = 0.0040           # m
tow_gap = 0.0015             # m

# Raw charge dimensions (meters): flat precursor used before clip forming.
charge_length = 0.060
charge_width = 0.022
charge_thickness = 0.0035

n_charge = (ceil(Int, charge_length / particle_spacing),
            ceil(Int, charge_width / particle_spacing),
            max(2, ceil(Int, charge_thickness / particle_spacing)))
charge_origin = (-0.5 * charge_length, -0.5 * charge_width, 0.0)

function build_glass_fiber_charge(particle_spacing, n_charge, charge_origin;
                                  matrix_density, glass_density,
                                  n_plies, fiber_vf_in_tow, fiber_vf_resin_rich,
                                  tow_width, tow_gap)
    nx, ny, nz = n_charge
    n_total = nx * ny * nz

    coordinates = Matrix{Float64}(undef, 3, n_total)
    velocity = zeros(3, n_total)
    mass = Vector{Float64}(undef, n_total)
    density = Vector{Float64}(undef, n_total)

    fiber_volume_fraction = Vector{Float64}(undef, n_total)
    fiber_direction = Matrix{Float64}(undef, 3, n_total)
    idx_fiber_rich = Int[]
    idx_matrix_rich = Int[]

    x0, y0, z0 = charge_origin
    ply_thickness = nz > 0 ? (nz * particle_spacing) / n_plies : particle_spacing
    tow_pitch = tow_width + tow_gap

    p = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        p += 1

        x = x0 + (i - 0.5) * particle_spacing
        y = y0 + (j - 0.5) * particle_spacing
        z = z0 + (k - 0.5) * particle_spacing

        coordinates[1, p] = x
        coordinates[2, p] = y
        coordinates[3, p] = z

        z_rel = max(0.0, z - z0)
        ply_id = clamp(floor(Int, z_rel / ply_thickness) + 1, 1, n_plies)
        ply_aligned_x = isodd(ply_id)

        selector = ply_aligned_x ? (y - y0) : (x - x0)
        in_tow = mod(selector, tow_pitch) <= tow_width

        vf_local = in_tow ? fiber_vf_in_tow : fiber_vf_resin_rich
        fiber_volume_fraction[p] = vf_local

        if ply_aligned_x
            fiber_direction[1, p] = 1.0
            fiber_direction[2, p] = 0.0
            fiber_direction[3, p] = 0.0
        else
            fiber_direction[1, p] = 0.0
            fiber_direction[2, p] = 1.0
            fiber_direction[3, p] = 0.0
        end

        rho_local = vf_local * glass_density + (1.0 - vf_local) * matrix_density
        density[p] = rho_local
        mass[p] = rho_local * particle_spacing^3

        if vf_local >= 0.5 * fiber_vf_in_tow
            push!(idx_fiber_rich, p)
        else
            push!(idx_matrix_rich, p)
        end
    end

    ic = InitialCondition(; coordinates, velocity, mass, density)

    return ic, fiber_volume_fraction, fiber_direction, idx_fiber_rich, idx_matrix_rich
end

polymer, fiber_volume_fraction, fiber_direction,
idx_fiber_rich, idx_matrix_rich =
    build_glass_fiber_charge(particle_spacing, n_charge, charge_origin;
                             matrix_density=matrix_density,
                             glass_density=glass_density,
                             n_plies=n_plies,
                             fiber_vf_in_tow=fiber_vf_in_tow,
                             fiber_vf_resin_rich=fiber_vf_resin_rich,
                             tow_width=tow_width,
                             tow_gap=tow_gap)

density_cylinder = mean(polymer.density)

vf_mean = mean(fiber_volume_fraction)
fiber_stiffness_efficiency = 0.35
E_composite_eff = matrix_E * (1.0 - vf_mean) + glass_E * vf_mean * fiber_stiffness_efficiency
cp_composite_eff = matrix_cp * (1.0 - vf_mean) + glass_cp * vf_mean
k_composite_eff = matrix_k * (1.0 - vf_mean) + glass_k * vf_mean

phase_vf_particle = copy(fiber_volume_fraction)
cp_particle = matrix_cp .* (1.0 .- phase_vf_particle) .+ glass_cp .* phase_vf_particle
k_particle = matrix_k .* (1.0 .- phase_vf_particle) .+ glass_k .* phase_vf_particle
temp_liq_particle = matrix_temp_liq .* (1.0 .- phase_vf_particle) .+ glass_temp_liq .* phase_vf_particle
tmelt_particle = matrix_tmelt .* (1.0 .- phase_vf_particle) .+ glass_tmelt .* phase_vf_particle
# Use phase-specific liquidus temperature as the flow trigger.
# Matrix-rich particles will flow (preheat 440K > ~412K threshold),
# but fiber-rich particles will safely remain elastic (440K < ~1033K threshold).
# PHYSICAL MODEL: In a fiber-reinforced thermoplastic the matrix governs the
# elastic↔viscous transition. Fibers stay solid but are carried by the molten
# matrix; their effect is captured through `viscosity_base_particle` (glass
# contribution ~1e16 Pa·s makes in-tow regions effectively rigid while still
# in the viscous regime). Using `matrix_temp_liq` uniformly therefore
#   (i) correctly activates flow everywhere once the matrix melts,
#   (ii) avoids a fictitious per-particle transition temperature from
#        averaging matrix (390 K) and glass (1500 K) Tm,
#   (iii) removes the tow/resin regime discontinuity that stresses the Jacobian.
local_regime_threshold_particle = fill(matrix_temp_liq, length(phase_vf_particle))
yield_base_particle = matrix_yield_stress .* (1.0 .- phase_vf_particle) .+
                      glass_yield_stress .* phase_vf_particle
hardening_base_particle = matrix_hardening .* (1.0 .- phase_vf_particle) .+
                         glass_hardening .* phase_vf_particle
viscosity_base_particle = matrix_viscosity .* (1.0 .- phase_vf_particle) .+
                          glass_viscosity .* phase_vf_particle
h_contact_particle = matrix_h_contact .* (1.0 .- phase_vf_particle) .+
                     glass_h_contact .* phase_vf_particle

orientation_scalar_particle = ones(length(phase_vf_particle))

# Implicit linear solves become ill-conditioned if effective viscosity is too large.
# Keep a finite numerical cap for robustness while preserving phase contrast.
viscosity_min_clip = 1.0e2
viscosity_max_clip = 5.0e7
temp_min_clip = 200.0
temp_max_clip = 2000.0

factor = 1.2
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = factor * particle_spacing

println("✓ CFK raw charge initialized:")
println("  - size [mm] = ", round.(SVector(charge_length, charge_width, charge_thickness) .* 1e3,
                                      digits=2))
println("  - particles = ", nparticles(polymer))
println("  - architecture = ", n_plies, " plies (0/90 alternating), tow width=", tow_width * 1e3,
        " mm, tow gap=", tow_gap * 1e3, " mm")
println("  - mean fibre volume fraction Vf = ", round(vf_mean, digits=4))
println("  - phase counts (fibre-rich/matrix-rich) = ", length(idx_fiber_rich), " / ",
        length(idx_matrix_rich))
println("  - homogenized rho = ", round(density_cylinder, digits=2),
        " kg/m^3, E = ", round(E_composite_eff / 1e9, digits=3), " GPa")
println("  - phase-specific tmelt matrix/glass = ", matrix_tmelt, " / ", glass_tmelt, " K")
println("  - mean per-particle regime threshold = ",
    round(mean(local_regime_threshold_particle), digits=2), " K")

# ==========================================================================================
# STEP 2: Visualize raw charge
# ==========================================================================================

coords = polymer.coordinates
# scatter3d(coords[1, :], coords[2, :], coords[3, :],
#           markersize=1, aspect_ratio=:equal, label="CFK raw charge particles")
# savefig("packing_cfk_charge.png")

# ==========================================================================================
# STEP 3: Build complex clip-style mold/floor tooling
# ==========================================================================================

floor_density = density_cylinder  # match fluid density for Adami extrapolation

function merge_rect_shapes(shapes)
    all_coords = reduce(hcat, map(s -> s.coordinates, shapes))
    all_vel = reduce(hcat, map(s -> s.velocity, shapes))
    all_mass = reduce(vcat, map(s -> s.mass, shapes))
    all_dens = reduce(vcat, map(s -> s.density, shapes))
    return InitialCondition(; coordinates=all_coords, velocity=all_vel,
                mass=all_mass, density=all_dens)
end

cyl_z_min = minimum(polymer.coordinates[3, :])
cyl_z_max = maximum(polymer.coordinates[3, :])

# Keep one-spacing clearance before first contact.
initial_gap = 0.5 * particle_spacing
initial_gap_upper = 0.5 * particle_spacing

z_surface     = cyl_z_min - initial_gap
z_top_surface = cyl_z_max + initial_gap_upper

tool_length = charge_length + 0.014
tool_width = charge_width + 0.014

tool_layers_base = 3
tool_layers_feature = 3
tool_feature_w = 0.0035

nx_tool = ceil(Int, tool_length / particle_spacing)
ny_tool = ceil(Int, tool_width / particle_spacing)
nf_tool = ceil(Int, tool_feature_w / particle_spacing)

tool_origin_x = -0.5 * tool_length
tool_origin_y = -0.5 * tool_width

floor_thickness = tool_layers_base * particle_spacing
z_bottom = z_surface - floor_thickness

# Female die style lower tool: base + two side rails + center rib.
floor_base = RectangularShape(particle_spacing, (nx_tool, ny_tool, tool_layers_base),
                  (tool_origin_x, tool_origin_y, z_bottom);
                  density=floor_density)
floor_left_rail = RectangularShape(particle_spacing, (nx_tool, nf_tool, tool_layers_feature),
                   (tool_origin_x, tool_origin_y,
                    z_bottom + floor_thickness - tool_layers_feature * particle_spacing);
                   density=floor_density)
floor_right_rail = RectangularShape(particle_spacing, (nx_tool, nf_tool, tool_layers_feature),
                    (tool_origin_x,
                     tool_origin_y + tool_width - tool_feature_w,
                     z_bottom + floor_thickness - tool_layers_feature * particle_spacing);
                    density=floor_density)
floor_center_rib = RectangularShape(particle_spacing,
                    (ceil(Int, 0.65 * nx_tool), nf_tool, tool_layers_feature),
                    (-0.325 * tool_length, -0.5 * tool_feature_w,
                     z_bottom + floor_thickness - tool_layers_feature * particle_spacing);
                    density=floor_density)
floor_front_wall = RectangularShape(particle_spacing,
                    (nf_tool, ny_tool, tool_layers_feature),
                    (tool_origin_x, tool_origin_y,
                     z_bottom + floor_thickness - tool_layers_feature * particle_spacing);
                    density=floor_density)
floor_back_wall = RectangularShape(particle_spacing,
                   (nf_tool, ny_tool, tool_layers_feature),
                   (tool_origin_x + tool_length - tool_feature_w, tool_origin_y,
                    z_bottom + floor_thickness - tool_layers_feature * particle_spacing);
                   density=floor_density)
floor_particles = merge_rect_shapes((floor_base, floor_left_rail, floor_right_rail, floor_center_rib,
                                     floor_front_wall, floor_back_wall))

z_top = z_top_surface + tool_layers_feature * particle_spacing
center_channel_w = 1.6 * tool_feature_w

# Male punch style upper tool: holder + split center punch (with center channel) + shoulders.
mold_holder = RectangularShape(particle_spacing, (nx_tool, ny_tool, tool_layers_base),
                   (tool_origin_x, tool_origin_y, z_top);
                   density=floor_density)
mold_center_punch = RectangularShape(particle_spacing,
                     (ceil(Int, 0.55 * nx_tool), nf_tool, tool_layers_feature),
                     (-0.275 * tool_length, -0.5 * center_channel_w - tool_feature_w,
                      z_top - tool_layers_feature * particle_spacing);
                     density=floor_density)
mold_center_punch_right = RectangularShape(particle_spacing,
                     (ceil(Int, 0.55 * nx_tool), nf_tool, tool_layers_feature),
                     (-0.275 * tool_length, 0.5 * center_channel_w,
                      z_top - tool_layers_feature * particle_spacing);
                     density=floor_density)
mold_left_shoulder = RectangularShape(particle_spacing,
                      (ceil(Int, 0.70 * nx_tool), nf_tool, tool_layers_feature),
                      (-0.35 * tool_length,
                       tool_origin_y + 0.22 * tool_width,
                       z_top - tool_layers_feature * particle_spacing);
                      density=floor_density)
mold_right_shoulder = RectangularShape(particle_spacing,
                       (ceil(Int, 0.70 * nx_tool), nf_tool, tool_layers_feature),
                       (-0.35 * tool_length,
                    tool_origin_y + 0.78 * tool_width - tool_feature_w,
                    z_top - tool_layers_feature * particle_spacing);
                       density=floor_density)
mold_front_lip = RectangularShape(particle_spacing,
                                            (nf_tool, ny_tool, tool_layers_feature),
                                            (tool_origin_x, tool_origin_y,
                                                z_top - tool_layers_feature * particle_spacing);
                                            density=floor_density)
mold_back_lip = RectangularShape(particle_spacing,
                                         (nf_tool, ny_tool, tool_layers_feature),
                                         (tool_origin_x + tool_length - tool_feature_w, tool_origin_y,
                                            z_top - tool_layers_feature * particle_spacing);
                                         density=floor_density)
mold_particles = merge_rect_shapes((mold_holder, mold_center_punch,
                          mold_center_punch_right,
                                                    mold_left_shoulder, mold_right_shoulder,
                                                    mold_front_lip, mold_back_lip))

floor_z_max = maximum(floor_particles.coordinates[3,:])
println("Gap = ", cyl_z_min - floor_z_max)
println("h   = ", factor* particle_spacing)
println("Contact at t=0? ", (cyl_z_min - floor_z_max) < factor * particle_spacing)
println("Tool particles (floor/mold) = ", nparticles(floor_particles), " / ",
    nparticles(mold_particles))

# ==========================================================================================
# THERMOMECHANICAL STATE INITIALIZATION (for elastoplastic-viscous-thermal regime)
# ==========================================================================================
n_cylinder_particles = nparticles(polymer)
alpha_plastic_state = Ref(zeros(n_cylinder_particles))  # accumulated plastic strain per particle (Ref avoids closure scoping issues)

# Persistent plastic deformation gradient tensor Fp for all particles
Fp_initial = zeros(3, 3, n_cylinder_particles)
for i in 1:n_cylinder_particles
    Fp_initial[:, :, i] .= Matrix{Float64}(I, 3, 3)
end
Fp_state = Ref(Fp_initial)

# Committed (frozen) plastic history — only updated after each accepted step.
# These are the read-only inputs to elastic_stress3d_trial! inside Newton residual
# evaluations, ensuring the residual is consistent across all Newton iterations.
alpha_committed = Ref(zeros(n_cylinder_particles))
Fp_committed    = Ref(copy(Fp_initial))

F_total_mold_state = Ref(0.0)                          # total force on mold (for tracking)

# Mold velocity (downward push, m/s)
mold_velocity_state = -0.02
# Use a longer ramp to avoid a sharp contact impulse when mold first engages.
t_ramp_mold = 2.5e-2

# PrescribedMotion for rigid tools:
# Floor: fully stationary (no motion object needed — clamped_particles_motion=nothing)
# Mold:  translates downward at constant velocity
mold_motion = PrescribedMotion(
    (x, t) -> begin
        # Smooth velocity ramp to avoid impulsive contact loading at t=0.
        z_shift = if t < t_ramp_mold
            0.5 * mold_velocity_state / t_ramp_mold * t^2
        else
            mold_velocity_state * (t - 0.5 * t_ramp_mold)
        end
        x + SVector(0.0, 0.0, z_shift)
    end,
    t -> true)  # always moving

# Boundary coordinate for thermal BC: (dimension_index, coordinate_value)
bound_coordinate_thermal = (3, minimum(floor_particles.coordinates[3, :]))

println("✓ Thermomechanical regime enabled:")
println("  - Plasticity: yield_stress = 150 MPa, hardening = 1e9 Pa")
println("  - Thermal effects: cp = 900 J/kg·K, k_thermal = 1.0, temp_melt = 700K")
println("  - Viscous regime: activated when T > matrix_temp_liq")
println("  - Initial temperature: 270 K")
println("  - Composite feedstock: glass fibres embedded in thermoplastic matrix")

# ==========================================================================================
# STEP 5: Build the Total Lagrangian SPH system for the cylinder
# ==========================================================================================

material_polymer = (density=density_cylinder, E=E_composite_eff, nu=0.3 ,beta=0.000,
                            temp=270.0, temp_ref=270.0,
                            cp=mean(cp_particle), k=mean(k_particle),
                            temp_liq=mean(temp_liq_particle),
                            h=mean(h_contact_particle),
                            hardening=mean(hardening_base_particle),
                            tmelt=mean(tmelt_particle),
                            viscosity=mean(viscosity_base_particle),
                            yield_stress=mean(yield_base_particle))

import PointNeighbors: DictionaryCellList

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=200)

# Floor: all particles clamped → no ODE integration, rigid boundary for contact
floor_system = TotalLagrangianSPHSystem(floor_particles,
                                           smoothing_kernel,
                                           smoothing_length,
                                           1e11,
                                           material_polymer.nu,
                                           material_polymer.beta,
                                           material_polymer.temp,
                                           material_polymer.temp_ref,
                                           material_polymer.cp,
                                           material_polymer.k,
                                           material_polymer.temp_liq,
                                           material_polymer.h,
                                           material_polymer.hardening,
                                           material_polymer.tmelt,
                                           material_polymer.yield_stress;
                                           clamped_particles=collect(1:nparticles(floor_particles)),
                                           acceleration=(0.0, 0.0, 0.0))
                                           #self_interaction_nhs=nhs_template)

# Mold: all particles clamped with prescribed downward motion — no ODE integration
mold_system = TotalLagrangianSPHSystem(mold_particles,
                                           smoothing_kernel,
                                           smoothing_length,
                                           1e11,
                                           material_polymer.nu,
                                           material_polymer.beta,
                                           material_polymer.temp,
                                           material_polymer.temp_ref,
                                           material_polymer.cp,
                                           material_polymer.k,
                                           material_polymer.temp_liq,
                                           material_polymer.h,
                                           material_polymer.hardening,
                                           material_polymer.tmelt,
                                           material_polymer.yield_stress;
                                           clamped_particles=collect(1:nparticles(mold_particles)),
                                           clamped_particles_motion=mold_motion,
                                           acceleration=(0.0, 0.0, 0.0),
                                           self_interaction_nhs=nhs_template)

# Quasi-static — prescribed displacement, no free fall

# ------------------------------------------------------------------------------------------
# 3-2-1 locator pins on the charge (Option 1: soft in-plane anchor)
# Purpose: suppress spurious rigid-body rotation (especially yaw about z) caused by
#   (a) particle-level asymmetry of the upper tool footprint, and
#   (b) anisotropic ply response under a nominally vertical press load.
# We fully clamp 2 particles on the bottom face of the charge, on the centerline y = 0,
# offset along x. Two pins on a horizontal line kill all 3 rigid-body in-plane DOFs
# (x-translation, y-translation, z-rotation) while leaving the rest of the blank free
# to deform. Using bottom-face particles is consistent with them resting on the flat
# floor (no artificial hovering). The artifact is confined to these ~2 particles out of
# thousands, so local stress is negligible.
# ------------------------------------------------------------------------------------------
function _find_nearest_particle_index(coords, target)
    best_idx = 1
    best_d2 = Inf
    @inbounds for p in axes(coords, 2)
        dx = coords[1, p] - target[1]
        dy = coords[2, p] - target[2]
        dz = coords[3, p] - target[3]
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2
            best_d2 = d2
            best_idx = p
        end
    end
    return best_idx
end

_charge_z_bottom = minimum(polymer.coordinates[3, :])
_pin_target_A = (0.0,                    0.0, _charge_z_bottom)
_pin_target_B = (+0.40 * charge_length,  0.0, _charge_z_bottom)
_pin_idx_A = _find_nearest_particle_index(polymer.coordinates, _pin_target_A)
_pin_idx_B = _find_nearest_particle_index(polymer.coordinates, _pin_target_B)
charge_locator_pins = unique([_pin_idx_A, _pin_idx_B])
println("Charge locator pins (3-2-1): indices = ", charge_locator_pins,
        "  at coords:")
for p in charge_locator_pins
    println("  pin ", p, " -> (",
            round(polymer.coordinates[1, p]; digits=5), ", ",
            round(polymer.coordinates[2, p]; digits=5), ", ",
            round(polymer.coordinates[3, p]; digits=5), ")")
end

cylinder_system = TotalLagrangianSPHSystem(polymer,
                                           smoothing_kernel,
                                           smoothing_length,
                                           material_polymer.E,      # realistic composite E (~11 GPa)
                                           material_polymer.nu,    # poisson ratio
                                           material_polymer.beta,             # beta
                                           material_polymer.temp,            # temp — scalar
                                           material_polymer.temp_ref,            # temp_ref — scalar
                                           material_polymer.cp,              # cp — scalar
                                           material_polymer.k,              # k — scalar
                                           material_polymer.temp_liq,              # temp_liq — scalar
                                           material_polymer.h,                  # h — scalar
                                           material_polymer.hardening,           # hardening  — scalar   
                                           material_polymer.tmelt,              # tmelt — scalar
                                           material_polymer.yield_stress;     # yield_stress
                                           acceleration=(0.0, 0.0, 0.0),
                                           self_interaction_nhs=nhs_template)
# NOTE ON LOCATOR PINS (partial-DOF, x/y only):
# `TotalLagrangianSPHSystem` only supports full-DOF clamping via `clamped_particles`,
# which would also freeze z and interfere with vertical compression at those particles.
# Instead, we enforce the pins as partial constraints (x, y only) inside the kick
# function below: for each pin particle we zero dv_x, dv_y (and v_x, v_y), leaving
# v_z / dv_z free so the charge can compress vertically normally at those locations.
# See `_apply_charge_pins!(...)` invoked at the end of `kick_implicit_visible!`.
# ==========================================================================================
# STEP 6: Semidiscretization and solve
# ==========================================================================================

semi = Semidiscretization(cylinder_system, floor_system, mold_system;
                          neighborhood_search=GridNeighborhoodSearch{3}(;
                              cell_list=DictionaryCellList{3}(),
                              search_radius=smoothing_length))

# ==========================================================================================
# PREHEATING PHASE: Uniform temperature initialization before compression
# ==========================================================================================
# Set cylinder to preheated temperature directly (uniform — no spatial gradient needed).
# Must use semi.systems[1] because Semidiscretization creates new system objects internally;
# the original cylinder_system reference is no longer the object registered in semi.
preheating_target_temp = 440.0  # Target: 440 K (between room temp 270K and melt 700K)

println("\n=== PREHEATING PHASE ===")
println("Setting cylinder temperature to $preheating_target_temp K...")
semi.systems[1].temp .= preheating_target_temp
println("✓ Cylinder preheated: T = $preheating_target_temp K")

# Preheat tools (floor + mold) to reduce the first-contact thermal shock that
# otherwise introduces a stiff transient and collapses the implicit solver dt.
# In real thermoforming / stamping of thermoplastic composites the tools are
# preheated close to (or slightly below) the blank temperature; here we pick
# a value that leaves a modest, physically meaningful ΔT for cooling, while
# removing the ~170 K shock that was freezing the solver at first engagement.
tool_preheat_temp = 400.0  # K — below blank (440 K) but well above cold-tool 270 K
semi.systems[2].temp .= tool_preheat_temp   # floor
semi.systems[3].temp .= tool_preheat_temp   # mold
println("✓ Floor preheated: T = $tool_preheat_temp K")
println("✓ Mold  preheated: T = $tool_preheat_temp K")
println("=== PREHEATING COMPLETE ===\n")

cylinder_height = maximum(polymer.coordinates[3,:]) - minimum(polymer.coordinates[3,:])
t_compress = 6 * cylinder_height * 0.5 / abs(mold_velocity_state)
tspan = (0.0, t_compress)

ode_base = semidiscretize(semi, tspan)

# Pre-allocate persistent buffers for implicit RHS stress-cache path.
v_stress_buf          = zeros(3, 3, n_cylinder_particles)
v_stress_elastic_buf  = zeros(3, 3, n_cylinder_particles)
v_stress_viscous_buf  = zeros(3, 3, n_cylinder_particles)
vel_grad_buf          = zeros(3, 3, n_cylinder_particles)

dt_cap = 1.0e-3
# Best available estimate of the current implicit step size seen by the residual.
# The RHS itself does not receive the integrator, so this is updated from callbacks
# and used by elastic_stress3d_trial! instead of a fixed dt_cap.
trial_dt_state = Ref(dt_cap)
nhs_updated_at_t = Ref(-Inf)
contact_diag_interval = 5000
rhs_eval_counter = Ref(0)
first_rhs_call = Ref(true)  # fires once to confirm RHS compilation is done
solver_diag_interval = 20
rhs_print_interval = 10
solver_diag_last_accept = Ref(0)
solver_diag_last_iter = Ref(0)
solver_diag_last_rhs = Ref(0)
solver_diag_dt_min = Ref(Inf)
solver_diag_dt_max = Ref(0.0)
solver_diag_dt_sum = Ref(0.0)
enable_contact_diag = false
contact_heat_flux_buf = zeros(n_cylinder_particles)
ys_particle_buf = zeros(n_cylinder_particles)
hard_particle_buf = zeros(n_cylinder_particles)
vis_particle_buf = zeros(n_cylinder_particles)
orientation_tensor_state = Ref(zeros(3, 3, n_cylinder_particles))

# One-time print sentinels — each fires exactly once when the condition first occurs.
first_viscous_regime    = Ref(true)
first_elastic_regime    = Ref(true)
first_floor_heat_xfer   = Ref(true)
first_mold_heat_xfer    = Ref(true)

# Lightweight RHS stage timing (accumulated ns across RHS calls).
rhs_time_pos_ns = Ref(0)
rhs_time_nhs_ns = Ref(0)
rhs_time_quant_ns = Ref(0)
rhs_time_implicit_ns = Ref(0)
rhs_time_pressure_ns = Ref(0)
rhs_time_boundary_ns = Ref(0)
rhs_time_final_ns = Ref(0)
rhs_time_stress_interact_ns = Ref(0)
rhs_time_thermal_ns = Ref(0)

# Snapshots used to print per-interval deltas in solver diagnostics.
rhs_time_pos_last_ns = Ref(0)
rhs_time_nhs_last_ns = Ref(0)
rhs_time_quant_last_ns = Ref(0)
rhs_time_implicit_last_ns = Ref(0)
rhs_time_pressure_last_ns = Ref(0)
rhs_time_boundary_last_ns = Ref(0)
rhs_time_final_last_ns = Ref(0)
rhs_time_stress_interact_last_ns = Ref(0)
rhs_time_thermal_last_ns = Ref(0)

function contact_condition(cyl_sys, floor_sys, mold_sys, i, contact_dist)
    z_i         = cyl_sys.current_coordinates[3, i]
    z_floor_top = maximum(floor_sys.current_coordinates[3, :])
    z_mold_bot  = minimum(mold_sys.current_coordinates[3, :])
    return (z_i - z_floor_top) <= contact_dist || (z_mold_bot - z_i) <= contact_dist
end

# function contact_stats(system_cyl, system_floor, system_mold, contact_dist)
#     z_floor_top = maximum(system_floor.current_coordinates[3, :])
#     z_mold_bot  = minimum(system_mold.current_coordinates[3, :])

#     n_floor_contact = 0
#     n_mold_contact = 0
#     n_floor_penetrated = 0
#     n_mold_penetrated = 0
#     min_gap_floor = Inf
#     min_gap_mold = Inf

#     @inbounds for i in 1:nparticles(system_cyl)
#         z_i = system_cyl.current_coordinates[3, i]
#         gap_floor = z_i - z_floor_top
#         gap_mold = z_mold_bot - z_i
#         min_gap_floor = min(min_gap_floor, gap_floor)
#         min_gap_mold = min(min_gap_mold, gap_mold)

#         if 0.0 <= gap_floor <= contact_dist
#             n_floor_contact += 1
#         elseif gap_floor < 0.0
#             n_floor_penetrated += 1
#         end

#         if 0.0 <= gap_mold <= contact_dist
#             n_mold_contact += 1
#         elseif gap_mold < 0.0
#             n_mold_penetrated += 1
#         end
#     end

#     return n_floor_contact, n_mold_contact, n_floor_penetrated, n_mold_penetrated,
#            min_gap_floor, min_gap_mold
# end

function compute_contact_heat_flux!(ext_heat_per_particle, system_cyl, system_floor,
                                    system_mold, particle_spacing)
    # In-place variant to avoid per-RHS allocations in implicit mode.
    fill!(ext_heat_per_particle, 0.0)

    z_floor_top = maximum(system_floor.current_coordinates[3, :])
    z_mold_bot  = minimum(system_mold.current_coordinates[3, :])

    T_floor = mean(system_floor.temp)
    T_mold = mean(system_mold.temp)

    h_contact = system_cyl.h
    # Smoothly ramp flux over a transition band rather than a step at the
    # threshold. This removes the C0 discontinuity in the RHS that otherwise
    # collapses the implicit solver's dt at first contact. Use a Hermite
    # smoothstep S(x) = x^2 * (3 - 2x) over x ∈ [0, 1], with S(0)=0, S(1)=1
    # and S'(0) = S'(1) = 0, giving a C1-smooth activation.
    contact_threshold = 0.25 * particle_spacing
    # Temperature smoothing bandwidth: use a small ΔT band so heat flux turns on
    # smoothly around T_cyl = T_tool instead of a hard `T_i > T_tool` branch.
    T_smooth_band = 5.0  # K — much smaller than the preheat ΔT so it has negligible
                          # physical effect but eliminates the other discontinuity.

    @inline smoothstep01(x) = begin
        y = clamp(x, 0.0, 1.0)
        y * y * (3.0 - 2.0 * y)
    end
    # Activation of flux as gap decreases from contact_threshold → 0.
    @inline gap_activation(gap) = 1.0 - smoothstep01(gap / contact_threshold)
    # Activation of flux as ΔT crosses 0 with a small smoothing band.
    @inline dT_activation(dT) = smoothstep01(dT / T_smooth_band)

    @inbounds for i in 1:nparticles(system_cyl)
        z_i = system_cyl.current_coordinates[3, i]
        T_i = system_cyl.temp[i]

        # -------- floor contact --------
        gap_floor = z_i - z_floor_top
        if 0.0 <= gap_floor <= contact_threshold
            dT = T_i - T_floor
            # smooth in BOTH gap and ΔT to keep the RHS C1-smooth
            flux_floor = h_contact * dT * gap_activation(gap_floor) * dT_activation(dT)
            if flux_floor > 0.0
                ext_heat_per_particle[i] = flux_floor
                if first_floor_heat_xfer[]
                    println(">>> Floor heat transfer started: T_cyl=", round(T_i, digits=2),
                            " K, T_floor=", round(T_floor, digits=2),
                            " K, gap=", round(gap_floor * 1e3, digits=3), " mm")
                    flush(stdout)
                    first_floor_heat_xfer[] = false
                end
            end
        end

        # -------- mold contact --------
        gap_mold = z_mold_bot - z_i
        if 0.0 <= gap_mold <= contact_threshold
            dT = T_i - T_mold
            flux_mold = h_contact * dT * gap_activation(gap_mold) * dT_activation(dT)
            if flux_mold > 0.0
                ext_heat_per_particle[i] = max(ext_heat_per_particle[i], flux_mold)
                if first_mold_heat_xfer[]
                    println(">>> Mold heat transfer started: T_cyl=", round(T_i, digits=2),
                            " K, T_mold=", round(T_mold, digits=2),
                            " K, gap=", round(gap_mold * 1e3, digits=3), " mm")
                    flush(stdout)
                    first_mold_heat_xfer[] = false
                end
            end
        end
    end

    return ext_heat_per_particle
end

function project_orientation_tensor!(A)
    A_sym = 0.5 .* (A .+ A')
    eig = eigen(Symmetric(A_sym))
    vals = clamp.(eig.values, 1.0e-8, 1.0)
    vals ./= sum(vals)
    A_proj = eig.vectors * Diagonal(vals) * eig.vectors'
    A .= 0.5 .* (A_proj .+ A_proj')
    return A
end

function initialize_orientation_tensor_state(fiber_direction; a0=0.85)
    n = size(fiber_direction, 2)
    A_state = zeros(3, 3, n)
    I3 = Matrix{Float64}(I, 3, 3)

    @inbounds for i in 1:n
        p_col = fiber_direction[:, i]
        nrm = norm(p_col)
        if nrm < eps(Float64)
            p = [1.0, 0.0, 0.0]
        else
            p = p_col / nrm
        end
        P = p * p'
        A_state[:, :, i] .= a0 .* P .+ 0.5 .* (1.0 - a0) .* (I3 .- P)
        project_orientation_tensor!(view(A_state, :, :, i))
    end

    return A_state
end

orientation_tensor_state[] = initialize_orientation_tensor_state(fiber_direction)

function update_flow_orientation_kinetics!(A_state, orientation_scalar, vel_grad, dt,
                                           fiber_direction;
                                           xi_ft, ci_ft, kappa_rsc,
                                           ci_ard_parallel, ci_ard_perp)
    I3 = Matrix{Float64}(I, 3, 3)
    dt_eff = max(dt, 1.0e-12)

    @inbounds for i in 1:size(A_state, 3)
        A = view(A_state, :, :, i)
        L = view(vel_grad, :, :, i)

        D = 0.5 .* (L .+ L')
        W = 0.5 .* (L .- L')
        gamma_dot = sqrt(2.0 * sum(D .* D))

        trAD = sum(A .* D)
        jeffery = W * A - A * W +
                  kappa_rsc * xi_ft * (D * A + A * D - 2.0 * trAD .* A)

        p_col = fiber_direction[:, i]
        pnorm = norm(p_col)
        if pnorm < eps(Float64)
            p = [1.0, 0.0, 0.0]
        else
            p = p_col / pnorm
        end

        D_r = ci_ard_perp .* I3 .+ (ci_ard_parallel - ci_ard_perp) .* (p * p')
        diffusion = 2.0 * ci_ft * gamma_dot .* (I3 .- 3.0 .* A)
        ard = 2.0 * gamma_dot .* (D_r .- tr(D_r) .* A)

        A .+= dt_eff .* (jeffery .+ diffusion .+ ard)
        project_orientation_tensor!(A)

        align = clamp(dot(p, A * p), 1.0 / 3.0, 1.0)
        align_norm = (align - 1.0 / 3.0) / (2.0 / 3.0)
        orientation_scalar[i] = 0.9 + 0.3 * align_norm
    end

    return nothing
end

function update_dual_phase_properties!(ys, hard, vis, temp, alpha)
    @inbounds for i in eachindex(temp)
        temp_i = temp[i]
        if !isfinite(temp_i)
            temp_i = matrix_temp_liq
        end
        temp_i = clamp(temp_i, temp_min_clip, temp_max_clip)

        tm_i = tmelt_particle[i]
        # Smoothly reduce solid strength above ~80% of local tmelt.
        melt_softening = clamp((temp_i - 0.8 * tm_i) / (0.4 * tm_i + eps(Float64)), 0.0, 1.0)
        solid_factor = 1.0 - melt_softening
        orient_scale = 1.0 + phase_vf_particle[i] * orientation_coupling_gain *
                             (orientation_scalar_particle[i] - 1.0)

        ys[i] = max(1.0e3, yield_base_particle[i] * solid_factor +
                           hardening_base_particle[i] * alpha[i] * solid_factor)
        ys[i] *= orient_scale
        hard[i] = hardening_base_particle[i] * solid_factor * orient_scale
        vis[i] = viscosity_base_particle[i] * (0.2 + 0.8 * solid_factor) * orient_scale

        # Numerical cap: glass η = 1e16 Pa·s otherwise dominates the Jacobian
        # conditioning and stalls Newton. Clamp to [viscosity_min_clip, viscosity_max_clip].
        vis[i] = clamp(vis[i], viscosity_min_clip, viscosity_max_clip)
    end
end

function update_implicit_stress_cache!(semi_local, v_ode, t)
    cyl_sys   = semi_local.systems[1]
    floor_sys = semi_local.systems[2]
    mold_sys  = semi_local.systems[3]

    rhs_eval_counter[] += 1
    # if enable_contact_diag && rhs_eval_counter[] > 0 &&
    #    mod(rhs_eval_counter[], contact_diag_interval) == 0
    #     near_contact_dist = smoothing_length
    #     n_floor_contact, n_mold_contact, n_floor_penetrated, n_mold_penetrated,
    #     min_gap_floor, min_gap_mold = contact_stats(cyl_sys, floor_sys, mold_sys,
    #                                                  near_contact_dist)
    #     # println("implicit rhs diag | t=", round(t, digits=6),
    #     #         " | mold_particles=", n_mold_contact,
    #     #         " | floor_particles=", n_floor_contact,
    #     #         " | mold_penetrated=", n_mold_penetrated,
    #     #         " | floor_penetrated=", n_floor_penetrated,
    #     #         " | min_gap_mold=", round(min_gap_mold, digits=6),
    #     #         " | min_gap_floor=", round(min_gap_floor, digits=6))
    # end

    # Solver-visible stress path: always use the custom constitutive model for the
    # cylinder, in both contact and non-contact states. The temperature selects the
    # constitutive regime per particle; contact only affects cross-system interaction forces.
    update_dual_phase_properties!(ys_particle_buf, hard_particle_buf, vis_particle_buf,
                                  cyl_sys.temp, alpha_committed[])
    n_hot = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        n_hot += cyl_sys.temp[particle] > local_regime_threshold_particle[particle]
    end
    n_total = length(cyl_sys.temp)
    n_cool = n_total - n_hot

    if n_hot > 0 && first_viscous_regime[]
        if first_viscous_regime[]
            println(">>> Regime: VISCOUS activated locally (hot particles=", n_hot,
                    "/", n_total, ", mean threshold=",
                    round(mean(local_regime_threshold_particle), digits=2),
                    " K) at t=", round(t, digits=6))
            flush(stdout)
            first_viscous_regime[] = false
        end
    end

    if n_cool > 0 && first_elastic_regime[]
        println(">>> Regime: ELASTIC/PLASTIC active locally (cool particles=", n_cool,
            "/", n_total, ", mean threshold=",
            round(mean(local_regime_threshold_particle), digits=2),
                " K) at t=", round(t, digits=6))
        flush(stdout)
        first_elastic_regime[] = false
    end

    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    if n_hot == n_total
        fill!(v_stress_buf, 0)
        fill!(vel_grad_buf, 0)
        stress_visc = TrixiParticles.viscous_stress3d_fast!(cyl_sys, v_cyl, vis_particle_buf,
                                    semi_local;
                                                             v_vis_buf=v_stress_buf,
                                                             vel_grad_buf=vel_grad_buf)
        TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), stress_visc)
    elseif n_hot == 0
        fill!(v_stress_buf, 0)
        stress_elas = TrixiParticles.elastic_stress3d_trial!(
            cyl_sys, ys_particle_buf, hard_particle_buf, vis_particle_buf,
            trial_dt_state[], alpha_committed[], Fp_committed[], semi_local;
            v_elas_buf=v_stress_buf)
        TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), stress_elas)
    else
        fill!(v_stress_viscous_buf, 0)
        fill!(v_stress_elastic_buf, 0)
        fill!(v_stress_buf, 0)
        fill!(vel_grad_buf, 0)
        stress_visc = TrixiParticles.viscous_stress3d_fast!(cyl_sys, v_cyl, vis_particle_buf,
                                                            semi_local;
                                                             v_vis_buf=v_stress_viscous_buf,
                                                             vel_grad_buf=vel_grad_buf)
        stress_elas = TrixiParticles.elastic_stress3d_trial!(
            cyl_sys, ys_particle_buf, hard_particle_buf, vis_particle_buf,
            trial_dt_state[], alpha_committed[], Fp_committed[], semi_local;
            v_elas_buf=v_stress_elastic_buf)

        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            stress_src = cyl_sys.temp[particle] > local_regime_threshold_particle[particle] ?
                         stress_visc : stress_elas
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] = stress_src[i, j, particle]
            end
        end

        TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), v_stress_buf)
    end
end

function kick_implicit_visible!(dv_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.set_zero!(dv_ode)

    if first_rhs_call[]
        println(">>> First RHS call reached (t=", t, ") — JIT compilation done, solver running")
        flush(stdout)
        first_rhs_call[] = false
    end

    rhs_eval_counter[] += 1
    if mod(rhs_eval_counter[], rhs_print_interval) == 0
        println(">>> RHS call #", rhs_eval_counter[], " (t=", round(t, digits=8), ") — GMRES iterating")
        flush(stdout)
    end

    # Sync system.temp from the ODE state before any physics evaluation.
    # This keeps system.temp consistent with the trial v during every Newton
    # iteration so that yield stress, viscosity, and stress caching all see
    # the same temperature that the ODE solver is currently evaluating.
    cyl_sys = semi_local.systems[1]
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        t_trial = v_cyl[NDIMS_CYL + 1, particle]
        if !isfinite(t_trial)
            t_trial = cyl_sys.temp[particle]
        end
        cyl_sys.temp[particle] = clamp(t_trial, temp_min_clip, temp_max_clip)
    end

    try
        # Expand update_systems_and_nhs so we can decide whether to skip the
        # built-in pk1_rho2 assembly for the cylinder before update_quantities!.
        t_pos_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        rhs_time_pos_ns[] += time_ns() - t_pos_start

        t_nhs_start = time_ns()
        if t != nhs_updated_at_t[]
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end
        rhs_time_nhs_ns[] += time_ns() - t_nhs_start

        # The cylinder uses the custom elastic/viscous stress cache in the residual,
        # so the built-in pk1_rho2 assembly is not needed here.
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(cyl_sys)

        t_quant_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        rhs_time_quant_ns[] += time_ns() - t_quant_start

        t_implicit_start = time_ns()
            TrixiParticles.update_implicit_sph!(semi_local, v_ode, u_ode, t)
        rhs_time_implicit_ns[] += time_ns() - t_implicit_start

        t_pressure_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_pressure!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        rhs_time_pressure_ns[] += time_ns() - t_pressure_start

        t_boundary_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_boundary_interpolation!(system, v, u, v_ode, u_ode,
                                                          semi_local, t)
        end
        rhs_time_boundary_ns[] += time_ns() - t_boundary_start

        t_final_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_final!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        rhs_time_final_ns[] += time_ns() - t_final_start

        t_stress_interact_start = time_ns()
        update_implicit_stress_cache!(semi_local, v_ode, t)
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi_local, t)
        rhs_time_stress_interact_ns[] += time_ns() - t_stress_interact_start
    finally
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = nothing
        TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
    end

    t_thermal_start = time_ns()
    # ========== THERMAL DIFFUSION AND CONTACT COOLING ==========
    floor_sys = semi_local.systems[2]
    mold_sys  = semi_local.systems[3]
    contact_heat_flux = compute_contact_heat_flux!(contact_heat_flux_buf, cyl_sys,
                                                   floor_sys, mold_sys, particle_spacing)
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    dx = particle_spacing
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        if contact_heat_flux[particle] > 0.0
            rho_i = cyl_sys.material_density[particle]
            cp_i = cp_particle[particle]
            dv_cyl[NDIMS_CYL + 1, particle] -= contact_heat_flux[particle] / (rho_i * cp_i * dx)
        end
    end
    TrixiParticles.thermal_rhs_sph3d!(cyl_sys, dv_cyl, v_cyl, 0.0,
                                      particle_spacing, bound_coordinate_thermal, semi_local)
    rhs_time_thermal_ns[] += time_ns() - t_thermal_start

    # ========== LOCATOR PIN ENFORCEMENT (partial-DOF, x/y only) ==========
    # Zero only the in-plane *acceleration* of the pinned charge particles.
    # We deliberately do NOT mutate v_cyl here: the implicit solver treats v_ode
    # as its state vector and finite-differences the RHS to build a Jacobian.
    # Silently modifying v inside the residual makes the Jacobian inconsistent
    # across Newton iterations and causes GMRES to stall / dt to shrink.
    # Since the pins start at v = 0 (see `velocity=zeros(3, n_total)` in
    # `build_glass_fiber_charge`) and we hold dv_x = dv_y = 0 every step, the
    # in-plane velocity of these particles remains exactly zero by integration.
    # Vertical (z) and temperature DOFs are left completely untouched, so
    # mold compression and heat transfer at the pinned particles proceed normally.
    @inbounds for pin in charge_locator_pins
        dv_cyl[1, pin] = 0.0
        dv_cyl[2, pin] = 0.0
    end

    return dv_ode
end

function drift_implicit_visible!(du_ode, v_ode, u_ode, semi_local, t)
    return TrixiParticles.drift!(du_ode, v_ode, u_ode, semi_local, t)
end

ode = DynamicalODEProblem(kick_implicit_visible!, drift_implicit_visible!,
                          Vector{Float64}(ode_base.u0.x[1]),
                          Vector{Float64}(ode_base.u0.x[2]),
                          tspan, semi)

# DTCAP_DISABLED: dt_cap_cb commented out — trial_dt_state[] is only consumed by
# elastic_stress3d_trial! inside the custom stress cache (currently STRESS_DISABLED).
# Re-enable together with the custom stress cache when needed.
# NOTE: use integrator.dtpropose (not integrator.dt) to avoid machine-epsilon lockup
# after SolutionSavingCallback forces a tiny landing step onto a scheduled output time.
# dt_cap_cb = DiscreteCallback(
#     (u, t, integrator) -> true,
#     function(integrator)
#         next_dt = min(integrator.dtpropose, dt_cap)
#         next_dt = max(next_dt, 1e-12)
#         trial_dt_state[] = next_dt
#         set_proposed_dt!(integrator, next_dt)
#     end
# )

# Time-based progress reporting works uniformly for explicit and implicit solvers.
t_start, t_end = tspan
progress_step_percent = 5.0
progress_dt = (progress_step_percent / 100.0) * (t_end - t_start)
next_progress_t = Ref(t_start + progress_dt)
progress_step_counter = Ref(0)
progress_heartbeat_interval = 5

# println("progress init | first percent mark at t=", round(next_progress_t[], digits=6),
#     " (", progress_step_percent, "%)")

# progress_cb = DiscreteCallback(
#     (u, t, integrator) -> true,
#     function (integrator)
#         progress_dt <= 0.0 && return
#         t = integrator.t
#         progress_step_counter[] += 1

#         if mod(progress_step_counter[], progress_heartbeat_interval) == 0
#             pct_hb = clamp(100.0 * (t - t_start) / (t_end - t_start), 0.0, 100.0)
#             println("progress hb | step=", progress_step_counter[],
#                     " | ", round(pct_hb, digits=3), "%",
#                     " | t=", round(t, digits=6), " / ", round(t_end, digits=6),
#                     " | dt=", integrator.dt)
#             flush(stdout)
#         end

#         if t + 1.0e-14 >= next_progress_t[]
#             pct = clamp(100.0 * (t - t_start) / (t_end - t_start), 0.0, 100.0)
#             println("progress | ", round(pct, digits=1), "% | t=",
#                     round(t, digits=6), " / ", round(t_end, digits=6),
#                     " | dt=", integrator.dt)
#             flush(stdout)

#             while next_progress_t[] <= t && next_progress_t[] < t_end
#                 next_progress_t[] += progress_dt
#             end
#         end
#     end
# )

callbacks = CallbackSet(
    # Commit plastic history (Fp, alpha, temp) once per accepted step.
    # The elastic_stress3d_trial! used inside the Newton residual is read-only;
    # this callback performs the one-shot mutation at the accepted solution.
    DiscreteCallback(
        (u, t, integrator) -> true,
        function(integrator)
            solver_diag_dt_min[] = min(solver_diag_dt_min[], integrator.dt)
            solver_diag_dt_max[] = max(solver_diag_dt_max[], integrator.dt)
            solver_diag_dt_sum[] += integrator.dt

            update_flow_orientation_kinetics!(orientation_tensor_state[],
                                              orientation_scalar_particle,
                                              vel_grad_buf,
                                              integrator.dt,
                                              fiber_direction;
                                              xi_ft=xi_ft,
                                              ci_ft=ci_ft,
                                              kappa_rsc=kappa_rsc,
                                              ci_ard_parallel=ci_ard_parallel,
                                              ci_ard_perp=ci_ard_perp)

            if integrator.stats.naccept > 0 &&
               mod(integrator.stats.naccept, solver_diag_interval) == 0
                accepted_delta = integrator.stats.naccept - solver_diag_last_accept[]
                iter_delta = integrator.iter - solver_diag_last_iter[]
                rhs_delta = rhs_eval_counter[] - solver_diag_last_rhs[]
                rejected_delta = max(iter_delta - accepted_delta, 0)
                avg_dt = solver_diag_dt_sum[] / max(accepted_delta, 1)
                avg_rhs_per_accept = rhs_delta / max(accepted_delta, 1)

                println("solver diag | accepted=", integrator.stats.naccept,
                        " | t=", round(integrator.t, digits=6),
                        " | dt_now=", integrator.dt,
                        " | dt_avg=", avg_dt,
                        " | dt_min=", solver_diag_dt_min[],
                        " | dt_max=", solver_diag_dt_max[],
                        " | rejected_since_last=", rejected_delta,
                        " | rhs_per_accept=", round(avg_rhs_per_accept, digits=2))

                if rhs_delta > 0
                    d_pos = rhs_time_pos_ns[] - rhs_time_pos_last_ns[]
                    d_nhs = rhs_time_nhs_ns[] - rhs_time_nhs_last_ns[]
                    d_quant = rhs_time_quant_ns[] - rhs_time_quant_last_ns[]
                    d_impl = rhs_time_implicit_ns[] - rhs_time_implicit_last_ns[]
                    d_press = rhs_time_pressure_ns[] - rhs_time_pressure_last_ns[]
                    d_bound = rhs_time_boundary_ns[] - rhs_time_boundary_last_ns[]
                    d_final = rhs_time_final_ns[] - rhs_time_final_last_ns[]
                    d_si = rhs_time_stress_interact_ns[] - rhs_time_stress_interact_last_ns[]
                    d_therm = rhs_time_thermal_ns[] - rhs_time_thermal_last_ns[]

                    println("rhs stage ms/eval | pos=", round(d_pos / rhs_delta / 1e6, digits=3),
                        " | nhs=", round(d_nhs / rhs_delta / 1e6, digits=3),
                        " | quant=", round(d_quant / rhs_delta / 1e6, digits=3),
                        " | impl=", round(d_impl / rhs_delta / 1e6, digits=3),
                        " | press=", round(d_press / rhs_delta / 1e6, digits=3),
                        " | bound=", round(d_bound / rhs_delta / 1e6, digits=3),
                        " | final=", round(d_final / rhs_delta / 1e6, digits=3),
                        " | stress+inter=", round(d_si / rhs_delta / 1e6, digits=3),
                        " | thermal=", round(d_therm / rhs_delta / 1e6, digits=3))

                    rhs_time_pos_last_ns[] = rhs_time_pos_ns[]
                    rhs_time_nhs_last_ns[] = rhs_time_nhs_ns[]
                    rhs_time_quant_last_ns[] = rhs_time_quant_ns[]
                    rhs_time_implicit_last_ns[] = rhs_time_implicit_ns[]
                    rhs_time_pressure_last_ns[] = rhs_time_pressure_ns[]
                    rhs_time_boundary_last_ns[] = rhs_time_boundary_ns[]
                    rhs_time_final_last_ns[] = rhs_time_final_ns[]
                    rhs_time_stress_interact_last_ns[] = rhs_time_stress_interact_ns[]
                    rhs_time_thermal_last_ns[] = rhs_time_thermal_ns[]
                end

                solver_diag_last_accept[] = integrator.stats.naccept
                solver_diag_last_iter[] = integrator.iter
                solver_diag_last_rhs[] = rhs_eval_counter[]
                solver_diag_dt_min[] = Inf
                solver_diag_dt_max[] = 0.0
                solver_diag_dt_sum[] = 0.0
            end

            cyl_sys = integrator.p.systems[1]
            n_cool = 0
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                n_cool += cyl_sys.temp[particle] <= local_regime_threshold_particle[particle]
            end

            if n_cool > 0
                update_dual_phase_properties!(ys_particle_buf, hard_particle_buf, vis_particle_buf,
                                              cyl_sys.temp, alpha_committed[])
                hot_particles = findall(i -> cyl_sys.temp[i] > local_regime_threshold_particle[i],
                                        eachindex(cyl_sys.temp))
                hot_alpha = alpha_committed[][hot_particles]
                hot_temp = cyl_sys.temp[hot_particles]
                hot_Fp = copy(Fp_committed[][:, :, hot_particles])

                # Commits Fp, alpha, and temp in-place using the actual accepted dt
                # for the locally cool particles only.
                trial_dt_state[] = integrator.dt
                TrixiParticles.elastic_stress3d_fast!(
                    cyl_sys, ys_particle_buf, hard_particle_buf, vis_particle_buf, integrator.dt,
                    alpha_committed[], Fp_committed[], integrator.p;
                    v_elas_buf=v_stress_buf)

                alpha_committed[][hot_particles] = hot_alpha
                cyl_sys.temp[hot_particles] = hot_temp
                Fp_committed[][:, :, hot_particles] .= hot_Fp

                # Write plastic-heated temperature back into ODE state
                v_wrap = TrixiParticles.wrap_v(integrator.u.x[1], cyl_sys, integrator.p)
                NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
                @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                    v_wrap[NDIMS_CYL + 1, particle] = cyl_sys.temp[particle]
                end
            end
        end),
    # progress_cb,
    SolutionSavingCallback(dt=1.0e-2,
                           prefix="molding_cfk_explicit_solution_2",
                           fiber_vf=(sys, data, t) -> nparticles(sys) == length(phase_vf_particle) ?
                                                      phase_vf_particle : fill(NaN, nparticles(sys))),
    InfoCallback(interval=2)
)

# linsolve_solver = KrylovKitJL_GMRES(; krylovdim=400,
#                                     atol=1e-6, rtol=5.0e-2,
#                                     maxiter=400, verbosity=0)
# MKLLUFactorization currently errors for this problem because the implicit
# state is an ArrayPartition, which does not satisfy the StridedArray path
# expected by MKL-backed factorization routines.
# Explicit Tsit5 is CFL-limited to ~1e-10 here due to stiffness/contact.
# Keep it commented for quick A/B tests.
# sol = solve(ode, Tsit5(); callback=callbacks, save_everystep=false,
#             abstol=1.0e-5, reltol=1.0e-3, dtmax=1.0e-4,
#             maxiters=10_000_000)

# Implicit solver disabled for mass-scaled explicit run
# linsolve_solver = LUFactorization()
# linsolve_solver = QRFactorization()

println("\n>>> Entering ODE solve (tspan=", tspan,
    ", implicit solver=TRBDF2, rhs_print_interval=", rhs_print_interval,
    ", solver_diag_interval=", solver_diag_interval, ")")
flush(stdout)

# Both KrylovKit.jl and Krylov.jl hit ArrayPartition type bugs with
# DynamicalODEProblem states, so use a direct solver. The Jacobian is built as
# a dense Matrix (not SparseMatrixCSC), so UMFPACK/KLU error. RFLUFactorization
# is recursive-blocked dense LU — typically 3-5x faster than plain LUFactorization.
linsolve_solver = RFLUFactorization()

# NLNewton settings tuned to reuse Jacobian as long as possible:
#  - κ=0.1         : loose Newton tolerance, fewer iterations needed to "converge"
#  - max_iter=20   : allow more iterations on stale Jacobian before rebuilding
#  - fast_convergence_cutoff=0.9 : keep reusing Jacobian even when contraction
#                                  rate is mediocre (default 0.2 is strict)
#  - always_new=false : do NOT rebuild every step
nlsolve = NLNewton(κ=1e-1, max_iter=20,
                   fast_convergence_cutoff=0.9,
                   always_new=false)

sol = Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
    solve(ode, TRBDF2(linsolve=linsolve_solver, autodiff=AutoFiniteDiff(),
                      nlsolve=nlsolve);
    # solve(ode, Rodas5P(linsolve=linsolve_solver, autodiff=AutoFiniteDiff());
    # solve(ode, Rosenbrock23(linsolve=linsolve_solver, autodiff=AutoFiniteDiff());
    # solve(ode, Tsit5();
            callback=callbacks,
            save_everystep=false,
            abstol=1.0e-5,
            reltol=1.0e-3,
            dtmax=1.0e-4,
            maxiters=10_000_000)
end
