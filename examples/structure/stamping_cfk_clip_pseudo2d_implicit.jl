# ==========================================================================================
# Pseudo-2D Thermomechanical Forming of a CFK Clip-Style Part (Implicit)
#
# Same physics stack as `stamping_cfk_clip_3d_implicit.jl`, but the domain is a
# thin slab — 1 particle thick along y — so it runs much faster for testing.
# Everything else (dual-phase constitutive model, viscous/elastic stress cache,
# plasticity history, flow-orientation kinetics, contact heat flux, implicit
# TRBDF2 solver) is unchanged.
#
# Tooling is a deep-draw clip geometry:
#   - Lower tool (female die): wide flat base + two tall side blocks leaving a
#     deep central cavity.
#   - Upper tool (male punch): full-width holder + long central protrusion that
#     mates exactly into the cavity (with one-particle clearance per side).
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
using Statistics
using LinearAlgebra
using StaticArrays
using SparseArrays
using Logging

include("../../src/schemes/structure/total_lagrangian_sph/fix_reimann.jl")

println("--- SIMULATION STARTING (pseudo-2D) ---")
println("Threads available: ", nthreads())
println("---------------------------------------")

# ==========================================================================================
# STEP 1: Build CFK raw-charge geometry (flat preform blank, 1-particle thick in y)
# ==========================================================================================

particle_spacing = 0.0011  # 1.5 mm — finer than 3D version because of the 1-particle slab

# Mass scaling kept for consistency with the 3D file (stays at 1.0 here).
mass_scaling = 1.0

# Composite feedstock (glass-fibre reinforced thermoplastic)
matrix_density = 1180.0 * mass_scaling
glass_density = 2550.0 * mass_scaling
matrix_E = 3.2e9
glass_E = 72.0e9
matrix_cp = 1800.0
glass_cp = 840.0
matrix_k = 0.22
glass_k = 1.10
# The physical conductivity of the feedstock is too low to cool the compressed
# pseudo-2D slab to solidification within the short debug horizon used here.
# Scale the effective conductivity explicitly so the thermal front reaches the
# inner particles on the same timescale as the mechanical test run.
thermal_conductivity_scale = 1.0
matrix_tmelt = 430.0
glass_tmelt = 1700.0
matrix_temp_liq = 425.0
glass_temp_liq = 1500.0
# Representative enthalpy of fusion for a thermoplastic matrix.
# This is treated as a material property, not a numerical tuning knob.
matrix_latent_heat = 1.10e5
thermal_softening_reference_temp = 270.0
matrix_viscosity = 8.0e4
matrix_viscosity_ref_temp = matrix_temp_liq
# Previous hot-flow form kept below as commented reference:
#   eta_0(T) = eta_ref * exp(E_a / R * (1 / T - 1 / T_ref))
#   eta(T, gamma_dot) = eta_0 / (1 + (lambda * gamma_dot)^(1 - n))
# matrix_viscosity_activation_energy = 6.0e4
# viscosity_cross_time_constant = 1.0e-2
# viscosity_cross_power_law_index = 0.35
# Active hot-flow form: zero-pressure Cross-WLF, anchored so eta_0(T_ref) = matrix_viscosity.
cross_wlf_A1 = 12.0
cross_wlf_A2 = 50.0
cross_wlf_transition_temp = matrix_viscosity_ref_temp
cross_wlf_critical_stress = matrix_viscosity / 1.0e-2
cross_wlf_power_law_index = 0.35
fiber_max_packing_fraction = 0.64
fiber_intrinsic_viscosity = 2.5
# Legacy placeholder from the earlier rule-of-mixtures viscosity model.
# The live hot-flow law below keeps viscosity matrix-dominated instead.
glass_viscosity = 1.0e16
matrix_yield_stress = 9.0e7
glass_yield_stress = 2.5e9
matrix_hardening = 8.0e8
glass_hardening = 8.0e9
matrix_h_contact = 2.0e5
glass_h_contact = 6.0e4

# Flow-orientation kinetics (Jeffery + RSC slowdown + ARD anisotropic diffusion)
fiber_aspect_ratio = 10000.0
xi_ft = (fiber_aspect_ratio^2 - 1.0) / (fiber_aspect_ratio^2 + 1.0)
ci_ft = 0.0
kappa_rsc = 0.35
ci_ard_parallel = 0.02
ci_ard_perp = 0.005
orientation_coupling_gain = 0.30

# Meso-architecture — alternating 0/90 plies with tow bands.
n_plies = 7          # with only ~3 layers through thickness, keep 2 plies (0, 90)
fiber_vf_in_tow = 0.58
fiber_vf_resin_rich = 0.02
tow_width = 0.0040
tow_gap = 0.0015

# Physical setup dimensions held fixed when particle spacing changes.
# These values match the current pseudo-2D baseline at particle_spacing = 1.1 mm.
charge_thickness_target = 0.0066
charge_bottom_clearance = 1.5 * particle_spacing
initial_gap_upper_target = 0.0033
base_thickness = 0.0033
holder_thickness = 0.0033
final_thickness_target = 0.00165
punch_side_clearance = 0.0011
# Thermal contact is evaluated from nearest particle-center distance in the x-z plane.
# In the drafted pseudo-2D tool geometry, a half-spacing gate is too tight and can
# miss real wall/floor contact on the discrete lattice, leaving temperature unchanged.
contact_heat_gap_threshold = particle_spacing
safety_margin = 0.0066
base_layers = max(1, ceil(Int, base_thickness / particle_spacing))

# Raw charge dimensions. charge_width = ONE particle in y → pseudo-2D slab.
# Compression-molding layout: charge is PRE-PLACED INSIDE the cavity, sitting
# on the cavity floor. Keep it small enough to fit in the narrow (bottom) part
# of the drafted trapezoidal cavity with margin > smoothing length per side,
# so contact-kernel interaction with the side walls stays zero at t = 0.
charge_length = 0.018             # 16 mm (doubled from 8 mm; cavity floor ≈ 16.9 mm wide)
charge_width = particle_spacing    # 1 particle thick along y
charge_thickness = charge_thickness_target

n_charge = (ceil(Int, charge_length / particle_spacing),
            1,                                             # y: exactly 1 particle
            max(2, ceil(Int, charge_thickness / particle_spacing)))

# Recenter the discrete particle lattice on the geometric mid-plane.
# When `ceil` adds an extra x-column, anchoring the charge at `-0.5 * charge_length`
# shifts the actual particle cloud off center and breaks left-right symmetry.
charge_length_discrete = n_charge[1] * particle_spacing
charge_thickness_discrete = n_charge[3] * particle_spacing
charge_top_target = 0.00165 + charge_thickness_target
charge_top_actual = charge_bottom_clearance + charge_thickness_discrete

# ==========================================================================================
# DEEP-DRAW TOOLING LAYOUT (x–z cross-section, 1 particle thick in y)
#
# COMPRESSION-MOLDING layout matching the clip-mold reference figure:
#   - Charge is PRE-PLACED INSIDE the cavity, resting on the cavity floor.
#   - Female die has a trapezoidal cavity with draft-angle side walls.
#   - Male punch is a mating trapezoid that descends and compresses the charge.
#   - Reference frame chosen so the cavity FLOOR is at z = 0 (charge rests here).
#
#   z = cavity_depth                    ← top face of die (side blocks / land)
#   z = 0                               ← cavity floor = charge bottom
#   z = -base_layers * ps               ← bottom of die base plate
#
# Cavity (open, no particles):
#   |x| < cavity_half_width_at(z),  z ∈ [0, cavity_depth]
#
# Male punch:
#   - Trapezoidal body of length `cavity_depth` that mates the cavity 1:1
#     (inset by one particle-spacing per side for clearance).
#   - Full-width holder plate above the body.
#   - No ejector pin.
# ==========================================================================================
tool_length = 0.040                       # 40 mm wide (charge is 12 mm; ~14 mm margin each side)
cavity_internal_width = 0.026             # 26 mm top width — accepts 12 mm charge with room
cavity_depth = 0.014                      # 14 mm deep
nz_cavity_wall = ceil(Int, cavity_depth / particle_spacing)
holder_layers = max(1, ceil(Int, holder_thickness / particle_spacing))
# Keep the actual charge placement spacing-dependent; the punch stroke then
# follows the current discretized charge top.
charge_origin = (-0.5 * charge_length_discrete, -0.5 * particle_spacing,
                 charge_bottom_clearance)
# ==========================================================================================

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
    x_center = x0 + 0.5 * nx * particle_spacing
    tow_cols_pseudo2d = max(1, round(Int, tow_width / particle_spacing))
    gap_cols_pseudo2d = max(1, round(Int, tow_gap / particle_spacing))
    tow_width_pseudo2d = tow_cols_pseudo2d * particle_spacing
    tow_pitch_pseudo2d = (tow_cols_pseudo2d + gap_cols_pseudo2d) * particle_spacing
    tow_pitch_cols_pseudo2d = tow_cols_pseudo2d + gap_cols_pseudo2d

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

        if ny == 1
            # Snap the pseudo-2D tow pattern to whole particle columns and center it on the
            # charge so the cross-section stays mirror-symmetric on the discrete lattice.
            phase_shift = ply_aligned_x ? 0.5 * tow_pitch_pseudo2d : 0.0
            distance_to_tow_center = mod((x - x_center) + phase_shift +
                                         0.5 * tow_pitch_pseudo2d,
                                         tow_pitch_pseudo2d) - 0.5 * tow_pitch_pseudo2d
            in_tow = abs(distance_to_tow_center) < 0.5 * tow_width_pseudo2d

            # Use a column-index selector tied to the charge mid-plane so mirrored columns
            # always receive the same classification after `ceil` changes the particle count.
            center_offset_twice = 2 * i - nx - 1
            phase_shift_twice = ply_aligned_x ? tow_pitch_cols_pseudo2d : 0
            wrapped_offset_twice = mod(center_offset_twice + phase_shift_twice +
                                       tow_pitch_cols_pseudo2d,
                                       2 * tow_pitch_cols_pseudo2d) - tow_pitch_cols_pseudo2d
            in_tow = abs(wrapped_offset_twice) <= tow_cols_pseudo2d - 1
        else
            selector = ply_aligned_x ? (y - y0) : (x - x0)
            in_tow = mod(selector, tow_pitch) <= tow_width
        end

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
k_composite_eff = thermal_conductivity_scale *
                  (matrix_k * (1.0 - vf_mean) + glass_k * vf_mean)

phase_vf_particle = copy(fiber_volume_fraction)
cp_particle = matrix_cp .* (1.0 .- phase_vf_particle) .+ glass_cp .* phase_vf_particle
k_particle = thermal_conductivity_scale .* (matrix_k .* (1.0 .- phase_vf_particle) .+
                                            glass_k .* phase_vf_particle)
temp_liq_particle = matrix_temp_liq .* (1.0 .- phase_vf_particle) .+ glass_temp_liq .* phase_vf_particle
tmelt_particle = matrix_tmelt .* (1.0 .- phase_vf_particle) .+ glass_tmelt .* phase_vf_particle
latent_heat_particle = matrix_latent_heat .* (1.0 .- phase_vf_particle)
local_regime_threshold_particle = fill(matrix_temp_liq, length(phase_vf_particle))
yield_base_particle = matrix_yield_stress .* (1.0 .- phase_vf_particle) .+
                      glass_yield_stress .* phase_vf_particle
hardening_base_particle = matrix_hardening .* (1.0 .- phase_vf_particle) .+
                         glass_hardening .* phase_vf_particle
# Keep the melt viscosity matrix-dominated. The live update below now uses a
# matrix Arrhenius law directly, so this array only carries the reference value
# for system initialization and diagnostics.
viscosity_base_particle = fill(matrix_viscosity, length(phase_vf_particle))
h_contact_particle = matrix_h_contact .* (1.0 .- phase_vf_particle) .+
                     glass_h_contact .* phase_vf_particle

orientation_scalar_particle = ones(length(phase_vf_particle))

viscosity_min_clip = 1.0e2
viscosity_max_clip = 5.0e7
temp_min_clip = 200.0
temp_max_clip = 2000.0

factor = 1.8
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = factor * particle_spacing

println("✓ CFK raw charge initialized (pseudo-2D slab):")
println("  - size [mm] = ", round.(StaticArrays.SVector(charge_length_discrete, charge_width,
                                              charge_thickness_discrete) .* 1e3,
                                      digits=2))
println("  - particles = ", nparticles(polymer))
println("  - architecture = ", n_plies, " plies (alternating), tow width=", tow_width * 1e3,
        " mm, tow gap=", tow_gap * 1e3, " mm")
println("  - mean fibre volume fraction Vf = ", round(vf_mean, digits=4))
println("  - phase counts (fibre-rich/matrix-rich) = ", length(idx_fiber_rich), " / ",
        length(idx_matrix_rich))
println("  - homogenized rho = ", round(density_cylinder, digits=2),
        " kg/m^3, E = ", round(E_composite_eff / 1e9, digits=3), " GPa")
println("  - effective conductivity scale = ", thermal_conductivity_scale,
    "x (pseudo-2D accelerated cooling)")
println("  - phase-specific tmelt matrix/glass = ", matrix_tmelt, " / ", glass_tmelt, " K")
println("  - matrix latent heat = ", matrix_latent_heat, " J/kg")

# ==========================================================================================
# STEP 3: Build mating deep-draw tooling (female die + male punch), 1 particle thick in y
# ==========================================================================================
floor_density = density_cylinder

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

# Option 1 — open the initial gap to ~2 * smoothing_length so the mold-motion
# ramp completes *before* first contact. This prevents the stiff t=0 contact +
# heat-flux transient that collapses TRBDF2's dt to ~1e-7.
initial_gap       = 2 * smoothing_length
initial_gap_upper = initial_gap_upper_target

# Reference frame: cavity FLOOR at z = 0 (this is where the charge sits).
# The charge is built at z0 = 0 (see charge_origin), so its bottom face
# automatically rests on the cavity floor.
#   - z = 0               → cavity floor = top of base plate
#   - z = cavity_depth    → top face of die (land/shoulder, outside the cavity)
#   - z = -base_layers*ps → bottom of die base plate
z_base_top   = 0.0
z_blocks_top = z_base_top + cavity_depth
z_base_bot   = z_base_top - base_layers * particle_spacing

# -------- Trapezoidal die + mating trapezoidal punch (angled walls / draft) --------
# Both die cavity and punch share the SAME master grid so they mate particle-for-
# particle, with a one-ps clearance per side. The cavity is widest at the top
# (where the charge enters) and narrows toward the floor — classic draft geometry
# as in the reference figure. The punch mirrors the cavity shape exactly, offset
# inward by one particle spacing per side.
nx_tool = ceil(Int, tool_length / particle_spacing)
ny_tool = 1
tool_origin_y = -0.5 * particle_spacing
tool_origin_x = -0.5 * tool_length

# Redefined geometry: set bottom width directly, derive draft angle.
cavity_top_half_w = 0.5 * cavity_internal_width   # widest (at z_blocks_top)
cavity_bot_half_w = 0.5 * (charge_length + 0.003) # 1.5mm clearance per side
wall_inset = cavity_top_half_w - cavity_bot_half_w
draft_angle = atan(wall_inset / cavity_depth)
@info "Derived draft angle for wider cavity" angle_deg=rad2deg(draft_angle)

# Cavity half-width as a function of z ∈ [z_base_top, z_blocks_top].
@inline function cavity_half_width_at(z)
    frac = (z - z_base_top) / (z_blocks_top - z_base_top)
    frac = clamp(frac, 0.0, 1.0)
    return cavity_bot_half_w + frac * (cavity_top_half_w - cavity_bot_half_w)
end

# Build die particles by filtering a rectangular grid:
#   - base plate: full rectangle from z_base_bot to z_base_top
#   - wall region: z ∈ [z_base_top, z_blocks_top], keep if |x| > half_w(z)
function build_die_particles(ps, x_min, x_max, y_center,
                             z_base_bot, z_base_top, z_blocks_top,
                             half_width_fn, density)
    coords = Float64[]
    nx = ceil(Int, (x_max - x_min) / ps)
    # Base plate layers
    nz_base = ceil(Int, (z_base_top - z_base_bot) / ps)
    for k in 1:nz_base
        z = z_base_bot + (k - 0.5) * ps
        for i in 1:nx
            x = x_min + (i - 0.5) * ps
            push!(coords, x, y_center, z)
        end
    end
    # Cavity-wall region (with trapezoidal hole)
    nz_walls = ceil(Int, (z_blocks_top - z_base_top) / ps)
    for k in 1:nz_walls
        z = z_base_top + (k - 0.5) * ps
        half_w = half_width_fn(z)
        for i in 1:nx
            x = x_min + (i - 0.5) * ps
            if abs(x) > half_w
                push!(coords, x, y_center, z)
            end
        end
    end
    n = length(coords) ÷ 3
    coords_m = reshape(coords, 3, n)
    mass = fill(density * ps^3, n)
    dens = fill(density, n)
    vel = zeros(3, n)
    return InitialCondition(; coordinates=coords_m, velocity=vel, mass=mass, density=dens)
end

floor_particles = build_die_particles(particle_spacing,
                                      tool_origin_x, tool_origin_x + nx_tool * particle_spacing,
                                      0.0,  # y-center (slab thickness = 1 ps centered on 0)
                                      z_base_bot, z_base_top, z_blocks_top,
                                      cavity_half_width_at,
                                      floor_density)

# -------- Male punch: trapezoidal body that truly MATES the cavity --------
# Construction strategy:
#   1. Define the SEATED position of the punch body (as if fully stroked down).
#      Seated z-range: [z_base_top + final_thickness, z_blocks_top + final_thickness]
#      where `final_thickness` is the residual part thickness at the cavity floor.
#   2. At each seated absolute z, the punch half-width equals the cavity
#      half-width at that z minus one ps per side (true mating profile — the
#      punch exactly fills the draft-walled cavity with 1 ps clearance).
#   3. Translate the body upward by `initial_stroke` so that at t=0 its bottom
#      sits `initial_gap_upper` above the charge top (= `charge_thickness`).
final_thickness = final_thickness_target                # residual part thickness at cavity floor
charge_top_z    = charge_top_target
z_punch_bot     = charge_top_z + initial_gap_upper      # start-position punch bottom
z_punch_top     = z_punch_bot + cavity_depth            # start-position punch top
initial_stroke  = z_punch_bot - (z_base_top + final_thickness)  # descent required to seat

@inline function punch_half_width_at_start(z_start)
    # Map starting z back to its seated z, then use the cavity profile.
    z_seated    = z_start - initial_stroke
    z_seated_cl = clamp(z_seated, z_base_top, z_blocks_top)
    half_w_die  = cavity_half_width_at(z_seated_cl)
    return max(0.5 * punch_side_clearance, half_w_die - punch_side_clearance)
end

function build_punch_particles(ps, x_min, x_max, y_center,
                               z_punch_bot, z_punch_top,
                               half_width_fn_at_start,
                               z_holder_bot, holder_layers, density)
    coords = Float64[]
    nx = ceil(Int, (x_max - x_min) / ps)

    # Trapezoidal punch body — profile from seated absolute z (via the map
    # inside `half_width_fn_at_start`).
    nz_body = ceil(Int, (z_punch_top - z_punch_bot) / ps)
    for k in 1:nz_body
        z = z_punch_bot + (k - 0.5) * ps
        half_w = half_width_fn_at_start(z)
        for i in 1:nx
            x = x_min + (i - 0.5) * ps
            if abs(x) < half_w
                push!(coords, x, y_center, z)
            end
        end
    end

    # Full-width holder plate above the punch body.
    for k in 1:holder_layers
        z = z_holder_bot + (k - 0.5) * ps
        for i in 1:nx
            x = x_min + (i - 0.5) * ps
            push!(coords, x, y_center, z)
        end
    end

    n = length(coords) ÷ 3
    coords_m = reshape(coords, 3, n)
    mass = fill(density * ps^3, n)
    dens = fill(density, n)
    vel = zeros(3, n)
    return InitialCondition(; coordinates=coords_m, velocity=vel, mass=mass, density=dens)
end

z_holder_bot = z_punch_top
mold_particles = build_punch_particles(particle_spacing,
                                       tool_origin_x, tool_origin_x + nx_tool * particle_spacing,
                                       0.0,
                                       z_punch_bot, z_punch_top,
                                       punch_half_width_at_start,
                                       z_holder_bot, holder_layers, floor_density)

floor_z_max = maximum(floor_particles.coordinates[3,:])
mold_z_min  = minimum(mold_particles.coordinates[3,:])
println("✓ Compression-mold tooling built (pseudo-2D, trapezoidal / drafted walls):")
println("  - tool_length = ", tool_length * 1e3, " mm, slab thickness y = ",
        particle_spacing * 1e3, " mm")
println("  - cavity: top width = ", cavity_internal_width * 1e3,
        " mm, bot width = ", round(2 * cavity_bot_half_w * 1e3; digits=2),
        " mm, depth = ", cavity_depth * 1e3, " mm, draft = ",
        round(rad2deg(draft_angle); digits=1), " deg")
println("  - charge inside cavity: length = ", charge_length * 1e3,
        " mm, thickness = ", charge_thickness * 1e3,
    " mm (initialized slightly above cavity floor for startup stability)")
println("  - male punch: mating trapezoid (1 ps clearance per side, no ejector)")
println("  - gap (mold bot ↔ charge top) = ", round(mold_z_min - cyl_z_max; digits=6),
        " m, initial stroke to seat = ", round(initial_stroke * 1e3; digits=3), " mm")
println("  - h = ", factor * particle_spacing, " m")
println("  - Contact at t=0? ",
        (cyl_z_min - (z_base_top - 0.5 * particle_spacing)) < factor * particle_spacing,
        " (floor), ",
        (mold_z_min - cyl_z_max) < factor * particle_spacing, " (mold)")
println("  - Tool particles (floor/mold) = ", nparticles(floor_particles), " / ",
        nparticles(mold_particles))

# Numerical safety rails for pseudo-2D debug runs.
# With all fundamental issues fixed (dt sync, 2D kernel correction, and plane-strain),
# these artificial limiters are no longer needed and just ruin Newton convergence.
enable_safety_clamps = false
enable_velocity_safety_clamps = false
enable_emergency_repair_clamp = true
x_safety_bound = 0.5 * tool_length + safety_margin
z_safety_min = z_base_bot - safety_margin
z_safety_max = z_punch_top + safety_margin
max_cylinder_speed = 2.0      # m/s
max_cylinder_accel = 1.0e5    # m/s^2

# ==========================================================================================
# THERMOMECHANICAL STATE INITIALIZATION
# ==========================================================================================
n_cylinder_particles = nparticles(polymer)
alpha_plastic_state = Ref(zeros(n_cylinder_particles))

Fp_initial = zeros(3, 3, n_cylinder_particles)
for i in 1:n_cylinder_particles
    Fp_initial[:, :, i] .= Matrix{Float64}(I, 3, 3)
end
Fp_state = Ref(Fp_initial)

alpha_committed = Ref(zeros(n_cylinder_particles))
Fp_committed    = Ref(copy(Fp_initial))

F_total_mold_state = Ref(0.0)

# Press speed. Higher than the 3D run so the pseudo-2D slab finishes quickly.
# DIAGNOSTIC (Option F): reduced from -0.5 to -0.05 m/s to test whether first
# punch–charge contact is the failure trigger. If failure time scales by ~10x
# (t ≈ 80 ms instead of 8 ms), contact is confirmed as the root cause.
mold_velocity_state = -0.05  # m/s
t_ramp_mold = 2.0e-3

# Mild velocity-proportional damping for the pseudo-2D slab.
# Keep baseline damping low to avoid stiffening the Newton residual; add extra
# damping only close to tool contact.
# NOTE: damping set to zero — artificial damping suppresses viscous velocity gradients
# and causes the charge to shrink. Physical viscosity (η = 8e4 Pa·s) handles dissipation.
velocity_damping_base = 0.0      # 1/s
velocity_damping_contact = 0.0   # 1/s

# Target stroke: drive the punch from its start position (z_punch_bot) down to
# the seated position, leaving a `final_thickness` residual layer on the floor.
target_stroke = z_punch_bot - (z_base_top + final_thickness)
t_full = (target_stroke / abs(mold_velocity_state)) + 0.5 * t_ramp_mold
t_down_end = 0.80 * t_full
z_shift_down_end = if t_down_end < t_ramp_mold
    0.5 * mold_velocity_state / t_ramp_mold * t_down_end^2
else
    mold_velocity_state * (t_down_end - 0.5 * t_ramp_mold)
end
t_compress = t_down_end
solidification_hold_max = 8.0e-2
solidification_fraction_threshold = 0.99
solidification_fraction_target = 0.95
post_solidification_dwell = 1.0e-2
retraction_duration_max = abs(z_shift_down_end) / abs(mold_velocity_state)
t_total = t_compress + solidification_hold_max + post_solidification_dwell +
      retraction_duration_max
println("  - press speed = ", abs(mold_velocity_state), " m/s, ramp = ", t_ramp_mold,
        " s, target stroke = ", round(target_stroke * 1e3; digits=2),
        ", t_down_end = ", round(t_down_end; digits=6),
        " s, t_motion = ", round(t_compress; digits=6),
        " s, solidification hold <= ", round(solidification_hold_max; digits=6),
    " s, post-solidification dwell = ", round(post_solidification_dwell; digits=6),
    " s, retract <= ", round(retraction_duration_max; digits=6), " s")

@inline function mold_z_shift_at_time(t)
    z_shift = if t < t_ramp_mold
        0.5 * mold_velocity_state / t_ramp_mold * t^2
    elseif t < t_down_end
        mold_velocity_state * (t - 0.5 * t_ramp_mold)
    else
        z_shift_down_end
    end
    return clamp(z_shift, -target_stroke, 0.0)
end

z_shift_motion_end = mold_z_shift_at_time(t_compress)
retraction_started = Ref(false)
retraction_complete = Ref(false)
retraction_start_time = Ref(Inf)
solidification_reached_time = Ref(Inf)

mold_motion = PrescribedMotion(
    (x, t) -> begin
        z_shift = if t <= t_compress
            mold_z_shift_at_time(t)
        elseif retraction_started[]
            z_shift_motion_end + abs(mold_velocity_state) * (t - retraction_start_time[])
        else
            z_shift_motion_end
        end
        z_shift = clamp(z_shift, z_shift_motion_end, 0.0)
        x + StaticArrays.SVector(0.0, 0.0, z_shift)
    end,
    t -> true)

bound_coordinate_thermal = (3, minimum(floor_particles.coordinates[3, :]))

println("✓ Thermomechanical regime configured:")
println("  - Viscous regime: activated when T > matrix_temp_liq = ", matrix_temp_liq, " K")
println("  - Composite feedstock: glass fibres embedded in thermoplastic matrix")

# ==========================================================================================
# STEP 5: Build the TLSPH systems
# ==========================================================================================
material_polymer = (density=density_cylinder, E=E_composite_eff, nu=0.3, beta=0.000,
                    temp=thermal_softening_reference_temp,
                    temp_ref=thermal_softening_reference_temp,
                    cp=mean(cp_particle), k=mean(k_particle),
                    temp_liq=mean(temp_liq_particle),
                    h=mean(h_contact_particle),
                    hardening=mean(hardening_base_particle),
                    tmelt=mean(tmelt_particle),
                    viscosity=mean(viscosity_base_particle),
                    yield_stress=mean(yield_base_particle))

# Contact penalty stiffness for clamped tools.
# Using 1e11 here causes an impulse-like response at first contact in this pseudo-2D setup.
# Keep tools only moderately stiffer than the charge in pseudo-2D to avoid
# impulse-like contact forces at first engagement.
E_boundary = 1.0 * material_polymer.E

import PointNeighbors: DictionaryCellList

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=200)

floor_system = TotalLagrangianSPHSystem(floor_particles,
                                        smoothing_kernel,
                                        smoothing_length,
                                        E_boundary,
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

mold_system = TotalLagrangianSPHSystem(mold_particles,
                                       smoothing_kernel,
                                       smoothing_length,
                                       E_boundary,
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

cylinder_system = TotalLagrangianSPHSystem(polymer,
                                           smoothing_kernel,
                                           smoothing_length,
                                           material_polymer.E,
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
                                           acceleration=(0.0, 0.0, 0.0),
                                           #penalty_force=PenaltyForceGanzenmueller(alpha=0.001),
                                           #artificial_stress=ArtificialStressGray(epsilon=0.01,
                                           #                                      exponent=2),
                                           viscosity=ArtificialViscosityMonaghan(alpha=0.02, beta=0.0),
                                           self_interaction_nhs=nhs_template)

# ==========================================================================================
# STEP 6: Semidiscretization and solve
# ==========================================================================================
semi = Semidiscretization(cylinder_system, floor_system, mold_system;
                          neighborhood_search=GridNeighborhoodSearch{3}(;
                              cell_list=DictionaryCellList{3}(),
                              search_radius=smoothing_length))

# Preheat: blank at 440 K, tools at 400 K (matches 3D file).
preheating_target_temp = 440.0
tool_preheat_temp = 320.0

println("\n=== PREHEATING PHASE ===")
semi.systems[1].temp .= preheating_target_temp
semi.systems[2].temp .= tool_preheat_temp
semi.systems[3].temp .= tool_preheat_temp
println("✓ Cylinder preheated: T = $preheating_target_temp K")
println("✓ Floor preheated:    T = $tool_preheat_temp K")
println("✓ Mold  preheated:    T = $tool_preheat_temp K")
println("=== PREHEATING COMPLETE ===\n")

tspan = (0.0, t_total)

ode_base = semidiscretize(semi, tspan)

# ==========================================================================================
# PSEUDO-2D TLSPH FIX: Compute the 2x2 Gradient Correction Matrix
# The default 3D correction_matrix_inversion_step! sees det(L) = 0 because all particles
# are coplanar in y, so it falls back to the identity matrix L_inv = I. This completely
# disables kernel correction, ruining TLSPH stability.
# We manually re-compute the x-z components of L, invert the 2x2 block, and set y=1.
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
        L = TrixiParticles.extract_smatrix(system.correction_matrix, system, particle)
        Lx = L[1, 1]; Lxz = L[1, 3]
        Lzx = L[3, 1]; Lz = L[3, 3]
        detL2d = Lx * Lz - Lxz * Lzx
        L_inv = Matrix(1.0I, 3, 3)
        
        # Regularize the determinant to avoid catastrophic jumps when detL2d < 1e-9
        # that drop kernel correction entirely and result in massive unphysical forces.
        detL2d_reg = sign(detL2d) * max(abs(detL2d), 1e-6)
        # Fall back to identity ONLY if the whole matrix is nearly zero to prevent NaNs
        if abs(Lx) + abs(Lz) + abs(Lxz) + abs(Lzx) > 1e-9
            L_inv[1, 1] = Lz / detL2d_reg
            L_inv[1, 3] = -Lxz / detL2d_reg
            L_inv[3, 1] = -Lzx / detL2d_reg
            L_inv[3, 3] = Lx / detL2d_reg
        end
        for j in 1:3, i in 1:3
            system.correction_matrix[i, j, particle] = L_inv[i, j]
        end
    end
end
println(">>> Pseudo-2D kernel correction applied.")
# ==========================================================================================

v_stress_buf          = zeros(3, 3, n_cylinder_particles)
v_stress_elastic_buf  = zeros(3, 3, n_cylinder_particles)
v_stress_viscous_buf  = zeros(3, 3, n_cylinder_particles)
vel_grad_buf          = zeros(3, 3, n_cylinder_particles)

nhs_updated_at_t = Ref(-Inf)
contact_diag_interval = 5000

enable_contact_diag = false
contact_heat_flux_buf = zeros(n_cylinder_particles)
ys_particle_buf = zeros(n_cylinder_particles)
hard_particle_buf = zeros(n_cylinder_particles)
vis_particle_buf = zeros(n_cylinder_particles)
thermal_softening_particle_buf = zeros(n_cylinder_particles)
liquid_fraction_particle_buf = zeros(n_cylinder_particles)
solid_fraction_particle_buf = zeros(n_cylinder_particles)
orientation_tensor_state = Ref(zeros(3, 3, n_cylinder_particles))
solidification_hold_announced = Ref(false)
solidification_complete = Ref(false)

first_viscous_regime    = Ref(true)
first_elastic_regime    = Ref(true)

# Fix C — match dt_cap with dtmax. The trial_dt_state is used in elastic_stress3d_trial!
# to scale the stress computation; if it doesn't match the actual dt being tried by the
# solver (especially before the first accepted step), the stress scaling is wrong, causing
# huge forces and particle ejection. Initialize dt_cap to match dtmax (set later at line ~1172).
dt_cap = 1.0e-5
trial_dt_state = Ref(dt_cap)

rhs_time_pos_ns = Ref(0)
rhs_time_nhs_ns = Ref(0)
rhs_time_quant_ns = Ref(0)
rhs_time_implicit_ns = Ref(0)
rhs_time_pressure_ns = Ref(0)
rhs_time_boundary_ns = Ref(0)
rhs_time_final_ns = Ref(0)
rhs_time_stress_interact_ns = Ref(0)
rhs_time_thermal_ns = Ref(0)

rhs_time_pos_last_ns = Ref(0)
rhs_time_nhs_last_ns = Ref(0)
rhs_time_quant_last_ns = Ref(0)
rhs_time_implicit_last_ns = Ref(0)
rhs_time_pressure_last_ns = Ref(0)
rhs_time_boundary_last_ns = Ref(0)
rhs_time_final_last_ns = Ref(0)
rhs_time_stress_interact_last_ns = Ref(0)
rhs_time_thermal_last_ns = Ref(0)

# Fine-grained timing for constitutive model functions (within update_implicit_stress_cache!)
rhs_time_properties_update_ns = Ref(0)       # update_dual_phase_properties!
rhs_time_elastic_trial_ns = Ref(0)           # elastic_stress3d_trial_pseudo2d!
rhs_time_viscous_stress_ns = Ref(0)          # viscous_stress_from_cached_grad!
rhs_time_stress_blend_ns = Ref(0)            # stress blending + NaN checking
rhs_time_stress_cache_ns = Ref(0)            # update_implicit_stress_cache!
rhs_time_system_interaction_ns = Ref(0)      # TrixiParticles.system_interaction!
rhs_time_source_terms_ns = Ref(0)            # TrixiParticles.add_source_terms!
rhs_time_contact_heat_ns = Ref(0)            # compute_contact_heat_flux!
rhs_time_thermal_sph_ns = Ref(0)             # thermal_rhs_sph3d! (inside thermal section)

# Callback timing
rhs_time_callback_vel_grad_ns = Ref(0)       # refresh_velocity_gradient_cache! (callback)
rhs_time_callback_orientation_ns = Ref(0)    # update_flow_orientation_kinetics! (callback)

# function contact_condition(cyl_sys, floor_sys, mold_sys, i, contact_dist)
#     z_i         = cyl_sys.current_coordinates[3, i]
#     z_floor_top = maximum(floor_sys.current_coordinates[3, :])
#     z_mold_bot  = minimum(mold_sys.current_coordinates[3, :])
#     return (z_i - z_floor_top) <= contact_dist || (z_mold_bot - z_i) <= contact_dist
# end

@inline soft_limit(x, lim) = lim * tanh(x / max(lim, eps(Float64)))

# @inline function soft_box(x, lo, hi)
#     c = 0.5 * (lo + hi)
#     r = max(0.5 * (hi - lo), eps(Float64))
#     return c + r * tanh((x - c) / r)
# end

function local_tool_surface_center_z(system_tool, x_target, z_target; side)
    best_dx = Inf
    best_z = side === :lower ? -Inf : Inf
    found = false

    @inbounds for p in 1:nparticles(system_tool)
        x_p = system_tool.current_coordinates[1, p]
        z_p = system_tool.current_coordinates[3, p]
        is_candidate = side === :lower ? (z_p <= z_target) : (z_p >= z_target)
        is_candidate || continue

        dx = abs(x_p - x_target)
        if dx < best_dx - eps(Float64)
            best_dx = dx
            best_z = z_p
            found = true
        elseif abs(dx - best_dx) <= eps(Float64)
            best_z = side === :lower ? max(best_z, z_p) : min(best_z, z_p)
            found = true
        end
    end

    if found
        return best_z
    end

    @inbounds for p in 1:nparticles(system_tool)
        x_p = system_tool.current_coordinates[1, p]
        z_p = system_tool.current_coordinates[3, p]
        dx = abs(x_p - x_target)
        if dx < best_dx - eps(Float64)
            best_dx = dx
            best_z = z_p
        elseif abs(dx - best_dx) <= eps(Float64)
            best_z = side === :lower ? max(best_z, z_p) : min(best_z, z_p)
        end
    end

    return best_z
end

@inline function tool_surface_gap(system_tool, x_target, z_target, particle_spacing; side)
    z_tool_center = local_tool_surface_center_z(system_tool, x_target, z_target; side=side)
    center_gap = side === :lower ? (z_target - z_tool_center) : (z_tool_center - z_target)
    return center_gap - particle_spacing
end

@inline function tool_thermal_gap(system_tool, x_target, z_target, particle_spacing)
    best_distance2 = Inf

    @inbounds for p in 1:nparticles(system_tool)
        dx = x_target - system_tool.current_coordinates[1, p]
        dz = z_target - system_tool.current_coordinates[3, p]
        distance2 = dx * dx + dz * dz
        if distance2 < best_distance2
            best_distance2 = distance2
        end
    end

    return sqrt(best_distance2) - particle_spacing
end

function compute_contact_heat_flux!(ext_heat_per_particle, system_cyl, system_floor,
                                    system_mold, particle_spacing)
    fill!(ext_heat_per_particle, 0.0)

    T_floor = mean(system_floor.temp)
    T_mold = mean(system_mold.temp)

    h_contact = system_cyl.h
    contact_threshold = contact_heat_gap_threshold
    T_smooth_band = 5.0

    @inline smoothstep01(x) = begin
        y = clamp(x, 0.0, 1.0)
        y * y * (3.0 - 2.0 * y)
    end
    @inline gap_activation(gap) = 1.0 - smoothstep01(gap / contact_threshold)
    @inline dT_activation(dT) = smoothstep01(abs(dT) / T_smooth_band)

    @inbounds for i in 1:nparticles(system_cyl)
        x_i = system_cyl.current_coordinates[1, i]
        z_i = system_cyl.current_coordinates[3, i]
        T_i = system_cyl.temp[i]
        h_contact_i = h_contact_particle[i]

        gap_floor = tool_thermal_gap(system_floor, x_i, z_i, particle_spacing)
        if gap_floor <= contact_threshold
            gap_floor_eff = clamp(gap_floor, 0.0, contact_threshold)
            dT = T_i - T_floor
            flux_floor = h_contact_i * dT * gap_activation(gap_floor_eff) * dT_activation(dT)
            if flux_floor != 0.0
                ext_heat_per_particle[i] += flux_floor
            end
        end

        gap_mold = tool_thermal_gap(system_mold, x_i, z_i, particle_spacing)
        if gap_mold <= contact_threshold
            gap_mold_eff = clamp(gap_mold, 0.0, contact_threshold)
            dT = T_i - T_mold
            flux_mold = h_contact_i * dT * gap_activation(gap_mold_eff) * dT_activation(dT)
            if flux_mold != 0.0
                ext_heat_per_particle[i] += flux_mold
            end
        end
    end

    return ext_heat_per_particle
end

@inline function effective_heat_capacity_particle(temp_i, particle)
    cp_eff = cp_particle[particle]
    latent_i = latent_heat_particle[particle]
    transition_span = max(matrix_tmelt - matrix_temp_liq, eps(Float64))

    if latent_i > 0.0 && matrix_temp_liq <= temp_i <= matrix_tmelt
        cp_eff += latent_i / transition_span
    end

    return cp_eff
end

@inline function liquid_fraction_particle(temp_i, particle)
    temp_liq_i = temp_liq_particle[particle]
    temp_melt_i = tmelt_particle[particle]
    transition_span = max(temp_melt_i - temp_liq_i, eps(Float64))
    return clamp((temp_i - temp_liq_i) / transition_span, 0.0, 1.0)
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

function refresh_velocity_gradient_cache!(system, v, semi, vel_grad_cache)
    fill!(vel_grad_cache, 0.0)

    initial_coords = TrixiParticles.initial_coordinates(system)
    nhs = TrixiParticles.get_neighborhood_search(system, system, semi)
    PointNeighbors.foreach_point_neighbor(
        initial_coords, initial_coords, nhs;
        parallelization_backend=TrixiParticles.PolyesterBackend()
    ) do particle, neighbor, pos_diff_initial, initial_distance
        initial_distance^2 < eps(TrixiParticles.initial_smoothing_length(system)^2) && return

        volume = @inbounds system.mass[neighbor] / system.material_density[neighbor]
        vel_diff = v[:, particle] - v[:, neighbor]
        grad_kernel = TrixiParticles.smoothing_kernel_grad(system, pos_diff_initial,
                                                           initial_distance, particle)
        result_v = volume * vel_diff * grad_kernel'

        for j in 1:TrixiParticles.ndims(system), i in 1:TrixiParticles.ndims(system)
            @inbounds vel_grad_cache[i, j, particle] -= result_v[i, j]
        end
    end

    return vel_grad_cache
end

function viscous_stress_from_cached_grad!(system, vis;
                                          v_vis_buf=nothing,
                                          vel_grad_buf,
                                          include_bulk_term=true,
                                          solid_fraction_buf=nothing,
                                          skip_solid_fraction=1.0)
    (; young_modulus, poisson_ratio) = system

    n_particles = size(vel_grad_buf, 3)
    v_vis = v_vis_buf !== nothing ? v_vis_buf : zeros(Float64, 3, 3, n_particles)
    K = young_modulus / (3 - 6 * poisson_ratio)

    Threads.@threads for particle in 1:n_particles
        if solid_fraction_buf !== nothing && solid_fraction_buf[particle] >= skip_solid_fraction
            @inbounds v_vis[:, :, particle] .= 0.0
            continue
        end

        F = TrixiParticles.deformation_gradient(system, particle)
        L_corr = @inbounds TrixiParticles.correction_matrix(system, particle)
        J = max(det(F), 1e-6)

        _L = vel_grad_buf[:, :, particle] * L_corr'
        d = 0.5 * (_L + _L')
        dev_d = d - 1 / 3 * tr(d) * I

        tau = 2 * vis[particle] * dev_d
        if include_bulk_term
            tau += K * log(J) * I
        end

        FinvT = pinv(F)'
        v_vis[:, :, particle] .= (tau * FinvT) * L_corr
    end

    return v_vis
end

function elastic_stress3d_trial_skip_liquid!(system, ys, hard, vis, dt, _alpha, _Fp, semi;
                                             solid_fraction_buf,
                                             v_elas_buf=nothing,
                                             skip_solid_fraction=0.0)
    (; deformation_grad, young_modulus, poisson_ratio, temp, tmelt, hardening) = system

    n_particles = size(deformation_grad, 3)
    v_elas = v_elas_buf !== nothing ? v_elas_buf : zeros(eltype(young_modulus), 3, 3, n_particles)

    mu = young_modulus / (2 + 2 * poisson_ratio)
    K = young_modulus / (3 - 6 * poisson_ratio)
    H_theta = 1.0 / (tmelt - system.temp_ref[1])

    Threads.@threads for particle in 1:n_particles
        if solid_fraction_buf[particle] <= skip_solid_fraction
            @inbounds v_elas[:, :, particle] .= 0.0
            continue
        end

        F = TrixiParticles.deformation_gradient(system, particle)
        Fp_local = StaticArrays.SMatrix{3, 3}(@view _Fp[:, :, particle])
        L_corr = @inbounds TrixiParticles.correction_matrix(system, particle)

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
        Fe = F_reg * Fp_inv
        J_e = max(det(Fe), 1e-6)
        be = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - 1 / 3 * tr(be_bar) * I
        tau = K * log(J_e) * I + mu * dev_be

        dev_tau = tau - 1 / 3 * tr(tau) * I
        yf = sqrt(1.5) * sqrt(sum(dev_tau .^ 2)) - (ys[particle] + hard[particle])

        if yf > 1e-6
            H_alpha_theta = hardening * (1 - H_theta * (temp[particle] - system.temp_ref[1]))
            if temp[particle] < 0.5 * tmelt
                delta_gamma = yf / (3 * mu + H_alpha_theta)
            else
                delta_gamma = yf * dt / max(vis[particle], 1e-8)
            end

            norm_dev = sqrt(sum(dev_tau .* dev_tau))
            if norm_dev > 1e-14 && isfinite(norm_dev)
                n_dir = dev_tau / norm_dev
                c = delta_gamma * sqrt(1.5)
                A = c * n_dir
                A2 = A * A
                sc = abs(c) < 1e-14 ? one(c) : sinh(c) / c
                cc = abs(c) < 1e-14 ? one(c) : (cosh(c) - 1) / (c * c)
                exp_A = one(StaticArrays.SMatrix{3, 3, eltype(n_dir)}) + sc * A + cc * A2
                Fp_local = exp_A * Fp_local
                Fp_local = Fp_local / cbrt(det(Fp_local))

                Fp_inv = inv(Fp_local)
                Fe = F_reg * Fp_inv
                J_e = max(det(Fe), 1e-6)
                be = Fe * Fe'
                be_bar = be / (J_e^(2 / 3))
                dev_be = be_bar - 1 / 3 * tr(be_bar) * I
                tau = K * log(J_e) * I + mu * dev_be
            end
        end

        FinvT = inv(F_reg)'
        v_elas[:, :, particle] .= (tau * FinvT) * L_corr
    end

    return v_elas
end

function update_dual_phase_properties!(ys, hard, vis, temp, alpha,
                                       thermal_softening_particle_buf,
                                       liquid_fraction_particle_buf,
                                       solid_fraction_particle_buf;
                                       vel_grad_buf=nothing)
    gas_constant = 8.31446261815324
    @inbounds for i in eachindex(temp)
        temp_i = temp[i]
        if !isfinite(temp_i)
            temp_i = matrix_temp_liq
        end
        temp_i = clamp(temp_i, temp_min_clip, temp_max_clip)

        liquid_fraction = liquid_fraction_particle(temp_i, i)
        solid_fraction = 1.0 - liquid_fraction
        liquid_fraction_particle_buf[i] = liquid_fraction
        solid_fraction_particle_buf[i] = solid_fraction

        h_theta = 1.0 / max(matrix_tmelt - thermal_softening_reference_temp, eps(Float64))
        thermal_softening = clamp(1.0 - h_theta * (temp_i - thermal_softening_reference_temp),
                                  0.0, 1.0)
        thermal_softening_particle_buf[i] = thermal_softening

        melt_softening = clamp((temp_i - 0.8 * matrix_tmelt) /
                       (0.4 * matrix_tmelt + eps(Float64)), 0.0, 1.0)
        solid_factor = 1.0 - melt_softening

        orient_scale = 1.0 + phase_vf_particle[i] * orientation_coupling_gain *
                             (orientation_scalar_particle[i] - 1.0)

        # ys[i] = max(1.0e3,
        #             (yield_base_particle[i] + hardening_base_particle[i] * alpha[i]) *
        #             thermal_softening)
        ys[i] = max(1.0e3,
                (yield_base_particle[i] + hardening_base_particle[i] * alpha[i]) *
                thermal_softening)
        ys[i] *= orient_scale
        # hard[i] = hardening_base_particle[i] * thermal_softening * orient_scale
        hard[i] = hardening_base_particle[i] * thermal_softening * orient_scale
        # Previous Arrhenius + Cross form retained here for comparison:
        # inv_temp = 1.0 / max(temp_i, eps(Float64))
        # inv_ref_temp = 1.0 / matrix_viscosity_ref_temp
        # eta_zero_shear = matrix_viscosity * exp(matrix_viscosity_activation_energy /
        #                                         gas_constant * (inv_temp - inv_ref_temp))

        temp_shift = temp_i - cross_wlf_transition_temp
        wlf_denom = cross_wlf_A2 + temp_shift
        if abs(wlf_denom) < 1.0e-8
            wlf_denom = sign(wlf_denom) * 1.0e-8
            if wlf_denom == 0.0
                wlf_denom = 1.0e-8
            end
        end
        eta_zero_shear = matrix_viscosity * exp(-cross_wlf_A1 * temp_shift / wlf_denom)
        if vel_grad_buf === nothing
            vis[i] = eta_zero_shear
        else
            L = @view vel_grad_buf[:, :, i]
            D = 0.5 * (L + L')
            shear_rate = sqrt(max(2.0 * sum(abs2, D), 0.0))
            # Previous Cross denominator retained here for comparison:
            # cross_denominator = 1.0 + (viscosity_cross_time_constant * shear_rate)^
            #                       (1.0 - viscosity_cross_power_law_index)
            cross_argument = eta_zero_shear * shear_rate / cross_wlf_critical_stress
            cross_denominator = 1.0 + cross_argument^
                                  (1.0 - cross_wlf_power_law_index)
            concentration_base = max(1.0 - phase_vf_particle[i] / fiber_max_packing_fraction,
                                     eps(Float64))
            concentration_factor = concentration_base^
                                   (-fiber_intrinsic_viscosity * fiber_max_packing_fraction)
            vis[i] = eta_zero_shear / cross_denominator * concentration_factor
        end
        # vis[i] = viscosity_base_particle[i] * (0.2 + 0.8 * solid_factor) * orient_scale
        vis[i] = clamp(vis[i], viscosity_min_clip, viscosity_max_clip)
    end
end

# -----------------------------------------------------------------------------
# Fix B — WARM-UP REGIME LOCK
# While the punch is still ramping up to press speed (t < t_warmup_end) the
# charge has no mechanical driver: freezing it in the pure-viscous branch
# avoids the spurious elastic↔viscous switching that collapses TRBDF2's dt.
# After t_warmup_end the normal local temperature threshold takes over.
# -----------------------------------------------------------------------------
const t_warmup_end = t_ramp_mold

# -----------------------------------------------------------------------------
# Option E — global elastic-stress switch.
# When `enable_elastic_stress = false`, every charge particle uses the viscous
# constitutive model for the entire run. When true, the charge uses the
# elastoplastic branch with paper-style thermal softening for yield stress and
# hardening, i.e. `1 - H_theta * (theta - theta0)` with
# `H_theta = 1 / (T_melt - theta0)`.
# -----------------------------------------------------------------------------
const enable_elastic_stress = true
const elastic_skip_solid_fraction = 0.05
const viscous_skip_solid_fraction = 0.95



function update_implicit_stress_cache!(semi_local, v_ode, t)
    cyl_sys   = semi_local.systems[1]
    floor_sys = semi_local.systems[2]
    mold_sys  = semi_local.systems[3]

    # ===== TIMING: Properties Update =====
    t_props_start = time_ns()
    update_dual_phase_properties!(ys_particle_buf, hard_particle_buf, vis_particle_buf,
                                  cyl_sys.temp, alpha_committed[],
                                  thermal_softening_particle_buf,
                                  liquid_fraction_particle_buf,
                                  solid_fraction_particle_buf;
                                  vel_grad_buf=vel_grad_buf)
    rhs_time_properties_update_ns[] += time_ns() - t_props_start

    # -------------------------------------------------------------------------
    # Liquid-state reset: particles that are fully softened (thermal_softening ≈ 0)
    # have no meaningful elastic reference configuration. Their Fp tracks
    # accumulated viscous flow and rapidly becomes singular, producing NaN in
    # elastic_stress3d_trial!. Reset Fp to identity and alpha to 0 for these
    # particles each step. When they later cool below t_liquid, they re-enter
    # the elastic regime with a fresh (and physically correct) reference state.
    # -------------------------------------------------------------------------
    let liquid_eps = eps(Float64), Fp = Fp_committed[], alpha = alpha_committed[]
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            if liquid_fraction_particle_buf[particle] <= liquid_eps
                Fp[1, 1, particle] = 1.0; Fp[1, 2, particle] = 0.0; Fp[1, 3, particle] = 0.0
                Fp[2, 1, particle] = 0.0; Fp[2, 2, particle] = 1.0; Fp[2, 3, particle] = 0.0
                Fp[3, 1, particle] = 0.0; Fp[3, 2, particle] = 0.0; Fp[3, 3, particle] = 1.0
                alpha[particle] = 0.0
            end
        end
    end

    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)

    # Unified stress calculation with smooth blending
    fill!(v_stress_viscous_buf, 0)
    fill!(v_stress_elastic_buf, 0)

    # ===== TIMING: Viscous Stress =====
    t_visc_start = time_ns()
    stress_visc = viscous_stress_from_cached_grad!(cyl_sys, vis_particle_buf;
                                                   v_vis_buf=v_stress_viscous_buf,
                                                   vel_grad_buf=vel_grad_buf,
                                                   include_bulk_term=enable_elastic_stress,
                                                   solid_fraction_buf=solid_fraction_particle_buf,
                                                   skip_solid_fraction=viscous_skip_solid_fraction)
    rhs_time_viscous_stress_ns[] += time_ns() - t_visc_start

    if !enable_elastic_stress
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] = stress_visc[i, j, particle]
            end
        end
        TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), v_stress_buf)
        return
    end

    # ===== TIMING: Elastic Trial =====
    t_elas_start = time_ns()
    stress_elas = elastic_stress3d_trial_skip_liquid!(
        cyl_sys, ys_particle_buf, hard_particle_buf, vis_particle_buf,
        trial_dt_state[], alpha_committed[], Fp_committed[], semi_local;
        solid_fraction_buf=solid_fraction_particle_buf,
        skip_solid_fraction=elastic_skip_solid_fraction,
        v_elas_buf=v_stress_elastic_buf)
    rhs_time_elastic_trial_ns[] += time_ns() - t_elas_start

    # ===== TIMING: Stress Blending =====
    t_blend_start = time_ns()
    n_bad_elas = 0
    first_bad = -1
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        elas_ok = true
        for j in 1:3, i in 1:3
            if !isfinite(stress_elas[i, j, particle])
                elas_ok = false
                break
            end
        end

        if elas_ok
            thermal_blend = thermal_softening_particle_buf[particle]
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] = thermal_blend * stress_elas[i, j, particle] +
                                               (1.0 - thermal_blend) * stress_visc[i, j, particle]
            end
            # # Pressure floor: only for fully softened (viscous) particles — molten material
            # # cannot sustain net tension. Skip for elastic particles (thermal_blend > 0.1)
            # # to avoid disturbing the elastic restoring pressure during Newton iterations.
            # p_mean = (v_stress_buf[1, 1, particle] + v_stress_buf[2, 2, particle] +
            #           v_stress_buf[3, 3, particle]) / 3
            # if p_mean > 0.0 && thermal_softening_particle_buf[particle] < 0.1
            #     v_stress_buf[1, 1, particle] -= p_mean
            #     v_stress_buf[2, 2, particle] -= p_mean
            #     v_stress_buf[3, 3, particle] -= p_mean
            # end
        else
            n_bad_elas += 1
            if first_bad < 0
                first_bad = particle
            end
            Fp_committed[][1, 1, particle] = 1.0
            Fp_committed[][1, 2, particle] = 0.0
            Fp_committed[][1, 3, particle] = 0.0
            Fp_committed[][2, 1, particle] = 0.0
            Fp_committed[][2, 2, particle] = 1.0
            Fp_committed[][2, 3, particle] = 0.0
            Fp_committed[][3, 1, particle] = 0.0
            Fp_committed[][3, 2, particle] = 0.0
            Fp_committed[][3, 3, particle] = 1.0
            alpha_committed[][particle] = 0.0
            cyl_sys.deformation_grad[1, 1, particle] = 1.0
            cyl_sys.deformation_grad[1, 2, particle] = 0.0
            cyl_sys.deformation_grad[1, 3, particle] = 0.0
            cyl_sys.deformation_grad[2, 1, particle] = 0.0
            cyl_sys.deformation_grad[2, 2, particle] = 1.0
            cyl_sys.deformation_grad[2, 3, particle] = 0.0
            cyl_sys.deformation_grad[3, 1, particle] = 0.0
            cyl_sys.deformation_grad[3, 2, particle] = 0.0
            cyl_sys.deformation_grad[3, 3, particle] = 1.0
            cyl_sys.current_coordinates[1, particle] = clamp(cyl_sys.current_coordinates[1, particle],
                                                             -x_safety_bound,
                                                             x_safety_bound)
            cyl_sys.current_coordinates[2, particle] = 0.0
            cyl_sys.current_coordinates[3, particle] = clamp(cyl_sys.current_coordinates[3, particle],
                                                             z_safety_min,
                                                             z_safety_max)
            v_cyl[1, particle] = 0.0
            v_cyl[2, particle] = 0.0
            v_cyl[3, particle] = 0.0
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] = stress_visc[i, j, particle]
            end
        end
    end
    rhs_time_stress_blend_ns[] += time_ns() - t_blend_start

    # if n_bad_elas > 0 && mod(rhs_eval_counter[], 2000) == 0
    #     println(">>> NaN guard: ", n_bad_elas, " particles had non-finite elastic stress at t=",
    #             round(t, digits=8), " (first idx=", first_bad,
    #             ", thermal_softening=",
    #             round(thermal_softening_particle_buf[first_bad], digits=4),
    #             ", temp=", round(cyl_sys.temp[first_bad], digits=2), " K)")
    #     flush(stdout)
    # end

    TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), v_stress_buf)
end

function commit_plastic_history_and_heat!(system, ys, hard, vis, dt, _alpha, _Fp)
    (; deformation_grad, young_modulus, poisson_ratio, temp, tmelt, hardening,
       material_density, cp, temp_ref) = system

    mu = young_modulus / (2 + 2 * poisson_ratio)
    K = young_modulus / (3 - 6 * poisson_ratio)
    H_theta = 1.0 / (tmelt - temp_ref[1])

    @inbounds for particle in TrixiParticles.eachparticle(system)
        F = TrixiParticles.deformation_gradient(system, particle)
        Fp_sm = StaticArrays.SMatrix{3,3}(@view _Fp[:, :, particle])

        det_F = det(F)
        F_reg = if !isfinite(det_F) || abs(det_F) < 1e-10
            F + 1e-4 * one(StaticArrays.SMatrix{3,3,eltype(F)})
        else
            F
        end

        d_fp = det(Fp_sm)
        if isfinite(d_fp) && abs(d_fp) > 1e-14
            Fp_sm = Fp_sm / cbrt(d_fp)
        end

        Fp_inv = inv(Fp_sm)
        Fe = F_reg * Fp_inv
        J_e = max(det(Fe), 1e-6)
        be = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - 1 / 3 * tr(be_bar) * I
        tau = K * log(J_e) * I + mu * dev_be

        dev_tau = tau - 1 / 3 * tr(tau) * I
        yf = sqrt(1.5) * sqrt(sum(dev_tau .^ 2)) - (ys[particle] + hard[particle])

        if yf > 1e-6
            H_alpha_theta = hardening * (1 - H_theta * (temp[particle] - temp_ref[1]))
            if temp[particle] < 0.5 * tmelt
                delta_gamma = yf / (3 * mu + H_alpha_theta)
            else
                delta_gamma = yf * dt / max(vis[particle], 1e-8)
            end

            norm_dev = sqrt(sum(dev_tau .* dev_tau))
            if norm_dev > 1e-14 && isfinite(norm_dev)
                n_dir = dev_tau / norm_dev
                c = delta_gamma * sqrt(1.5)
                A = c * n_dir
                A2 = A * A
                sc = abs(c) < 1e-14 ? one(c) : sinh(c) / c
                cc = abs(c) < 1e-14 ? one(c) : (cosh(c) - 1) / (c * c)
                exp_A = one(StaticArrays.SMatrix{3,3,eltype(n_dir)}) + sc * A + cc * A2
                Fp_old = Fp_sm
                Fp_old_inv = inv(Fp_old)
                Fp_sm = exp_A * Fp_sm
                Fp_sm = Fp_sm / cbrt(det(Fp_sm))

                Fp_inv = inv(Fp_sm)
                Fe = F_reg * Fp_inv
                J_e = max(det(Fe), 1e-6)
                be = Fe * Fe'
                be_bar = be / (J_e^(2 / 3))
                dev_be = be_bar - 1 / 3 * tr(be_bar) * I
                tau = K * log(J_e) * I + mu * dev_be

                _alpha[particle] += delta_gamma

                dFp_total = Fp_sm - Fp_old
                depsilon = 0.5 * (dFp_total * Fp_old_inv + (dFp_total * Fp_old_inv)')
                plastic_work = sum(tau .* depsilon)
                # temp[particle] += 0.9 * plastic_work / (material_density[particle] * cp) * dt
            end
        end

        for j in 1:3, i in 1:3
            _Fp[i, j, particle] = Fp_sm[i, j]
        end
    end

    return nothing
end

function kick_implicit_visible!(dv_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.set_zero!(dv_ode)

    cyl_sys = semi_local.systems[1]
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)

    # Pseudo-2D projection: enforce zero out-of-plane velocity in trial states.
    # This suppresses spurious y-mode growth in the one-particle-thick slab.
    TrixiParticles.foreach_system(semi_local) do system
        v_sys = TrixiParticles.wrap_v(v_ode, system, semi_local)
        @inbounds for particle in TrixiParticles.each_integrated_particle(system)
            v_sys[2, particle] = 0.0
        end
    end

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        if enable_velocity_safety_clamps
            v_cyl[1, particle] = soft_limit(v_cyl[1, particle], max_cylinder_speed)
            v_cyl[3, particle] = soft_limit(v_cyl[3, particle], max_cylinder_speed)
        end
        t_trial = v_cyl[NDIMS_CYL + 1, particle]
        if !isfinite(t_trial)
            t_trial = cyl_sys.temp[particle]
        end
        cyl_sys.temp[particle] = clamp(t_trial, temp_min_clip, temp_max_clip)
    end

    try
        t_pos_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        # Safety clamp for pseudo-2D charge coordinates during nonlinear iterations.
        # This prevents rare transient outliers from polluting the neighborhood search
        # and VTK output.
        if enable_safety_clamps
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                cyl_sys.current_coordinates[1, particle] = clamp(cyl_sys.current_coordinates[1, particle],
                                                                 -x_safety_bound, x_safety_bound)
                cyl_sys.current_coordinates[2, particle] = 0.0
                cyl_sys.current_coordinates[3, particle] = clamp(cyl_sys.current_coordinates[3, particle],
                                                                 z_safety_min, z_safety_max)
            end
        end

        rhs_time_pos_ns[] += time_ns() - t_pos_start

        t_nhs_start = time_ns()
        if t != nhs_updated_at_t[]
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end
        rhs_time_nhs_ns[] += time_ns() - t_nhs_start

        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(cyl_sys)

        t_quant_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        rhs_time_quant_ns[] += time_ns() - t_quant_start

        # ---- Pseudo-2D plane-strain fix for deformation gradient ----
        # In a 1-particle-thick y slab, pos_diff_current[2] == 0 for every neighbor pair,
        # so the SPH sum gives F[2, :] = 0 for every charge particle.  This makes
        # det(F) = 0  →  J = 1e-6  →  K·log(J) ≈ -37 GPa (catastrophic tensile stress),
        # which is why the charge collapses immediately.
        # Fix: enforce plane-strain in y by setting the y-row to [0, 1, 0] and zeroing
        # the y off-diagonal column.  det(F) then equals the 2-D Jacobian of the x–z
        # sub-block, which is the correct volumetric measure for plane-strain.
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            cyl_sys.deformation_grad[2, 1, particle] = 0.0
            cyl_sys.deformation_grad[2, 2, particle] = 1.0  # no y-stretch
            cyl_sys.deformation_grad[2, 3, particle] = 0.0
            cyl_sys.deformation_grad[1, 2, particle] = 0.0  # y doesn't drive x or z strain
            cyl_sys.deformation_grad[3, 2, particle] = 0.0
        end

        # In pure-viscous mode the deformation gradient is only an auxiliary quantity
        # used by TLSPH internals; it carries no elastic history that must be preserved.
        # If a particle inverts (J <= 0) or nearly collapses, reset that particle's F to
        # plane-strain identity before the stress/contact assembly to avoid a later stall
        # from nonphysical F^{-T} / penalty-force usage.
        if enable_emergency_repair_clamp
            u_cyl = TrixiParticles.wrap_u(u_ode, cyl_sys, semi_local)
            repaired_F_particles = 0
            worst_J = Inf
            worst_J_particle = 0
            J_min = enable_elastic_stress ? 2.0e-1 : 5.0e-2
            J_max = enable_elastic_stress ? 5.0 : 10.0
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                F11 = cyl_sys.deformation_grad[1, 1, particle]
                F12 = cyl_sys.deformation_grad[1, 2, particle]
                F13 = cyl_sys.deformation_grad[1, 3, particle]
                F21 = cyl_sys.deformation_grad[2, 1, particle]
                F22 = cyl_sys.deformation_grad[2, 2, particle]
                F23 = cyl_sys.deformation_grad[2, 3, particle]
                F31 = cyl_sys.deformation_grad[3, 1, particle]
                F32 = cyl_sys.deformation_grad[3, 2, particle]
                F33 = cyl_sys.deformation_grad[3, 3, particle]
                J = F11 * (F22 * F33 - F23 * F32) -
                    F12 * (F21 * F33 - F23 * F31) +
                    F13 * (F21 * F32 - F22 * F31)
                if J < worst_J
                    worst_J = J
                    worst_J_particle = particle
                end
                if !isfinite(J) || J <= J_min || J >= J_max
                    cyl_sys.deformation_grad[1, 1, particle] = 1.0
                    cyl_sys.deformation_grad[1, 2, particle] = 0.0
                    cyl_sys.deformation_grad[1, 3, particle] = 0.0
                    cyl_sys.deformation_grad[2, 1, particle] = 0.0
                    cyl_sys.deformation_grad[2, 2, particle] = 1.0
                    cyl_sys.deformation_grad[2, 3, particle] = 0.0
                    cyl_sys.deformation_grad[3, 1, particle] = 0.0
                    cyl_sys.deformation_grad[3, 2, particle] = 0.0
                    cyl_sys.deformation_grad[3, 3, particle] = 1.0
                    if enable_elastic_stress
                        Fp_committed[][1, 1, particle] = 1.0
                        Fp_committed[][1, 2, particle] = 0.0
                        Fp_committed[][1, 3, particle] = 0.0
                        Fp_committed[][2, 1, particle] = 0.0
                        Fp_committed[][2, 2, particle] = 1.0
                        Fp_committed[][2, 3, particle] = 0.0
                        Fp_committed[][3, 1, particle] = 0.0
                        Fp_committed[][3, 2, particle] = 0.0
                        Fp_committed[][3, 3, particle] = 1.0
                        alpha_committed[][particle] = 0.0
                    end

                    u_cyl[1, particle] = clamp(u_cyl[1, particle], -x_safety_bound,
                                               x_safety_bound)
                    u_cyl[2, particle] = 0.0
                    u_cyl[3, particle] = clamp(u_cyl[3, particle], z_safety_min,
                                               z_safety_max)

                    cyl_sys.current_coordinates[1, particle] = clamp(cyl_sys.current_coordinates[1, particle],
                                                                     -x_safety_bound,
                                                                     x_safety_bound)
                    cyl_sys.current_coordinates[2, particle] = 0.0
                    cyl_sys.current_coordinates[3, particle] = clamp(cyl_sys.current_coordinates[3, particle],
                                                                     z_safety_min,
                                                                     z_safety_max)

                    v_cyl[1, particle] = 0.0
                    v_cyl[2, particle] = 0.0
                    v_cyl[3, particle] = 0.0

                    repaired_F_particles += 1
                end
            end
        end
        # ---- End pseudo-2D fix ----

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

        t_stress_cache_start = time_ns()
        update_implicit_stress_cache!(semi_local, v_ode, t)
        rhs_time_stress_cache_ns[] += time_ns() - t_stress_cache_start

        t_system_interaction_start = time_ns()
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
        rhs_time_system_interaction_ns[] += time_ns() - t_system_interaction_start

        t_source_terms_start = time_ns()
        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi_local, t)
        rhs_time_source_terms_ns[] += time_ns() - t_source_terms_start

        rhs_time_stress_interact_ns[] += time_ns() - t_stress_interact_start
    finally
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = nothing
        TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
    end

    t_thermal_start = time_ns()
    floor_sys = semi_local.systems[2]
    mold_sys  = semi_local.systems[3]

    t_contact_heat_start = time_ns()
    contact_heat_flux = compute_contact_heat_flux!(contact_heat_flux_buf, cyl_sys,
                                                   floor_sys, mold_sys, particle_spacing)
    rhs_time_contact_heat_ns[] += time_ns() - t_contact_heat_start

    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    dx = particle_spacing
    contact_damp_dist = 0.8 * smoothing_length

    t_thermal_sph_start = time_ns()
    TrixiParticles.thermal_rhs_sph3d!(cyl_sys, dv_cyl, v_cyl, 0.0,
                                      particle_spacing, bound_coordinate_thermal, semi_local)
    rhs_time_thermal_sph_ns[] += time_ns() - t_thermal_sph_start

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        cp_eff = effective_heat_capacity_particle(cyl_sys.temp[particle], particle)
        dv_cyl[NDIMS_CYL + 1, particle] *= cyl_sys.cp / cp_eff

        if contact_heat_flux[particle] != 0.0
            rho_i = cyl_sys.material_density[particle]
            dv_cyl[NDIMS_CYL + 1, particle] -= contact_heat_flux[particle] /
                                              (rho_i * cp_eff * dx)
        end
    end
    rhs_time_thermal_ns[] += time_ns() - t_thermal_start

    # Velocity damping in x/z to stabilize first-contact transients in the 1-particle slab.
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        x_i = cyl_sys.current_coordinates[1, particle]
        z_i = cyl_sys.current_coordinates[3, particle]
        near_floor = tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower) <=
                     contact_damp_dist
        near_mold = tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper) <=
                    contact_damp_dist
        damping_local = (near_floor || near_mold) ? velocity_damping_contact : velocity_damping_base

        # FIX: The negative damping term is accumulating out-of-control when combined with
        # the implicit TRBDF2 integrator's iterative updates, functioning as a non-linear
        # positive feedback loop. Remove it!
        # dv_cyl[1, particle] -= damping_local * v_cyl[1, particle]
        # dv_cyl[3, particle] -= damping_local * v_cyl[3, particle]
        if enable_velocity_safety_clamps
            dv_cyl[1, particle] = soft_limit(dv_cyl[1, particle], max_cylinder_accel)
            dv_cyl[3, particle] = soft_limit(dv_cyl[3, particle], max_cylinder_accel)
        end
    end
    TrixiParticles.foreach_system(semi_local) do system
        dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
        @inbounds for particle in TrixiParticles.each_integrated_particle(system)
            dv_sys[2, particle] = 0.0
        end
    end

    return dv_ode
end

function drift_implicit_visible!(du_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.drift!(du_ode, v_ode, u_ode, semi_local, t)

    # Pseudo-2D projection: prevent out-of-plane displacement drift.
    TrixiParticles.foreach_system(semi_local) do system
        du_sys = TrixiParticles.wrap_u(du_ode, system, semi_local)
        @inbounds for particle in TrixiParticles.each_integrated_particle(system)
            du_sys[2, particle] = 0.0
        end
    end

    return du_ode
end

ode = DynamicalODEProblem(kick_implicit_visible!, drift_implicit_visible!,
                          Vector{Float64}(ode_base.u0.x[1]),
                          Vector{Float64}(ode_base.u0.x[2]),
                          tspan, semi)

t_start, t_end = tspan

# Local compatibility helper: provide a periodic callback without requiring
# DiffEqCallbacks.jl in this environment.
function PeriodicCallback(affect!, dt; save_positions=(false, false))
    next_t = Ref(dt)
    condition(u, t, integrator) = t + 1.0e-14 >= next_t[]

    function wrapped_affect!(integrator)
        affect!(integrator)
        while next_t[] <= integrator.t + 1.0e-14
            next_t[] += dt
        end
    end

    return DiscreteCallback(condition, wrapped_affect!;
                            save_positions=save_positions)
end

function print_timing_snapshot(integrator)
    ns_to_s(ns) = ns / 1.0e9

    t_pos_s = ns_to_s(rhs_time_pos_ns[])
    t_nhs_s = ns_to_s(rhs_time_nhs_ns[])
    t_quant_s = ns_to_s(rhs_time_quant_ns[])
    t_implicit_s = ns_to_s(rhs_time_implicit_ns[])
    t_pressure_s = ns_to_s(rhs_time_pressure_ns[])
    t_boundary_s = ns_to_s(rhs_time_boundary_ns[])
    t_final_s = ns_to_s(rhs_time_final_ns[])
    t_stress_interact_s = ns_to_s(rhs_time_stress_interact_ns[])
    t_thermal_s = ns_to_s(rhs_time_thermal_ns[])
    t_props_s = ns_to_s(rhs_time_properties_update_ns[])
    t_elastic_s = ns_to_s(rhs_time_elastic_trial_ns[])
    t_visc_s = ns_to_s(rhs_time_viscous_stress_ns[])
    t_stress_cache_s = ns_to_s(rhs_time_stress_cache_ns[])
    t_system_interaction_s = ns_to_s(rhs_time_system_interaction_ns[])
    t_source_terms_s = ns_to_s(rhs_time_source_terms_ns[])
    t_callback_vel_grad_s = ns_to_s(rhs_time_callback_vel_grad_ns[])
    t_callback_orient_s = ns_to_s(rhs_time_callback_orientation_ns[])

    t_rhs_major_total = t_pos_s + t_nhs_s + t_quant_s + t_implicit_s + t_pressure_s +
                        t_boundary_s + t_final_s + t_stress_interact_s + t_thermal_s

    println("\n>>> TIMING SNAPSHOT at t=", round(integrator.t; digits=6),
            " s, iter=", integrator.iter)
    println("    major RHS total       = ", round(t_rhs_major_total; digits=4), " s")
    println("    stress/interactions   = ", round(t_stress_interact_s; digits=4), " s")
        println("      stress cache        = ", round(t_stress_cache_s; digits=4), " s")
        println("      system interaction  = ", round(t_system_interaction_s; digits=4), " s")
        println("      source terms        = ", round(t_source_terms_s; digits=4), " s")
    println("    thermal RHS           = ", round(t_thermal_s; digits=4), " s")
    println("    implicit SPH          = ", round(t_implicit_s; digits=4), " s")
    println("    elastic trial         = ", round(t_elastic_s; digits=4), " s")
    println("    viscous stress        = ", round(t_visc_s; digits=4), " s")
    println("    properties update     = ", round(t_props_s; digits=4), " s")
    println("    callback vel_grad     = ", round(t_callback_vel_grad_s; digits=4), " s")
    println("    callback orientation  = ", round(t_callback_orient_s; digits=4), " s")
    flush(stdout)
end

callbacks = CallbackSet(
    DiscreteCallback(
        (u, t, integrator) -> integrator.iter > 0 && mod(integrator.iter, 1) == 0,
        function(integrator)
            cyl_sys = integrator.p.systems[1]
            v_wrap_state = TrixiParticles.wrap_v(integrator.u.x[1], cyl_sys, integrator.p)

            t_vel_grad_start = time_ns()
            refresh_velocity_gradient_cache!(cyl_sys, v_wrap_state, integrator.p, vel_grad_buf)
            rhs_time_callback_vel_grad_ns[] += time_ns() - t_vel_grad_start

            # t_orient_start = time_ns()
            # update_flow_orientation_kinetics!(orientation_tensor_state[],
            #                                   orientation_scalar_particle,
            #                                   vel_grad_buf,
            #                                   integrator.dt,
            #                                   fiber_direction;
            #                                   xi_ft=xi_ft,
            #                                   ci_ft=ci_ft,
            #                                   kappa_rsc=kappa_rsc,
            #                                   ci_ard_parallel=ci_ard_parallel,
            #                                   ci_ard_perp=ci_ard_perp)
            # rhs_time_callback_orientation_ns[] += time_ns() - t_orient_start

            u_wrap = TrixiParticles.wrap_u(integrator.u.x[2], cyl_sys, integrator.p)
            if enable_emergency_repair_clamp
                repaired_particles = 0
                @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                    x = u_wrap[1, particle]
                    z = u_wrap[3, particle]
                    vx = v_wrap_state[1, particle]
                    vz = v_wrap_state[3, particle]

                    F11 = cyl_sys.deformation_grad[1, 1, particle]
                    F12 = cyl_sys.deformation_grad[1, 2, particle]
                    F13 = cyl_sys.deformation_grad[1, 3, particle]
                    F21 = cyl_sys.deformation_grad[2, 1, particle]
                    F22 = cyl_sys.deformation_grad[2, 2, particle]
                    F23 = cyl_sys.deformation_grad[2, 3, particle]
                    F31 = cyl_sys.deformation_grad[3, 1, particle]
                    F32 = cyl_sys.deformation_grad[3, 2, particle]
                    F33 = cyl_sys.deformation_grad[3, 3, particle]
                    J = F11 * (F22 * F33 - F23 * F32) -
                        F12 * (F21 * F33 - F23 * F31) +
                        F13 * (F21 * F32 - F22 * F31)

                    J_min = enable_elastic_stress ? 2.0e-1 : 5.0e-2
                    J_max = enable_elastic_stress ? 5.0 : 10.0

                    invalid_state = !isfinite(x) || !isfinite(z) || !isfinite(vx) || !isfinite(vz) ||
                                    x < -x_safety_bound || x > x_safety_bound ||
                                    z < z_safety_min || z > z_safety_max ||
                                    abs(vx) > max_cylinder_speed || abs(vz) > max_cylinder_speed ||
                                    !isfinite(J) || J <= J_min || J >= J_max

                    if invalid_state
                        u_wrap[1, particle] = clamp(x, -x_safety_bound, x_safety_bound)
                        u_wrap[2, particle] = 0.0
                        u_wrap[3, particle] = clamp(z, z_safety_min, z_safety_max)

                        cyl_sys.current_coordinates[1, particle] = u_wrap[1, particle]
                        cyl_sys.current_coordinates[2, particle] = 0.0
                        cyl_sys.current_coordinates[3, particle] = u_wrap[3, particle]

                        v_wrap_state[1, particle] = 0.0
                        v_wrap_state[2, particle] = 0.0
                        v_wrap_state[3, particle] = 0.0

                        cyl_sys.deformation_grad[1, 1, particle] = 1.0
                        cyl_sys.deformation_grad[1, 2, particle] = 0.0
                        cyl_sys.deformation_grad[1, 3, particle] = 0.0
                        cyl_sys.deformation_grad[2, 1, particle] = 0.0
                        cyl_sys.deformation_grad[2, 2, particle] = 1.0
                        cyl_sys.deformation_grad[2, 3, particle] = 0.0
                        cyl_sys.deformation_grad[3, 1, particle] = 0.0
                        cyl_sys.deformation_grad[3, 2, particle] = 0.0
                        cyl_sys.deformation_grad[3, 3, particle] = 1.0
                        if enable_elastic_stress
                            Fp_committed[][1, 1, particle] = 1.0
                            Fp_committed[][1, 2, particle] = 0.0
                            Fp_committed[][1, 3, particle] = 0.0
                            Fp_committed[][2, 1, particle] = 0.0
                            Fp_committed[][2, 2, particle] = 1.0
                            Fp_committed[][2, 3, particle] = 0.0
                            Fp_committed[][3, 1, particle] = 0.0
                            Fp_committed[][3, 2, particle] = 0.0
                            Fp_committed[][3, 3, particle] = 1.0
                            alpha_committed[][particle] = 0.0
                        end
                        repaired_particles += 1
                    elseif enable_velocity_safety_clamps
                        v_wrap_state[1, particle] = soft_limit(vx, max_cylinder_speed)
                        v_wrap_state[2, particle] = 0.0
                        v_wrap_state[3, particle] = soft_limit(vz, max_cylinder_speed)
                    end
                end
            end

            if enable_elastic_stress && integrator.t >= t_warmup_end
                update_dual_phase_properties!(ys_particle_buf, hard_particle_buf, vis_particle_buf,
                                              cyl_sys.temp, alpha_committed[],
                                              thermal_softening_particle_buf,
                                              liquid_fraction_particle_buf,
                                              solid_fraction_particle_buf)

                commit_plastic_history_and_heat!(cyl_sys,
                                                 ys_particle_buf,
                                                 hard_particle_buf,
                                                 vis_particle_buf,
                                                 integrator.dt,
                                                 alpha_committed[],
                                                 Fp_committed[])

                v_wrap = TrixiParticles.wrap_v(integrator.u.x[1], cyl_sys, integrator.p)
                NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
                @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                    v_wrap[NDIMS_CYL + 1, particle] = cyl_sys.temp[particle]
                end
            end

            if integrator.t >= t_compress && !solidification_hold_announced[]
                println(">>> Entering solidification hold at t=", round(integrator.t; digits=6),
                    " s (solid fraction threshold=",
                    round(100 * solidification_fraction_threshold; digits=1),
                    "%, target fraction=", round(100 * solidification_fraction_target; digits=1),
                        "%)")
                flush(stdout)
                solidification_hold_announced[] = true
            end

            if integrator.t >= t_compress && !solidification_complete[]
                solidified_count = count(fs >= solidification_fraction_threshold for fs in
                             solid_fraction_particle_buf)
                solidified_fraction = solidified_count / length(cyl_sys.temp)
                if solidified_fraction >= solidification_fraction_target
                    println(">>> Solidification target reached at t=",
                            round(integrator.t; digits=6), " s (",
                            round(100 * solidified_fraction; digits=1),
                        "% of charge at or above ",
                        round(100 * solidification_fraction_threshold; digits=1),
                        "% solid fraction)")
                    flush(stdout)
                    solidification_complete[] = true
                    solidification_reached_time[] = integrator.t
                    println(">>> Holding closed for post-solidification dwell of ",
                            round(post_solidification_dwell; digits=6),
                            " s before mold retraction")
                    flush(stdout)
                end
            end

            if solidification_complete[] && !retraction_started[]
                dwell_elapsed = integrator.t - solidification_reached_time[]
                if dwell_elapsed >= post_solidification_dwell
                    retraction_started[] = true
                    retraction_start_time[] = integrator.t
                    println(">>> Starting mold retraction at t=",
                            round(integrator.t; digits=6), " s")
                    flush(stdout)
                end
            end

            if retraction_started[] && !retraction_complete[]
                retract_elapsed = integrator.t - retraction_start_time[]
                if retract_elapsed >= retraction_duration_max
                    println(">>> Mold retraction complete at t=",
                            round(integrator.t; digits=6), " s")
                    flush(stdout)
                    retraction_complete[] = true
                    terminate!(integrator)
                end
            end
        end),
    PeriodicCallback(print_timing_snapshot, 2.0e-2,
                     save_positions=(false, false)),
    SolutionSavingCallback(dt=5e-3,
                           prefix="molding_cfk_pseudo2d_elastic_3_solution",
                           max_coordinates=Inf,
                           fiber_vf=(sys, data, t) -> nparticles(sys) == length(phase_vf_particle) ?
                                                      phase_vf_particle : fill(NaN, nparticles(sys))),
    InfoCallback(interval=2)
)

println("\n>>> Entering ODE solve (tspan=", tspan,
    ", implicit solver=TRBDF2)")
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
          callback=callbacks,
          save_everystep=false,
          # Fix A — relaxed tolerances. abstol=1e-5 was binding on stresses ~1e8 Pa
              # relax abstol further to avoid Newton iteration stalls. The forces are huge (~1e8),
          # so 1e-1 is still virtually zero.
          abstol=1.0e-3,
          reltol=1.0e-3,
          dtmax=1.0e-3,
          maxiters=10_000_000)
end

println("\n=== SIMULATION COMPLETE ===")
println("Solver retcode: ", sol.retcode)
println("\n" * "="^80)
println("COMPREHENSIVE TIMING DIAGNOSTICS")
println("="^80)

# Convert nanoseconds to seconds
ns_to_s(ns) = ns / 1.0e9

# Major RHS phases
t_pos_s = ns_to_s(rhs_time_pos_ns[])
t_nhs_s = ns_to_s(rhs_time_nhs_ns[])
t_quant_s = ns_to_s(rhs_time_quant_ns[])
t_implicit_s = ns_to_s(rhs_time_implicit_ns[])
t_pressure_s = ns_to_s(rhs_time_pressure_ns[])
t_boundary_s = ns_to_s(rhs_time_boundary_ns[])
t_final_s = ns_to_s(rhs_time_final_ns[])
t_stress_interact_s = ns_to_s(rhs_time_stress_interact_ns[])
t_thermal_s = ns_to_s(rhs_time_thermal_ns[])

# Fine-grained constitutive phases
t_props_s = ns_to_s(rhs_time_properties_update_ns[])
t_elastic_s = ns_to_s(rhs_time_elastic_trial_ns[])
t_visc_s = ns_to_s(rhs_time_viscous_stress_ns[])
t_blend_s = ns_to_s(rhs_time_stress_blend_ns[])
t_stress_cache_s = ns_to_s(rhs_time_stress_cache_ns[])
t_system_interaction_s = ns_to_s(rhs_time_system_interaction_ns[])
t_source_terms_s = ns_to_s(rhs_time_source_terms_ns[])
t_contact_heat_s = ns_to_s(rhs_time_contact_heat_ns[])
t_thermal_sph_s = ns_to_s(rhs_time_thermal_sph_ns[])

# Callback phases
t_callback_vel_grad_s = ns_to_s(rhs_time_callback_vel_grad_ns[])
t_callback_orient_s = ns_to_s(rhs_time_callback_orientation_ns[])

t_total_s = sol.t[end]

println("\n>>> MAJOR RHS PHASES (accumulated over all RHS evaluations):")
println(string("    Position updates:        ", lpad(round(t_pos_s, digits=4), 10), " s"))
println(string("    Neighborhood search:     ", lpad(round(t_nhs_s, digits=4), 10), " s"))
println(string("    Quantity updates (F):    ", lpad(round(t_quant_s, digits=4), 10), " s"))
println(string("    Implicit SPH:            ", lpad(round(t_implicit_s, digits=4), 10), " s"))
println(string("    Pressure update:         ", lpad(round(t_pressure_s, digits=4), 10), " s"))
println(string("    Boundary interpolation:  ", lpad(round(t_boundary_s, digits=4), 10), " s"))
println(string("    Final quantities:        ", lpad(round(t_final_s, digits=4), 10), " s"))
println(string("    Stress & interactions:   ", lpad(round(t_stress_interact_s, digits=4), 10), " s"))
println(string("    Thermal RHS:             ", lpad(round(t_thermal_s, digits=4), 10), " s"))
t_rhs_major_total = t_pos_s + t_nhs_s + t_quant_s + t_implicit_s + t_pressure_s +
                    t_boundary_s + t_final_s + t_stress_interact_s + t_thermal_s
println(string("    ──────────────────────────────────────────────────────────"))
println(string("    TOTAL Major Phases:      ", lpad(round(t_rhs_major_total, digits=4), 10), " s"))

println("\n>>> FINE-GRAINED CONSTITUTIVE PHASES (within update_implicit_stress_cache!):")
println(string("    Properties update:       ", lpad(round(t_props_s, digits=4), 10), " s"))
println(string("    Elastic trial stress:    ", lpad(round(t_elastic_s, digits=4), 10), " s"))
println(string("    Viscous stress:          ", lpad(round(t_visc_s, digits=4), 10), " s"))
println(string("    Stress blending + NaN:   ", lpad(round(t_blend_s, digits=4), 10), " s"))
t_stress_cache_total = t_props_s + t_elastic_s + t_visc_s + t_blend_s
println(string("    ──────────────────────────────────────────────────────────"))
println(string("    TOTAL Stress Cache:      ", lpad(round(t_stress_cache_total, digits=4), 10), " s"))

println("\n>>> FINE-GRAINED STRESS / INTERACTION PHASES:")
println(string("    Stress cache call:       ", lpad(round(t_stress_cache_s, digits=4), 10), " s"))
println(string("    System interaction:      ", lpad(round(t_system_interaction_s, digits=4), 10), " s"))
println(string("    Source terms:            ", lpad(round(t_source_terms_s, digits=4), 10), " s"))

println("\n>>> THERMAL PHASES:")
println(string("    Contact heat flux:       ", lpad(round(t_contact_heat_s, digits=4), 10), " s"))
println(string("    Thermal RHS SPH3D:       ", lpad(round(t_thermal_sph_s, digits=4), 10), " s"))
println(string("    Other thermal overhead:  ", lpad(round(t_thermal_s - t_contact_heat_s - t_thermal_sph_s, digits=4), 10), " s"))

println("\n>>> CALLBACK PHASES (periodic, every step):")
println(string("    Velocity gradient cache: ", lpad(round(t_callback_vel_grad_s, digits=4), 10), " s"))
println(string("    Orientation kinetics:    ", lpad(round(t_callback_orient_s, digits=4), 10), " s"))

println("\n>>> RANK ORDERING (by cumulative time spent):")
timing_dict = Dict(
    "Position updates" => t_pos_s,
    "Stress & interactions" => t_stress_interact_s,
    "Thermal RHS" => t_thermal_s,
    "Implicit SPH" => t_implicit_s,
    "Quantity updates" => t_quant_s,
    "Elastic trial" => t_elastic_s,
    "Pressure update" => t_pressure_s,
    "Boundary interp" => t_boundary_s,
    "Properties update" => t_props_s,
    "Viscous stress" => t_visc_s,
    "Callback vel_grad" => t_callback_vel_grad_s,
    "Contact heat flux" => t_contact_heat_s,
    "Callback orient" => t_callback_orient_s,
    "Final quantities" => t_final_s,
    "Neighborhood search" => t_nhs_s,
    "Stress blending" => t_blend_s,
    "Thermal SPH3D" => t_thermal_sph_s,
)
sorted_timings = sort(collect(timing_dict), by=x -> x[2], rev=true)
for (i, (name, time_s)) in enumerate(sorted_timings)
    pct = 100.0 * time_s / t_rhs_major_total
    println(string(lpad(i, 2), ". ", rpad(name, 25), 
                   lpad(round(time_s, digits=4), 10), " s  (", 
                   lpad(round(pct, digits=1), 5), "%)"))
end

println("\n" * "="^80)
println("INTERPRETATION:")
println("  • Top 3 functions account for ~", 
        round(100.0 * (sorted_timings[1][2] + sorted_timings[2][2] + sorted_timings[3][2]) / t_rhs_major_total, digits=1),
        "% of RHS runtime")
println("  • Stress & interactions includes: update_implicit_stress_cache! +")
println("    system_interaction! + add_source_terms!")
println("  • Thermal includes: thermal_rhs_sph3d! + heat flux + assembly")
println("  • If 'Stress & interactions' dominates, the main cost is in")
println("    elastic_stress3d_trial! or viscous_stress_from_cached_grad!")
println("="^80 * "\n")

