# ==========================================================================================
# 2D Oscillating Elastic Beam (Cantilever) Simulation
#
# This example simulates the oscillation of a 2D elastic beam (cantilever)
# clamped at one end and subjected to gravity. It uses the Total Lagrangian SPH (TLSPH)
# method for structure mechanics.
#
# Based on:
# J. O'Connor and B.D. Rogers
# "A fluid-structure interaction model for free-surface flows and
# flexible structures using smoothed particle hydrodynamics on a GPU",
# Journal of Fluids and Structures, Volume 104, 2021.
# DOI: 10.1016/j.jfluidstructs.2021.103312
# ==========================================================================================
using TrixiParticles
using OrdinaryDiffEq
using ADTypes
using LinearSolve
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
# STEP 1: Load and pack the cylinder geometry
# ==========================================================================================

file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "Cylinder.stl")
geometry = load_geometry(file)

particle_spacing = 0.002
boundary_thickness = 3 * particle_spacing

signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

density_cylinder = 1000.0  # kg/m³ — realistic polymer density

point_in_geometry_algorithm = WindingNumberJacobson(; geometry)

shape_sampled = ComplexShape(geometry; particle_spacing, density=density_cylinder,
                             point_in_geometry_algorithm, pad_initial_particle_grid=0.002)

shape_sampled.mass .= density_cylinder * TrixiParticles.volume(geometry) / nparticles(shape_sampled)

background_pressure = 1.0

factor = 1.2
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = factor * particle_spacing

# ==========================================================================================
# STEP 2: Pack the cylinder particles
# ==========================================================================================

packing_system = ParticlePackingSystem(shape_sampled;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=signed_distance_field,  # provide SDF!
                                       background_pressure=background_pressure)

semi_pack = Semidiscretization(packing_system)

tspan_pack = (0.0, 10000.0)
ode_pack = semidiscretize(semi_pack, tspan_pack)

sol_pack = solve(ode_pack, RDPK3SpFSAL35();
                 abstol=1e-7, reltol=1e-4, save_everystep=false, maxiters=100,
                 callback=CallbackSet(UpdateCallback()))

# Extract packed initial condition for the cylinder
polymer = InitialCondition(sol_pack, packing_system, semi_pack)

# # Rotate cylinder 90° so axis is along X (horizontal) and curved surface faces down
# # Swap Y and Z coordinates
# new_coords = copy(polymer.coordinates)
# new_coords[2,:], new_coords[3,:] = polymer.coordinates[3,:], polymer.coordinates[2,:]
# polymer.coordinates .= new_coords

# ==========================================================================================
# STEP 3: Visualise packed cylinder
# ==========================================================================================

coords = polymer.coordinates
scatter3d(coords[1, :], coords[2, :], coords[3, :],
          markersize=1, aspect_ratio=:equal, label="Cylinder particles")
savefig("packing.png")

# ==========================================================================================
# STEP 4: Build the mold/floor wall boundary
# ==========================================================================================

floor_density = density_cylinder  # match fluid density for Adami extrapolation

# Position floor just 1 particle_spacing below cylinder
cyl_z_min    = minimum(polymer.coordinates[3,:])
cyl_z_max    = maximum(polymer.coordinates[3,:])
# floor_origin = cyl_z_min - factor * 0.001

# floor_particles = RectangularShape(particle_spacing, (30,30,4),
#                                    (-0.01, -0.01, floor_origin - 2*particle_spacing);
#                                    density=floor_density)

# z_surface = cyl_z_min - factor * 0.0015
# z_top_surface = cyl_z_max + 0.0015

# Gap scales with particle_spacing: contact onset and initial transient are then identical
# in particle-spacing units across all mesh resolutions.
initial_gap = 0.0000 + 0.05 * particle_spacing  # 0.5mm + half spacing to ensure no initial contact

z_surface     = cyl_z_min - initial_gap
z_top_surface = cyl_z_max + initial_gap

# 2. Calculate the origin (the bottom-back-left corner) 
# based on how many layers you want. 
# Thickness = (number_of_layers - 1) * ps
num_layers = 3

floor_thickness = (num_layers) * particle_spacing
z_bottom = z_surface - floor_thickness

z_top = z_top_surface

# 3. Create the shape using the calculated bottom
# Plate count scales with particle_spacing to maintain the same physical coverage (25mm)
n_plate = ceil(Int, 0.025 / particle_spacing)

floor_particles = RectangularShape(particle_spacing, (n_plate, n_plate, num_layers),
                                   (-0.01, -0.01, z_bottom);
                                   density=floor_density)

mold_particles = RectangularShape(particle_spacing, (n_plate, n_plate, num_layers),
        (-0.01, -0.01, z_top);
        density=floor_density)

floor_z_max = maximum(floor_particles.coordinates[3,:])
println("Gap = ", cyl_z_min - floor_z_max)
println("h   = ", factor* particle_spacing)
println("Contact at t=0? ", (cyl_z_min - floor_z_max) < factor * particle_spacing)

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
mold_velocity_state = -0.05
# Use a longer ramp to avoid a sharp contact impulse when mold first engages.
t_ramp_mold = 1.0e-2

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
println("  - Viscous regime: activated when T > 0.5 * T_melt")
println("  - Initial temperature: 270 K")

# ==========================================================================================
# STEP 5: Build the Total Lagrangian SPH system for the cylinder
# ==========================================================================================

material_polymer = (density=1500.0, E=20e9, nu=0.3 ,beta=0.000, temp=270.0, temp_ref=270.0, cp=900.0 , k=1.0,
                            temp_liq=390.0,h= 70000.0,hardening= 1e9,tmelt=700.0, viscosity=100000.0, yield_stress=150e6)
local_regime_threshold = 0.5 * material_polymer.tmelt

import PointNeighbors: DictionaryCellList

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=200)

# Floor: all particles clamped → no ODE integration, rigid boundary for contact
floor_system = TotalLagrangianSPHSystem(floor_particles,
                                           smoothing_kernel,
                                           smoothing_length,
                                           1e7,
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
                                           1e7,
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

cylinder_system = TotalLagrangianSPHSystem(polymer,
                                           smoothing_kernel,
                                           smoothing_length,
                                           1e5,                     #E-moudulus
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
preheating_target_temp = 400.0  # Target: 400 K (between room temp 270K and melt 700K)

println("\n=== PREHEATING PHASE ===")
println("Setting cylinder temperature to $preheating_target_temp K...")
semi.systems[1].temp .= preheating_target_temp
println("✓ Cylinder preheated: T = $preheating_target_temp K")
println("=== PREHEATING COMPLETE ===\n")

cylinder_height = maximum(polymer.coordinates[3,:]) - minimum(polymer.coordinates[3,:])
t_compress = cylinder_height * 0.5 / 0.05  # time to compress 10% height at 50mm/s
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
solver_diag_interval = 100
solver_diag_last_accept = Ref(0)
solver_diag_last_iter = Ref(0)
solver_diag_last_rhs = Ref(0)
solver_diag_dt_min = Ref(Inf)
solver_diag_dt_max = Ref(0.0)
solver_diag_dt_sum = Ref(0.0)
enable_contact_diag = false
contact_heat_flux_buf = zeros(n_cylinder_particles)

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
    contact_threshold = 0.25 * particle_spacing

    @inbounds for i in 1:nparticles(system_cyl)
        z_i = system_cyl.current_coordinates[3, i]
        T_i = system_cyl.temp[i]

        gap_floor = z_i - z_floor_top
        if 0.0 <= gap_floor <= contact_threshold && T_i > T_floor
            ext_heat_per_particle[i] = h_contact * (T_i - T_floor)
            if first_floor_heat_xfer[]
                println(">>> Floor heat transfer started: T_cyl=", round(T_i, digits=2),
                        " K, T_floor=", round(T_floor, digits=2),
                        " K, gap=", round(gap_floor * 1e3, digits=3), " mm")
                flush(stdout)
                first_floor_heat_xfer[] = false
            end
        end

        gap_mold = z_mold_bot - z_i
        if 0.0 <= gap_mold <= contact_threshold && T_i > T_mold
            ext_heat_per_particle[i] = max(ext_heat_per_particle[i], h_contact * (T_i - T_mold))
            if first_mold_heat_xfer[]
                println(">>> Mold heat transfer started: T_cyl=", round(T_i, digits=2),
                        " K, T_mold=", round(T_mold, digits=2),
                        " K, gap=", round(gap_mold * 1e3, digits=3), " mm")
                flush(stdout)
                first_mold_heat_xfer[] = false
            end
        end
    end

    return ext_heat_per_particle
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
    ys, hard, vis = TrixiParticles.update_properties!(cyl_sys, alpha_committed[], semi_local,
                                                      material_polymer.viscosity)
    n_hot = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        n_hot += cyl_sys.temp[particle] > local_regime_threshold
    end
    n_total = length(cyl_sys.temp)
    n_cool = n_total - n_hot

    if n_hot > 0 && first_viscous_regime[]
        if first_viscous_regime[]
            println(">>> Regime: VISCOUS activated locally (hot particles=", n_hot,
                    "/", n_total, ", threshold=", local_regime_threshold,
                    " K) at t=", round(t, digits=6))
            flush(stdout)
            first_viscous_regime[] = false
        end
    end

    if n_cool > 0 && first_elastic_regime[]
        println(">>> Regime: ELASTIC/PLASTIC active locally (cool particles=", n_cool,
                "/", n_total, ", threshold=", local_regime_threshold,
                " K) at t=", round(t, digits=6))
        flush(stdout)
        first_elastic_regime[] = false
    end

    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    if n_hot == n_total
        fill!(v_stress_buf, 0)
        fill!(vel_grad_buf, 0)
        stress_visc = TrixiParticles.viscous_stress3d_fast!(cyl_sys, v_cyl, vis, semi_local;
                                                             v_vis_buf=v_stress_buf,
                                                             vel_grad_buf=vel_grad_buf)
        TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), stress_visc)
    elseif n_hot == 0
        fill!(v_stress_buf, 0)
        stress_elas = TrixiParticles.elastic_stress3d_trial!(
            cyl_sys, ys, hard, vis, trial_dt_state[], alpha_committed[], Fp_committed[], semi_local;
            v_elas_buf=v_stress_buf)
        TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), stress_elas)
    else
        fill!(v_stress_viscous_buf, 0)
        fill!(v_stress_elastic_buf, 0)
        fill!(v_stress_buf, 0)
        fill!(vel_grad_buf, 0)
        stress_visc = TrixiParticles.viscous_stress3d_fast!(cyl_sys, v_cyl, vis, semi_local;
                                                             v_vis_buf=v_stress_viscous_buf,
                                                             vel_grad_buf=vel_grad_buf)
        stress_elas = TrixiParticles.elastic_stress3d_trial!(
            cyl_sys, ys, hard, vis, trial_dt_state[], alpha_committed[], Fp_committed[], semi_local;
            v_elas_buf=v_stress_elastic_buf)

        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            stress_src = cyl_sys.temp[particle] > local_regime_threshold ? stress_visc : stress_elas
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
    if mod(rhs_eval_counter[], 500) == 0
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
        cyl_sys.temp[particle] = v_cyl[NDIMS_CYL + 1, particle]
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
    rho_cyl = cyl_sys.material_density[1]
    cp_cyl = cyl_sys.cp
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        if contact_heat_flux[particle] > 0.0
            dv_cyl[NDIMS_CYL + 1, particle] -= contact_heat_flux[particle] / (rho_cyl * cp_cyl * dx)
        end
    end
    TrixiParticles.thermal_rhs_sph3d!(cyl_sys, dv_cyl, v_cyl, 0.0,
                                      particle_spacing, bound_coordinate_thermal, semi_local)
    rhs_time_thermal_ns[] += time_ns() - t_thermal_start

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
progress_heartbeat_interval = 20

# println("progress init | first percent mark at t=", round(next_progress_t[], digits=6),
#     " (", progress_step_percent, "%)")

# progress_cb = DiscreteCallback(
#     (u, t, integrator) -> true,
#     function (integrator)
#         progress_dt <= 0.0 && return
#         t = integrator.t
#         progress_step_counter[] += 1

#         # Heartbeat independent of simulated time progress; useful when dt is tiny.
#         if mod(progress_step_counter[], progress_heartbeat_interval) == 0
#             pct_hb = clamp(100.0 * (t - t_start) / (t_end - t_start), 0.0, 100.0)
#             println("progress hb | step=", progress_step_counter[],
#                     " | ", round(pct_hb, digits=3), "%",
#                     " | t=", round(t, digits=6), " / ", round(t_end, digits=6),
#                     " | dt=", integrator.dt)
#         end

#         if t + 1.0e-14 >= next_progress_t[]
#             pct = clamp(100.0 * (t - t_start) / (t_end - t_start), 0.0, 100.0)
#             println("progress | ", round(pct, digits=1), "% | t=",
#                     round(t, digits=6), " / ", round(t_end, digits=6),
#                     " | dt=", integrator.dt)

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
                n_cool += cyl_sys.temp[particle] <= local_regime_threshold
            end

            if n_cool > 0
                ys_c, hard_c, vis_c = TrixiParticles.update_properties!(
                    cyl_sys, alpha_committed[], integrator.p, material_polymer.viscosity)
                hot_particles = findall(>(local_regime_threshold), cyl_sys.temp)
                hot_alpha = alpha_committed[][hot_particles]
                hot_temp = cyl_sys.temp[hot_particles]
                hot_Fp = copy(Fp_committed[][:, :, hot_particles])

                # Commits Fp, alpha, and temp in-place using the actual accepted dt
                # for the locally cool particles only.
                trial_dt_state[] = integrator.dt
                TrixiParticles.elastic_stress3d_fast!(
                    cyl_sys, ys_c, hard_c, vis_c, integrator.dt,
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
    SolutionSavingCallback(dt=2.0e-3, prefix="molding_cfrp_3d_thermo_implicit_krylov_mod2"),
    InfoCallback(interval=10)
)

# krylovdim must stay large enough to avoid KrylovKit restart-triggered ArrayPartition
# type mismatch (Vector{Float64} vs ArrayPartition in Givens rotation during restart).
# Small krylovdim (≤40) causes frequent restarts which trigger this KrylovKit bug.
# With dtmax=1e-4 the Newton system is mild enough that krylovdim=120 converges fast.
gmres_solver = KrylovKitJL_GMRES(; krylovdim=40,
                                   atol=1e-6, rtol=5.0e-2,
                                   maxiter=200, verbosity=0)

sol = Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
    solve(ode, TRBDF2(linsolve=gmres_solver, autodiff=AutoFiniteDiff());
            callback=callbacks,
            save_everystep=false,
            abstol=1.0e-5,   # loosened from 1e-7 for screening speed
            reltol=1.0e-3,   # loosened from 1e-4 for screening speed
            dtmax=1.0e-4,    # tightened: 1ms was too stiff for Newton at contact onset
            maxiters=10_000_000)
end
