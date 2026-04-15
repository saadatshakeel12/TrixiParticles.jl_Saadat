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
using PointNeighbors
using Plots
using Base.Threads
using Statistics  # for mean() function
using LinearAlgebra # for Identity matrix I

println("--- SIMULATION STARTING ---")
println("Threads available: ", nthreads())
println("---------------------------")

# ==========================================================================================
# STEP 1: Load and pack the cylinder geometry
# ==========================================================================================

file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "Cylinder.stl")
geometry = load_geometry(file)

particle_spacing = 0.0015
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

factor = 1.5
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
initial_gap = 0.0015 + 0.5 * particle_spacing  # 1.5mm + half spacing to ensure no initial contact

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

F_total_mold_state = Ref(0.0)                          # total force on mold (for tracking)

# Mold velocity (downward push, m/s)
mold_velocity_state = -0.05
# Use a longer ramp to avoid a sharp contact impulse when mold first engages.
t_ramp_mold = 2.0e-3

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
                            temp_liq=390.0,h= 1000.0,hardening= 1e9,tmelt=700.0, viscosity=100000.0, yield_stress=150e6)

import PointNeighbors: DictionaryCellList

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=600)

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
                                           acceleration=(0.0, 0.0, 0.0),
                                           self_interaction_nhs=nhs_template)

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

ode = semidiscretize(semi, tspan)

# Pre-allocate persistent stress/velocity-gradient buffers to eliminate per-step heap allocations.
# These are reused across callback invocations; functions zero them before use.
v_stress_buf  = zeros(3, 3, n_cylinder_particles)   # output of elastic/viscous stress
vel_grad_buf  = zeros(3, 3, n_cylinder_particles)   # intermediate velocity gradient (viscous path)

# Thermal SPH is O(N·K) but thermally negligible per step (Fo per Δt ~ 1e-10).
# Run it every N_therm mechanical steps — equivalent dt_therm still 1e8× inside CFL.
therm_counter = Ref(0)
N_therm = 200  # thermal update interval (mechanical steps)
dt_cap = 1.0e-6
contact_diag_interval = 5000

# ==========================================================================================
# THERMOMECHANICAL CALLBACK: Elastoplastic-Viscous-Thermal Compression
# ==========================================================================================
# This callback integrates thermomechanical effects: plasticity, viscous flow, thermal diffusion

thermomechanical_cb = DiscreteCallback(
    (u, t, integrator) -> true,  # call every step
    function (integrator)
        semi  = integrator.p
        v_ode = integrator.u.x[1]
        u_ode = integrator.u.x[2]
        dt    = integrator.dt
        t     = integrator.t

        # Access systems
        cyl_sys   = semi.systems[1]
        floor_sys = semi.systems[2]
        mold_sys  = semi.systems[3]

        # ========== STEP 1: Update temperature-dependent material properties ==========
        ys, hard, vis = TrixiParticles.update_properties!(cyl_sys, alpha_plastic_state[], semi, material_polymer.viscosity)
        vis = clamp.(vis, 1.0e-1, 1.0e6)

        # ========== STEP 2: Regime check ==========
        temp_avg = sum(cyl_sys.temp) / length(cyl_sys.temp)

        # ========== STEP 3: Thermal diffusion + contact heat transfer ==========
        h_contact = cyl_sys.h
        T_tool    = material_polymer.temp_ref

        # Current z-positions of tool surfaces (updated by PrescribedMotion automatically)
        z_floor_top = maximum(floor_sys.current_coordinates[3, :])
        z_mold_bot  = minimum(mold_sys.current_coordinates[3, :])
        contact_dist = particle_spacing

        n_floor_contact = 0
        n_mold_contact = 0
        n_floor_penetrated = 0
        n_mold_penetrated = 0
        min_gap_floor = Inf
        min_gap_mold = Inf
        @inbounds for i in 1:nparticles(cyl_sys)
            z_i = cyl_sys.current_coordinates[3, i]
            rho_cp_dx = cyl_sys.material_density[i] * cyl_sys.cp * particle_spacing
            gap_floor = z_i - z_floor_top
            gap_mold = z_mold_bot - z_i
            min_gap_floor = min(min_gap_floor, gap_floor)
            min_gap_mold = min(min_gap_mold, gap_mold)
            # Floor contact: particle is at most one spacing above the floor surface
            if 0.0 <= gap_floor <= contact_dist
                n_floor_contact += 1
                cyl_sys.temp[i] += h_contact * (T_tool - cyl_sys.temp[i]) / rho_cp_dx * dt
            elseif gap_floor < 0.0
                n_floor_penetrated += 1
            end
            # Mold contact: particle is at most one spacing below the mold bottom surface
            if 0.0 <= gap_mold <= contact_dist
                n_mold_contact += 1
                cyl_sys.temp[i] += h_contact * (T_tool - cyl_sys.temp[i]) / rho_cp_dx * dt
            elseif gap_mold < 0.0
                n_mold_penetrated += 1
            end
        end

        # Contact diagnostics: confirms when mold-floor contact zones become active.
        if therm_counter[] > 0 && mod(therm_counter[], contact_diag_interval) == 0
            println("contact diag | t=", round(t, digits=6),
                    " | mold_particles=", n_mold_contact,
                    " | floor_particles=", n_floor_contact,
                    " | mold_penetrated=", n_mold_penetrated,
                    " | floor_penetrated=", n_floor_penetrated,
                    " | min_gap_mold=", round(min_gap_mold, digits=6),
                    " | min_gap_floor=", round(min_gap_floor, digits=6))
        end

        # SPH thermal diffusion — throttled: thermal timescale >> mechanical Δt
        therm_counter[] += 1
        if mod(therm_counter[], N_therm) == 0
            dt_therm = dt * N_therm  # effective thermal timestep
            TrixiParticles.update_temperature_sph3d!(cyl_sys, dt_therm, 0.0,
                                  particle_spacing, bound_coordinate_thermal, semi)
        end

        # ========== STEP 4: Compute elastoplastic/viscous stress and store in cache ==========
        # Only enable the custom thermomechanical stress path near contact.
        # Before contact, keep default TLSPH PK1 stress to avoid pre-contact blow-up.
        near_contact = (n_mold_contact + n_floor_contact > 0) ||
                       (min_gap_mold < 2 * contact_dist) ||
                       (min_gap_floor < 2 * contact_dist)
        if near_contact
            if temp_avg > 0.5 * material_polymer.tmelt
                v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi)
                fill!(v_stress_buf, 0)
                fill!(vel_grad_buf, 0)
                stress_visc = TrixiParticles.viscous_stress3d_fast!(cyl_sys, v_cyl, vis, semi;
                                                                     v_vis_buf=v_stress_buf,
                                                                     vel_grad_buf=vel_grad_buf)
                TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), stress_visc)
            else
                fill!(v_stress_buf, 0)
                stress_elas, alpha_plastic_state[], Fp_state[] = TrixiParticles.elastic_stress3d_fast!(
                    cyl_sys, ys, hard, vis, dt, alpha_plastic_state[], Fp_state[], semi;
                    v_elas_buf=v_stress_buf)
                TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), stress_elas)
            end
        else
            TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
        end

        # Steps 5 & 6 (floor/mold constraint enforcement) removed:
        # floor is all-clamped (stationary), mold is all-clamped with PrescribedMotion.
        # Both are automatically handled by apply_prescribed_motion! in update_positions!.
    end
)

dt_cap_cb = DiscreteCallback(
    (u, t, integrator) -> true,
    integrator -> set_proposed_dt!(integrator, min(integrator.dt, dt_cap))
)

# Time-based progress reporting works uniformly for explicit and implicit solvers.
t_start, t_end = tspan
progress_step_percent = 5.0
progress_dt = (progress_step_percent / 100.0) * (t_end - t_start)
next_progress_t = Ref(t_start + progress_dt)
progress_step_counter = Ref(0)
progress_heartbeat_interval = 20

println("progress init | first percent mark at t=", round(next_progress_t[], digits=6),
    " (", progress_step_percent, "%)")

progress_cb = DiscreteCallback(
    (u, t, integrator) -> true,
    function (integrator)
        progress_dt <= 0.0 && return
        t = integrator.t
        progress_step_counter[] += 1

        # Heartbeat independent of simulated time progress; useful when dt is tiny.
        if mod(progress_step_counter[], progress_heartbeat_interval) == 0
            pct_hb = clamp(100.0 * (t - t_start) / (t_end - t_start), 0.0, 100.0)
            println("progress hb | step=", progress_step_counter[],
                    " | ", round(pct_hb, digits=3), "%",
                    " | t=", round(t, digits=6), " / ", round(t_end, digits=6),
                    " | dt=", integrator.dt)
        end

        if t + 1.0e-14 >= next_progress_t[]
            pct = clamp(100.0 * (t - t_start) / (t_end - t_start), 0.0, 100.0)
            println("progress | ", round(pct, digits=1), "% | t=",
                    round(t, digits=6), " / ", round(t_end, digits=6),
                    " | dt=", integrator.dt)

            while next_progress_t[] <= t && next_progress_t[] < t_end
                next_progress_t[] += progress_dt
            end
        end
    end
)

callbacks = CallbackSet(
    UpdateCallback(),
    thermomechanical_cb,
    progress_cb,
    #dt_cap_cb,
    SolutionSavingCallback(dt=0.005, prefix="molding_cfrp_3d_thermo"),
    # Contact with stiff penalty is sensitive to explicit step size.
    StepsizeCallback(cfl=0.5),
    InfoCallback(interval=100)
)

sol = solve(ode, RDPK3SpFSAL35();
            save_everystep=false,
            dtmax=dt_cap,
            maxiters=10_000_000,
            callback=callbacks)

# sol = solve(ode, TRBDF2(); 
#       callback=callbacks, 
#       save_everystep=false, 
#       maxiters=10_000_000)

# sol = solve(ode, Rosenbrock23(autodiff=false);
#     callback=callbacks,
#     save_everystep=false,
#     maxiters=10_000_000)
