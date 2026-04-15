# ==========================================================================================
# 3D Stamping — Isothermal Elastoplastic (no preheating, no viscous regime)
#
# Identical geometry, contact, and solver to stamping_cfrp_3d_2_implicit.jl
# but with:
#   - All temperatures fixed at 298 K (room temperature)
#   - No preheating phase
#   - No viscous stress branch
#   - No thermal diffusion RHS
#   - v_nvariables = ndims (temperature NOT an ODE state)
#   - Only elastic_stress3d_trial! / elastic_stress3d_fast! used
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
using Statistics
using LinearAlgebra
using SparseArrays
using Logging

println("--- ISOTHERMAL ELASTOPLASTIC SIMULATION STARTING ---")
println("Threads available: ", nthreads())
println("----------------------------------------------------")

# ==========================================================================================
# STEP 1: Pack cylinder geometry (same as thermal file)
# ==========================================================================================

file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "Cylinder.stl")
geometry = load_geometry(file)

particle_spacing = 0.0005
boundary_thickness = 3 * particle_spacing

signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

density_cylinder = 905.0  # kg/m^3, typical PP homopolymer density at room temperature

point_in_geometry_algorithm = WindingNumberJacobson(; geometry)

shape_sampled = ComplexShape(geometry; particle_spacing, density=density_cylinder,
                             point_in_geometry_algorithm, pad_initial_particle_grid=0.002)

shape_sampled.mass .= density_cylinder * TrixiParticles.volume(geometry) / nparticles(shape_sampled)

background_pressure = 1.0
factor = 1.2
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = factor * particle_spacing

# ==========================================================================================
# STEP 2: Pack
# ==========================================================================================

packing_system = ParticlePackingSystem(shape_sampled;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=signed_distance_field,
                                       background_pressure=background_pressure)

semi_pack = Semidiscretization(packing_system)
tspan_pack = (0.0, 10000.0)
ode_pack = semidiscretize(semi_pack, tspan_pack)

sol_pack = solve(ode_pack, RDPK3SpFSAL35();
                 abstol=1e-7, reltol=1e-4, save_everystep=false, maxiters=100,
                 callback=CallbackSet(UpdateCallback()))

polymer = InitialCondition(sol_pack, packing_system, semi_pack)

# Free packing-phase memory to prevent OOM during the main simulation
sol_pack = nothing
ode_pack = nothing
semi_pack = nothing
packing_system = nothing
signed_distance_field = nothing
shape_sampled = nothing
GC.gc()

# Rescale the packed STL cylinder to match the benchmark specimen from
# Jerabek–Major–Lang (2010): diameter = 8 mm, length = 12 mm.
# The longest axis is treated as the cylinder axis; the other two axes are scaled
# to the target diameter.
target_diameter = 8.0e-3
target_length = 12.0e-3
coords_poly = polymer.coordinates
mins_poly = vec(minimum(coords_poly, dims=2))
maxs_poly = vec(maximum(coords_poly, dims=2))
extents_poly = maxs_poly - mins_poly
axis_idx = argmax(extents_poly)
radial_idxs = filter(i -> i != axis_idx, 1:3)
center_poly = 0.5 .* (mins_poly + maxs_poly)

scale_factors = ones(eltype(coords_poly), 3)
scale_factors[axis_idx] = target_length / extents_poly[axis_idx]
for idx in radial_idxs
    scale_factors[idx] = target_diameter / extents_poly[idx]
end

@inbounds for d in 1:3
    coords_poly[d, :] .-= center_poly[d]
    coords_poly[d, :] .*= scale_factors[d]
    coords_poly[d, :] .+= center_poly[d]
end

mins_scaled = vec(minimum(coords_poly, dims=2))
maxs_scaled = vec(maximum(coords_poly, dims=2))
extents_scaled = maxs_scaled - mins_scaled
println("✓ Cylinder rescaled to paper geometry:")
println("  - diameter targets: 8 mm, 8 mm")
println("  - length target: 12 mm")
println("  - achieved extents [mm] = ", round.(extents_scaled .* 1e3, digits=3))

# ==========================================================================================
# STEP 3: Visualise
# ==========================================================================================

# Skip plotting during production runs to avoid extra startup/JIT cost in headless mode.
# coords = polymer.coordinates
# scatter3d(coords[1, :], coords[2, :], coords[3, :],
#           markersize=1, aspect_ratio=:equal, label="Cylinder particles")
# savefig("packing_isothermal.png")

# ==========================================================================================
# STEP 4: Build floor/mold
# ==========================================================================================

floor_density = density_cylinder

cyl_z_min = minimum(polymer.coordinates[3, :])
cyl_z_max = maximum(polymer.coordinates[3, :])

initial_gap   = 1.0 * particle_spacing
z_surface     = cyl_z_min - initial_gap
z_top_surface = cyl_z_max + initial_gap

num_layers      = 3
floor_thickness = num_layers * particle_spacing
z_bottom        = z_surface - floor_thickness
z_top           = z_top_surface

n_plate = ceil(Int, 0.025 / particle_spacing)

floor_particles = RectangularShape(particle_spacing, (n_plate, n_plate, num_layers),
                                   (-0.01, -0.01, z_bottom); density=floor_density)

mold_particles = RectangularShape(particle_spacing, (n_plate, n_plate, num_layers),
                                  (-0.01, -0.01, z_top); density=floor_density)

floor_z_max = maximum(floor_particles.coordinates[3, :])
println("Gap = ", cyl_z_min - floor_z_max)
println("h   = ", factor * particle_spacing)
println("Contact at t=0? ", (cyl_z_min - floor_z_max) < factor * particle_spacing)

# ==========================================================================================
# STEP 5: Material and system definitions
# ==========================================================================================

# Paper conditions: polypropylene homopolymer tested at 23°C under quasi-static
# compression. The excerpt gives the material class and test conditions, but not a full
# constitutive table, so the values below use representative room-temperature PP(H)
# properties consistent with the benchmark setup.
T_room = 296.15   # K = 23°C

material_polymer = (density=905.0, E=1.5e9, nu=0.42, beta=0.0,
                    temp=T_room, temp_ref=T_room, cp=1900.0, k=0.22,
                    temp_liq=433.15, h=70000.0, hardening=8.0e7,
                    # tmelt=433.15 would put T_room(296K) above 0.5*tmelt(216K), triggering
                    # the viscoplastic branch (Δγ = yf*dt/η).  With dt~1e-10 and η=1e5,
                    # Δγ≈1e-15 — negligible plastic flow; solver stuck at yield surface.
                    # Set tmelt=700 so 0.5*tmelt=350K > T_room → rate-independent branch
                    # (Δγ = yf/(3μ+H)); no dt-dependence, robust for any step size.
                    tmelt=700.0, viscosity=1.0e5, yield_stress=3.5e7)

n_cylinder_particles = nparticles(polymer)

# Plastic history — committed (frozen) arrays updated only on accepted steps
Fp_initial = zeros(3, 3, n_cylinder_particles)
for i in 1:n_cylinder_particles
    Fp_initial[:, :, i] .= Matrix{Float64}(I, 3, 3)
end
alpha_committed = Ref(zeros(n_cylinder_particles))
Fp_committed    = Ref(copy(Fp_initial))

# Pre-allocated buffers for update_properties! (avoids 3 × N allocations per RHS call)
ys_buf   = zeros(n_cylinder_particles)
hard_buf = zeros(n_cylinder_particles)
vis_buf  = zeros(n_cylinder_particles)

mold_velocity_state = -0.05
t_ramp_mold = 1.0e-2

mold_motion = PrescribedMotion(
    (x, t) -> begin
        z_shift = if t < t_ramp_mold
            0.5 * mold_velocity_state / t_ramp_mold * t^2
        else
            mold_velocity_state * (t - 0.5 * t_ramp_mold)
        end
        x + SVector(0.0, 0.0, z_shift)
    end,
    t -> true)

import PointNeighbors: DictionaryCellList

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=500)

# Boundary E is a contact-penalty parameter (all boundary particles are clamped).
# E=1e11 creates contact forces ~67× stiffer than the cylinder, causing dt collapse
# at ~13% compression in the explicit solver.
# E = 3×E_cylinder keeps boundary 3× stiffer (adequate repulsion) while keeping
# the CFL limit at ~1e-6 s (compatible with explicit stepping).
E_boundary = 3.0 * material_polymer.E   # 4.5e9 Pa

floor_system = TotalLagrangianSPHSystem(floor_particles,
                                        smoothing_kernel, smoothing_length,
                                        E_boundary,
                                        material_polymer.nu, material_polymer.beta,
                                        material_polymer.temp, material_polymer.temp_ref,
                                        material_polymer.cp, material_polymer.k,
                                        material_polymer.temp_liq, material_polymer.h,
                                        material_polymer.hardening, material_polymer.tmelt,
                                        material_polymer.yield_stress;
                                        clamped_particles=collect(1:nparticles(floor_particles)),
                                        acceleration=(0.0, 0.0, 0.0),
                                        self_interaction_nhs=nhs_template)

mold_system = TotalLagrangianSPHSystem(mold_particles,
                                       smoothing_kernel, smoothing_length,
                                       E_boundary,
                                       material_polymer.nu, material_polymer.beta,
                                       material_polymer.temp, material_polymer.temp_ref,
                                       material_polymer.cp, material_polymer.k,
                                       material_polymer.temp_liq, material_polymer.h,
                                       material_polymer.hardening, material_polymer.tmelt,
                                       material_polymer.yield_stress;
                                       clamped_particles=collect(1:nparticles(mold_particles)),
                                       clamped_particles_motion=mold_motion,
                                       acceleration=(0.0, 0.0, 0.0),
                                       self_interaction_nhs=nhs_template)

cylinder_system = TotalLagrangianSPHSystem(polymer,
                                           smoothing_kernel, smoothing_length,
                                           material_polymer.E,
                                           material_polymer.nu, material_polymer.beta,
                                           material_polymer.temp, material_polymer.temp_ref,
                                           material_polymer.cp, material_polymer.k,
                                           material_polymer.temp_liq, material_polymer.h,
                                           material_polymer.hardening, material_polymer.tmelt,
                                           material_polymer.yield_stress;
                                           acceleration=(0.0, 0.0, 0.0),
                                           self_interaction_nhs=nhs_template)

semi = Semidiscretization(cylinder_system, floor_system, mold_system;
                          neighborhood_search=GridNeighborhoodSearch{3}(;
                              cell_list=DictionaryCellList{3}(),
                              search_radius=smoothing_length))

println("✓ Isothermal elastoplastic: T=", T_room, " K for all systems")
println("  - yield_stress=", material_polymer.yield_stress / 1e6, " MPa")
println("  - hardening=", material_polymer.hardening / 1e9, " GPa")
println("  - E=", cylinder_system.young_modulus / 1e9, " GPa (cylinder)")

cylinder_height = cyl_z_max - cyl_z_min
t_compress = cylinder_height * 0.5 / 0.05
tspan = (0.0, t_compress)

ode_base = semidiscretize(semi, tspan)

# ==========================================================================================
# STEP 6: Implicit RHS infrastructure (elastoplastic only, no thermal)
# ==========================================================================================

v_stress_buf = zeros(3, 3, n_cylinder_particles)
vel_grad_buf = zeros(3, 3, n_cylinder_particles)   # unused here but kept for API compat

trial_dt_state    = Ref(1.0e-3)
nhs_updated_at_t  = Ref(-Inf)
rhs_eval_counter  = Ref(0)
first_rhs_call    = Ref(true)

solver_diag_interval    = 100
solver_diag_last_accept = Ref(0)
solver_diag_last_iter   = Ref(0)
solver_diag_last_rhs    = Ref(0)
solver_diag_dt_min      = Ref(Inf)
solver_diag_dt_max      = Ref(0.0)
solver_diag_dt_sum      = Ref(0.0)

function update_isothermal_stress_cache!(semi_local, v_ode, t)
    cyl_sys = semi_local.systems[1]

    ys, hard, vis = TrixiParticles.update_properties!(cyl_sys, alpha_committed[],
                                                      semi_local,
                                                      material_polymer.viscosity;
                                                      yield_stress_buf=ys_buf,
                                                      hardening_buf=hard_buf,
                                                      viscosity_buf=vis_buf)
    fill!(v_stress_buf, 0)
    stress_elas = TrixiParticles.elastic_stress3d_trial!(
        cyl_sys, ys, hard, vis, trial_dt_state[],
        alpha_committed[], Fp_committed[], semi_local;
        v_elas_buf=v_stress_buf)
    TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), stress_elas)
end

function kick_isothermal!(dv_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.set_zero!(dv_ode)

    # if first_rhs_call[]
    #     println(">>> First RHS call (t=", t, ") — JIT done, solver running")
    #     flush(stdout)
    #     first_rhs_call[] = false
    # end

    rhs_eval_counter[] += 1
    # if mod(rhs_eval_counter[], 100) == 0
    #     println(">>> RHS call #", rhs_eval_counter[], " (t=", round(t, digits=8), ") — GMRES iterating")
    #     flush(stdout)
    # end

    cyl_sys = semi_local.systems[1]

    TrixiParticles.foreach_system(semi_local) do system
        v = TrixiParticles.wrap_v(v_ode, system, semi_local)
        u = TrixiParticles.wrap_u(u_ode, system, semi_local)
        TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
    end

    if t != nhs_updated_at_t[]
        TrixiParticles.update_nhs!(semi_local, u_ode)
        nhs_updated_at_t[] = t
    end

    TrixiParticles.foreach_system(semi_local) do system
        v = TrixiParticles.wrap_v(v_ode, system, semi_local)
        u = TrixiParticles.wrap_u(u_ode, system, semi_local)
        TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi_local, t)
    end

    TrixiParticles.update_implicit_sph!(semi_local, v_ode, u_ode, t)

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

    update_isothermal_stress_cache!(semi_local, v_ode, t)
    TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
    TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.STRESS_TENSOR_CACHE[] = nothing

    return dv_ode
end

function drift_isothermal!(du_ode, v_ode, u_ode, semi_local, t)
    return TrixiParticles.drift!(du_ode, v_ode, u_ode, semi_local, t)
end

ode = DynamicalODEProblem(kick_isothermal!, drift_isothermal!,
                          Vector{Float64}(ode_base.u0.x[1]),
                          Vector{Float64}(ode_base.u0.x[2]),
                          tspan, semi)

# ==========================================================================================
# STEP 7: Callbacks
# ==========================================================================================

callbacks = CallbackSet(
    # Velocity clamping: cap each particle velocity at v_max to prevent explosion
    # from unresolved contact impulses. 2× mold speed is physically generous.
    let v_max = 2.0 * abs(mold_velocity_state)  # 0.1 m/s
        DiscreteCallback(
            (u, t, integrator) -> true,
            function(integrator)
                v_ode = integrator.u.x[1]
                cyl_sys = integrator.p.systems[1]
                v_wrap = TrixiParticles.wrap_v(v_ode, cyl_sys, integrator.p)
                clamped = 0
                @inbounds for p in axes(v_wrap, 2)
                    vx, vy, vz = v_wrap[1, p], v_wrap[2, p], v_wrap[3, p]
                    speed = sqrt(vx^2 + vy^2 + vz^2)
                    if speed > v_max
                        scale = v_max / speed
                        v_wrap[1, p] *= scale
                        v_wrap[2, p] *= scale
                        v_wrap[3, p] *= scale
                        clamped += 1
                    end
                end
                if clamped > 0
                    u_modified!(integrator, true)
                end
            end)
    end,
    DiscreteCallback(
        (u, t, integrator) -> true,
        function(integrator)
            solver_diag_dt_min[] = min(solver_diag_dt_min[], integrator.dt)
            solver_diag_dt_max[] = max(solver_diag_dt_max[], integrator.dt)
            solver_diag_dt_sum[] += integrator.dt

                if integrator.stats.naccept > 0 && mod(integrator.stats.naccept, 1000) == 0
                accepted_delta      = integrator.stats.naccept - solver_diag_last_accept[]
                iter_delta          = integrator.iter - solver_diag_last_iter[]
                rhs_delta           = rhs_eval_counter[] - solver_diag_last_rhs[]
                rejected_delta      = max(iter_delta - accepted_delta, 0)
                avg_dt              = solver_diag_dt_sum[] / max(accepted_delta, 1)
                avg_rhs_per_accept  = rhs_delta / max(accepted_delta, 1)

                println("solver diag | accepted=", integrator.stats.naccept,
                    " | t=", round(integrator.t, digits=6),
                    " | dt_now=", integrator.dt,
                    " | dt_avg=", avg_dt,
                    " | dt_min=", solver_diag_dt_min[],
                    " | dt_max=", solver_diag_dt_max[],
                    " | rejected_since_last=", rejected_delta,
                    " | rhs_per_accept=", round(avg_rhs_per_accept, digits=2),
                    " | mem_MB=", round((Sys.total_memory()-Sys.free_memory())/1e6, digits=0))
                flush(stdout)

                solver_diag_last_accept[] = integrator.stats.naccept
                solver_diag_last_iter[]   = integrator.iter
                solver_diag_last_rhs[]    = rhs_eval_counter[]
                solver_diag_dt_min[]      = Inf
                solver_diag_dt_max[]      = 0.0
                solver_diag_dt_sum[]      = 0.0
                end

            # Periodic incremental GC to prevent heap fragmentation from residual
            # allocations (e.g. SVD fallback, Polyester thread metadata) accumulating
            # across millions of micro-steps when dt collapses near contact.
            if mod(integrator.stats.naccept, 1000) == 0
                GC.gc(false)  # fast incremental collection
            end

            # Commit plastic history for all particles (all are cool at 298 K)
            cyl_sys = integrator.p.systems[1]
            ys_c, hard_c, vis_c = TrixiParticles.update_properties!(
                cyl_sys, alpha_committed[], integrator.p, material_polymer.viscosity;
                yield_stress_buf=ys_buf, hardening_buf=hard_buf, viscosity_buf=vis_buf)
            trial_dt_state[] = integrator.dt
            TrixiParticles.elastic_stress3d_fast!(
                cyl_sys, ys_c, hard_c, vis_c, integrator.dt,
                alpha_committed[], Fp_committed[], integrator.p;
                v_elas_buf=v_stress_buf)

            # Sync the J2 elastoplastic stress into pk1_rho2 so that VTU
            # output (cauchy_stress / von_mises_stress) reflects the actual
            # stress used in the momentum equation, not the default SVK model.
            # v_stress_buf = (tau * F^{-T}) * L_corr  =  P_corrected
            # pk1_rho2     = P_corrected / rho^2
            @inbounds for p in axes(v_stress_buf, 3)
                rho2_inv = 1.0 / cyl_sys.material_density[p]^2
                for j in 1:3, i in 1:3
                    cyl_sys.pk1_rho2[i, j, p] = v_stress_buf[i, j, p] * rho2_inv
                end
            end

            # det(Fp) diagnostic: print every 5000 accepted steps.
            # if mod(integrator.stats.naccept, 5000) == 0
            #     Fp = Fp_committed[]
            #     n_p = size(Fp, 3)
            #     det_min = Inf; det_max = -Inf; det_nan = 0
            #     @inbounds for i in 1:n_p
            #         d = det(@view Fp[:, :, i])
            #         if isnan(d) || isinf(d)
            #             det_nan += 1
            #         else
            #             det_min = min(det_min, d)
            #             det_max = max(det_max, d)
            #         end
            #     end
            #     println("Fp diag | step=", integrator.stats.naccept,
            #             " | det_min=", round(det_min, sigdigits=4),
            #             " | det_max=", round(det_max, sigdigits=4),
            #             " | nan/inf=", det_nan)
            #     flush(stdout)
            #     flush(stdout)
            # end
            # Diagnostic print to confirm pk1_rho2 sync is running (every 5000 steps)
            # if integrator.stats.naccept % 5000 == 0
            #     println("[pk1_rho2 sync] step=", integrator.stats.naccept,
            #             " | s11=", round(cyl_sys.pk1_rho2[1,1,1]*cyl_sys.material_density[1]^2/1e6, digits=2),
            #             " | s33=", round(cyl_sys.pk1_rho2[3,3,1]*cyl_sys.material_density[1]^2/1e6, digits=2))
            # end
        end),
    SolutionSavingCallback(dt=2.0e-3, prefix="stamping_isothermal_elastoplastic_0.0005"),
    InfoCallback(interval=10)
)

# ==========================================================================================
# STEP 8: Solve
# ==========================================================================================

# Keep TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM set permanently for the cylinder.
# This prevents the VTU writer's update_systems_and_nhs → update_quantities! from
# overwriting pk1_rho2 with the default SVK stress. The committed-step callback
# syncs the correct J2 elastoplastic stress into pk1_rho2 after each accepted step.
TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(cylinder_system)

sol = solve(ode, RDPK3SpFSAL35();
            callback=callbacks,
            save_everystep=false,
            abstol=1.0e-6,
            reltol=1.0e-4,
            dtmax=1.0e-4,
            dtmin=1.0e-9,
            force_dtmin=true,
            maxiters=10_000_000)
