# ==========================================================================================
# 3D Couette Flow — Pure Viscous Validation
#
# Validates viscous_stress3d_fast! in isolation using a Couette flow benchmark.
#
# Setup:
#   - Rectangular block of particles between two plates (z-direction)
#   - Bottom plate: clamped (v_x = 0)
#   - Top plate:    moves at constant v_x = V_wall
#   - Analytical steady-state: v_x(z) = V_wall * (z - z_bot) / H
#   - Shear stress: σ_xz = η * V_wall / H
#
# No elasticity, no plasticity — only viscous stress drives the flow.
# Explicit time integration (RDPK3SpFSAL35) — no GMRES needed.
# ==========================================================================================
using TrixiParticles
using OrdinaryDiffEq
using PointNeighbors
using Plots
using Base.Threads
using Statistics
using LinearAlgebra

println("--- COUETTE VISCOUS VALIDATION ---")
println("Threads available: ", nthreads())
println("----------------------------------")

# ==========================================================================================
# STEP 1: Parameters
# ==========================================================================================
initialize_from_rest = true
transient_series_terms = 400
include_boundary_effects_in_plots = true

particle_spacing = 0.0005         # 0.5 mm — gives 40 particles across H
H = 0.02                          # channel height = 20 mm
L = 0.005                         # domain size in x,y (small — just needs lateral neighbors)
V_wall = 0.01                     # top wall velocity in x-direction [m/s]
viscosity_val = 1.0e4             # Pa·s (same as PP viscosity)
density_val = 905.0               # kg/m³
T_room = 293.15                   # K

# Wall/boundary-support tuning knobs.
# `num_layers_wall` controls how many fixed wall particle layers are present.
# `h_over_dx` controls the smoothing length via h = (h/dx) * dx, which changes
# how strongly wall particles are seen by nearby fluid particles.
#
# Previous settings kept here for quick rollback:
# num_layers_wall = 3
# h_over_dx = 1.2
num_layers_wall = 5
h_over_dx = 1.5

smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = h_over_dx * particle_spacing

# Analytical Couette solution
shear_rate_analytical = V_wall / H
stress_xz_analytical = viscosity_val * shear_rate_analytical
println("Analytical shear rate: ", shear_rate_analytical, " s⁻¹")
println("Analytical σ_xz:      ", stress_xz_analytical, " Pa")
println("Viscous diffusion time τ = ρH²/η = ", density_val * H^2 / viscosity_val, " s")
println("Mode: ", initialize_from_rest ? "transient start from rest" : "steady-profile preservation")
println("Boundary effects in plots: ", include_boundary_effects_in_plots ? "on" : "off")
println("Wall layers: ", num_layers_wall)
println("Boundary support h/dx: ", h_over_dx, "  (h = ", smoothing_length, " m)")

# ==========================================================================================
# STEP 2: Build geometry — fluid block + two wall plates
# ==========================================================================================
nx = ceil(Int, L / particle_spacing)
ny = nx
nz_fluid = ceil(Int, H / particle_spacing)

# Fluid block: occupies z in [0, H]
fluid_particles = RectangularShape(particle_spacing,
    (nx, ny, nz_fluid),
    (0.0, 0.0, 0.0);
    density=density_val)

n_fluid = nparticles(fluid_particles)
println("Fluid particles: ", n_fluid)

# Bottom wall: below z=0
bottom_particles = RectangularShape(particle_spacing,
    (nx, ny, num_layers_wall),
    (0.0, 0.0, -num_layers_wall * particle_spacing);
    density=density_val)

n_bottom = nparticles(bottom_particles)

# Top wall: above z=H
top_particles = RectangularShape(particle_spacing,
    (nx, ny, num_layers_wall),
    (0.0, 0.0, H);
    density=density_val)

n_top = nparticles(top_particles)

# Merge all particles into a single InitialCondition so the velocity gradient
# SPH loop sees wall neighbors.  Order: bottom | fluid | top
all_coords = hcat(bottom_particles.coordinates, fluid_particles.coordinates, top_particles.coordinates)
all_vel    = hcat(bottom_particles.velocity,    fluid_particles.velocity,    top_particles.velocity)
all_mass   = vcat(bottom_particles.mass,        fluid_particles.mass,        top_particles.mass)
all_dens   = vcat(bottom_particles.density,     fluid_particles.density,     top_particles.density)

n_all = n_bottom + n_fluid + n_top

# Particle index ranges
idx_bottom = 1:n_bottom
idx_fluid  = (n_bottom + 1):(n_bottom + n_fluid)
idx_top    = (n_bottom + n_fluid + 1):n_all

all_ic = InitialCondition(; coordinates=all_coords, velocity=all_vel,
                            mass=all_mass, density=all_dens)

println("Total particles: ", n_all, " (bottom=", n_bottom, " fluid=", n_fluid, " top=", n_top, ")")

# ==========================================================================================
# STEP 3: Material + single system (viscous only)
# ==========================================================================================
# Very soft E so elastic PK1 stress is negligible compared to viscous stress.
E_soft = 1.0e2    # 100 Pa

material = (
    nu=0.3, beta=0.0,
    temp=T_room, temp_ref=T_room,
    cp=1900.0, k=0.22,
    temp_liq=433.15, h=70000.0,
    hardening=0.0, tmelt=433.15,
    yield_stress=1.0e12,         # effectively infinite — no plasticity
    viscosity=viscosity_val
)

import PointNeighbors: DictionaryCellList
nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=500)

# Single system — all particles integrated (no clamped).
# Boundary velocity enforcement is done manually in the kick function.
the_system = TotalLagrangianSPHSystem(all_ic,
    smoothing_kernel, smoothing_length,
    E_soft,
    material.nu, material.beta,
    material.temp, material.temp_ref,
    material.cp, material.k,
    material.temp_liq, material.h,
    material.hardening, material.tmelt,
    material.yield_stress;
    acceleration=(0.0, 0.0, 0.0),
    self_interaction_nhs=nhs_template)

semi = Semidiscretization(the_system;
    neighborhood_search=GridNeighborhoodSearch{3}(;
        cell_list=DictionaryCellList{3}(),
        search_radius=smoothing_length))

# ==========================================================================================
# STEP 4: Viscous-only stress cache + kick function
# ==========================================================================================
v_stress_buf = zeros(3, 3, n_all)
vel_grad_buf = zeros(3, 3, n_all)
nhs_updated_at_t = Ref(-Inf)
rhs_counter = Ref(0)

function update_viscous_stress_cache!(semi_local, v_ode, t)
    sys = semi_local.systems[1]
    v_wrap = TrixiParticles.wrap_v(v_ode, sys, semi_local)
    _, _, vis = TrixiParticles.update_properties!(sys, zeros(n_all),
                                                  semi_local, material.viscosity)
    fill!(v_stress_buf, 0)
    fill!(vel_grad_buf, 0)
    stress_visc = TrixiParticles.viscous_stress3d_fast!(sys,
        v_wrap,  # actual velocity — includes wall particles with enforced BCs
        vis, semi_local;
        v_vis_buf=v_stress_buf, vel_grad_buf=vel_grad_buf)
    TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(sys), stress_visc)
end

function kick_viscous!(dv_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.set_zero!(dv_ode)

    rhs_counter[] += 1
    if rhs_counter[] == 1
        println(">>> First RHS call (t=", t, ") — JIT done")
        flush(stdout)
    end
    if mod(rhs_counter[], 5000) == 0
        println(">>> RHS #", rhs_counter[], " t=", round(t, digits=6))
        flush(stdout)
    end

    sys = semi_local.systems[1]
    v_wrap = TrixiParticles.wrap_v(v_ode, sys, semi_local)

    # Enforce boundary velocities BEFORE physics evaluation so the velocity
    # gradient sees the correct wall velocities.
    @inbounds for i in idx_bottom
        v_wrap[1, i] = 0.0   # v_x = 0 (stationary bottom)
        v_wrap[2, i] = 0.0
        v_wrap[3, i] = 0.0
    end
    @inbounds for i in idx_top
        v_wrap[1, i] = V_wall  # v_x = V_wall (moving top)
        v_wrap[2, i] = 0.0
        v_wrap[3, i] = 0.0
    end

    try
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        if t != nhs_updated_at_t[]
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end

        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(sys)

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

        update_viscous_stress_cache!(semi_local, v_ode, t)
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi_local, t)
    finally
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = nothing
        TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
    end

    # Zero out dv for boundary particles — keep their velocity fixed
    dv_wrap = TrixiParticles.wrap_v(dv_ode, sys, semi_local)
    @inbounds for i in idx_bottom
        dv_wrap[1, i] = 0.0; dv_wrap[2, i] = 0.0; dv_wrap[3, i] = 0.0
    end
    @inbounds for i in idx_top
        dv_wrap[1, i] = 0.0; dv_wrap[2, i] = 0.0; dv_wrap[3, i] = 0.0
    end

    return dv_ode
end

function drift_viscous!(du_ode, v_ode, u_ode, semi_local, t)
    return TrixiParticles.drift!(du_ode, v_ode, u_ode, semi_local, t)
end

function couette_transient_velocity(z, t, H, V_wall, tau_visc; n_terms=transient_series_terms)
    if z <= 0.0
        return 0.0
    elseif z >= H
        return V_wall
    elseif t <= 0.0
        return 0.0
    end

    series_sum = 0.0
    @inbounds for n in 1:n_terms
        decay = exp(-(n^2) * pi^2 * t / tau_visc)
        series_sum += ((-1)^n / n) * sinpi(n * z / H) * decay
    end

    return V_wall * z / H + 2.0 * V_wall / pi * series_sum
end

function analytical_velocity(z, t, H, V_wall, tau_visc; initialize_from_rest=initialize_from_rest)
    if initialize_from_rest
        return couette_transient_velocity(z, t, H, V_wall, tau_visc)
    end

    if z <= 0.0
        return 0.0
    elseif z >= H
        return V_wall
    end

    return V_wall * z / H
end

function transient_profile_metrics(sol, sys, semi, idx_fluid, H, L,
                                   particle_spacing, smoothing_length,
                                   n_bins, V_wall, tau_visc_eval;
                                   include_boundary_effects=false)
    time_hist = Float64[]
    rel_l2_hist = Float64[]
    linf_hist = Float64[]
    total_sq_err = 0.0
    total_count = 0

    for (t_snapshot, state) in zip(sol.t, sol.u)
        t_snapshot <= 0.0 && continue

        v_snapshot = TrixiParticles.wrap_v(state.x[1], sys, semi)
        u_snapshot = TrixiParticles.wrap_u(state.x[2], sys, semi)
        z_snapshot, vx_snapshot = collect_comparison_profile(v_snapshot, u_snapshot, sys,
                                                             idx_fluid, H, L,
                                                             particle_spacing,
                                                             smoothing_length;
                                                             include_boundary_effects=include_boundary_effects)
        z_snapshot_min, z_snapshot_max = comparison_z_limits(z_snapshot, H)
        z_bins_snapshot, vx_bins_snapshot = bin_average_profile(z_snapshot, vx_snapshot, H, n_bins;
                                    z_lo=z_snapshot_min,
                                    z_hi=z_snapshot_max)
        isempty(vx_bins_snapshot) && continue

        vx_exact_snapshot = [couette_transient_velocity(z, t_snapshot, H, V_wall, tau_visc_eval)
                             for z in z_bins_snapshot]
        diff_snapshot = vx_bins_snapshot .- vx_exact_snapshot
        l2_snapshot = sqrt(mean(diff_snapshot .^ 2))
        linf_snapshot = maximum(abs.(diff_snapshot))

        push!(time_hist, t_snapshot / tau_visc)
        push!(rel_l2_hist, l2_snapshot / V_wall * 100)
        push!(linf_hist, linf_snapshot / V_wall * 100)

        total_sq_err += sum(diff_snapshot .^ 2)
        total_count += length(diff_snapshot)
    end

    rmse = total_count > 0 ? sqrt(total_sq_err / total_count) : Inf
    return time_hist, rel_l2_hist, linf_hist, rmse
end

function fit_effective_viscosity(sol, sys, semi, idx_fluid, H, L,
                                 particle_spacing, smoothing_length,
                                 n_bins, V_wall, density_val, viscosity_val;
                                 include_boundary_effects=false)
    scale_candidates = exp10.(range(log10(0.05), log10(5.0), length=121))
    best_viscosity = viscosity_val
    best_tau = density_val * H^2 / viscosity_val
    best_rmse = Inf

    for scale in scale_candidates
        viscosity_candidate = viscosity_val * scale
        tau_candidate = density_val * H^2 / viscosity_candidate
        _, _, _, rmse = transient_profile_metrics(sol, sys, semi, idx_fluid, H, L,
                                                  particle_spacing, smoothing_length,
                                                  n_bins, V_wall, tau_candidate;
                                                  include_boundary_effects=include_boundary_effects)
        if rmse < best_rmse
            best_rmse = rmse
            best_viscosity = viscosity_candidate
            best_tau = tau_candidate
        end
    end

    return best_viscosity, best_tau, best_rmse
end

function comparison_z_limits(z_pos_cmp, H)
    if isempty(z_pos_cmp)
        return 0.0, H
    end

    return minimum(z_pos_cmp), maximum(z_pos_cmp)
end

function collect_region_profile(v_state, u_state, idx_range, L, particle_spacing)
    z_pos = Float64[]
    vx_vals = Float64[]

    x_mid = 0.5 * L
    y_mid = 0.5 * L
    xy_tol = 1.5 * particle_spacing

    for i in idx_range
        x_i = u_state[1, i]
        y_i = u_state[2, i]
        if (abs(x_i - x_mid) <= xy_tol) && (abs(y_i - y_mid) <= xy_tol)
            push!(z_pos, u_state[3, i])
            push!(vx_vals, v_state[1, i])
        end
    end

    return z_pos, vx_vals
end

function collect_comparison_profile(v_state, u_state, sys, idx_fluid, H, L,
                                    particle_spacing, smoothing_length;
                                    include_boundary_effects=false)
    z_pos_cmp = Float64[]
    vx_sph_cmp = Float64[]

    z_interior_lo_cmp = 3.0 * smoothing_length
    z_interior_hi_cmp = H - 3.0 * smoothing_length
    x_mid = 0.5 * L
    y_mid = 0.5 * L
    xy_tol = 1.5 * particle_spacing
    idx_compare = include_boundary_effects ? axes(u_state, 2) : idx_fluid

    for i in idx_compare
        x_i = u_state[1, i]
        y_i = u_state[2, i]
        z_i = u_state[3, i]
        vx_i = v_state[1, i]

        z_ok = include_boundary_effects ? true : (z_interior_lo_cmp <= z_i <= z_interior_hi_cmp)

        if z_ok &&
           (abs(x_i - x_mid) <= xy_tol) &&
           (abs(y_i - y_mid) <= xy_tol)
            push!(z_pos_cmp, z_i)
            push!(vx_sph_cmp, vx_i)
        end
    end

    if isempty(z_pos_cmp)
        fallback_label = include_boundary_effects ? "all-particle comparison" : "interior-only comparison"
        println("WARNING: centerline subset empty; falling back to ", fallback_label)
        for i in idx_compare
            z_i = u_state[3, i]
            z_ok = include_boundary_effects ? true : (z_interior_lo_cmp <= z_i <= z_interior_hi_cmp)
            if z_ok
                push!(z_pos_cmp, z_i)
                push!(vx_sph_cmp, v_state[1, i])
            end
        end
    end

    return z_pos_cmp, vx_sph_cmp
end

function bin_average_profile(z_pos_cmp, vx_sph_cmp, H, n_bins;
                             z_lo=nothing, z_hi=nothing)
    z_min = isnothing(z_lo) ? 0.0 : z_lo
    z_max = isnothing(z_hi) ? H : z_hi
    z_bin_edges = range(z_min, z_max, length=n_bins + 1)
    z_bin_centers = Float64[]
    vx_bin_avg = Float64[]

    for k in 1:n_bins
        z_left, z_right = z_bin_edges[k], z_bin_edges[k + 1]
        mask = findall(z -> (k == n_bins ? (z_left <= z <= z_right) : (z_left <= z < z_right)),
                       z_pos_cmp)
        if !isempty(mask)
            push!(z_bin_centers, mean(z_pos_cmp[mask]))
            push!(vx_bin_avg, mean(vx_sph_cmp[mask]))
        end
    end

    return z_bin_centers, vx_bin_avg
end

# ==========================================================================================
# STEP 5: ODE setup + solve (explicit)
# ==========================================================================================
# Viscous diffusion time: τ = ρH²/η
tau_visc = density_val * H^2 / viscosity_val
# In transient mode, run long enough to observe diffusion from rest while staying
# short enough that TLSPH remains close to the small-strain viscous benchmark.
# In steady mode, keep the original very short preservation test.
t_end_factor = initialize_from_rest ? 0.10 : 0.01
t_end = t_end_factor * tau_visc
tspan = (0.0, t_end)
println("τ_visc = ", tau_visc, " s")
println("t_end  = ", t_end, " s  (", t_end_factor, " × τ_visc)")

ode_base = semidiscretize(semi, tspan)

# Initialize velocities using the PHYSICAL channel boundaries (z=0 to z=H),
# NOT the full particle extent including wall layers.
# - Bottom wall particles (z < 0): v_x = 0
# - Fluid particles (0 ≤ z ≤ H):  either 0 (transient) or V_wall * z / H (steady)
# - Top wall particles (z > H):    v_x = V_wall
v0 = ode_base.u0.x[1]
sys = semi.systems[1]
v0_wrap = TrixiParticles.wrap_v(v0, sys, semi)
@inbounds for i in 1:n_all
    z_i = sys.initial_coordinates[3, i]
    if z_i <= 0.0
        v0_wrap[1, i] = 0.0          # bottom wall
    elseif z_i >= H
        v0_wrap[1, i] = V_wall       # top wall
    else
        v0_wrap[1, i] = initialize_from_rest ? 0.0 : V_wall * z_i / H
    end
    v0_wrap[2, i] = 0.0
    v0_wrap[3, i] = 0.0
end
if initialize_from_rest
    println("✓ Velocities initialized: fluid at rest, bottom fixed, top wall moving at V_wall")
else
    println("✓ Velocities initialized: v_x = V_wall*z/H in fluid, 0 at bottom, V_wall at top")
end

# ==========================================================================================
# STEP 5a: Instantaneous diagnostics at t=0  (velocity gradient + stress)
# ==========================================================================================
if initialize_from_rest
    println("\n=== TRANSIENT VALIDATION SETUP ===")
    println("Fluid starts from rest; analytical comparison uses the transient Couette solution.")
    println("Saved snapshots will be compared against the exact unsteady profile u(z, t).")
    println("=== END SETUP ===\n")
else
    println("\n=== INSTANTANEOUS DIAGNOSTICS (t=0) ===")

    # Evaluate one RHS call to populate vel_grad_buf and v_stress_buf
    dv_diag = zero(ode_base.u0.x[1])
    kick_viscous!(dv_diag, ode_base.u0.x[1], ode_base.u0.x[2], semi, 0.0)

    # Use only INTERIOR fluid particles (skip 3 layers from each wall) to avoid
    # boundary artifacts in the comparison.
    z_interior_lo = 3.0 * smoothing_length
    z_interior_hi = H - 3.0 * smoothing_length
    idx_interior = [i for i in idx_fluid if z_interior_lo <= sys.initial_coordinates[3, i] <= z_interior_hi]
    n_interior = length(idx_interior)
    println("Interior fluid particles: ", n_interior, " / ", n_fluid, " (excluding 3h from each wall)")

    # Analytical values use physical channel H
    dvx_dz_analytical = V_wall / H
    sigma_xz_analytical = viscosity_val * dvx_dz_analytical

    # Report velocity gradient ∂vx/∂z for INTERIOR fluid particles
    dvx_dz_values = [vel_grad_buf[1, 3, i] for i in idx_interior]
    dvx_dz_mean = mean(dvx_dz_values)
    println("Velocity gradient ∂vx/∂z (interior only):")
    println("  Analytical : ", round(dvx_dz_analytical, sigdigits=4), " s⁻¹")
    println("  SPH mean   : ", round(dvx_dz_mean, sigdigits=4), " s⁻¹")
    println("  SPH std    : ", round(std(dvx_dz_values), sigdigits=4), " s⁻¹")
    println("  Rel. error : ", round(abs(dvx_dz_mean - dvx_dz_analytical) / dvx_dz_analytical * 100, sigdigits=3), "%")

    # Report σ_xz for INTERIOR particles
    # v_stress_buf stores (τ · F^{-T}) · L_corr. At t=0, F=I, so this is τ · L_corr.
    # For interior particles L_corr ≈ I, so this ≈ τ_xz = η * ∂vx/∂z.
    sigma_xz_values = [v_stress_buf[1, 3, i] for i in idx_interior]
    sigma_xz_mean = mean(sigma_xz_values)
    println("\nShear stress σ_xz (interior, from stress cache):")
    println("  Analytical : ", round(sigma_xz_analytical, sigdigits=4), " Pa")
    println("  SPH mean   : ", round(sigma_xz_mean, sigdigits=4), " Pa")
    println("  SPH std    : ", round(std(sigma_xz_values), sigdigits=4), " Pa")
    println("  Rel. error : ", round(abs(sigma_xz_mean - sigma_xz_analytical) / abs(sigma_xz_analytical) * 100, sigdigits=3), "%")

    # Report acceleration of INTERIOR fluid particles (should ≈ 0 for uniform stress)
    dv_wrap = TrixiParticles.wrap_v(dv_diag, sys, semi)
    ax_values = [dv_wrap[1, i] for i in idx_interior]
    println("\nAcceleration dvx/dt (interior fluid particles):")
    println("  Mean : ", round(mean(ax_values), sigdigits=4), " m/s²")
    println("  Std  : ", round(std(ax_values), sigdigits=4), " m/s²")
    println("  Max  : ", round(maximum(abs.(ax_values)), sigdigits=4), " m/s²")
    println("  (should be ≈ 0 for steady Couette — ∇·σ = 0 for uniform stress)")

    # Also report ALL fluid particles for comparison
    dvx_dz_all = [vel_grad_buf[1, 3, i] for i in idx_fluid]
    sigma_xz_all = [v_stress_buf[1, 3, i] for i in idx_fluid]
    ax_all = [dv_wrap[1, i] for i in idx_fluid]
    println("\n--- For reference (ALL fluid particles including near-wall): ---")
    println("  ∂vx/∂z: mean=", round(mean(dvx_dz_all), sigdigits=4), " std=", round(std(dvx_dz_all), sigdigits=4))
    println("  σ_xz:   mean=", round(mean(sigma_xz_all), sigdigits=4), " std=", round(std(sigma_xz_all), sigdigits=4))
    println("  dvx/dt: mean=", round(mean(ax_all), sigdigits=4), " std=", round(std(ax_all), sigdigits=4), " max=", round(maximum(abs.(ax_all)), sigdigits=4))
    println("=== END DIAGNOSTICS ===\n")
end

# ==========================================================================================
# STEP 5b: Short time integration to verify profile maintenance
# ==========================================================================================
ode = DynamicalODEProblem(kick_viscous!, drift_viscous!,
                          ode_base.u0.x[1], ode_base.u0.x[2],
                          tspan, semi)

save_times = collect(range(0.0, t_end, length=initialize_from_rest ? 15 : 6))
output_prefix = initialize_from_rest ? "couette_viscous_transient" : "couette_viscous"

callbacks = CallbackSet(
    SolutionSavingCallback(dt=t_end / 5.0, prefix=output_prefix),
    InfoCallback(interval=50)
)

println("=== SOLVING (explicit, short run) ===")
sol = solve(ode, RDPK3SpFSAL35();
    callback=callbacks,
    save_everystep=false,
    saveat=save_times,
    abstol=1.0e-6,
    reltol=1.0e-4,
    maxiters=50_000_000)
println("=== SOLVE COMPLETE ===")

# ==========================================================================================
# STEP 6: Post-processing — extract v_x(z) profile and compare to analytical
# ==========================================================================================
println("\n=== POST-PROCESSING ===")
sys = semi.systems[1]
v_final = sol.u[end].x[1]
u_final = sol.u[end].x[2]

v_wrap = TrixiParticles.wrap_v(v_final, sys, semi)
u_wrap = TrixiParticles.wrap_u(u_final, sys, semi)

z_pos_cmp, vx_sph_cmp = collect_comparison_profile(v_wrap, u_wrap, sys, idx_fluid, H, L,
                           particle_spacing, smoothing_length;
                           include_boundary_effects=include_boundary_effects_in_plots)

comparison_total = include_boundary_effects_in_plots ? n_all : n_fluid
comparison_label = include_boundary_effects_in_plots ?
    " (boundary effects on: full stack + centerline strip)" :
    " (boundary effects off: interior + centerline strip)"

println("Comparison particles used: ", length(z_pos_cmp), " / ", comparison_total,
        comparison_label)

# Bin-average by z for a clean profile.
n_bins = include_boundary_effects_in_plots ? (nz_fluid + 2 * num_layers_wall) : nz_fluid
z_plot_min, z_plot_max = comparison_z_limits(z_pos_cmp, H)
z_bin_centers, vx_bin_avg = bin_average_profile(z_pos_cmp, vx_sph_cmp, H, n_bins;
                                                z_lo=z_plot_min, z_hi=z_plot_max)

z_bottom_plot = Float64[]
vx_bottom_plot = Float64[]
z_fluid_plot = Float64[]
vx_fluid_plot = Float64[]
z_top_plot = Float64[]
vx_top_plot = Float64[]

if include_boundary_effects_in_plots && !initialize_from_rest
    z_bottom_plot, vx_bottom_plot = collect_region_profile(v_wrap, u_wrap, idx_bottom, L,
                                                           particle_spacing)
    z_fluid_plot, vx_fluid_plot = collect_region_profile(v_wrap, u_wrap, idx_fluid, L,
                                                         particle_spacing)
    z_top_plot, vx_top_plot = collect_region_profile(v_wrap, u_wrap, idx_top, L,
                                                     particle_spacing)
end

# Analytical profile
z_analytical = range(z_plot_min, z_plot_max, length=100)
vx_analytical = [analytical_velocity(z, sol.t[end], H, V_wall, tau_visc) for z in z_analytical]

eta_eff = viscosity_val
tau_eff = tau_visc
fit_rmse = NaN
vx_analytical_eff = copy(vx_analytical)
time_hist_nominal = Float64[]
rel_l2_hist_nominal = Float64[]
linf_hist_nominal = Float64[]
time_hist_eff = Float64[]
rel_l2_hist_eff = Float64[]
linf_hist_eff = Float64[]

if initialize_from_rest
    eta_eff, tau_eff, fit_rmse = fit_effective_viscosity(sol, sys, semi, idx_fluid, H, L,
                                                         particle_spacing, smoothing_length,
                                                         n_bins, V_wall, density_val,
                                                         viscosity_val;
                                                         include_boundary_effects=include_boundary_effects_in_plots)
    vx_analytical_eff = [couette_transient_velocity(z, sol.t[end], H, V_wall, tau_eff)
                         for z in z_analytical]
    time_hist_nominal, rel_l2_hist_nominal, linf_hist_nominal, _ =
        transient_profile_metrics(sol, sys, semi, idx_fluid, H, L,
                                  particle_spacing, smoothing_length,
                                  n_bins, V_wall, tau_visc;
                                  include_boundary_effects=include_boundary_effects_in_plots)
    time_hist_eff, rel_l2_hist_eff, linf_hist_eff, _ =
        transient_profile_metrics(sol, sys, semi, idx_fluid, H, L,
                                  particle_spacing, smoothing_length,
                                  n_bins, V_wall, tau_eff;
                                  include_boundary_effects=include_boundary_effects_in_plots)
end

# Plot
p = plot(z_analytical .* 1e3, vx_analytical .* 1e3,
    label=initialize_from_rest ? "Analytical (nominal η)" : "Analytical",
    linewidth=2, linestyle=:dash,
    xlabel="z [mm]", ylabel="vx [mm/s]",
    title=initialize_from_rest ? "Couette Flow — Transient Viscous Validation" :
                                 (include_boundary_effects_in_plots ?
                                  "Couette Flow — Viscous (boundary effects on)" :
                                  "Couette Flow — Viscous (boundary effects off)"))
if initialize_from_rest
    plot!(p, z_analytical .* 1e3, vx_analytical_eff .* 1e3,
        label="Analytical (fitted η_eff)", linewidth=2)
end
if include_boundary_effects_in_plots && !initialize_from_rest
    scatter!(z_bottom_plot .* 1e3, vx_bottom_plot .* 1e3,
        label="SPH bottom wall", markersize=5, markerstrokewidth=0.5)
    scatter!(z_fluid_plot .* 1e3, vx_fluid_plot .* 1e3,
        label="SPH fluid", markersize=5, markerstrokewidth=0.5)
    scatter!(z_top_plot .* 1e3, vx_top_plot .* 1e3,
        label="SPH top wall", markersize=5, markerstrokewidth=0.5)
else
    scatter!(z_bin_centers .* 1e3, vx_bin_avg .* 1e3,
        label=include_boundary_effects_in_plots ? "SPH (binned, boundary effects on)" :
                                                 "SPH (binned, boundary effects off)",
        markersize=5)
end
profile_plot_file = initialize_from_rest ? "couette_viscous_transient_profile.png" :
                                           "couette_viscous_validation.png"
savefig(p, profile_plot_file)
println("✓ Plot saved: ", profile_plot_file)

# Quantitative error
vx_analytical_at_bins = [analytical_velocity(z, sol.t[end], H, V_wall, tau_visc) for z in z_bin_centers]
L2_err = sqrt(mean((vx_bin_avg .- vx_analytical_at_bins).^2))
Linf_err = maximum(abs.(vx_bin_avg .- vx_analytical_at_bins))
println("L2  error (vx): ", round(L2_err, sigdigits=4), " m/s")
println("L∞  error (vx): ", round(Linf_err, sigdigits=4), " m/s")
println("Relative L2 error: ", round(L2_err / V_wall * 100, sigdigits=3), "%")

if initialize_from_rest
    vx_eff_at_bins = [couette_transient_velocity(z, sol.t[end], H, V_wall, tau_eff)
                      for z in z_bin_centers]
    L2_err_eff = sqrt(mean((vx_bin_avg .- vx_eff_at_bins).^2))
    Linf_err_eff = maximum(abs.(vx_bin_avg .- vx_eff_at_bins))
    println("Estimated η_eff: ", round(eta_eff, sigdigits=5), " Pa·s")
    println("η_eff / η_nominal: ", round(eta_eff / viscosity_val, sigdigits=4))
    println("Best-fit τ_eff / τ_nominal: ", round(tau_eff / tau_visc, sigdigits=4))
    println("Fit RMSE over transient snapshots: ", round(fit_rmse, sigdigits=4), " m/s")
    println("Final relative L2 error vs fitted η_eff solution: ",
            round(L2_err_eff / V_wall * 100, sigdigits=3), "%")
    println("Final relative L∞ error vs fitted η_eff solution: ",
            round(Linf_err_eff / V_wall * 100, sigdigits=3), "%")

    p_err = plot(time_hist_nominal, rel_l2_hist_nominal,
        label="Relative L2 error (nominal η)", linewidth=2,
        xlabel="t / τ_visc", ylabel="Error [% of V_wall]",
        title="Couette Flow — Transient Convergence")
    plot!(p_err, time_hist_nominal, linf_hist_nominal,
        label="Relative L∞ error (nominal η)", linewidth=2, linestyle=:dash)
    plot!(p_err, time_hist_eff, rel_l2_hist_eff,
        label="Relative L2 error (fitted η_eff)", linewidth=2, linestyle=:dot)
    plot!(p_err, time_hist_eff, linf_hist_eff,
        label="Relative L∞ error (fitted η_eff)", linewidth=2, linestyle=:dashdot)
    savefig(p_err, "couette_viscous_transient_convergence.png")
    println("✓ Plot saved: couette_viscous_transient_convergence.png")
    println("Transient snapshots saved: ", length(time_hist_nominal))
    println("Final relative L2 error vs transient analytical solution: ",
            round(last(rel_l2_hist_nominal), sigdigits=3), "%")
end
println("=== DONE ===")
