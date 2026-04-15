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
particle_spacing = 0.0005         # 0.5 mm — gives 40 particles across H
H = 0.02                          # channel height = 20 mm
L = 0.005                         # domain size in x,y (small — just needs lateral neighbors)
V_wall = 0.01                     # top wall velocity in x-direction [m/s]
viscosity_val = 1.0e4             # Pa·s (same as PP viscosity)
density_val = 905.0               # kg/m³
T_room = 293.15                   # K

factor = 1.2
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = factor * particle_spacing

# Analytical Couette solution
shear_rate_analytical = V_wall / H
stress_xz_analytical = viscosity_val * shear_rate_analytical
println("Analytical shear rate: ", shear_rate_analytical, " s⁻¹")
println("Analytical σ_xz:      ", stress_xz_analytical, " Pa")
println("Viscous diffusion time τ = ρH²/η = ", density_val * H^2 / viscosity_val, " s")

# ==========================================================================================
# STEP 2: Build geometry — fluid block + two wall plates
# ==========================================================================================
num_layers_wall = 3
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
nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=200)

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

# ==========================================================================================
# STEP 5: ODE setup + solve (explicit)
# ==========================================================================================
# Viscous diffusion time: τ = ρH²/η
tau_visc = density_val * H^2 / viscosity_val
# Run for only 0.01τ — TLSPH is Lagrangian, particles drift with v_x, so F deviates
# from I over time.  A short run validates that the viscous stress *instantaneously*
# preserves the linear profile (∇·σ ≈ 0 for uniform shear stress).
t_end = 0.01 * tau_visc
tspan = (0.0, t_end)
println("τ_visc = ", tau_visc, " s")
println("t_end  = ", t_end, " s  (0.01 × τ_visc — short to keep F ≈ I)")

ode_base = semidiscretize(semi, tspan)

# Initialize velocities using the PHYSICAL channel boundaries (z=0 to z=H),
# NOT the full particle extent including wall layers.
# - Bottom wall particles (z < 0): v_x = 0
# - Fluid particles (0 ≤ z ≤ H):  v_x = V_wall * z / H  (linear Couette)
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
        v0_wrap[1, i] = V_wall * z_i / H   # fluid: linear
    end
    v0_wrap[2, i] = 0.0
    v0_wrap[3, i] = 0.0
end
println("✓ Velocities initialized: v_x = V_wall*z/H in fluid, 0 at bottom, V_wall at top")

# ==========================================================================================
# STEP 5a: Instantaneous diagnostics at t=0  (velocity gradient + stress)
# ==========================================================================================
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

# ==========================================================================================
# STEP 5b: Short time integration to verify profile maintenance
# ==========================================================================================
ode = DynamicalODEProblem(kick_viscous!, drift_viscous!,
                          ode_base.u0.x[1], ode_base.u0.x[2],
                          tspan, semi)

callbacks = CallbackSet(
    SolutionSavingCallback(dt=t_end / 5.0, prefix="couette_viscous"),
    InfoCallback(interval=50)
)

println("=== SOLVING (explicit, short run) ===")
sol = solve(ode, RDPK3SpFSAL35();
    callback=callbacks,
    save_everystep=false,
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

# Build a comparison subset that excludes near-wall and side-boundary particles.
z_pos_cmp = Float64[]
vx_sph_cmp = Float64[]

z_interior_lo_cmp = 3.0 * smoothing_length
z_interior_hi_cmp = H - 3.0 * smoothing_length
x_mid = 0.5 * L
y_mid = 0.5 * L
xy_tol = 1.5 * particle_spacing

for i in idx_fluid
    x_i = u_wrap[1, i]
    y_i = u_wrap[2, i]
    z_i = u_wrap[3, i]
    vx_i = v_wrap[1, i]

    if (z_interior_lo_cmp <= z_i <= z_interior_hi_cmp) &&
       (abs(x_i - x_mid) <= xy_tol) &&
       (abs(y_i - y_mid) <= xy_tol)
        push!(z_pos_cmp, z_i)
        push!(vx_sph_cmp, vx_i)
    end
end

# Fallback for very coarse/small cases: if centerline strip is empty,
# still compare interior particles only.
if isempty(z_pos_cmp)
    println("WARNING: centerline subset empty; falling back to interior-only comparison")
    for i in idx_fluid
        z_i = u_wrap[3, i]
        if z_interior_lo_cmp <= z_i <= z_interior_hi_cmp
            push!(z_pos_cmp, z_i)
            push!(vx_sph_cmp, v_wrap[1, i])
        end
    end
end

println("Comparison particles used: ", length(z_pos_cmp), " / ", n_fluid,
        " (interior + centerline strip)")

# Bin-average by z for a clean profile.
n_bins = nz_fluid
z_bin_edges = range(0.0, H, length=n_bins + 1)
z_bin_centers = Float64[]
vx_bin_avg = Float64[]
for k in 1:n_bins
    z_lo, z_hi = z_bin_edges[k], z_bin_edges[k + 1]
    mask = findall(z -> z_lo <= z < z_hi, z_pos_cmp)
    if !isempty(mask)
        push!(z_bin_centers, mean(z_pos_cmp[mask]))
        push!(vx_bin_avg, mean(vx_sph_cmp[mask]))
    end
end

# Analytical profile
z_analytical = range(0.0, H, length=100)
vx_analytical = V_wall .* z_analytical ./ H

# Plot
p = plot(z_analytical .* 1e3, vx_analytical .* 1e3,
    label="Analytical", linewidth=2, linestyle=:dash,
    xlabel="z [mm]", ylabel="vx [mm/s]",
    title="Couette Flow — Viscous (non-boundary comparison)")
scatter!(z_bin_centers .* 1e3, vx_bin_avg .* 1e3,
    label="SPH (binned, non-boundary)", markersize=5)
savefig(p, "couette_viscous_validation.png")
println("✓ Plot saved: couette_viscous_validation.png")

# Quantitative error
vx_analytical_at_bins = V_wall .* z_bin_centers ./ H
L2_err = sqrt(mean((vx_bin_avg .- vx_analytical_at_bins).^2))
Linf_err = maximum(abs.(vx_bin_avg .- vx_analytical_at_bins))
println("L2  error (vx): ", round(L2_err, sigdigits=4), " m/s")
println("L∞  error (vx): ", round(Linf_err, sigdigits=4), " m/s")
println("Relative L2 error: ", round(L2_err / V_wall * 100, sigdigits=3), "%")
println("=== DONE ===")
