# ==========================================================================================
# 3D Shear-Wave Decay — Pure Viscous Transient Validation (Periodic Box)
#
# Validates viscous_stress3d_fast! in transient mode without wall effects.
#
# Setup:
#   - Single periodic block of particles
#   - Initial velocity: v_x(z, 0) = V0 * sin(k z), with k = 2π / Lz
#   - No elasticity/plasticity effects intended (very soft E, viscous-only stress cache)
#   - Analytical mode decay: A(t) = A0 * exp(-nu * k^2 * t), nu = eta / rho
#
# Why this benchmark:
#   - Isolates viscosity from boundary-wall transfer artifacts
#   - Allows direct eta_eff estimation from transient amplitude decay
# ==========================================================================================
using TrixiParticles
using OrdinaryDiffEq
using PointNeighbors
using Plots
using Base.Threads
using Statistics
using LinearAlgebra

# Check if ODE solver succeeded without requiring SciMLBase.
# OrdinaryDiffEq retcodes may be symbols or enum-like objects depending on versions.
successful_retcode(sol) = (sol.retcode == :Success || occursin("Success", string(sol.retcode)))

println("--- SHEAR-WAVE DECAY VISCOUS VALIDATION (PERIODIC) ---")
println("Threads available: ", nthreads())
println("-------------------------------------------------------")

# ==========================================================================================
# STEP 1: Parameters
# ==========================================================================================
particle_spacing = 0.0007
# Keep x/y box lengths sufficiently large for periodic GridNeighborhoodSearch
# when TLSPH initializes the frozen self-interaction NHS.
Lx = 0.0144
Ly = 0.0144
Lz = 0.036

V0 = 0.00015                       # gentler excitation to avoid numerical blow-up
viscosity_val = 1.0e3              # lower viscosity to reduce explicit stiffness
density_val = 905.0                # density [kg/m^3]
T_room = 293.15

# Robust post-processing settings.
fit_time_fraction = 0.5            # fit over early-to-mid decay for a more reliable eta_eff estimate
mode_purity_threshold = 0.15       # validation requires fit-window mean |A2/A1| <= this threshold
eta_ratio_bounds = (0.90, 1.10)    # validation target for eta_eff / eta_nominal
min_fit_samples = 6                # require enough snapshots in fit window for stable slope estimation
n_save_samples = 48                # denser sampling than before to improve eta_eff fit robustness

h_over_dx = 1.3
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = h_over_dx * particle_spacing

nx = max(4, ceil(Int, Lx / particle_spacing))
ny = max(4, ceil(Int, Ly / particle_spacing))
nz = max(8, ceil(Int, Lz / particle_spacing))

# Use exact box lengths implied by integer particle counts.
Lx_eff = nx * particle_spacing
Ly_eff = ny * particle_spacing
Lz_eff = nz * particle_spacing

k_wave = 2.0 * pi / Lz_eff
nu_nominal = viscosity_val / density_val
tau_mode = 1.0 / (nu_nominal * k_wave^2)
t_end_factor = 0.25
t_end = t_end_factor * tau_mode

println("Grid: nx=", nx, " ny=", ny, " nz=", nz)
println("Effective lengths: Lx=", Lx_eff, " Ly=", Ly_eff, " Lz=", Lz_eff, " m")
println("h/dx=", h_over_dx, "  h=", smoothing_length, " m")
println("k_wave=", k_wave, " 1/m")
println("tau_mode (nominal)=", tau_mode, " s")
println("V0=", V0, " m/s")
println("t_end=", t_end, " s  (", t_end_factor, " * tau_mode)")

# ==========================================================================================
# STEP 2: Build periodic particle block
# ==========================================================================================
fluid_particles = RectangularShape(particle_spacing,
    (nx, ny, nz),
    (0.0, 0.0, 0.0);
    density=density_val)

n_particles = nparticles(fluid_particles)
println("Particles: ", n_particles)

periodic_box = PeriodicBox(min_corner=[0.0, 0.0, 0.0],
                           max_corner=[Lx_eff, Ly_eff, Lz_eff])

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=400,
                                                periodic_box=periodic_box)

# ==========================================================================================
# STEP 3: TLSPH system (viscous-only intent)
# ==========================================================================================
# Keep E very small so non-viscous response is negligible.
E_soft = 1.0e2

material = (
    nu=0.3, beta=0.0,
    temp=T_room, temp_ref=T_room,
    cp=1900.0, k=0.22,
    temp_liq=433.15, h=70000.0,
    hardening=0.0, tmelt=433.15,
    yield_stress=1.0e12,
    viscosity=viscosity_val
)

system = TotalLagrangianSPHSystem(fluid_particles,
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

semi = Semidiscretization(system;
    neighborhood_search=GridNeighborhoodSearch{3}(; periodic_box,
                                                   search_radius=smoothing_length))

# ==========================================================================================
# STEP 4: Viscous-only stress cache + custom kick/drift
# ==========================================================================================
v_stress_buf = zeros(3, 3, n_particles)
vel_grad_buf = zeros(3, 3, n_particles)
nhs_updated_at_t = Ref(-Inf)
rhs_counter = Ref(0)

function update_viscous_stress_cache!(semi_local, v_ode, viscosity_0)
    sys_local = semi_local.systems[1]
    v_wrap = TrixiParticles.wrap_v(v_ode, sys_local, semi_local)
    _, _, vis = TrixiParticles.update_properties!(sys_local, zeros(n_particles),
                                                  semi_local, viscosity_0)
    fill!(v_stress_buf, 0)
    fill!(vel_grad_buf, 0)
    stress_visc = TrixiParticles.viscous_stress3d_fast!(sys_local,
        v_wrap,
        vis, semi_local;
        v_vis_buf=v_stress_buf, vel_grad_buf=vel_grad_buf)
    TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(sys_local), stress_visc)
end

function kick_viscous!(dv_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.set_zero!(dv_ode)

    rhs_counter[] += 1
    if rhs_counter[] == 1
        println(">>> First RHS call (t=", t, ")")
        flush(stdout)
    end
    if mod(rhs_counter[], 4000) == 0
        println(">>> RHS #", rhs_counter[], " t=", round(t, digits=8))
        flush(stdout)
    end

    sys_local = semi_local.systems[1]

    try
        TrixiParticles.foreach_system(semi_local) do system_local
            v = TrixiParticles.wrap_v(v_ode, system_local, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system_local, semi_local)
            TrixiParticles.update_positions!(system_local, v, u, v_ode, u_ode, semi_local, t)
        end

        if t != nhs_updated_at_t[]
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end

        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(sys_local)

        TrixiParticles.foreach_system(semi_local) do system_local
            v = TrixiParticles.wrap_v(v_ode, system_local, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system_local, semi_local)
            TrixiParticles.update_quantities!(system_local, v, u, v_ode, u_ode, semi_local, t)
        end

        TrixiParticles.update_implicit_sph!(semi_local, v_ode, u_ode, t)

        TrixiParticles.foreach_system(semi_local) do system_local
            v = TrixiParticles.wrap_v(v_ode, system_local, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system_local, semi_local)
            TrixiParticles.update_pressure!(system_local, v, u, v_ode, u_ode, semi_local, t)
        end

        TrixiParticles.foreach_system(semi_local) do system_local
            v = TrixiParticles.wrap_v(v_ode, system_local, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system_local, semi_local)
            TrixiParticles.update_boundary_interpolation!(system_local, v, u, v_ode, u_ode,
                                                          semi_local, t)
        end

        TrixiParticles.foreach_system(semi_local) do system_local
            v = TrixiParticles.wrap_v(v_ode, system_local, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system_local, semi_local)
            TrixiParticles.update_final!(system_local, v, u, v_ode, u_ode, semi_local, t)
        end

        update_viscous_stress_cache!(semi_local, v_ode, material.viscosity)
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi_local, t)
    finally
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = nothing
        TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
    end

    return dv_ode
end

function drift_viscous!(du_ode, v_ode, u_ode, semi_local, t)
    return TrixiParticles.drift!(du_ode, v_ode, u_ode, semi_local, t)
end

# ==========================================================================================
# STEP 5: Initialize sine-wave velocity and solve
# ==========================================================================================
tspan = (0.0, t_end)
ode_base = semidiscretize(semi, tspan)

v0 = ode_base.u0.x[1]
sys = semi.systems[1]
v0_wrap = TrixiParticles.wrap_v(v0, sys, semi)

@inbounds for i in 1:n_particles
    z_i = sys.initial_coordinates[3, i]
    v0_wrap[1, i] = V0 * sin(k_wave * z_i)
    v0_wrap[2, i] = 0.0
    v0_wrap[3, i] = 0.0
end
println("Initialized v_x(z,0)=V0*sin(k z)")

ode = DynamicalODEProblem(kick_viscous!, drift_viscous!,
                          ode_base.u0.x[1], ode_base.u0.x[2],
                          tspan, semi)

save_times = collect(range(0.0, t_end, length=n_save_samples))

callbacks = CallbackSet(
    SolutionSavingCallback(dt=10.0 * t_end, prefix="shear_wave_decay_viscous"),
    InfoCallback(interval=500)
)

println("=== SOLVING (explicit) ===")
fixed_dt = tau_mode / 5000.0
println("Using fixed dt=", fixed_dt, " s (adaptive=false)")
sol = solve(ode, RDPK3SpFSAL35();
    callback=callbacks,
    adaptive=false,
    dt=fixed_dt,
    save_everystep=false,
    saveat=save_times,
    maxiters=50_000_000)
println("=== SOLVE COMPLETE ===")
println("Solver reached final time successfully: ", successful_retcode(sol))

# ==========================================================================================
# STEP 6: Post-processing (mode decay + eta_eff fit)
# ==========================================================================================
function mode_projection(vx, z, k, mode)
    return 2.0 * mean(vx .* sin.(mode * k .* z))
end

function fit_eta_eff(time_vec, amp_vec, density, k)
    ids = findall(i -> time_vec[i] > 0.0 && amp_vec[i] > 0.0, eachindex(time_vec))
    length(ids) >= 2 || error("Not enough positive-amplitude samples to fit eta_eff")

    t_fit = time_vec[ids]
    y_fit = log.(amp_vec[ids])
    X = hcat(ones(length(t_fit)), t_fit)
    coeff = X \ y_fit

    slope = coeff[2]
    nu_eff = -slope / k^2
    eta_eff = density * nu_eff

    return eta_eff, nu_eff, coeff[1], slope
end

function fit_sample_ids(time_vec, amp_vec, max_fit_time)
    return findall(i -> 0.0 < time_vec[i] <= max_fit_time && amp_vec[i] > 0.0,
                   eachindex(time_vec))
end

time_hist = Float64[]
amp_hist = Float64[]
amp_analytical_nominal = Float64[]
harmonic_ratio = Float64[]

for (t_snapshot, state) in zip(sol.t, sol.u)
    v_snapshot = TrixiParticles.wrap_v(state.x[1], sys, semi)
    u_snapshot = TrixiParticles.wrap_u(state.x[2], sys, semi)

    vx = vec(v_snapshot[1, :])
    z = vec(u_snapshot[3, :])

    A1 = mode_projection(vx, z, k_wave, 1)
    A2 = mode_projection(vx, z, k_wave, 2)

    push!(time_hist, t_snapshot)
    push!(amp_hist, abs(A1))
    push!(harmonic_ratio, abs(A2) / max(abs(A1), eps()))
    push!(amp_analytical_nominal, V0 * exp(-nu_nominal * k_wave^2 * t_snapshot))
end

solver_ok = successful_retcode(sol)
max_fit_time = fit_time_fraction * t_end
fit_ids = fit_sample_ids(time_hist, amp_hist, max_fit_time)
all_positive_fit_ids = fit_sample_ids(time_hist, amp_hist, Inf)

eta_eff = NaN
nu_eff = NaN
logA0_fit = NaN
slope_fit = NaN
amp_analytical_eff = fill(NaN, length(time_hist))

fit_window_purity = isempty(fit_ids) ? Inf : mean(harmonic_ratio[fit_ids])
ids_for_fit = length(fit_ids) >= 2 ? fit_ids : all_positive_fit_ids
can_fit_eta = length(ids_for_fit) >= 2

if can_fit_eta
    eta_eff, nu_eff, logA0_fit, slope_fit = fit_eta_eff(time_hist[ids_for_fit], amp_hist[ids_for_fit],
                                                        density_val, k_wave)
    if isfinite(eta_eff) && isfinite(nu_eff)
        amp_analytical_eff = exp.(logA0_fit .+ slope_fit .* time_hist)
    end
end

eta_ratio = can_fit_eta ? (eta_eff / viscosity_val) : NaN
passes_fit_samples = length(fit_ids) >= min_fit_samples
passes_mode_purity = fit_window_purity <= mode_purity_threshold
passes_eta_ratio = can_fit_eta && isfinite(eta_ratio) && eta_ratio_bounds[1] <= eta_ratio <= eta_ratio_bounds[2]
validation_pass = solver_ok && can_fit_eta && passes_fit_samples && passes_mode_purity && passes_eta_ratio

println("\n=== DECAY FIT RESULTS ===")
println("Nominal eta: ", viscosity_val, " Pa·s")
println("Nominal nu: ", nu_nominal, " m^2/s")
println("Mean |A2/A1| (mode purity): ", round(mean(harmonic_ratio), sigdigits=5))
println("Final |A2/A1|: ", round(last(harmonic_ratio), sigdigits=5))
println("Mean |A2/A1| in fit window: ", round(fit_window_purity, sigdigits=5))
println("Fit samples in window: ", length(fit_ids), " (required >= ", min_fit_samples, ")")

if length(fit_ids) < 3
    println("WARNING: not enough early-window samples; eta_eff fit used all available positive samples.")
elseif fit_window_purity > mode_purity_threshold
    println("WARNING: mode purity exceeds threshold in fit window; eta_eff fit forced anyway.")
elseif !can_fit_eta
    println("WARNING: eta_eff fit could not be computed (insufficient positive samples overall).")
else
    println("Estimated eta_eff: ", round(eta_eff, sigdigits=6), " Pa·s")
    println("eta_eff / eta_nominal: ", round(eta_eff / viscosity_val, sigdigits=5))
    println("Estimated nu_eff: ", round(nu_eff, sigdigits=6), " m^2/s")
end

if can_fit_eta
    println("Acceptance window for eta_eff / eta_nominal: [", eta_ratio_bounds[1], ", ", eta_ratio_bounds[2], "]")
end

println("Validation checks:")
println("  solver success: ", solver_ok)
println("  enough fit samples: ", passes_fit_samples)
println("  mode purity in fit window: ", passes_mode_purity)
println("  eta ratio in bounds: ", passes_eta_ratio)
println(validation_pass ? "✓ VISCOUS VALIDATION PASSED" : "✗ VISCOUS VALIDATION FAILED")

# Plot: amplitude decay
p_decay = plot(time_hist ./ tau_mode, amp_hist .* 1e3,
    label="SPH |A1(t)|", marker=:circle, linewidth=2,
    xlabel="t / tau_mode(nominal)", ylabel="Mode amplitude [mm/s]",
    title="Shear-Wave Decay — Viscous Validation")
plot!(p_decay, time_hist ./ tau_mode, amp_analytical_nominal .* 1e3,
    label="Analytical (nominal eta)", linewidth=2, linestyle=:dash)
if can_fit_eta && isfinite(eta_eff)
    plot!(p_decay, time_hist ./ tau_mode, amp_analytical_eff .* 1e3,
        label="Analytical (fitted eta_eff)", linewidth=2, linestyle=:dot)
end
savefig(p_decay, "shear_wave_decay_viscous_amplitude.png")
println("✓ Plot saved: shear_wave_decay_viscous_amplitude.png")

# Plot: final profile vs analytical curves
v_final = TrixiParticles.wrap_v(sol.u[end].x[1], sys, semi)
u_final = TrixiParticles.wrap_u(sol.u[end].x[2], sys, semi)

z_final = vec(u_final[3, :])
vx_final = vec(v_final[1, :])

n_bins = nz
z_edges = range(0.0, Lz_eff, length=n_bins + 1)
z_centers = Float64[]
vx_bin = Float64[]
for b in 1:n_bins
    z_lo, z_hi = z_edges[b], z_edges[b + 1]
    ids = findall(z -> z_lo <= z < z_hi, z_final)
    if !isempty(ids)
        push!(z_centers, mean(z_final[ids]))
        push!(vx_bin, mean(vx_final[ids]))
    end
end

t_final = sol.t[end]
vx_nominal = [V0 * sin(k_wave * z) * exp(-nu_nominal * k_wave^2 * t_final) for z in z_centers]
vx_eff = can_fit_eta ? [V0 * sin(k_wave * z) * exp(-nu_eff * k_wave^2 * t_final)
                        for z in z_centers] : Float64[]

p_profile = plot(z_centers .* 1e3, vx_nominal .* 1e3,
    label="Analytical (nominal eta)", linewidth=2, linestyle=:dash,
    xlabel="z [mm]", ylabel="v_x [mm/s]",
    title="Shear-Wave Final Profile")
scatter!(p_profile, z_centers .* 1e3, vx_bin .* 1e3,
    label="SPH (z-binned)", markersize=5)
savefig(p_profile, "shear_wave_decay_viscous_profile.png")
println("✓ Plot saved: shear_wave_decay_viscous_profile.png")

println("=== DONE ===")
