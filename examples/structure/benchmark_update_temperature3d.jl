# ==========================================================================================
# Benchmark: update_temperature_sph3d! against 1D transient conduction reference
# ==========================================================================================
# Reference model:
# Semi-infinite solid, initially at T0, with constant surface heat flux q'' at z=0.
# Analytical solution (Carslaw & Jaeger) used widely in SPH heat-transfer validation,
# e.g. Cleary & Monaghan, "Conduction Modelling Using Smoothed Particle Hydrodynamics"
# (J. Comput. Phys., 1999).
#
# This script builds a static 3D block and advances temperature only using
# `update_temperature_sph3d!`, then compares SPH centerline profile vs analytical profile
# at selected times.
# ==========================================================================================

using TrixiParticles
using PointNeighbors
using LinearAlgebra
using Statistics
using Plots

import PointNeighbors: DictionaryCellList, FullGridCellList

# ------------------------------------------------------
# User parameters
# ------------------------------------------------------
particle_spacing = 10e-4
factor = 1.15
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = factor * particle_spacing

# Geometry: long in z so the heated boundary behaves approximately semi-infinite.
# Keep particle count moderate for fast benchmark turnaround.
nx, ny, nz = 8, 8, 60
origin = (0.0, 0.0, 0.0)

rho = 1500.0          # kg/m^3
cp = 900.0            # J/(kg K)
k_thermal = 1.0       # W/(m K)
alpha = k_thermal / (rho * cp)

T0 = 270.0            # K
q_flux = 2.0e4        # W/m^2 (applied at z-min boundary via ext_heat)

# Thermal integration controls
# NOTE: update_temperature_sph3d! internally clips per-step |dT| to 0.1 K.
# dt must satisfy:  dt < dT_max / (q'' / (rho*cp*ps))  = 0.1*rho*cp*ps/q''.
# For the default material that limit is ~5e-3 s; use 1e-3 for better temporal accuracy.
dt = 1.0e-3
t_end = 2.0
snapshot_times = [0.5, 1.0, 1.5, 2.0]

benchmark_modes = [
    (name="physical", artificial_diffusion_coeff=0.0,
    label="SPH (artificial_diffusion_coeff=0.0)")
]

# ------------------------------------------------------
# Build static 3D TLSPH system
# ------------------------------------------------------
shape = RectangularShape(particle_spacing, (nx, ny, nz), origin; density=rho)

material_polymer = (
    density=rho,
    E=1.0e6,
    nu=0.30,
    beta=0.0,
    temp=T0,
    temp_ref=T0,
    cp=cp,
    k=k_thermal,
    temp_liq=390.0,
    h=1000.0,
    hardening=1.0e9,
    tmelt=700.0,
    viscosity=1.0e5,
    yield_stress=150.0e6
)

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=2000)

system = TotalLagrangianSPHSystem(shape,
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
                                  self_interaction_nhs=nhs_template)

min_corner = vec(minimum(system.current_coordinates, dims=2))
max_corner = vec(maximum(system.current_coordinates, dims=2))
cell_list = FullGridCellList(; min_corner, max_corner, max_points_per_cell=2500)

semi = Semidiscretization(system;
                          neighborhood_search=GridNeighborhoodSearch{3}(;
                              cell_list,
                              search_radius=smoothing_length))

# Semidiscretization deep-copies systems; use the registered instance from here on.
system = semi.systems[1]

# Heat is applied to first particle layer near z-min
z_min = minimum(system.current_coordinates[3, :])
bound_coordinate = (3, z_min + 0.5 * particle_spacing)

# ------------------------------------------------------
# Analytical reference: semi-infinite solid with constant heat flux at z=0
# ------------------------------------------------------
function temp_ref_semi_infinite(z, t, T_init, q0, k, alpha)
    if t <= 0
        return T_init
    end

    # Complementary error-function approximation (Abramowitz & Stegun 7.1.26)
    # to avoid requiring external packages in this benchmark script.
    erfc_approx(x) = begin
        p = 0.3275911
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        s = sign(x)
        xx = abs(x)
        tloc = 1.0 / (1.0 + p * xx)
        erf_x = 1.0 - (((((a5 * tloc + a4) * tloc + a3) * tloc + a2) * tloc + a1) * tloc) * exp(-xx * xx)
        1.0 - s * erf_x
    end

    eta = z / (2 * sqrt(alpha * t))
    dT = (2 * q0 / k) * sqrt(alpha * t / pi) * exp(-eta^2) -
         (q0 * z / k) * erfc_approx(eta)
    return T_init + dT
end

function temp_ref_layer_average(z_left, z_right, t, T_init, q0, k, alpha; n_quad=16)
    dz = z_right - z_left
    dz <= 0 && return temp_ref_semi_infinite(z_left, t, T_init, q0, k, alpha)

    acc = 0.0
    for q in 0:(n_quad - 1)
        xi = (q + 0.5) / n_quad
        z = z_left + xi * dz
        acc += temp_ref_semi_infinite(z, t, T_init, q0, k, alpha)
    end

    return acc / n_quad
end

function layer_average_profile(coords, temp, z_min, dz)
    z = coords[3, :]
    n_layers = Int(floor((maximum(z) - z_min) / dz)) + 1

    z_center = Float64[]
    z_left = Float64[]
    z_right = Float64[]
    T_layer = Float64[]

    for layer in 0:(n_layers - 1)
        zl = z_min + layer * dz
        zr = zl + dz
        mask = (z .>= zl) .& (z .< zr)
        if any(mask)
            push!(z_center, 0.5 * (zl + zr) - z_min)
            push!(z_left, zl - z_min)
            push!(z_right, zr - z_min)
            push!(T_layer, mean(temp[mask]))
        end
    end

    return z_center, z_left, z_right, T_layer
end

function slice_temperature_contours(coords, temp, z_min, t, T_init, q0, k, alpha,
                                    particle_spacing)
    x = coords[1, :]
    y = coords[2, :]
    z = coords[3, :]

    y_unique = sort(unique(vec(y)))
    y_slice = y_unique[cld(length(y_unique), 2)]
    slice_mask = abs.(y .- y_slice) .<= 0.25 * particle_spacing

    x_slice = x[slice_mask]
    z_slice = z[slice_mask] .- z_min
    temp_slice = temp[slice_mask]

    x_unique = sort(unique(x_slice))
    z_unique = sort(unique(z_slice))

    T_sph_grid = fill(NaN, length(x_unique), length(z_unique))
    T_ref_grid = fill(NaN, length(x_unique), length(z_unique))

    for idx in eachindex(x_slice)
        ix = findfirst(==(x_slice[idx]), x_unique)
        iz = findfirst(==(z_slice[idx]), z_unique)

        layer_left = z_slice[idx]
        layer_right = layer_left + particle_spacing

        T_sph_grid[ix, iz] = temp_slice[idx]
        T_ref_grid[ix, iz] = temp_ref_layer_average(layer_left, layer_right, t,
                                                    T_init, q0, k, alpha)
    end

    return x_unique .* 1000, z_unique .* 1000, T_sph_grid, T_ref_grid,
           T_sph_grid .- T_ref_grid, y_slice
end

function make_temperature_contour_figure(snapshot_fields, coords, z_min, snapshot_times,
                                         T0, q_flux, k_thermal, alpha,
                                         particle_spacing, mode_label)
    contour_data = Dict{Float64, NamedTuple{(:x_mm, :z_mm, :T_sph, :T_ref, :T_err, :y_slice),
                                            Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64},
                                                  Matrix{Float64}, Matrix{Float64}, Float64}}}()

    temp_min = Inf
    temp_max = -Inf
    err_max = 0.0

    for t_snap in snapshot_times
        x_mm, z_mm, T_sph_grid, T_ref_grid, T_err_grid, y_slice =
            slice_temperature_contours(coords, snapshot_fields[t_snap], z_min, t_snap,
                                       T0, q_flux, k_thermal, alpha, particle_spacing)

        contour_data[t_snap] = (; x_mm, z_mm, T_sph=T_sph_grid, T_ref=T_ref_grid,
                                 T_err=T_err_grid, y_slice)

        temp_min = min(temp_min, minimum(T_sph_grid), minimum(T_ref_grid))
        temp_max = max(temp_max, maximum(T_sph_grid), maximum(T_ref_grid))
        err_max = max(err_max, maximum(abs.(T_err_grid)))
    end

    temp_levels = collect(range(temp_min, temp_max; length=18))
    err_levels = collect(range(-err_max, err_max; length=18))
    subplots = Any[]

    for t_snap in snapshot_times
        contour_state = contour_data[t_snap]
        time_label = "t=$(round(t_snap, digits=3)) s"

        push!(subplots,
              contourf(contour_state.x_mm, contour_state.z_mm, contour_state.T_sph';
                       title="SPH, $time_label",
                       xlabel="x (mm)",
                       ylabel="Depth z (mm)",
                       levels=temp_levels,
                       clims=(temp_min, temp_max),
                       fill=true,
                       c=:thermal,
                       aspect_ratio=:equal))

        push!(subplots,
              contourf(contour_state.x_mm, contour_state.z_mm, contour_state.T_ref';
                       title="Analytical, $time_label",
                       xlabel="x (mm)",
                       ylabel="Depth z (mm)",
                       levels=temp_levels,
                       clims=(temp_min, temp_max),
                       fill=true,
                       c=:thermal,
                       aspect_ratio=:equal))

        push!(subplots,
              contourf(contour_state.x_mm, contour_state.z_mm, contour_state.T_err';
                       title="SPH - analytical, $time_label",
                       xlabel="x (mm)",
                       ylabel="Depth z (mm)",
                       levels=err_levels,
                       clims=(-err_max, err_max),
                       fill=true,
                       c=:balance,
                       aspect_ratio=:equal))
    end

    y_slice_mm = contour_data[first(snapshot_times)].y_slice * 1000

    return plot(subplots...;
                layout=(length(snapshot_times), 3),
                size=(1350, 320 * length(snapshot_times)),
                plot_title="Temperature contours at mid-plane y=$(round(y_slice_mm, digits=3)) mm ($(mode_label))",
                margin=5Plots.mm)
end

function run_benchmark_mode!(system, semi, artificial_diffusion_coeff, dt, t_end,
                             q_flux, particle_spacing, bound_coordinate,
                             temperature_increment_limit,
                             snapshot_times, z_min, T0, k_thermal, alpha)
    system.temp .= T0

    n_steps = Int(round(t_end / dt))
    snapshot_data = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
    snapshot_fields = Dict{Float64, Vector{Float64}}()

    for step in 1:n_steps
        t = step * dt
        update_temperature_sph3d!(system, dt, q_flux, particle_spacing, bound_coordinate, semi;
                                  artificial_diffusion_coeff=artificial_diffusion_coeff,
                                  temperature_increment_limit=temperature_increment_limit)

        for t_snap in snapshot_times
            if abs(t - t_snap) <= 0.5 * dt && !haskey(snapshot_data, t_snap)
                z_prof, z_left, z_right, T_sph = layer_average_profile(system.current_coordinates,
                                                                       system.temp,
                                                                       z_min,
                                                                       particle_spacing)
                T_ref = [temp_ref_layer_average(zl, zr, t_snap, T0, q_flux, k_thermal, alpha)
                         for (zl, zr) in zip(z_left, z_right)]
                snapshot_data[t_snap] = (z_prof, T_sph, T_ref)
                snapshot_fields[t_snap] = copy(system.temp)
            end
        end
    end

    system.temp .= T0
    surface_history = Float64[]
    for step in 1:n_steps
        update_temperature_sph3d!(system, dt, q_flux, particle_spacing, bound_coordinate, semi;
                                  artificial_diffusion_coeff=artificial_diffusion_coeff,
                                  temperature_increment_limit=temperature_increment_limit)
        z = system.current_coordinates[3, :]
        surface_mask = z .<= (z_min + 0.75 * particle_spacing)
        push!(surface_history, mean(system.temp[surface_mask]))
    end

    return snapshot_data, surface_history, snapshot_fields
end

# ------------------------------------------------------
# Time integration: thermal-only update
# ------------------------------------------------------
n_steps = Int(round(t_end / dt))
mode_results = Dict{String, Any}()

println("--- Thermal Benchmark Start ---")
println("particles = ", nparticles(system),
        ", ps = ", particle_spacing,
        ", dt = ", dt,
        ", t_end = ", t_end)
println("q'' = ", q_flux, " W/m^2, alpha = ", alpha, " m^2/s")

for mode in benchmark_modes
    println("\nMode: ", mode.label)
    snapshot_data, surface_history, snapshot_fields = run_benchmark_mode!(system, semi,
                                                         mode.artificial_diffusion_coeff,
                                                         dt, t_end, q_flux,
                                                         particle_spacing,
                                                         bound_coordinate,
                                                         typemax(Float64),
                                                         snapshot_times,
                                                         z_min, T0,
                                                         k_thermal, alpha)
    mode_results[mode.name] = (snapshot_data=snapshot_data,
                               surface_history=surface_history,
                               snapshot_fields=snapshot_fields,
                               label=mode.label,
                               coeff=mode.artificial_diffusion_coeff)
end

# ------------------------------------------------------
# Error metrics + profile line plots
# ------------------------------------------------------
for mode in benchmark_modes
    snapshot_data = mode_results[mode.name].snapshot_data
    snapshot_fields = mode_results[mode.name].snapshot_fields

    plt_profile = plot(title="update_temperature_sph3d! benchmark: $(mode.label)",
                       xlabel="Depth from heated surface z (mm)",
                       ylabel="Temperature (K)",
                       legend=:bottomright,
                       size=(900, 550))

    println("\n--- Error Summary ($(mode.label)) ---")
    println(rpad("time [s]", 12), rpad("RMSE [K]", 14), rpad("Linf [K]", 14), "Rel L2 [%]")

    time_slices = sort(collect(keys(snapshot_data)))
    color_cycle = palette(:tab10)

    for (i, t_snap) in enumerate(time_slices)
        z_prof, T_sph, T_ref = snapshot_data[t_snap]
        curve_color = color_cycle[mod1(i, length(color_cycle))]

        err = T_sph .- T_ref
        rmse = sqrt(mean(err .^ 2))
        linf = maximum(abs.(err))
        rel_l2 = 100 * norm(err) / max(norm(T_ref .- T0), eps())

        println(rpad(string(round(t_snap, digits=4)), 12),
                rpad(string(round(rmse, digits=6)), 14),
                rpad(string(round(linf, digits=6)), 14),
                string(round(rel_l2, digits=4)))

        z_mm_lp = 1000 .* z_prof
      plot!(plt_profile, z_mm_lp, T_sph, lw=2, color=curve_color,
              label="SPH t=$(round(t_snap, digits=3)) s")
      plot!(plt_profile, z_mm_lp, T_ref, ls=:dash, lw=2, color=curve_color,
              label="Ref t=$(round(t_snap, digits=3)) s")

          plt_profile_step = plot(title="update_temperature_sph3d! profile: $(mode.label), t=$(round(t_snap, digits=3)) s",
                          xlabel="Depth from heated surface z (mm)",
                          ylabel="Temperature (K)",
                          legend=:bottomright,
                          size=(900, 550))
          plot!(plt_profile_step, z_mm_lp, T_sph, lw=2, color=curve_color,
              label="SPH")
          plot!(plt_profile_step, z_mm_lp, T_ref, ls=:dash, lw=2, color=curve_color,
              label="Ref")

          t_tag = replace(string(round(t_snap, digits=3)), "." => "p")
          savefig(plt_profile_step,
                "benchmark_update_temperature3d_profiles_$(mode.name)_t$(t_tag).png")
        
    end
    savefig(plt_profile, "benchmark_update_temperature3d_profiles_$(mode.name).png")

    plt_contours = make_temperature_contour_figure(snapshot_fields,
                                                   system.current_coordinates,
                                                   z_min,
                                                   snapshot_times,
                                                   T0,
                                                   q_flux,
                                                   k_thermal,
                                                   alpha,
                                                   particle_spacing,
                                                   mode.label)
    savefig(plt_contours, "benchmark_update_temperature3d_contours_$(mode.name).png")
end

# Near-surface layer temperature evolution check using the same first-layer average
# quantity as the SPH measurement.
t_vec = range(dt, t_end, length=n_steps)
T_surface_ref = [temp_ref_layer_average(0.0, particle_spacing, t, T0, q_flux,
                                        k_thermal, alpha) for t in t_vec]

plt_surface = plot(xlabel="Time (s)", ylabel="Surface temperature (K)",
                   title="Heated-surface temperature evolution",
                   size=(850, 500))
for mode in benchmark_modes
    surface_history = mode_results[mode.name].surface_history
    plot!(plt_surface, collect(t_vec), surface_history, lw=2,
          label=mode.label)
end
plot!(plt_surface, collect(t_vec), T_surface_ref, ls=:dash, lw=2,
      label="Analytical layer average")
savefig(plt_surface, "benchmark_update_temperature3d_surface.png")

println("\nSaved:")
for mode in benchmark_modes
    println("  - benchmark_update_temperature3d_profiles_$(mode.name).png")
    for t_snap in snapshot_times
        t_tag = replace(string(round(t_snap, digits=3)), "." => "p")
        println("  - benchmark_update_temperature3d_profiles_$(mode.name)_t$(t_tag).png")
    end
    println("  - benchmark_update_temperature3d_contours_$(mode.name).png")
end
println("  - benchmark_update_temperature3d_surface.png")
println("\nBenchmark completed.")

