using TrixiParticles
using PointNeighbors
using Printf

# ==========================================================================================
# Surrogate benchmark for the compression-force curves reported in
#   Cao et al. (2019), JNNFM 274, 104186.
#
# IMPORTANT:
#   This example uses the repository's existing thermo-mechanical TLSPH compression loop.
#   It does NOT implement the paper's modified PTT viscoelastic constitutive model.
#   The output is therefore suitable as a benchmark scaffold and comparison harness,
#   not as a paper-faithful constitutive validation on its own.
# ==========================================================================================

const DEFAULT_TEMPS_C = [270.0, 280.0, 290.0, 300.0]
const DEFAULT_SPEEDS_MM_S = [0.01]

parse_list(env_key, default_values) = begin
    raw = strip(get(ENV, env_key, ""))
    isempty(raw) && return copy(default_values)
    return [parse(Float64, strip(item)) for item in split(raw, ',') if !isempty(strip(item))]
end

temperatures_C = parse_list("TP_PC_TEMPS_C", DEFAULT_TEMPS_C)
speeds_mm_s = parse_list("TP_PC_SPEEDS_MM_S", DEFAULT_SPEEDS_MM_S)

t_total = parse(Float64, get(ENV, "TP_PC_TMAX_S", "45.0"))
dt = parse(Float64, get(ENV, "TP_PC_DT_S", "5.0e-4"))
output_dt = parse(Float64, get(ENV, "TP_PC_OUTPUT_DT_S", "0.25"))

output_directory = normpath(joinpath(@__DIR__, "..", "..", "out_pc_compression_benchmark"))
mkpath(output_directory)

history_csv_path = joinpath(output_directory, "compression_force_history.csv")
open(history_csv_path, "w") do io
    println(io, "case_id,temp_C,speed_mm_s,time_s,force_N")
end

function build_benchmark_system(temp_C)
    temp_K = temp_C + 273.15

    specimen_length = 0.160
    specimen_thickness = 0.004
    n_particles_y = 9
    particle_spacing = specimen_thickness / (n_particles_y - 1)
    n_particles_x = round(Int, specimen_length / particle_spacing)

    polymer = RectangularShape(particle_spacing,
                               (n_particles_x, n_particles_y),
                               (0.0, 0.0);
                               density=1200.0,
                               place_on_shell=true,
                               coordinates_eltype=Float64)

    smoothing_kernel = WendlandC2Kernel{2}()
    smoothing_length = 2.0 * particle_spacing

    # These parameters are placeholders for the current thermo-mechanical TLSPH model.
    # They should be calibrated if this scaffold is used for fitting to the paper curves.
    material_polymer = (
        density=1200.0,
        E=5.0e6,
        nu=0.30,
        beta=0.0,
        temp=temp_K,
        temp_ref=temp_K,
        cp=2050.0,
        k=0.17,
        temp_liq=temp_K - 1.0,
        h=5.0e4,
        hardening=1.0e4,
        tmelt=700.0,
        viscosity=0.132,
        yield_stress=5.0e5,
    )

    X0 = polymer.coordinates
    ymin = minimum(X0[2, :])
    tol = 1e-8 * particle_spacing
    fixed = findall(i -> X0[2, i] <= ymin + tol, eachparticle(polymer))

    system = TotalLagrangianSPHSystem(polymer,
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
                                      clamped_particles=fixed,
                                      acceleration=(0.0, 0.0),
                                      penalty_force=nothing,
                                      viscosity=material_polymer.viscosity,
                                      clamped_particles_motion=nothing,
                                      self_interaction_nhs=:default)

    semi = Semidiscretization(system;
                              neighborhood_search=nothing,
                              parallelization_backend=PolyesterBackend())

    semi.systems[1].temp .= temp_K

    top_y = maximum(semi.systems[1].current_coordinates[2, :])
    initial_gap = particle_spacing
    y_mold = top_y + initial_gap
    bound_coordinate = (2, ymin + 2.0 * particle_spacing)

    return (; semi, fixed, y_mold, bound_coordinate, particle_spacing, temp_K)
end

function run_case!(io, case_id, temp_C, speed_mm_s)
    cfg = build_benchmark_system(temp_C)
    semi = cfg.semi
    fixed = cfg.fixed
    y_mold = cfg.y_mold
    bound_coordinate = cfg.bound_coordinate
    particle_spacing = cfg.particle_spacing
    temp_K = cfg.temp_K

    system = semi.systems[1]
    n_particles = size(system.current_coordinates, 2)

    vel = zeros(eltype(system), size(system.current_coordinates, 1), n_particles)
    alpha = zeros(eltype(system), 1, n_particles)

    speed_m_s = speed_mm_s * 1e-3
    n_steps = round(Int, t_total / dt)
    next_output_t = 0.0
    force_N = 0.0

    println("Running case ", case_id,
            " | T = ", temp_C, " C",
            " | v = ", speed_mm_s, " mm/s",
            " | steps = ", n_steps)

    for step in 0:n_steps
        t = step * dt

        if t + 1e-12 >= next_output_t || step == 0 || step == n_steps
            @printf(io, "%s,%.3f,%.6f,%.8f,%.8f\n",
                    case_id, temp_C, speed_mm_s, t, max(force_N, 0.0))
            next_output_t += output_dt
        end

        step == n_steps && continue

        y_mold -= speed_m_s * dt
        vel, alpha, force_N = thermomechanical_loop(system,
                                                    temp_K,
                                                    y_mold,
                                                    particle_spacing,
                                                    bound_coordinate,
                                                    dt,
                                                    vel,
                                                    fixed,
                                                    alpha,
                                                    0.0,
                                                    speed_m_s,
                                                    semi)
    end

    flush(io)
end

open(history_csv_path, "a") do io
    for speed_mm_s in speeds_mm_s
        for temp_C in temperatures_C
            case_id = @sprintf("pc_T%03.0fC_v%.4f", temp_C, speed_mm_s)
            run_case!(io, case_id, temp_C, speed_mm_s)
        end
    end
end

println("Saved compression-force history to ", history_csv_path)
