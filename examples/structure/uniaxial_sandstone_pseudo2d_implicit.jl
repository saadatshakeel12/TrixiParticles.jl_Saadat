# ==========================================================================================
# Pseudo-2D Uniaxial Compression of a Sandstone Specimen
#
# Based on:
#   "Evaluation of Accuracy and Stability of the Classical SPH Method Under Uniaxial
#   Compression"
#
# This setup models a rectangular sandstone specimen compressed between rigid top and bottom
# platens. The platens are represented as clamped TLSPH particles. The lower platen remains
# fixed while the upper platen moves downward at a prescribed speed.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
particle_spacing = 1.0e-3

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 70.0e-3)
compression_speed = 1.5e-3
output_directory = "/SPH_Code/TrixiParticles.jl_Saadat/out_sandstone_uniaxial"
save_dt = 1.0e-3
stress_history_dt = 1.0e-3

specimen_size = (25.0e-3, 50.0e-3)
sandstone = (density=2200.0, E=14.0e7, nu=0.2)

# The integrated specimen starts one particle spacing above the fixed platen.
sample_origin = (0.0, particle_spacing)
top_platen_y = sample_origin[2] + specimen_size[2] + particle_spacing

n_particles_x = round(Int, specimen_size[1] / particle_spacing) + 1
n_particles_y = round(Int, specimen_size[2] / particle_spacing) + 1

sample = RectangularShape(particle_spacing, (n_particles_x, n_particles_y), sample_origin,
                          density=sandstone.density, place_on_shell=true,
                          coordinates_eltype=Float64)
bottom_platen = RectangularShape(particle_spacing, (n_particles_x, 1), (0.0, 0.0),
                                 density=sandstone.density, place_on_shell=true,
                                 coordinates_eltype=Float64)
top_platen = RectangularShape(particle_spacing, (n_particles_x, 1), (0.0, top_platen_y),
                              density=sandstone.density, place_on_shell=true,
                              coordinates_eltype=Float64)

structure = union(sample, bottom_platen, top_platen)

n_sample_particles = nparticles(sample)
n_bottom_particles = nparticles(bottom_platen)
n_top_particles = nparticles(top_platen)

bottom_platen_ids = collect((n_sample_particles + 1):(n_sample_particles + n_bottom_particles))
top_platen_ids = collect((n_sample_particles + n_bottom_particles + 1):
                         (n_sample_particles + n_bottom_particles + n_top_particles))
clamped_particles = vcat(bottom_platen_ids, top_platen_ids)

top_platen_threshold = top_platen_y - particle_spacing / 2

function platen_movement(x, t)
    if x[2] < top_platen_threshold
        return x
    end

    return x + SVector(0.0, -compression_speed * t)
end

is_moving(t) = t <= last(tspan)
platen_motion = PrescribedMotion(platen_movement, is_moving)

# ==========================================================================================
# ==== Structure
smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()
penalty_force = PenaltyForceGanzenmueller(alpha=0.1)
viscosity = ArtificialViscosityMonaghan(alpha=0.01, beta=0.0)

structure_system = TotalLagrangianSPHSystem(structure, smoothing_kernel, smoothing_length,
                                            sandstone.E, sandstone.nu,
                                            clamped_particles=clamped_particles,
                                            clamped_particles_motion=platen_motion,
                                            acceleration=(0.0, 0.0),
                                            penalty_force=penalty_force,
                                            viscosity=viscosity,
                                            self_interaction_nhs=:default)

function nearest_particle_id(coordinates, point, n_particles)
    best_particle = 1
    best_distance = Inf

    for particle in 1:n_particles
        dx = coordinates[1, particle] - point[1]
        dy = coordinates[2, particle] - point[2]
        distance = dx^2 + dy^2

        if distance < best_distance
            best_distance = distance
            best_particle = particle
        end
    end

    return best_particle
end

sample_mid_x = specimen_size[1] / 2
probe_points = (
    lower=SVector(sample_mid_x, sample_origin[2] + 0.25 * specimen_size[2]),
    middle=SVector(sample_mid_x, sample_origin[2] + 0.50 * specimen_size[2]),
    upper=SVector(sample_mid_x, sample_origin[2] + 0.75 * specimen_size[2])
)

const PROBE_PARTICLES = (
    lower=nearest_particle_id(structure_system.initial_coordinates, probe_points.lower,
                              structure_system.n_integrated_particles),
    middle=nearest_particle_id(structure_system.initial_coordinates, probe_points.middle,
                               structure_system.n_integrated_particles),
    upper=nearest_particle_id(structure_system.initial_coordinates, probe_points.upper,
                              structure_system.n_integrated_particles)
)

axial_stress(system, particle) = TrixiParticles.cauchy_stress(system)[2, 2, particle]

axial_stress_lower(system, data, t) = axial_stress(system, PROBE_PARTICLES.lower)
axial_stress_middle(system, data, t) = axial_stress(system, PROBE_PARTICLES.middle)
axial_stress_upper(system, data, t) = axial_stress(system, PROBE_PARTICLES.upper)
punch_displacement(system, data, t) = compression_speed * t

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(structure_system, neighborhood_search=nothing,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=1000)
saving_callback = SolutionSavingCallback(dt=save_dt, prefix="",
                                         output_directory=output_directory)
stress_history_callback = PostprocessCallback(; dt=stress_history_dt,
                                              output_directory=output_directory,
                                              filename="stress_history",
                                              write_file_interval=0,
                                              axial_stress_lower,
                                              axial_stress_middle,
                                              axial_stress_upper,
                                              punch_displacement)

callbacks = CallbackSet(info_callback, saving_callback, stress_history_callback)

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8, reltol=1e-6, dtmax=min(save_dt, stress_history_dt),
            save_everystep=false, callback=callbacks);
