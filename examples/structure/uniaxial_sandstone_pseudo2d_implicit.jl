using TrixiParticles
using OrdinaryDiffEq
using StaticArrays
using Statistics

# ==========================================================================================
# Pseudo-2D Uniaxial Compression of Crossley Sandstone
#
# Paper-backed benchmark setup for the classical SPH method under uniaxial compression.
# The sample is modeled as a pseudo-2D 3D TLSPH column that is one particle thick in
# the out-of-plane direction. Bottom platen particles are fixed, while the top platen
# particles are prescribed to move downward at a constant speed.
# ==========================================================================================

# ==========================================================================================
# ==== Resolution and geometry
particle_spacing = 0.0025
specimen_width = 0.025
specimen_height = 0.05
platen_layers = 1

# ==========================================================================================
# ==== Material parameters (Crossley sandstone)
material_density = 2300.0
bulk_modulus = 12.2e9
shear_modulus = 2.67e9

young_modulus_sample = 9 * bulk_modulus * shear_modulus / (3 * bulk_modulus + shear_modulus)
poisson_ratio_sample = (3 * bulk_modulus - 2 * shear_modulus) /
                       (2 * (3 * bulk_modulus + shear_modulus))
platen_stiffness_ratio = 100.0

# ==========================================================================================
# ==== Loading and solver controls
punch_speed = 1.5e-3
target_engineering_strain = 0.01
tspan = (0.0, target_engineering_strain * specimen_height / punch_speed)
diagnostics_interval = 200
output_dt = max(last(tspan) / 50, eps(last(tspan)))
postprocess_dt = output_dt

penalty_force = PenaltyForceGanzenmueller(alpha=0.1)
viscosity = ArtificialViscosityMonaghan(alpha=0.02)
source_terms = nothing
max_dt = particle_spacing / sqrt(bulk_modulus / material_density) / 10

# ==========================================================================================
# ==== Setup
specimen_height >= particle_spacing ||
    throw(ArgumentError("`specimen_height` must be at least one particle spacing"))
specimen_width >= particle_spacing ||
    throw(ArgumentError("`specimen_width` must be at least one particle spacing"))
platen_layers >= 1 || throw(ArgumentError("`platen_layers` must be at least 1"))
punch_speed > 0 || throw(ArgumentError("`punch_speed` must be positive"))

n_particles_x = round(Int, specimen_width / particle_spacing) + 1
n_particles_z = round(Int, specimen_height / particle_spacing) + 2 * platen_layers + 1

origin = (-specimen_width / 2, -particle_spacing / 2, 0.0)
structure = RectangularShape(particle_spacing, (n_particles_x, 1, n_particles_z), origin,
                             density=material_density, place_on_shell=true,
                             coordinates_eltype=Float64)

z_levels = sort(unique(structure.coordinates[3, :]))
bottom_levels = Set(z_levels[1:platen_layers])
top_levels = Set(z_levels[(end - platen_layers + 1):end])

bottom_clamped_particles = findall(z -> z in bottom_levels, vec(structure.coordinates[3, :]))
top_clamped_particles = findall(z -> z in top_levels, vec(structure.coordinates[3, :]))
clamped_particles = vcat(bottom_clamped_particles, top_clamped_particles)

young_modulus = fill(young_modulus_sample, nparticles(structure))
poisson_ratio = fill(poisson_ratio_sample, nparticles(structure))
young_modulus[bottom_clamped_particles] .= platen_stiffness_ratio * young_modulus_sample
young_modulus[top_clamped_particles] .= platen_stiffness_ratio * young_modulus_sample

n_top_clamped_particles = length(top_clamped_particles)
moving_particles = (nparticles(structure) - n_top_clamped_particles + 1):nparticles(structure)
target_platen_displacement = target_engineering_strain * specimen_height
loading_duration = min(last(tspan), target_platen_displacement / punch_speed)

function movement_function(x, t)
    displacement = punch_speed * min(t, loading_duration)
    return SVector(x[1], x[2], x[3] - displacement)
end

is_moving(t) = t <= loading_duration
prescribed_motion = PrescribedMotion(movement_function, is_moving;
                                     moving_particles=collect(moving_particles))

smoothing_length = sqrt(3) * particle_spacing
smoothing_kernel = WendlandC2Kernel{3}()

structure_system = TotalLagrangianSPHSystem(structure, smoothing_kernel, smoothing_length,
                                            young_modulus, poisson_ratio,
                                            clamped_particles=clamped_particles,
                                            clamped_particles_motion=prescribed_motion,
                                            penalty_force=penalty_force,
                                            viscosity=viscosity,
                                            source_terms=source_terms,
                                            self_interaction_nhs=:default)

# ==========================================================================================
# ==== Postprocessing
const INITIAL_SPECIMEN_HEIGHT = specimen_height

function engineering_strain(system, data, t)
    z_coords = @view system.current_coordinates[3, 1:system.n_integrated_particles]
    return (INITIAL_SPECIMEN_HEIGHT - (maximum(z_coords) - minimum(z_coords))) /
           INITIAL_SPECIMEN_HEIGHT
end

function axial_stress(system, data, t)
    sigma = TrixiParticles.cauchy_stress(system)
    return mean(abs, @view sigma[3, 3, 1:system.n_integrated_particles])
end

function average_von_mises_stress(system, data, t)
    sigma_vm = TrixiParticles.von_mises_stress(system)
    return mean(@view sigma_vm[1:system.n_integrated_particles])
end

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(structure_system,
                          neighborhood_search=nothing,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=diagnostics_interval)
saving_callback = SolutionSavingCallback(dt=output_dt, prefix="")
postprocess_callback = PostprocessCallback(; dt=postprocess_dt,
                                           filename="uniaxial_sandstone_pseudo2d_implicit",
                                           engineering_strain, axial_stress,
                                           average_von_mises_stress)
callbacks = CallbackSet(info_callback, saving_callback, postprocess_callback)

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8,
            reltol=1e-6,
            dtmax=max_dt,
            save_everystep=false,
            callback=callbacks);
