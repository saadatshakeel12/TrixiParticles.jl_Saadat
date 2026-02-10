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
# ==========================================================================================

@inline initial_coordinates(system::TotalLagrangianSPHSystem) = system.initial_coordinates

@inline system_smoothing_kernel(system) = system.smoothing_kernel

@inline function _smoothing_length(system, particle)
    return system.smoothing_length
end

@inline system_correction(system) = nothing

@inline function current_coordinates(u, system::TotalLagrangianSPHSystem)
    return system.current_coordinates
end

@inline normalization_factor(::WendlandC2Kernel{2}, h) = 7 / (pi * h^2 * 4)

@inline function kernel_deriv(kernel::WendlandC2Kernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    result = -5 * (1 - q / 2)^3 * q

    # Zero out result if q >= 2
    result = ifelse(q < 2,
                    normalization_factor(kernel, h) * result * inner_deriv, zero(q))

    return result
end

@inline function kernel_grad(kernel, pos_diff, distance, h)
    # For `distance == 0`, the analytical gradient is zero, but the code divides by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`,
    # `distance^2` is in the order of `h^2`, hence the comparison `distance^2 < eps(h^2)`.
    # Note that this is faster than `distance < sqrt(eps(h^2))`.
    # Also note that `sqrt(eps(h^2)) != eps(h)`.
    distance^2 < eps(h^2) && return zero(pos_diff)

    return kernel_deriv(kernel, distance, h) / distance * pos_diff
end

@inline function corrected_kernel_grad(kernel, pos_diff, distance, h, correction, system,
                                       particle)
    return kernel_grad(kernel, pos_diff, distance, h)
end

@inline function smoothing_kernel_grad(system, pos_diff, distance, particle)
    return corrected_kernel_grad(system_smoothing_kernel(system), pos_diff,
                                 distance, _smoothing_length(system, particle),
                                 system_correction(system), system, particle)
end

Base.@propagate_inbounds function correction_matrix(system, particle)
    extract_smatrix(system.correction_matrix, system, particle)
end

@inline function update_temperature_sph!(system, dt, particle_spacing, x_heater,
    T_heater,h , semi)
    # Unpack system properties
    (; mass, material_density, temp, temp_ref, cp, k, current_coordinates) = system

    # Temporary storage for ΔT
    dT = zeros(length(temp_ref))

    dx = particle_spacing

    for i in 1:length(temp)
        if current_coordinates[1,i] < x_heater
            dq = 5e4   # W/m²
            dT[i] += dq / (material_density[i] * cp * dx)
        end
    end

    # Loop over all particles and neighbors (SPH)
    initial_coords = initial_coordinates(system)
    TrixiParticles.PointNeighbors.foreach_point_neighbor(system, system, initial_coords, initial_coords,
                           semi) do particle, neighbor, r, initial_distance2

        # Skip zero distance (same particle)
        #initial_distance^2 < eps(system.smoothing_length^2) && return

        # Particle volumes
        rho = material_density[1]
        volume = @inbounds mass[neighbor] / rho

        ##artificial thermal diffusion to reduce oscillations
        temp[particle] += 0.005 * (temp[neighbor] - temp[particle])

        # Distance vector

        @views r_vec =
            current_coordinates[:, particle] .-
            current_coordinates[:, neighbor]     
            
        r_vec = convert.(eltype(system), r_vec)

        initial_distance2 = TrixiParticles.dot(r_vec, r_vec)
        
        initial_distance2 < eps(smoothing_length^2) && return
        r = sqrt(initial_distance2)

        # Kernel gradient
        grad_kernel = smoothing_kernel_grad(system, r_vec,
                                            r, particle)

        # # Multiply by correction matrix (optional, improves consistency)
        #L = @inbounds correction_matrix(system, particle)

        # gradW = L' * grad_kernel   # corrected gradient
        
        gradW = grad_kernel

        # Harmonic mean of thermal conductivity
        #k_ij = 2 * k[particle] * k[neighbor] / (k[particle] + k[neighbor])
        k_ij = k

        ## SPH conduction contribution
        #dT[particle] += volume * k_ij / (material_density[particle] * cp) *
         #               TrixiParticles.dot(r_vec, gradW) * (temp[neighbor] - temp[particle]) / (initial_distance^2)
    
        _eps = 1e-3                # regularisation factor
        h2 = system.smoothing_length^2
        r2= initial_distance2 + _eps * h2

        val = TrixiParticles.dot(r_vec, gradW)

        if val > 0
            @warn "Positive r·∇W detected" val
        end

        flux = volume * k / (rho * cp) *
                        (temp[neighbor] - temp[particle]) *
                        val /(r2)

        dT[particle] += flux
        

    end

    println("max dT = ", maximum(abs.(dT)))
    println("nonzero dT count = ", count(!iszero, dT))

    # # Update temperature with flux limiter
    dT_max = 0.1
    @inbounds for i in 1:length(temp)
        temp_ref = temp[i]

        temp[i] += dt * dT[i]

        dT_act = temp[i] - temp_ref
        if abs(dT_act) > dT_max
            temp[i] = temp_ref + sign(dT_act) * dT_max
        end
    end
    
    println("dE = ", sum(dT .* mass ./ material_density))

    #E = sum(temp .* mass ./ material_density)
    #println("Total energy = ",E)

    return temp
end

# ==== Resolution
n_particles_y = 5

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
tspan = (0.0, 1.0)

rec_size = (length=0.2, thickness=0.05)
#material_plunger = (density=1000.0, E=1.4e6, nu=0.4)
material_polymer = (density=1000.0, E=1.4e6, nu=0.4 ,beta=0.000, alpha=0.000, temp=270.0, temp_ref=270.0, cp=1500.0 , k=50.0,
                            temp_liq=350.0,h= 5000.0,hardening= 1.4e4,tmelt=600.0)
#material_container=(density=1000.0, E=1.4e6, nu=0.4)
#clamp_radius = 0.05

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = rec_size.thickness / (n_particles_y - 1)

# # Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius
# clamped_particles = SphereShape(particle_spacing, clamp_radius + particle_spacing / 2,
#                                 (0.0, elastic_beam.thickness / 2), material.density,
#                                 cutout_min=(0.0, 0.0),
#                                 cutout_max=(clamp_radius, elastic_beam.thickness),
#                                 place_on_shell=true, coordinates_eltype=Float64)

# n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)


n_particles_per_dimension = (round(Int, rec_size.length / particle_spacing), n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for structures.
# We therefore need to pass `place_on_shell=true`.

#plunger = RectangularShape(particle_spacing, n_particles_per_dimension,
                        # (0.0, 0.2), density=material_plunger.density, place_on_shell=true,
                        # coordinates_eltype=Float64)

polymer = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (0.0, 0.0), density=material_polymer.density, place_on_shell=true,
                        coordinates_eltype=Float64)
                        
#container = RectangularShape(particle_spacing, n_particles_per_dimension,
                        #(0.0, 0.0), density=material_container.density, place_on_shell=true,
                        #coordinates_eltype=Float64)

#structure = union(clamped_particles, beam)

# ==========================================================================================
# ==== Structure
smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

#structure_system_plunger = TotalLagrangianSPHSystem(plunger, smoothing_kernel, smoothing_length,
                                            # material_plunger.E, material_plunger.nu,0,0,0,0,
                                            # acceleration=(0.0, -gravity),
                                            # penalty_force=nothing, viscosity=nothing,
                                            # clamped_particles_motion=nothing,
                                            # self_interaction_nhs=:default)

structure_system_polymer = TotalLagrangianSPHSystem(polymer, smoothing_kernel, smoothing_length,
                                            material_polymer.E, material_polymer.nu,
                                            material_polymer.beta, material_polymer.alpha,
                                            material_polymer.temp, material_polymer.temp_ref, material_polymer.cp, material_polymer.k,
                                            material_polymer.temp_liq,material_polymer.h,material_polymer.hardening,
                                            material_polymer.tmelt,
                                            acceleration=(0.0, 0),
                                            penalty_force=nothing, viscosity=nothing,
                                            clamped_particles_motion=nothing,
                                            self_interaction_nhs=:default)

#structure_system_container = TotalLagrangianSPHSystem(container, smoothing_kernel, smoothing_length,
                                            # material_container.E, material_container.nu,0,0,0,0,
                                            # acceleration=(0.0, 0),
                                            # penalty_force=nothing, viscosity=nothing,
                                            # clamped_particles_motion=nothing,
                                            # self_interaction_nhs=:default)

# ==========================================================================================
# ==== Simulation
# Note that the `neighborhood_search` passed here is not used if the simulation
# consists of a single `TotalLagrangianSPHSystem`.
# Instead, the neighborhood search passed to the `TotalLagrangianSPHSystem` is used.
semi = Semidiscretization(structure_system_polymer,
                          neighborhood_search=nothing,
                          parallelization_backend=PolyesterBackend())

# ------------- APPLY HEAT SOURCE AT SURFACE ----------------
# This enforces boundary temperature before updating ΔT
x_heater = 0.01   # example: left edge at x=0
T_heater = 1500 # example heater temperature
h = 100
##CFL-criteria
#dt_bc1 = vec((structure_system_polymer.material_density .* structure_system_polymer.cp .* particle_spacing) ./ h)
##Explicit Diffusion - criteria
#dt_bc2 = vec((structure_system_polymer.material_density .* structure_system_polymer.cp .* particle_spacing^2) ./ 2*structure_system_polymer.k)
dt = 0.1
t_total = 2500       # seconds to preheat

n_steps = round(Int, t_total / dt)

for step in 1:n_steps
    temp_updated = update_temperature_sph!(semi.systems[1],dt,particle_spacing,
    x_heater, T_heater, h, semi)
end

@show structure_system_polymer.temp        # shows the full 1D vector of particle temperatures
@show size(structure_system_polymer.temp)

using Plots

coords = structure_system_polymer.current_coordinates  # 2×N
xs = coords[1, :]
ys = coords[2, :]

x_unique = sort(unique(xs))
y_unique = sort(unique(ys))

n_x = length(unique(x_unique))
n_y = length(unique(y_unique))

Tgrid = fill(NaN, n_x, n_y)

for i in eachindex(structure_system_polymer.temp)
    ix = findfirst(==(xs[i]), x_unique)
    iy = findfirst(==(ys[i]), y_unique)
    Tgrid[ix, iy] = structure_system_polymer.temp[i]
end

heatmap(x_unique, y_unique, Tgrid',
        aspect_ratio=1,
        title="Temperature distribution")
savefig("temperature.png")

# material_polymer_preheated = (density=1000.0, E=1.4e6, nu=0.4 ,beta=0.000, alpha=0.000, temp_updated=271.0, temp_ref=270.0)

# structure_system_polymer_preheated = TotalLagrangianSPHSystem(polymer, smoothing_kernel, smoothing_length,
#                                             material_polymer_preheated.E, material_polymer_preheated.nu,
#                                             material_polymer_preheated.beta, material_polymer_preheated.alpha,
#                                             material_polymer_preheated.temp, material_polymer_preheated.temp_ref,
#                                             acceleration=(0.0, 0),
#                                             penalty_force=nothing, viscosity=nothing,
#                                             clamped_particles_motion=nothing,
#                                             self_interaction_nhs=:default)

# semi = Semidiscretization(structure_system_polymer_preheated,
#                           neighborhood_search=nothing,
#                           parallelization_backend=PolyesterBackend())

# ode = semidiscretize(semi, tspan)


# info_callback = InfoCallback(interval=1000)

# # Track the position of the particle in the middle of the tip of the beam.
# middle_particle_id = Int(n_particles_per_dimension[1] * (n_particles_per_dimension[2] + 1) /
#                          2)

# # Make these constants because global variables in the functions below are slow
# const STARTPOSITION_X = polymer.coordinates[1, middle_particle_id]
# const STARTPOSITION_Y = polymer.coordinates[2, middle_particle_id]

# function deflection_x(system, data, t)
#     return data.coordinates[1, middle_particle_id] - STARTPOSITION_X
# end

# function deflection_y(system, data, t)
#     return data.coordinates[2, middle_particle_id] - STARTPOSITION_Y
# end

# saving_callback = SolutionSavingCallback(dt=0.01, prefix="",
#                                          deflection_x=deflection_x,
#                                          deflection_y=deflection_y)

# callbacks = CallbackSet(info_callback, saving_callback)

# @show minimum(structure_system_polymer.lame_mu)
# @show maximum(structure_system_polymer.lame_mu)

# @show minimum(structure_system_polymer.lame_lambda)
# @show maximum(structure_system_polymer.lame_lambda)

# @show minimum(structure_system_polymer.young_modulus)
# @show maximum(structure_system_polymer.young_modulus)

# @show minimum(structure_system_polymer.poisson_ratio)
# @show maximum(structure_system_polymer.poisson_ratio)

# @show minimum(structure_system_polymer.temp)
# @show maximum(structure_system_polymer.temp)

# # Use a Runge-Kutta method with automatic (error based) time step size control
# sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks)

# using Plots 
# plot(sol)
