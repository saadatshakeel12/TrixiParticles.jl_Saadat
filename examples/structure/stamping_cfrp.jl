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

# @inline initial_coordinates(system::TotalLagrangianSPHSystem) = system.initial_coordinates

# @inline system_smoothing_kernel(system) = system.smoothing_kernel

# @inline function _smoothing_length(system, particle)
#     return system.smoothing_length
# end

# @inline system_correction(system) = nothing

# @inline function current_coordinates(u, system::TotalLagrangianSPHSystem)
#     return system.current_coordinates
# end

# @inline normalization_factor(::WendlandC2Kernel{2}, h) = 7 / (pi * h^2 * 4)

# @inline function kernel_deriv(kernel::WendlandC2Kernel, r::Real, h)
#     inner_deriv = 1 / h
#     q = r * inner_deriv

#     result = -5 * (1 - q / 2)^3 * q

#     # Zero out result if q >= 2
#     result = ifelse(q < 2,
#                     normalization_factor(kernel, h) * result * inner_deriv, zero(q))

#     return result
# end

# @inline function kernel_grad(kernel, pos_diff, distance, h)
#     # For `distance == 0`, the analytical gradient is zero, but the code divides by zero.
#     # To account for rounding errors, we check if `distance` is almost zero.
#     # Since the coordinates are in the order of the smoothing length `h`,
#     # `distance^2` is in the order of `h^2`, hence the comparison `distance^2 < eps(h^2)`.
#     # Note that this is faster than `distance < sqrt(eps(h^2))`.
#     # Also note that `sqrt(eps(h^2)) != eps(h)`.
#     distance^2 < eps(h^2) && return zero(pos_diff)
#     # q = distance / h
#     # if q >= 2
#     #     return zero(pos_diff)
#     # end

#     return kernel_deriv(kernel, distance, h) / distance * pos_diff
# end

# @inline function corrected_kernel_grad(kernel, pos_diff, distance, h, correction, system,
#                                        particle)
#     return kernel_grad(kernel, pos_diff, distance, h)
# end

# @inline function smoothing_kernel_grad(system, pos_diff, distance, particle)
#     return corrected_kernel_grad(system_smoothing_kernel(system), pos_diff,
#                                  distance, _smoothing_length(system, particle),
#                                  system_correction(system), system, particle)
# end

# Base.@propagate_inbounds function correction_matrix(system, particle)
#     extract_smatrix(system.correction_matrix, system, particle)
# end

# @inline function update_temperature_sph!(system, dt, particle_spacing, x_heater,
#     T_heater,h , semi)
#     # Unpack system properties
#     (; mass, material_density, temp, temp_ref, cp, k, current_coordinates) = system

#     # Temporary storage for ΔT
#     dT = zeros(length(temp_ref))

#     dx = particle_spacing

#     for i in 1:length(temp)
#         if current_coordinates[1,i] < x_heater
#             dq = 5e4   # W/m²
#             dT[i] += dq / (material_density[i] * cp * dx)
#         end
#     end

#     # Loop over all particles and neighbors (SPH)
#     initial_coords = initial_coordinates(system)
#     TrixiParticles.PointNeighbors.foreach_point_neighbor(system, system, initial_coords, initial_coords,
#                            semi) do particle, neighbor, r, initial_distance2

#         # Skip zero distance (same particle)
#         #initial_distance^2 < eps(system.smoothing_length^2) && return

#         # Particle volumes
#         rho = material_density[1]
#         volume = @inbounds mass[neighbor] / rho

#         ##artificial thermal diffusion to reduce oscillations
#         temp[particle] += 0.005 * (temp[neighbor] - temp[particle])

#         # Distance vector

#         @views r_vec =
#             current_coordinates[:, particle] .-
#             current_coordinates[:, neighbor]     
            
#         r_vec = convert.(eltype(system), r_vec)

#         initial_distance2 = TrixiParticles.dot(r_vec, r_vec)
        
#         initial_distance2 < eps(system.smoothing_length^2) && return
#         r = sqrt(initial_distance2)

#         # Kernel gradient
#         grad_kernel = smoothing_kernel_grad(system, r_vec,
#                                             r, particle)

#         # # Multiply by correction matrix (optional, improves consistency)
#         #L = @inbounds correction_matrix(system, particle)

#         # gradW = L' * grad_kernel   # corrected gradient
        
#         gradW = grad_kernel

#         # Harmonic mean of thermal conductivity
#         #k_ij = 2 * k[particle] * k[neighbor] / (k[particle] + k[neighbor])
#         k_ij = k

#         ## SPH conduction contribution
#         #dT[particle] += volume * k_ij / (material_density[particle] * cp) *
#          #               TrixiParticles.dot(r_vec, gradW) * (temp[neighbor] - temp[particle]) / (initial_distance^2)
    
#         _eps = 1e-3                # regularisation factor
#         h2 = system.smoothing_length^2
#         r2= initial_distance2 + _eps * h2

#         val = TrixiParticles.dot(r_vec, gradW)

#         if val > 0
#             @warn "Positive r·∇W detected" val
#         end

#         flux = volume * k / (rho * cp) *
#                         (temp[neighbor] - temp[particle]) *
#                         val /(r2)

#         dT[particle] += flux
        

#     end

#     #println("max dT = ", maximum(abs.(dT)))
#     #println("nonzero dT count = ", count(!iszero, dT))

#     # # Update temperature with flux limiter
#     dT_max = 0.1
#     @inbounds for i in 1:length(temp)
#         temp_ref = temp[i]

#         temp[i] += dt * dT[i]

#         dT_act = temp[i] - temp_ref
#         if abs(dT_act) > dT_max
#             temp[i] = temp_ref + sign(dT_act) * dT_max
#         end
#     end
    
#     #println("dE = ", sum(dT .* mass ./ material_density))

#     #E = sum(temp .* mass ./ material_density)
#     #println("Total energy = ",E)

#     return temp
# end

# ==== Resolution
n_particles_y = 6

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
#tspan = (0.0, 1.0)

rec_size = (length=0.2, thickness=0.06)
#material_plunger = (density=1000.0, E=1.4e6, nu=0.4)
material_polymer = (density=1600.0, E=12e10, nu=0.3 ,beta=0.000, temp=270.0, temp_ref=270.0, cp=1200.0 , k=10.0,
                            temp_liq=390.0,h= 1000000.0,hardening= 1.4e4,tmelt=700.0, viscosity=100000.0, yield_stress=600e6)


# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = rec_size.thickness / (n_particles_y - 1)


n_particles_per_dimension = (round(Int, rec_size.length / particle_spacing), n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for structures.
# We therefore need to pass `place_on_shell=true`.

polymer = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (0.0, -0.01), density=material_polymer.density, place_on_shell=true,
                        coordinates_eltype=Float64)

# ==========================================================================================
# ==== Structure
smoothing_length = 2 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

# X0 = initial_coordinates(semi.systems[1])
# ymin = minimum(X0[2, :])

# tol = 1e-8 * particle_spacing

# fixed = findall(i -> X0[2, i] ≤ ymin + tol, eachparticle(semi.systems[1]))

fixed = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

structure_system_polymer = TotalLagrangianSPHSystem(polymer, smoothing_kernel, smoothing_length,
                                            material_polymer.E, material_polymer.nu,
                                            material_polymer.beta,
                                            material_polymer.temp, material_polymer.temp_ref, material_polymer.cp, material_polymer.k,
                                            material_polymer.temp_liq,material_polymer.h,material_polymer.hardening, material_polymer.yield_stress,
                                            material_polymer.tmelt;
                                            clamped_particles=fixed,
                                            acceleration=(0.0, 0),
                                            penalty_force=nothing, viscosity=material_polymer.viscosity,
                                            clamped_particles_motion=nothing,
                                            self_interaction_nhs=:default)

# ==========================================================================================
# ==== Simulation
# Note that the `neighborhood_search` passed here is not used if the simulation
# consists of a single `TotalLagrangianSPHSystem`.
# Instead, the neighborhood search passed to the `TotalLagrangianSPHSystem` is used.
semi = Semidiscretization(structure_system_polymer,
                          neighborhood_search=nothing,
                          parallelization_backend=PolyesterBackend())


##CFL-criteria
#dt_bc1 = vec((structure_system_polymer.material_density .* structure_system_polymer.cp .* particle_spacing) ./ h)
##Explicit Diffusion - criteria
#dt_bc2 = vec((structure_system_polymer.material_density .* structure_system_polymer.cp .* particle_spacing^2) ./ 2*structure_system_polymer.k)
dt_p = 0.1
t_preheat = 2500       # seconds to preheat
dt_c = 0.000001
t_comp_cooling = 0.08  #seconds under compression and cooling
temp_mold = 270
global y_mold = 0.000000001
global y_particle = 0.06
global v_mold = 0.00000008
bound_coordinate1 = (1,0.01)
bound_coordinate = (2,0.05)

n_of_particles = size(semi.systems[1].current_coordinates,2)
n_of_fixed_particles = n_of_particles - length(fixed)

n_pre_steps = round(Int, t_preheat / dt_p)

n_comp_steps = round(Int, t_comp_cooling / dt_c)

for step in 1:n_pre_steps
    # temp_updated = update_temperature_sph!(semi.systems[1],dt_p,particle_spacing,
    # x_heater, T_heater, h, semi)
    q = 5e4
    update_temperature_sph!(semi.systems[1], dt_p, q ,particle_spacing, bound_coordinate1, semi)
end

using Plots

coords = structure_system_polymer.current_coordinates  # 2×N
xs = coords[1, 1:n_of_fixed_particles]
ys = coords[2, 1:n_of_fixed_particles]

x_unique = sort(unique(xs))

y_unique = sort(unique(ys))


n_x = length(unique(x_unique))
n_y = length(unique(y_unique))

Tgrid = fill(NaN, n_x, n_y)

for i in eachindex(structure_system_polymer.temp[1:n_of_fixed_particles])
    ix = findfirst(==(xs[i]), x_unique)
    iy = findfirst(==(ys[i]), y_unique)
    Tgrid[ix, iy] = structure_system_polymer.temp[i]
end

heatmap(x_unique, y_unique, Tgrid',
        aspect_ratio=1,
        title="Temperature distribution")
savefig("temperature_preheat.png")

global vel = zeros(eltype(semi.systems[1]), size(semi.systems[1].current_coordinates,1),n_of_particles)

@assert all(semi.systems[1].mass .> 0) 
@assert all(semi.systems[1].material_density .> 0)
@assert all(semi.systems[1].smoothing_length .> 0)
@assert all(isfinite.(smoothing_length))

x_hist = Vector{Vector{Float64}}()
y_hist = Vector{Vector{Float64}}()
T_hist = Vector{Vector{Float64}}()
v_hist = []
step_hist = []
alpha_hist = Vector{Vector{Float64}}()

#global time = 0

global _alpha = zeros(eltype(semi.systems[1]),1,n_of_particles)
global F_total_mold = 0.0

for step in 1:n_comp_steps
    
    global y_mold -= v_mold * dt_c
   
    global vel, _alpha,F_total_mold = thermomechanical_loop(semi.systems[1], temp_mold, y_mold ,particle_spacing, bound_coordinate ,dt_c,vel,fixed,_alpha,F_total_mold, v_mold, semi)

    if step%1000 == 0.0
        push!(x_hist, copy(semi.systems[1].current_coordinates[1,1:n_of_fixed_particles]))
        push!(y_hist, copy(semi.systems[1].current_coordinates[2,1:n_of_fixed_particles]))
        push!(T_hist, copy(semi.systems[1].temp[1:n_of_fixed_particles]))
        push!(v_hist, copy(vel[2,8]))
        push!(step_hist, copy(step))
        _alpha_flat = vec(_alpha)
        push!(alpha_hist, copy(_alpha_flat[1:n_of_fixed_particles]))
    end

    F_target = 0.00

    if abs(F_total_mold)>F_target
        F_error = F_total_mold - F_target
        global v_mold -= 1e-20 * F_error
    end

    #global time += dt_c
    #println("Time: ",time)
    #println("particle : ",8," vel : ",vel[2,8])
end

coords = structure_system_polymer.current_coordinates  # 2×N
xs = coords[1, 1:n_of_fixed_particles]
ys = coords[2, 1:n_of_fixed_particles]

println("current coordinates xs",xs[75:80])
println("current coordinates ys",ys[75:80])
# println("fixed: ",fixed)

anim = @animate for n in 1:length(x_hist)

    scatter(
        x_hist[n], y_hist[n],
        marker_z = T_hist[n],   # color = temperature
        markersize = 4,
        clims = (340, 380),     # fix color scale!
        color = :thermal,
        xlabel = "x",
        ylabel = "y",
        title = "Time step = $n",
        aspect_ratio = 1,
        ylims = (-0.02, 0.08) 
    )
end

gif(anim, "polymer_deformation.gif", fps = 20)

plot(step_hist, v_hist, label="Particle 8", xlabel="Time (s)", ylabel="Velocity (m/s)", lw=2)
savefig("velocity_plot_particle_8.png")

x_unique = sort(unique(xs))
y_unique = sort(unique(ys))


n_x = length(unique(x_unique))
n_y = length(unique(y_unique))

Tgrid = fill(NaN, n_x, n_y)

for i in eachindex(structure_system_polymer.temp[1:n_of_fixed_particles])
    ix = findfirst(==(xs[i]), x_unique)
    iy = findfirst(==(ys[i]), y_unique)
    Tgrid[ix, iy] = structure_system_polymer.temp[i]
end

heatmap(x_unique, y_unique, Tgrid',
        aspect_ratio=1,
        title="Temperature distribution")
savefig("temperature_mold.png")

anim2 = @animate for n in 1:length(x_hist)
    xs = x_hist[n]
    ys = y_hist[n]
    x_unique = sort(unique(xs)) 
    y_unique = sort(unique(ys))
    n_x = length(unique(x_unique))
    n_y = length(unique(y_unique))
    Agrid = fill(NaN, n_x, n_y)

    for i in eachindex(alpha_hist[n])
        ix = findfirst(==(xs[i]), x_unique)
        iy = findfirst(==(ys[i]), y_unique)
        alpha_current = alpha_hist[n]
        Agrid[ix, iy] = alpha_current[i]
    end

    heatmap(x_unique, y_unique,Agrid,
            aspect_ratio=1,
            title="alpha distribution")
end
gif(anim2, "polymer_alpha.gif", fps = 20)