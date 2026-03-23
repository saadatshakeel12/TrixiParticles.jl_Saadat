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
using Plots
# ==========================================================================================

file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "Cylinder.stl")
geometry = load_geometry(file)
#plot(geometry, showaxis=false, label=nothing, color=:black)

particle_spacing = 0.001
boundary_thickness = 3 * particle_spacing


signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

sdf_ic = InitialCondition(; coordinates=stack(signed_distance_field.positions),
                          density=1.0, particle_spacing=particle_spacing)

density = 1.0
boundary_sampled = sample_boundary(signed_distance_field; boundary_density=density,
                                   boundary_thickness)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry)                                   

shape_sampled = ComplexShape(geometry; particle_spacing, density=density,
                             point_in_geometry_algorithm , pad_initial_particle_grid=0.002)

shape_sampled.mass .= density * TrixiParticles.volume(geometry) / nparticles(shape_sampled);

background_pressure = 1.0

smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = 1.5 * particle_spacing

packing_system = ParticlePackingSystem(shape_sampled;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=nothing, background_pressure)

semi = Semidiscretization(packing_system)

# Use a high `tspan` to guarantee that the simulation runs for at least `maxiters`
tspan = (0, 10000.0)
ode = semidiscretize(semi, tspan)

maxiters = 100
callbacks = CallbackSet(UpdateCallback())
time_integrator = RDPK3SpFSAL35()

sol = solve(ode, time_integrator;
            abstol=1e-7, reltol=1e-4, save_everystep=false, maxiters=maxiters,
            callback=callbacks)

polymer = InitialCondition(sol, packing_system, semi)

# plot(packed_ic)

coords = semi.systems[1].initial_condition.coordinates
scatter3d(coords[1, :], coords[2, :], coords[3, :], 
          markersize=1, aspect_ratio=:equal)
savefig("packing.png")

# # ==========================================================================================
# # ==== Experiment Setup

material_polymer = (density=16000.0, E=12e10, nu=0.3 ,beta=0.000, temp=270.0, temp_ref=270.0, cp=1200.0 , k=20.0,
                            temp_liq=390.0,h= 100000.0,hardening= 1.4e6,tmelt=700.0, viscosity=100000.0, yield_stress=600e6)

# ==========================================================================================
# ==== Structure
smoothing_length = 1.4 * particle_spacing
smoothing_kernel = WendlandC2Kernel{3}()

X0 = polymer.coordinates
zmin = minimum(X0[3, :])

tol = 1e-8 * particle_spacing

fixed = findall(i -> X0[3, i] ≤ zmin + tol, eachparticle(polymer))
#fixed = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

structure_system_polymer = TotalLagrangianSPHSystem(polymer, smoothing_kernel, smoothing_length,
                                            material_polymer.E, material_polymer.nu,
                                            material_polymer.beta,
                                            material_polymer.temp, material_polymer.temp_ref, material_polymer.cp, material_polymer.k,
                                            material_polymer.temp_liq,material_polymer.h,material_polymer.hardening, material_polymer.yield_stress,
                                            material_polymer.tmelt;
                                            clamped_particles=fixed,
                                            acceleration=(0.0,0.0,0),
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
dt_p = 0.05
t_preheat = 200      # seconds to preheat
dt_c = 0.00000005
t_comp_cooling = 0.0000001 #seconds under compression and cooling
temp_mold = 270
global y_mold = 0.012
global v_mold = 0.00001
bound_coordinate1 = (3,-0.011)
bound_coordinate = (3,0.011)

n_of_particles = size(semi.systems[1].current_coordinates,2)
n_of_fixed_particles = n_of_particles - length(fixed)

n_pre_steps = round(Int, t_preheat / dt_p)

n_comp_steps = round(Int, t_comp_cooling / dt_c)

for step in 1:n_pre_steps
    q = 5e3
    update_temperature_sph3d!(semi.systems[1], dt_p, q ,particle_spacing, bound_coordinate1, semi)
end

coords = structure_system_polymer.current_coordinates  # 2×N
xs = coords[1, 1:n_of_fixed_particles]
ys = coords[2, 1:n_of_fixed_particles]
zs = coords[3, 1:n_of_fixed_particles]

scatter3d(xs, ys, zs,
          marker_z = structure_system_polymer.temp[1:n_of_fixed_particles],
          markersize = 4)
savefig("temperature_preheat.png")

global vel = zeros(eltype(semi.systems[1]), size(semi.systems[1].current_coordinates,1),n_of_particles)

@assert all(semi.systems[1].mass .> 0) 
@assert all(semi.systems[1].material_density .> 0)
@assert all(semi.systems[1].smoothing_length .> 0)
@assert all(isfinite.(smoothing_length))

x_hist = Vector{Vector{Float64}}()
y_hist = Vector{Vector{Float64}}()
z_hist = Vector{Vector{Float64}}()
T_hist = Vector{Vector{Float64}}()
stress_hist = Float64[]
strain_hist = Float64[]


#alpha_hist = Vector{Vector{Float64}}()
global time = 0

global _alpha = zeros(eltype(semi.systems[1]),1,n_of_particles)
global F_total_mold = 0.0
original_height = maximum(semi.systems[1].initial_coordinates[3,:]) - minimum(semi.systems[1].initial_coordinates[3,:])

for step in 1:n_comp_steps
    
    global y_mold -= v_mold * dt_c
   
    global vel, _alpha,F_total_mold = thermomechanical_loop3d(semi.systems[1], temp_mold, y_mold ,particle_spacing, bound_coordinate ,dt_c,vel,fixed,_alpha,F_total_mold, v_mold, semi)

    current_height = maximum(semi.systems[1].current_coordinates[3,:]) - minimum(semi.systems[1].current_coordinates[3,:])
    true_strain = log(current_height/original_height)

    volume = sum((semi.systems[1].mass)./(semi.systems[1].material_density))
    area = volume / current_height
    true_stress = abs(F_total_mold)/area

    if step%1 == 0.0
        push!(x_hist, copy(semi.systems[1].current_coordinates[1,1:n_of_fixed_particles]))
        push!(y_hist, copy(semi.systems[1].current_coordinates[2,1:n_of_fixed_particles]))
        push!(z_hist, copy(semi.systems[1].current_coordinates[3,1:n_of_fixed_particles]))
        push!(T_hist, copy(semi.systems[1].temp[1:n_of_fixed_particles]))
        push!(stress_hist, copy(true_stress))
        push!(strain_hist, copy(true_strain))
        #_alpha_flat = vec(_alpha)
        #push!(alpha_hist, copy(_alpha_flat[1:n_of_fixed_particles]))
    end

    #global time += dt_c
    #println("Time: ",time)

end

coords = structure_system_polymer.current_coordinates  # 2×N
xs = coords[1, 1:n_of_fixed_particles]
ys = coords[2, 1:n_of_fixed_particles]
zs = coords[3, 1:n_of_fixed_particles]

println("current coordinates xs",xs[75:80])
println("current coordinates ys",ys[75:80])
println("current coordinates zs",zs[75:80])
# println("fixed: ",fixed)

anim = @animate for n in 1:length(x_hist)

    scatter(
        x_hist[n], y_hist[n],z_hist[n],
        marker_z = T_hist[n],   # color = temperature
        markersize = 4,
        clims = (100, 600),     # fix color scale!
        color = :thermal,
        xlabel = "x",
        ylabel = "y",
        title = "Time step = $n",
        aspect_ratio = 1,
        ylims = (-0.015, 0.015),
        xlims = (-0.015, 0.015),
        zlims = (-0.015, 0.015)   
    )
end

gif(anim, "polymer_deformation.gif", fps = 20)

plot(strain_hist, stress_hist, label="True Stress vs True Strain", xlabel="Strain", ylabel="Stress", lw=2)
savefig("stress_vs_strain.png")

scatter3d(xs, ys, zs,
          marker_z = structure_system_polymer.temp[1:n_of_fixed_particles],
          markersize = 4)

savefig("temperature_mold.png")


# original_height = maximum(semi.systems[1].initial_coordinates[3,:]) - minimum(semi.systems[1].initial_coordinates[3,:])

#     current_height = maximum(semi.systems[1].current_coordinates[3,:]) - minimum(semi.systems[1].current_coordinates[3,:])
#     true_strain = original_height/current_height
    
#     volume = sum((semi.systems[1].mass)./(semi.systems[1].material_density))
#     area = volume / current_height
#     true_stress = F_total_mold/area
#         push!(stress_hist, copy(true_stress))
#         push!(strain_hist, copy(true_strain))