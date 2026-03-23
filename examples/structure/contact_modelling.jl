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
using PointNeighbors
using Plots
using Base.Threads
println("--- SIMULATION STARTING ---")
println("Threads available: ", nthreads())
println("---------------------------")

# ==========================================================================================
# STEP 1: Load and pack the cylinder geometry
# ==========================================================================================

file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "Cylinder.stl")
geometry = load_geometry(file)

particle_spacing = 0.001
boundary_thickness = 3 * particle_spacing

signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

density_cylinder = 1000.0  # kg/m³ — realistic polymer density

point_in_geometry_algorithm = WindingNumberJacobson(; geometry)

shape_sampled = ComplexShape(geometry; particle_spacing, density=density_cylinder,
                             point_in_geometry_algorithm, pad_initial_particle_grid=0.002)

shape_sampled.mass .= density_cylinder * TrixiParticles.volume(geometry) / nparticles(shape_sampled)

background_pressure = 1.0

smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
smoothing_length = 1.5 * particle_spacing

# ==========================================================================================
# STEP 2: Pack the cylinder particles
# ==========================================================================================

packing_system = ParticlePackingSystem(shape_sampled;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=signed_distance_field,  # provide SDF!
                                       background_pressure=background_pressure)

semi_pack = Semidiscretization(packing_system)

tspan_pack = (0.0, 10000.0)
ode_pack = semidiscretize(semi_pack, tspan_pack)

sol_pack = solve(ode_pack, RDPK3SpFSAL35();
                 abstol=1e-7, reltol=1e-4, save_everystep=false, maxiters=100,
                 callback=CallbackSet(UpdateCallback()))

# Extract packed initial condition for the cylinder
polymer = InitialCondition(sol_pack, packing_system, semi_pack)

# Rotate cylinder 90° so axis is along X (horizontal) and curved surface faces down
# Swap Y and Z coordinates
new_coords = copy(polymer.coordinates)
new_coords[2,:], new_coords[3,:] = polymer.coordinates[3,:], polymer.coordinates[2,:]
polymer.coordinates .= new_coords

# ==========================================================================================
# STEP 3: Visualise packed cylinder
# ==========================================================================================

coords = polymer.coordinates
scatter3d(coords[1, :], coords[2, :], coords[3, :],
          markersize=1, aspect_ratio=:equal, label="Cylinder particles")
savefig("packing.png")

# ==========================================================================================
# STEP 4: Build the mold/floor wall boundary
# ==========================================================================================

n_particles_per_dimension = (20, 20, 4)
floor_density = density_cylinder  # match fluid density for Adami extrapolation

# Position floor just 1 particle_spacing below cylinder
cyl_z_min    = minimum(polymer.coordinates[3,:])
floor_origin = cyl_z_min - 1.5 * particle_spacing

floor_particles = RectangularShape(particle_spacing, (20,20,3),
                                   (-0.01, -0.01, floor_origin - 2*particle_spacing);
                                   density=floor_density)

cyl_z_min = minimum(polymer.coordinates[3,:])
floor_z_max = maximum(floor_particles.coordinates[3,:])
println("Gap = ", cyl_z_min - floor_z_max)
println("h   = ", 1.5 * particle_spacing)
println("Contact at t=0? ", (cyl_z_min - floor_z_max) < 1.5 * particle_spacing)                                   

n_boundary = nparticles(floor_particles)

boundary_model = BoundaryModelDummyParticles(
    fill(floor_density, n_boundary),          # per-particle density Vector
    floor_particles.mass,                     # per-particle mass from InitialCondition
    PressureMirroring(),
    smoothing_kernel,
    smoothing_length
)

floor_system = WallBoundarySystem(floor_particles, boundary_model)

# ==========================================================================================
# STEP 5: Build the Total Lagrangian SPH system for the cylinder
# ==========================================================================================

material_polymer = (density=1500.0, E=20e9, nu=0.3 ,beta=0.000, temp=270.0, temp_ref=270.0, cp=900.0 , k=1.0,
                            temp_liq=390.0,h= 1000.0,hardening= 1e9,tmelt=700.0, viscosity=100000.0, yield_stress=150e6)

import PointNeighbors: DictionaryCellList

n_cyl = nparticles(polymer)

cylinder_boundary_model = BoundaryModelDummyParticles(
    fill(1000.0, n_cyl),
    packing_system.mass,
    AdamiPressureExtrapolation(),
    smoothing_kernel,
    smoothing_length
)

# Quasi-static — prescribed displacement, no free fall
polymer.velocity[3, :] .= -0.00    # 5 mm/s downward
nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=600)
#source_terms = SourceTermDamping(damping_coefficient=50.0)
cylinder_system = TotalLagrangianSPHSystem(polymer,
                                           smoothing_kernel,
                                           smoothing_length,
                                           1e6,                     #E-moudulus
                                           material_polymer.nu,    # poisson ratio
                                           material_polymer.beta,             # beta
                                           material_polymer.temp,            # temp — scalar
                                           material_polymer.temp_ref,            # temp_ref — scalar
                                           material_polymer.cp,              # cp — scalar
                                           material_polymer.k,              # k — scalar
                                           material_polymer.temp_liq,              # temp_liq — scalar
                                           material_polymer.h,                  # h — scalar
                                           material_polymer.hardening,           # hardening  — scalar   
                                           material_polymer.tmelt,              # tmelt — scalar
                                           material_polymer.yield_stress;     # yield_stress
                                           acceleration=(0.0, 0.0, -9.8),
                                           boundary_model=cylinder_boundary_model,
                                           #source_terms=source_terms,
                                           #penalty_force=PenaltyForceGanzenmueller(alpha=0.1),
                                           self_interaction_nhs=nhs_template)
                                           #GridNeighborhoodSearch{3}(;
                                              # cell_list=DictionaryCellList{3}()))
# ==========================================================================================
# STEP 6: Semidiscretization and solve
# ==========================================================================================

semi = Semidiscretization(cylinder_system, floor_system;
                          neighborhood_search=GridNeighborhoodSearch{3}(;
                              cell_list=DictionaryCellList{3}(),
                              search_radius=smoothing_length))

gap   = minimum(polymer.coordinates[3,:]) - maximum(floor_particles.coordinates[3,:])
t_fall = sqrt(2 * gap / 9.81)   # time to fall the gap distance

tspan  = (0.0,12*t_fall)    # 3x fall time to see contact + compression

ode = semidiscretize(semi, tspan)

callbacks = CallbackSet(
    UpdateCallback(),
    SolutionSavingCallback(dt=0.005, prefix="cylinder_drop_new2"),
    StepsizeCallback(cfl=0.1),  # TrixiParticles built-in CFL control
    InfoCallback(interval=100)
)

sol = solve(ode, RDPK3SpFSAL35();
            save_everystep=false,
            maxiters=10_000_000,
            callback=callbacks)
