@doc raw"""
    TotalLagrangianSPHSystem(initial_condition, smoothing_kernel, smoothing_length,
                             young_modulus, poisson_ratio;
                             n_clamped_particles=0,
                             clamped_particles=Int[],
                             clamped_particles_motion=nothing,
                             acceleration=ntuple(_ -> 0.0, NDIMS),
                             penalty_force=nothing, viscosity=nothing,
                             source_terms=nothing, boundary_model=nothing,
                             self_interaction_nhs=:default)

System for particles of an elastic structure.

A Total Lagrangian framework is used wherein the governing equations are formulated such that
all relevant quantities and operators are measured with respect to the
initial configuration (O’Connor & Rogers 2021, Belytschko et al. 2000).
See [Total Lagrangian SPH](@ref tlsph) for more details on the method.

# Arguments
- `initial_condition`:  Initial condition representing the system's particles.
- `young_modulus`:      Young's modulus.
- `poisson_ratio`:      Poisson ratio.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).

# Keywords
- `n_clamped_particles` (deprecated): Number of clamped particles that are fixed and not integrated
                         to clamp the structure. Note that the clamped particles must be the **last**
                         particles in the `InitialCondition`. See the info box below.
                         This keyword is deprecated and will be removed in a future release.
                         Instead pass `clamped_particles` with the explicit particle indices to be clamped.
- `clamped_particles`: Indices specifying the clamped particles that are fixed
                       and not integrated to clamp the structure.
- `clamped_particles_motion`: Prescribed motion of the clamped particles.
                    If `nothing` (default), the clamped particles are fixed.
                    See [`PrescribedMotion`](@ref) for details.
- `boundary_model`: Boundary model to compute the hydrodynamic density and pressure for
                    fluid-structure interaction (see [Boundary Models](@ref boundary_models)).
- `penalty_force`:  Penalty force to ensure regular particle position under large deformations
                    (see [`PenaltyForceGanzenmueller`](@ref)).
- `viscosity`:      Artificial viscosity model to stabilize both the TLSPH and the FSI.
                    Currently, only [`ArtificialViscosityMonaghan`](@ref) is supported.
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
- `source_terms`:   Additional source terms for this system. Has to be either `nothing`
                    (by default), or a function of `(coords, velocity, density, pressure)`
                    (which are the quantities of a single particle), returning a `Tuple`
                    or `SVector` that is to be added to the acceleration of that particle.
                    See, for example, [`SourceTermDamping`](@ref).
- `self_interaction_nhs`: Neighborhood search for self-interaction.
                    Being a total Lagrangian formulation, the neighborhood search
                    needs to find neighbors in the initial configuration only.
                    Precomputing concrete neighbor lists is therefore much more efficient
                    than the [`GridNeighborhoodSearch`](@ref) typically used for fluid
                    systems. By default, a [`PrecomputedNeighborhoodSearch`](@ref) is used,
                    which precomputes and stores the neighbor lists at initialization.
                    This NHS is significantly faster (~2x on CPUs, up to 10x on GPUs)
                    than the [`GridNeighborhoodSearch`](@ref).
                    Note that when the default value is used here, the keyword argument
                    `transpose_backend` of the [`PrecomputedNeighborhoodSearch`](@ref)
                    is set to `true` when running on GPUs and `false` on CPUs.
                    Alternatively, a user-defined neighborhood search can be passed here.

!!! note
    If specifying the clamped particles manually (via `n_clamped_particles`),
    the clamped particles must be the **last** particles in the `InitialCondition`.
    To do so, e.g. use the `union` function:
    ```jldoctest; output = false, setup = :(clamped_particles = RectangularShape(0.1, (1, 4), (0.0, 0.0), density=1.0); beam = RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0))
    structure = union(beam, clamped_particles)

    # output
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ InitialCondition                                                                                 │
    │ ════════════════                                                                                 │
    │ #dimensions: ……………………………………………… 2                                                                │
    │ #particles: ………………………………………………… 16                                                               │
    │ particle spacing: ………………………………… 0.1                                                              │
    │ eltype: …………………………………………………………… Float64                                                          │
    │ coordinate eltype: ……………………………… Float64                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    where `beam` and `clamped_particles` are of type [`InitialCondition`](@ref).
"""
struct TotalLagrangianSPHSystem{BM, NDIMS, ELTYPE <: Real, IC, ARRAY1D, ARRAY2D, ARRAY3D,
                                YM, PR, LL, LM, K, PF, V, ST, M, IM, NHS,
                                C, temp, temp_ref, YS} <: AbstractStructureSystem{NDIMS}
    initial_condition   :: IC
    initial_coordinates :: ARRAY2D # Array{ELTYPE, 2}: [dimension, particle]
    # `current_coordinates` contains `u` plus coordinates of the fixed particles
    current_coordinates      :: ARRAY2D # Array{ELTYPE, 2}: [dimension, particle]
    mass                     :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    correction_matrix        :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    pk1_rho2                 :: ARRAY3D # PK1 corrected divided by rho^2: [i, j, particle]
    deformation_grad         :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    #deformation_grad_init    :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    material_density         :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    n_integrated_particles   :: Int64
    young_modulus            :: YM
    poisson_ratio            :: PR
    lame_lambda              :: LL
    lame_mu                  :: LM
    smoothing_kernel         :: K
    smoothing_length         :: ELTYPE
    acceleration             :: SVector{NDIMS, ELTYPE}
    boundary_model           :: BM
    penalty_force            :: PF
    viscosity                :: V
    source_terms             :: ST
    clamped_particles_motion :: M
    clamped_particles_moving :: IM
    self_interaction_nhs     :: NHS
    cache                    :: C
    beta                     :: Float64
    #alpha                    :: alpha
    temp                     :: temp
    temp_ref                 :: temp_ref
    cp                       :: Float64
    k                        :: Float64      
    temp_liq                 :: Float64
    h                        :: Float64
    hardening                :: Float64
    tmelt                    :: Float64
    yield_stress             :: YS
end

function TotalLagrangianSPHSystem(initial_condition, smoothing_kernel, smoothing_length,
                                  young_modulus, poisson_ratio, beta , temp, temp_ref, cp, k, temp_liq,h,hardening,tmelt,yield_stress;
                                  n_clamped_particles=0,
                                  clamped_particles=Int[],
                                  clamped_particles_motion=nothing,
                                  acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                                      ndims(smoothing_kernel)),
                                  penalty_force=nothing, viscosity=nothing,
                                  source_terms=nothing, boundary_model=nothing,
                                  self_interaction_nhs=:default)
    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)
    n_particles = nparticles(initial_condition)

    if ndims(smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    # Make acceleration an SVector
    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    # Backwards compatibility: `n_clamped_particles` is deprecated.
    # Emit a deprecation warning and (if the user didn't supply explicit indices)
    # convert the old `n_clamped_particles` convention to `clamped_particles`.
    if n_clamped_particles != 0
        Base.depwarn("keyword `n_clamped_particles` is deprecated and will be removed in a future release; " *
                     "pass `clamped_particles` (Vector{Int} of indices) instead.",
                     :n_clamped_particles)
        if isempty(clamped_particles)
            clamped_particles = collect((n_particles - n_clamped_particles + 1):n_particles)
        else
            throw(ArgumentError("Either `n_clamped_particles` or `clamped_particles` can be specified, not both."))
        end
    end

    # Handle clamped particles
    if !isempty(clamped_particles)
        @assert allunique(clamped_particles) "`clamped_particles` contains duplicate particle indices"

        n_clamped_particles = length(clamped_particles)
        initial_condition_sorted = deepcopy(initial_condition)
        young_modulus_sorted = copy(young_modulus)
        poisson_ratio_sorted = copy(poisson_ratio)
        beta_sorted = copy(beta)
        move_particles_to_end!(initial_condition_sorted, clamped_particles)
        move_particles_to_end!(young_modulus_sorted, clamped_particles)
        move_particles_to_end!(poisson_ratio_sorted, clamped_particles)
        move_particles_to_end!(beta_sorted, clamped_particles)
    else
        initial_condition_sorted = initial_condition
        young_modulus_sorted = young_modulus
        poisson_ratio_sorted = poisson_ratio
        beta_sorted = beta
    end

    initial_coordinates = copy(initial_condition_sorted.coordinates)
    current_coordinates = copy(initial_condition_sorted.coordinates)
    mass = copy(initial_condition_sorted.mass)
    material_density = copy(initial_condition_sorted.density)
    temp = fill(temp, n_particles)
    temp_ref = fill(temp_ref, n_particles)
    #alpha = fill(alpha, n_particles)
    correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)
    pk1_rho2 = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)
    deformation_grad = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)
    # deformation_grad_init = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)

    n_integrated_particles = n_particles - n_clamped_particles

    lame_lambda = @. young_modulus_sorted * poisson_ratio_sorted /
                     ((1 + poisson_ratio_sorted) *
                      (1 - 2 * poisson_ratio_sorted))
    lame_mu = @. (young_modulus_sorted / 2) / (1 + poisson_ratio_sorted)

    ismoving = Ref(!isnothing(clamped_particles_motion))
    initialize_prescribed_motion!(clamped_particles_motion, initial_condition_sorted,
                                  n_clamped_particles)

    cache = create_cache_tlsph(clamped_particles_motion, initial_condition_sorted)

    return TotalLagrangianSPHSystem(initial_condition_sorted, initial_coordinates,
                                    current_coordinates, mass, correction_matrix,
                                    pk1_rho2, deformation_grad, material_density,
                                    n_integrated_particles, young_modulus_sorted,
                                    poisson_ratio_sorted,
                                    lame_lambda, lame_mu, smoothing_kernel,
                                    smoothing_length, acceleration_, boundary_model,
                                    penalty_force, viscosity, source_terms,
                                    clamped_particles_motion, ismoving,
                                    self_interaction_nhs, cache, beta_sorted , temp, temp_ref, cp, k, temp_liq,
                                    h,hardening,tmelt,yield_stress)
end

# Initialize self-interaction neighborhood search if not provided by the user
# (which means `self_interaction_nhs === :default`). This cannot be done in the constructor
# because we need both the parallelization backend (to optimize the memory layout)
# and the NHS of the `Semidiscretization` (to copy the `PeriodicBox`).
function initialize_self_interaction_nhs(system::TotalLagrangianSPHSystem,
                                         neighborhood_search, parallelization_backend)
    periodic_box = extract_periodic_box(neighborhood_search)

    if system.self_interaction_nhs === :default
        if parallelization_backend isa KernelAbstractions.GPU
            # On GPUs, transpose the backend for an optimized memory access pattern
            template = PrecomputedNeighborhoodSearch{ndims(system)}(transpose_backend=true;
                                                                    periodic_box)
        else
            # On CPUs, use the default configuration to optimize cache hits
            template = PrecomputedNeighborhoodSearch{ndims(system)}(; periodic_box)
        end
    elseif isnothing(system.self_interaction_nhs)
        template = TrivialNeighborhoodSearch{ndims(system)}()
    else
        # User supplied custom NHS — use it as template but still initialize it
        template = system.self_interaction_nhs
    end

    # Create concrete NHS from template
    search_radius = compact_support(system, system)
    self_interaction_nhs = copy_neighborhood_search(template, search_radius,
                                                    nparticles(system))

    # Note that for large numbers of particles, initialization can take a while
    PointNeighbors.initialize!(self_interaction_nhs,
                               system.initial_coordinates, system.initial_coordinates,
                               parallelization_backend=PolyesterBackend())

    # Self-interaction only requires neighbors in the initial configuration,
    # so the self-interaction NHS will never be updated.
    # For the `PrecomputedNeighborhoodSearch`, "freezing" strips all data structures
    # that are only required for updating (and not necessarily GPU-compatible)
    self_interaction_nhs = PointNeighbors.freeze_neighborhood_search(self_interaction_nhs)

    @info "To create the self-interaction neighborhood search of a " *
          "`TotalLagrangianSPHSystem`, a deep copy of the system is created inside " *
          "the `Semidiscretization`. Use `system = semi.systems[i]` to access " *
          "simulation data."

    return TotalLagrangianSPHSystem(system.initial_condition,
                                    system.initial_coordinates,
                                    system.current_coordinates, system.mass,
                                    system.correction_matrix, system.pk1_rho2,
                                    system.deformation_grad, system.material_density,
                                    system.n_integrated_particles, system.young_modulus,
                                    system.poisson_ratio, system.lame_lambda,
                                    system.lame_mu, system.smoothing_kernel,
                                    system.smoothing_length, system.acceleration,
                                    system.boundary_model, system.penalty_force,
                                    system.viscosity, system.source_terms,
                                    system.clamped_particles_motion,
                                    system.clamped_particles_moving,
                                    self_interaction_nhs, system.cache, system.beta, system.temp, system.temp_ref,
                                    system.cp, system.k, system.temp_liq, system.h, system.hardening, system.tmelt, system.yield_stress)
end

extract_periodic_box(::Nothing) = nothing
extract_periodic_box(nhs) = nhs.periodic_box

create_cache_tlsph(::Nothing, initial_condition) = (;)

function create_cache_tlsph(::PrescribedMotion, initial_condition)
    velocity = zero(initial_condition.velocity)
    acceleration = zero(initial_condition.velocity)

    return (; velocity, acceleration)
end

@inline function Base.eltype(::TotalLagrangianSPHSystem{<:Any, <:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

@inline function v_nvariables(system::TotalLagrangianSPHSystem)
    return ndims(system)
end

@inline function v_nvariables(system::TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    return ndims(system) + 1
end

@inline function n_integrated_particles(system::TotalLagrangianSPHSystem)
    system.n_integrated_particles
end

@inline initial_coordinates(system::TotalLagrangianSPHSystem) = system.initial_coordinates

@inline function current_coordinates(u, system::TotalLagrangianSPHSystem)
    return system.current_coordinates
end

@propagate_inbounds function current_coords(system::TotalLagrangianSPHSystem, particle)
    # For this system, the current coordinates are stored in the system directly,
    # so we don't need a `u` array. This function is only to be used in this file
    # when no `u` is available.
    current_coords(nothing, system, particle)
end

@propagate_inbounds function current_velocity(v, system::TotalLagrangianSPHSystem, particle)
    if particle <= system.n_integrated_particles
        return extract_svector(v, system, particle)
    end

    return current_clamped_velocity(v, system, system.clamped_particles_motion, particle)
end

@inline function current_clamped_velocity(v, system, prescribed_motion, particle)
    (; cache, clamped_particles_moving) = system

    if clamped_particles_moving[]
        return extract_svector(cache.velocity, system, particle)
    end

    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_clamped_velocity(v, system, prescribed_motion::Nothing, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_velocity(v, system::TotalLagrangianSPHSystem)
    error("`current_velocity(v, system)` is not implemented for `TotalLagrangianSPHSystem`")
end

@propagate_inbounds function viscous_velocity(v, system::TotalLagrangianSPHSystem, particle)
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@propagate_inbounds function current_density(v, system::TotalLagrangianSPHSystem)
    return current_density(v, system.boundary_model, system)
end

# In fluid-structure interaction, use the "hydrodynamic pressure" of the structure particles
# corresponding to the chosen boundary model.
@propagate_inbounds function current_pressure(v, system::TotalLagrangianSPHSystem)
    return current_pressure(v, system.boundary_model, system)
end

@propagate_inbounds function hydrodynamic_mass(system::TotalLagrangianSPHSystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

@propagate_inbounds function correction_matrix(system, particle)
    extract_smatrix(system.correction_matrix, system, particle)
end

@propagate_inbounds function deformation_gradient(system, particle)
    extract_smatrix(system.deformation_grad, system, particle)
end
@propagate_inbounds function pk1_rho2(system, particle)
    extract_smatrix(system.pk1_rho2, system, particle)
end

@propagate_inbounds function young_modulus(system::TotalLagrangianSPHSystem, particle)
    return young_modulus(system, system.young_modulus, particle)
end

@inline function young_modulus(::TotalLagrangianSPHSystem, young_modulus, particle)
    return young_modulus
end

@propagate_inbounds function young_modulus(::TotalLagrangianSPHSystem,
                                           young_modulus::AbstractVector, particle)
    return young_modulus[particle]
end

@propagate_inbounds function poisson_ratio(system::TotalLagrangianSPHSystem, particle)
    return poisson_ratio(system, system.poisson_ratio, particle)
end

@inline function poisson_ratio(::TotalLagrangianSPHSystem, poisson_ratio, particle)
    return poisson_ratio
end

@inline function poisson_ratio(::TotalLagrangianSPHSystem,
                               poisson_ratio::AbstractVector, particle)
    return poisson_ratio[particle]
end

function initialize!(system::TotalLagrangianSPHSystem, semi)
    (; correction_matrix) = system

    initial_coords = initial_coordinates(system)

    density_fun(particle) = system.material_density[particle]

    # Calculate correction matrix
    compute_gradient_correction_matrix!(correction_matrix, system, initial_coords,
                                        density_fun, semi)
end

function update_positions!(system::TotalLagrangianSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; current_coordinates, clamped_particles_motion) = system

    # `current_coordinates` stores the coordinates of both integrated and clamped particles.
    # Copy the coordinates of the integrated particles from `u`.
    @threaded semi for particle in each_integrated_particle(system)
        for i in 1:ndims(system)
            current_coordinates[i, particle] = u[i, particle]
        end
    end

    apply_prescribed_motion!(system, clamped_particles_motion, semi, t)
end

function apply_prescribed_motion!(system::TotalLagrangianSPHSystem,
                                  prescribed_motion::PrescribedMotion, semi, t)
    (; clamped_particles_moving, current_coordinates, cache) = system
    (; acceleration, velocity) = cache

    prescribed_motion(current_coordinates, velocity, acceleration, clamped_particles_moving,
                      system, semi, t)

    return system
end

function apply_prescribed_motion!(system::TotalLagrangianSPHSystem, ::Nothing, semi, t)
    return system
end

function update_quantities!(system::TotalLagrangianSPHSystem, v, u, v_ode, u_ode, semi, t, dt=1e-5)
    # Precompute PK1 stress tensor
    @trixi_timeit timer() "stress tensor" compute_pk1_corrected!(system, v, semi, dt)
    
    return system
end

function update_boundary_interpolation!(system::TotalLagrangianSPHSystem, v, u,
                                        v_ode, u_ode, semi, t)
    (; boundary_model) = system

    # Only update boundary model
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)
end

@inline function compute_pk1_corrected!(system, v, semi, dt=1e-6)
    (; deformation_grad, pk1_rho2, material_density) = system

    calc_deformation_grad!(deformation_grad, system, v, dt, false, semi)

    # F1 = deformation_grad[:,:,1]
    # println("det(F[1]) = ", det(F1))
    # println("F[1] = ", F1)

    @threaded semi for particle in eachparticle(system)
        pk1_particle = @inbounds pk1_stress_tensor(system, particle)

        pk1_particle_corrected = pk1_particle *
                                 @inbounds correction_matrix(system, particle)
        rho2_inv = 1 / @inbounds material_density[particle]^2

        for j in 1:ndims(system), i in 1:ndims(system)
            # Precompute PK1 / rho^2 to avoid repeated divisions in the interaction loop
            @inbounds pk1_rho2[i, j, particle] = pk1_particle_corrected[i, j] * rho2_inv
        end
    end
end

@inline function update_properties!(system,alpha, semi)
    (; temp, temp_ref, yield_stress, hardening, tmelt, viscosity) = system
    viscosity = fill(viscosity,length(temp))
    H = 1/(tmelt.-temp_ref[1])         #thermal softening modulus
    yield_stress = yield_stress .* (1 .- H.*(temp .- temp_ref) ) 
    hardening = hardening.*(1 .- H .*(temp .- temp_ref) )*alpha
    E=100000 # activation energy for flow
    R=8.314  #gas constant
    viscosity = viscosity .* exp.(E ./(R.* temp))
    return yield_stress, hardening, viscosity
end

@inline function update_temperature_sph!(system, dt, ext_heat ,particle_spacing, bound_coordinate, semi)
    # Unpack system properties
    (; mass, material_density, temp, temp_ref, cp, k, current_coordinates, smoothing_length) = system

    # Temporary storage for ΔT
    dT = zeros(length(temp_ref))

    dx = particle_spacing

    for i in 1:length(temp)
        if current_coordinates[bound_coordinate[1],i] < bound_coordinate[2]
            dq = ext_heat   # W/m²
            dT[i] += dq / (material_density[i] * cp * dx)
        end
    end

    # Loop over all particles and neighbors (SPH)
    initial_coords = initial_coordinates(system)
    foreach_point_neighbor(system, system, initial_coords, initial_coords,
                           semi) do particle, neighbor, r, initial_distance2


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
    
    #println("dE = ", sum(dT .* mass ./ material_density))

    return temp
end

@inline function thermomechanical_loop(system, temp_mold, y_mold ,particle_spacing, bound_coordinate ,dt, vel,fixed, alpha,F_total_mold, v_mold, semi)
    (;temp_liq, temp, h) = system
    
    ys, hard, vis = update_properties!(system,alpha, semi)

    if y_mold <= 0.05 
        # println("ys: ",ys[1:5])
        #println("hard:",hard)
        # println("vis:",vis[1:5])
        #println("alpha1:",alpha)
        vel,coor,alpha,F_total_mold = update_v_x(system, dt, y_mold ,particle_spacing ,temp_liq,vel, fixed,ys, hard, vis, alpha,F_total_mold, v_mold, semi)
        system.current_coordinates .= coor
    end

    if y_mold > 0.05
        q = 0
        update_temperature_sph!(system, dt, q ,particle_spacing, bound_coordinate, semi)
    else
        temp_t = 0.0
        count = 0.0
        for i in 1:length(temp)
            if system.current_coordinates[bound_coordinate[1],i] < bound_coordinate[2]
               temp_t += temp[i]
               count +=1
            end
        end
        temp_avg = sum(temp_t) / count 
        q = h*(temp_mold-temp_avg)
        update_temperature_sph!(system, dt, q ,particle_spacing, bound_coordinate, semi)
    end

    return vel, alpha, F_total_mold

end


@inline function update_v_x(system, dt, y_mold, particle_spacing ,temp_liq, vel, fixed,ys, hard, vis, alpha,F_total_mold, v_mold ,semi)
    (;current_coordinates, temp, young_modulus, initial_coordinates) = system

    temp_avg = sum(temp) / length(temp) 

    stress = zeros(eltype(system), size(current_coordinates,1), size(current_coordinates,1), size(current_coordinates,2))

    if temp_avg > temp_liq
        stress = viscous_stress!(system,vis,dt,fixed,semi)
    else
        stress, alpha = elastic_stress!(system,ys,hard,vis,dt,alpha,fixed,semi)
    end

    #k_n = 5.0 * young_modulus / particle_spacing
    k_n = 1e7
    
    acceleration,F_total_mold = momentum(system, y_mold, k_n ,stress, fixed,vel,F_total_mold, v_mold,particle_spacing, semi)

    #println("acc:",acceleration)
    #println("vel:",vel)
    # for i in fixed
    #     acceleration[:, i] .= 0.0
    #     vel[:, i] .= 0.0
    #     #current_coordinates[:, i] .= initial_coordinates[:, i]
    # end

    vel .+= acceleration.*dt

    # for particle in length(temp)
    #     vel[1, particle] += 1e-6 * randn()
    # end

    current_coordinates += vel.*dt

    return vel,current_coordinates, alpha,F_total_mold
end

@inline function momentum(system, y_mold, k_n, stress, fixed,vel,F_total_mold, v_mold,particle_spacing, semi)
    (;mass, material_density, current_coordinates, smoothing_length, young_modulus) = system

    _acceleration = zeros(eltype(system), size(current_coordinates,1), size(current_coordinates,2))
    initial_coords = initial_coordinates(system)

    foreach_point_neighbor(system, system, initial_coords, initial_coords,semi) do particle, neighbor, r, distance2

        @views r_vec =
            current_coordinates[:, particle] .-
            current_coordinates[:, neighbor]     
            
        r_vec = convert.(eltype(system), r_vec)
        r_norm = sqrt(TrixiParticles.dot(r_vec, r_vec)) + 1e-12  # epsilon in denominator

        ##Skip too-close particles (optional)
        if r_norm < 1e-1
            return
        end

        distance2 = TrixiParticles.dot(r_vec, r_vec)
        
        distance2 < eps(smoothing_length^2) && return
        r = sqrt(distance2)

        # Kernel gradient
        grad_kernel = smoothing_kernel_grad(system, r_vec,
                                            r, particle)
        gradW = grad_kernel

        #print("grad_kernel: ",grad_kernel)

        stress_term = stress[:,:,particle] / material_density[particle]^2 +
            stress[:,:,neighbor] / material_density[neighbor]^2

        _acceleration[:, particle] .+= mass[neighbor] * (stress_term * gradW)

    end

    normal = SVector(0.00, -1.0)   #top_mold
    @threaded semi for particle in eachparticle(system)
        gap = y_mold - current_coordinates[2,particle]
        # if particle==1000
        #     println("gap = ", gap)
        # end
        if gap < 0.0
            gamma_n = 10 * sqrt(k_n * mass[particle])
            #delta = max(gap, -0.1*smoothing_length)
            #delta = max(gap, -0.0000105)
            f_contact = -k_n * gap * normal
            f_contact -= gamma_n * dot(vel[:,particle]-v_mold*normal, normal) * normal
            F_total_mold += dot(f_contact, normal)
            _acceleration[:,particle] .+= f_contact / mass[particle]
            gamma_lat = 0.1 * gamma_n  # smaller than vertical damping
            _acceleration[1,particle] -= gamma_lat * vel[1,particle] / mass[particle]
        end

        r_min = particle_spacing # particle spacing or slightly less
        if !(particle in fixed)
            dist_to_b = current_coordinates[2, particle] - 0.00  
            if dist_to_b < r_min
                gamma_n = 10 * sqrt(k_n * mass[particle])
                normal_bottom = SVector(0.00, 1.0)
                F_rep = -1e2 * dist_to_b * normal_bottom
                F_rep -= gamma_n * dot(vel[:,particle], normal_bottom) * normal_bottom
            else
                F_rep = 0
            end
            _acceleration[:,particle] .+= F_rep / mass[particle]
        end
    end

    return _acceleration, F_total_mold
end

@inline function viscous_stress!(system,vis,dt,fixed, semi)
    (; deformation_grad, young_modulus, poisson_ratio) = system

    v_vis = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2)
                    ,size(deformation_grad,3))
    F,b,d = calc_deformation_grad!(deformation_grad, system,dt,fixed, semi)
    K = young_modulus/(3-6*poisson_ratio)

    @threaded semi for particle in eachparticle(system)
        det_F = max(det(F[:,:,particle]), 1e-6)
        J= sqrt(det_F)
        dev_d = zeros(eltype(system), size(d,1), size(d,2))
        dev_d = d[:,:,particle] - 1/2* (tr(d[:,:,particle]))*I
        for j in 1:ndims(system), i in 1:ndims(system)
            # Precompute PK1 / rho^2 to avoid repeated divisions in the interaction loop
            @inbounds v_vis[i, j, particle] = K*log(J)*I+2*vis[particle]*dev_d[i, j]
        end
    end

    return v_vis
end

@inline function elastic_stress!(system,ys, hard, vis, dt, _alpha ,fixed, semi)
    (; deformation_grad,young_modulus,poisson_ratio,temp, tmelt, hardening, material_density, cp, temp) = system

    v_elas = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2)
                    ,size(deformation_grad,3))
    
    F,b,d = calc_deformation_grad!(deformation_grad, system,dt, fixed, semi)
    
    Fp = zeros(eltype(system), size(F,1), size(F,2) ,size(F,3))

    mu = young_modulus/(2+2*poisson_ratio)
    K = young_modulus/(3-6*poisson_ratio)


    @threaded semi for particle in eachparticle(system)
        Fp[:,:,particle] .= Matrix{Float64}(I, ndims(system), ndims(system))
        detF = det(F[:,:,particle])

        J = max(detF, 1e-6)

        dev_b = zeros(eltype(system), size(b,1), size(b,2))
        dev_b = b[:,:,particle] - 1/2* (tr(b[:,:,particle]))*I       

        @inbounds v_elas[:, :, particle] .= K*log(J)*I+mu*dev_b

        yf = sqrt(1.5)*sqrt(sum((v_elas[:, :, particle]- 1/2* (tr(v_elas[:, :, particle]))*I).^ 2))- (ys[particle]-hard[particle])
        #yf_max = 10000000
        #yf = max(yf, 0.0)
        #yf = min(yf, yf_max)    
        #println("yf_value", yf)
        #print("particle: ",particle)
        # println("J = ", J)
        # println("norm(dev_b) = ", norm(dev_b))
        # println("hardening = ", hard[particle])
        # println("alpha = ", _alpha[particle])


        if yf > 1e-6
            if temp[particle] < 0.5*tmelt
                strain_rate = yf/(3*mu + hardening)     
            else
                strain_rate = yf*dt/max(vis[particle],1e-8)

                # # Isochoric part of b
                # J_b = max(det(b[:,:,particle]), 1e-6)
                # be_bar = zeros(eltype(system), size(b,1), size(b,2),1)
                # be_bar = b[:,:,particle] / J_b^(1/2)   ##iso_choric b
                # dev_be = be_bar - 1/2* (tr(be_bar))*I
                # dev_be .= strain_rate* dt *dev_be 
                # det_be = max(det(be_bar), 1e-6)
                # J_e = sqrt(det_be)
                # @inbounds v_elas[:, :, particle] .= K*log(J_e)*I+mu*dev_be
            end
            
            dev_v_elas = v_elas[:, :, particle] - 1/2* (tr(v_elas[:, :, particle]))*I
            norm_dev = sqrt(sum(dev_v_elas .* dev_v_elas))
            n = dev_v_elas/norm_dev
            dFp = strain_rate * n * dt
            Fp_old = Fp[:,:,particle]
            Fp[:,:,particle] += dFp * Fp[:,:,particle]
            Fe = F[:,:,particle] * inv(Fp[:,:,particle])
            J = det(Fe)
            be = Fe * Fe'   # left Cauchy-Green
            dev_be = be - 1/2*tr(be)*I
            v_elas[:, :, particle] .= K*log(J)*I + mu*dev_be 
            alpha_dot = strain_rate
            _alpha[particle] += alpha_dot  

            depsilon = 0.5 * ((Fp[:,:,particle] - Fp_old) * inv(Fp_old) + ((Fp[:,:,particle] - Fp_old) * inv(Fp_old))')
            plastic_work = sum(v_elas[:,:,particle] .* depsilon)
            beta = 0.9
            temp[particle] += beta * plastic_work / (material_density[particle]*cp) * dt
        end
    end
    return v_elas, _alpha
end
    

@inline function calc_deformation_grad!(deformation_grad, system, v, dt, fixed, semi)
    (; mass, material_density, temp, temp_ref) = system

    b = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2), size(deformation_grad,3))
    d = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2), size(deformation_grad,3))
    velocity_grad = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2), size(deformation_grad,3))
   
    for i in 1:length(temp)
        deformation_grad[:,:,i] .= zeros(eltype(system), ndims(system), ndims(system))
        b[:,:,i] .= Matrix{Float64}(I, ndims(system), ndims(system))
        d[:,:,i] .= Matrix{Float64}(I, ndims(system), ndims(system))
    end

    # Loop in INITIAL configuration — standard Total Lagrangian SPH
    initial_coords = initial_coordinates(system)
    nhs = get_neighborhood_search(system, system, semi)
    PointNeighbors.foreach_point_neighbor(
        initial_coords, initial_coords, nhs;
        parallelization_backend=PolyesterBackend()
    ) do particle, neighbor, pos_diff_initial, initial_distance
        initial_distance^2 < eps(initial_smoothing_length(system)^2) && return

        volume = @inbounds mass[neighbor] / material_density[neighbor]
        pos_diff_current = @inbounds current_coords(system, particle) -
                                     current_coords(system, neighbor)
        pos_diff_current = convert.(eltype(system), pos_diff_current)
        vel_diff = v[:, particle] - v[:, neighbor]

        grad_kernel = smoothing_kernel_grad(system, pos_diff_initial,
                                            initial_distance, particle)
        L = @inbounds correction_matrix(system, particle)

        result   = volume * pos_diff_current * grad_kernel' * L'
        result_v = volume * vel_diff         * grad_kernel' * L'

        for j in 1:ndims(system), i in 1:ndims(system)
            @inbounds deformation_grad[i, j, particle] -= result[i, j]
            @inbounds velocity_grad[i, j, particle]    -= result_v[i, j]
        end
    end

    # Add identity and compute b, d
    for particle in eachparticle(system)
        F  = deformation_grad[:,:,particle]
        b[:,:,particle] = F * F'
        _L = velocity_grad[:,:,particle]
        d[:,:,particle] = 0.5 * (_L + _L')
    end

    return deformation_grad, b, d
end

# First Piola-Kirchhoff stress tensor
@propagate_inbounds function pk1_stress_tensor(system, particle)
    (; lame_lambda, lame_mu) = system

    F = deformation_gradient(system, particle)
    S = pk2_stress_tensor(F, lame_lambda, lame_mu, particle)

    return F * S
end

# Second Piola-Kirchhoff stress tensor
@propagate_inbounds function pk2_stress_tensor(F, lame_lambda::AbstractVector,
                                               lame_mu::AbstractVector, particle)

    # Compute the Green-Lagrange strain
    E = (transpose(F) * F - I) / 2

    return lame_lambda[particle] * tr(E) * I + 2 * lame_mu[particle] * E
end

# Second Piola-Kirchhoff stress tensor
@inline function pk2_stress_tensor(F, lame_lambda, lame_mu, particle)

    # Compute the Green-Lagrange strain
    E = (transpose(F) * F - I) / 2

    return lame_lambda * tr(E) * I + 2 * lame_mu * E
end

function write_u0!(u0, system::TotalLagrangianSPHSystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices((ndims(system), each_integrated_particle(system)))
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

function write_v0!(v0, system::TotalLagrangianSPHSystem)
    (; initial_condition, boundary_model) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices((ndims(system), each_integrated_particle(system)))
    copyto!(v0, indices, initial_condition.velocity, indices)

    write_v0!(v0, boundary_model, system)

    return v0
end

function write_v0!(v0, model, system::TotalLagrangianSPHSystem)
    return v0
end

function write_v0!(v0, ::BoundaryModelDummyParticles{ContinuityDensity},
                   system::TotalLagrangianSPHSystem)
    (; cache) = system.boundary_model
    (; initial_density) = cache

    for particle in each_integrated_particle(system)
        # Set particle densities
        v0[ndims(system) + 1, particle] = initial_density[particle]
    end

    return v0
end

function restart_with!(system::TotalLagrangianSPHSystem, v, u)
    for particle in each_integrated_particle(system)
        system.current_coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
    end

    # This is dispatched in the boundary system.jl file
    restart_with!(system, system.boundary_model, v, u)
end

# An explanation of these equation can be found in
# J. Lubliner, 2008. Plasticity theory.
# See here below Equation 5.3.21 for the equation for the equivalent stress.
# The von-Mises stress is one form of equivalent stress, where sigma is the deviatoric stress.
# See pages 32 and 123.
function von_mises_stress(system)
    von_mises_stress_vector = zeros(eltype(system.pk1_rho2), nparticles(system))

    @threaded default_backend(von_mises_stress_vector) for particle in
                                                           each_integrated_particle(system)
        von_mises_stress_vector[particle] = von_mises_stress(system, particle)
    end

    return von_mises_stress_vector
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function von_mises_stress(system, particle::Integer)
    F= deformation_gradient(system, particle)
    J = det(F)
    P = pk1_rho2(system, particle) * system.material_density[particle]^2
    sigma = (1.0 / J) * P * F'

    # Calculate deviatoric stress tensor
    s = sigma - (1.0 / 3.0) * tr(sigma) * I

    return sqrt(3.0 / 2.0 * sum(s .^ 2))
end

# An explanation of these equation can be found in
# J. Lubliner, 2008. Plasticity theory.
# See here page 473 for the relation between the `pk1`, the first Piola-Kirchhoff tensor,
# and the Cauchy stress.
function cauchy_stress(system::TotalLagrangianSPHSystem)
    NDIMS = ndims(system)

    cauchy_stress_tensors = zeros(eltype(system.pk1_rho2), NDIMS, NDIMS,
                                  nparticles(system))

    @threaded default_backend(cauchy_stress_tensors) for particle in
                                                         each_integrated_particle(system)
        F = deformation_gradient(system, particle)
        J = det(F)
        P = pk1_rho2(system, particle) * system.material_density[particle]^2
        sigma = (1.0 / J) * P * F'
        cauchy_stress_tensors[:, :, particle] = sigma
    end

    return cauchy_stress_tensors
end

function kirchhoff_stress(system::TotalLagrangianSPHSystem)
    NDIMS = ndims(system)

    kirchhoff_stress_tensors = zeros(eltype(system.pk1_rho2), NDIMS, NDIMS,
                                  nparticles(system))

    @threaded default_backend(kirchhoff_stress_tensors) for particle in
                                                         each_integrated_particle(system)
        F = deformation_gradient(system, particle)
        J = det(F)
        P = pk1_rho2(system, particle) * system.material_density[particle]^2
        sigma = (1.0 / J) * P * F'
        kirchhoff_stress_tensors[:, :, particle] = sigma
    end

    return kirchhoff_stress_tensors
end

function calculate_dt(v_ode, u_ode, cfl_number, system::TotalLagrangianSPHSystem, semi)
    # TODO variable smoothing length
    smoothing_length_ = initial_smoothing_length(system)

    # Compute bulk modulus from Young's modulus and Poisson's ratio.
    # See the table at the end of https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
    # TODO Should we compute the sound speed per particle and then use the maximum?
    E = maximum(system.young_modulus)
    K = E / (ndims(system) * (1 - 2 * maximum(system.poisson_ratio)))

    # Newton–Laplace equation
    sound_speed = sqrt(K / minimum(system.material_density))

    # According to Eq. 19 in Sun et al. (2021) "An accurate FSI-SPH modeling..."
    return cfl_number * smoothing_length_ / sound_speed
end

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline function viscosity_model(system::TotalLagrangianSPHSystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::Union{AbstractFluidSystem, OpenBoundarySystem},
                                 neighbor_system::TotalLagrangianSPHSystem)
    return neighbor_system.boundary_model.viscosity
end

function system_data(system::TotalLagrangianSPHSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    (; mass, material_density, deformation_grad, young_modulus,
     poisson_ratio, lame_lambda, lame_mu) = system

    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    initial_coordinates_ = initial_coordinates(system)
    velocity = [current_velocity(v, system, particle) for particle in eachparticle(system)]
    acceleration = system_data_acceleration(dv, system, system.clamped_particles_motion)
    pk1_corrected = [pk1_rho2(system, particle) * system.material_density[particle]^2
                     for particle in eachparticle(system)]

    return (; coordinates, initial_coordinates=initial_coordinates_, velocity, mass,
            material_density, deformation_grad, pk1_corrected, young_modulus, poisson_ratio,
            lame_lambda, lame_mu, acceleration)
end

function system_data_acceleration(dv, system::TotalLagrangianSPHSystem, ::Nothing)
    return dv
end

function system_data_acceleration(dv, system::TotalLagrangianSPHSystem, ::PrescribedMotion)
    clamped_particles = (n_integrated_particles(system) + 1):nparticles(system)
    return hcat(dv, view(system.cache.acceleration, :, clamped_particles))
end

function available_data(::TotalLagrangianSPHSystem)
    return (:coordinates, :initial_coordinates, :velocity, :mass, :material_density,
            :deformation_grad, :pk1_corrected, :young_modulus, :poisson_ratio,
            :lame_lambda, :lame_mu, :acceleration)
end

function Base.show(io::IO, system::TotalLagrangianSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "TotalLagrangianSPHSystem{", ndims(system), "}(")
    print(io, "", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.boundary_model)
    print(io, ", ", system.penalty_force)
    print(io, ", ", system.viscosity)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::TotalLagrangianSPHSystem)
    @nospecialize system # reduce precompilation time

    function display_param(param)
        if param isa AbstractVector
            min_val = round(minimum(param), digits=3)
            max_val = round(maximum(param), digits=3)
            return "min = $(min_val), max = $(max_val)"
        else
            return string(param)
        end
    end

    if get(io, :compact, false)
        show(io, system)
    else
        n_clamped_particles = nparticles(system) - n_integrated_particles(system)

        summary_header(io, "TotalLagrangianSPHSystem{$(ndims(system))}")
        summary_line(io, "total #particles", nparticles(system))
        summary_line(io, "#clamped particles", n_clamped_particles)
        summary_line(io, "Young's modulus", display_param(system.young_modulus))
        summary_line(io, "Poisson ratio", display_param(system.poisson_ratio))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "boundary model", system.boundary_model)
        summary_line(io, "penalty force", system.penalty_force)
        summary_line(io, "viscosity", system.viscosity)
        summary_footer(io)
    end
end
