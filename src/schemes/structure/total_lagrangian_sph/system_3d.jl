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
@inline function update_temperature_sph3d!(system, dt, ext_heat, particle_spacing, bound_coordinate, semi;
                                           artificial_diffusion_coeff=0.005,
                                           temperature_increment_limit=0.1)
    # Unpack system properties
    (; mass, material_density, temp, temp_ref, cp, k, current_coordinates, smoothing_length) = system

    # Temporary storage for ΔT
    dT = zeros(length(temp_ref))

    dx = particle_spacing

    rho_wall = material_density[1]
    for i in 1:length(temp)
        if current_coordinates[bound_coordinate[1],i] < bound_coordinate[2]
            dT[i] += ext_heat / (rho_wall * cp * dx)
        end
    end

    # Loop over all particles and neighbors (SPH)
    initial_coords = initial_coordinates(system)

    # --- Pass 1: Shepard normalization factor ---
    # W_sum[i] = sum_j V_j * W(r_ij).  For a fully-embedded particle this
    # approaches 1; near the wall it drops to ~0.5 because the kernel support
    # is truncated by the boundary.  Dividing the conduction flux by W_sum
    # restores consistency (Randles & Libersky 1996 renormalization).
    W_sum = zeros(eltype(temp), length(temp))
    foreach_point_neighbor(system, system, initial_coords, initial_coords,
                           semi) do particle, neighbor, r_nb, _
        rho_nb  = material_density[1]
        vol_nb  = mass[neighbor] / rho_nb
        r_scalar = sqrt(TrixiParticles.dot(r_nb, r_nb))
        W_sum[particle] += vol_nb * smoothing_kernel(system, r_scalar, particle)
    end
    clamp!(W_sum, 0.1, Inf)   # guard against near-zero at isolated boundary nodes

    # --- Pass 2: SPH thermal conduction with truncated-support renormalization ---
    foreach_point_neighbor(system, system, initial_coords, initial_coords,
                           semi) do particle, neighbor, r, initial_distance2


        # Particle volumes
        rho = material_density[1]
        volume = @inbounds mass[neighbor] / rho

        ##artificial thermal diffusion to reduce oscillations
        temp[particle] += artificial_diffusion_coeff * (temp[neighbor] - temp[particle])

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
                (temp[particle] - temp[neighbor]) *
                val /(r2)

        # Truncated-support renormalization: scale by 1/W_sum to recover
        # Laplacian consistency near the wall where kernel support is cut off
        dT[particle] += flux / W_sum[particle]

    end

    #println("max dT = ", maximum(abs.(dT)))
    #println("nonzero dT count = ", count(!iszero, dT))

    # # Update temperature with flux limiter
    dT_max = temperature_increment_limit
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

# Compute the thermal RHS dT/dt for use as an ODE state variable.
# Writes dT/dt into dv[ndims(system)+1, particle] for each integrated particle.
# No mutation of system.temp — the ODE solver integrates the temperature row of v.
# Call this at the end of kick_implicit_visible! after all momentum terms are set.
@inline function thermal_rhs_sph3d!(system, dv, v, ext_heat, particle_spacing, bound_coordinate, semi)
    (; mass, material_density, temp, cp, k, current_coordinates, smoothing_length) = system

    NDIMS      = ndims(system)
    n_particles = nparticles(system)
    rho        = material_density[1]
    dT         = zeros(eltype(system), n_particles)

    # Boundary heat flux source term
    dx = particle_spacing
    for i in 1:n_particles
        if current_coordinates[bound_coordinate[1], i] < bound_coordinate[2]
            dT[i] += ext_heat / (rho * cp * dx)
        end
    end

    initial_coords = initial_coordinates(system)

    # Shepard normalization for truncated-kernel consistency near boundaries
    W_sum = zeros(eltype(system), n_particles)
    foreach_point_neighbor(system, system, initial_coords, initial_coords,
                           semi) do particle, neighbor, r_nb, _
        vol_nb   = mass[neighbor] / rho
        r_scalar = sqrt(TrixiParticles.dot(r_nb, r_nb))
        W_sum[particle] += vol_nb * smoothing_kernel(system, r_scalar, particle)
    end
    clamp!(W_sum, 0.1, Inf)

    # SPH thermal conduction: dT/dt += (k / rho*cp) * Laplacian(T)
    foreach_point_neighbor(system, system, initial_coords, initial_coords,
                           semi) do particle, neighbor, r_nb, _
        volume = mass[neighbor] / rho

        @views r_vec = current_coordinates[:, particle] .- current_coordinates[:, neighbor]
        r_vec = convert.(eltype(system), r_vec)

        r2 = TrixiParticles.dot(r_vec, r_vec)
        r2 < eps(smoothing_length^2) && return
        r_dist = sqrt(r2)

        grad_kernel = smoothing_kernel_grad(system, r_vec, r_dist, particle)
        _eps = 1e-3
        h2   = smoothing_length^2

        val = TrixiParticles.dot(r_vec, grad_kernel)
        # val should be ≤ 0 for a radial kernel; clamp positive roundoff to zero
        val = min(val, zero(val))

        flux = volume * k / (rho * cp) *
               (temp[particle] - temp[neighbor]) * val / (r2 + _eps * h2)
        dT[particle] += flux / W_sum[particle]
    end

    # Write dT/dt into the temperature row of dv
    for particle in each_integrated_particle(system)
        dv[NDIMS + 1, particle] += dT[particle]
    end

    return dv
end

@inline function thermomechanical_loop3d(system, temp_mold, y_mold ,particle_spacing, bound_coordinate ,dt, vel,fixed, alpha,F_total_mold, v_mold, semi)
    (;temp_liq, temp, h) = system
    
    ys, hard, vis = update_properties!(system,alpha, semi)

    if y_mold <= 0.012 
        vel,coor,alpha,F_total_mold = update_v_x3d(system, dt, y_mold ,particle_spacing ,temp_liq,vel, fixed,ys, hard, vis, alpha,F_total_mold, v_mold, semi)
        system.current_coordinates .= coor
    end

    if y_mold > 0.012
        q = 0
        update_temperature_sph3d!(system, dt, q ,particle_spacing, bound_coordinate, semi)
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
        update_temperature_sph3d!(system, dt, q ,particle_spacing, bound_coordinate, semi)
    end

    return vel, alpha, F_total_mold

end


@inline function update_v_x3d(system, dt, y_mold, particle_spacing ,temp_liq, vel, fixed,ys, hard, vis, alpha, F_total_mold, v_mold ,semi)
    (;current_coordinates, temp, young_modulus, initial_coordinates,poisson_ratio,mass, material_density) = system

    temp_avg = sum(temp) / length(temp) 

    #F_total_mold = 0.0

    wave_speed = 0.0

    stress,F_total_mold = zeros(eltype(system), size(current_coordinates,1), size(current_coordinates,1), size(current_coordinates,2))

    if temp_avg > temp_liq
        stress, wave_speed = viscous_stress3d!(system,vis,dt,fixed,semi)
    else
        stress, alpha = elastic_stress3d!(system,ys,hard,vis,dt,alpha,fixed,semi)
    end

    #k_n = 5.0 * young_modulus / particle_spacing
    #k_n = 100000
    
    acceleration, F_total_mold = momentum3d(system, y_mold ,stress,vel, v_mold,particle_spacing,F_total_mold, wave_speed, semi)

    vel .+= acceleration.*dt

    # @threaded semi for particle in eachparticle(system)
    #     volume = mass[particle]/material_density[particle]
    #     F_total_mold += stress[3,3,particle]*(volume)
    # end

    # total_volume = sum((mass)./(material_density))

    # F_total_mold /= total_volume

    # for particle in fixed
    #     acceleration[:, particle] .= 0.0
    #     vel[:, particle] .= 0.0
    #     current_coordinates[:, particle] .= initial_coordinates[:, particle]
    # end

    current_coordinates += vel.*dt

    return vel,current_coordinates, alpha, F_total_mold
end

@inline function momentum3d(system, y_mold, stress,vel, v_mold,particle_spacing,F_total_mold, wave_speed , semi)
    (;mass, material_density, current_coordinates, smoothing_length, young_modulus, poisson_ratio) = system

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

    normal = SVector(0.00,0.00, 1.0)   #top_mold
    
    @threaded semi for particle in eachparticle(system)
        t_star = 0.0
        
        if wave_speed==0.0
            wave_speed = sqrt(young_modulus*(1.0-poisson_ratio) / (material_density[particle]*(1.0+poisson_ratio)*(1.0-2.0*poisson_ratio)))
        else
            wave_speed = wave_speed/material_density[particle]
        end
            # gap = y_mold - current_coordinates[3,particle]
        # if gap < 0.0
        #     gamma_n = 10 * sqrt(k_n * mass[particle])
        #     #delta = max(gap, -0.1*smoothing_length)
        #     #delta = max(gap, -0.0000105)
        #     f_contact = -k_n * gap * normal
        #     f_contact -= gamma_n * dot(vel[:,particle]-v_mold*normal, normal) * normal
        #     F_total_mold += dot(f_contact, normal)
        #     _acceleration[:,particle] .+= f_contact / mass[particle]
        #    # gamma_lat = 0.1 * gamma_n  # smaller than vertical damping
        #     #_acceleration[1,particle] -= gamma_lat * vel[1,particle] / mass[particle]
        # end
        z_i = current_coordinates[3,particle]
        v_rel = dot(vel[:,particle] .- v_mold, normal)
        if z_i >= y_mold - 2*smoothing_length
            P = stress[:,:,particle]
            traction = P * normal
            if v_rel < 0
                Z = material_density[particle] * wave_speed
                t_star = traction - Z*(dot(vel[:,particle],normal)-v_mold)*normal
                _acceleration[:,particle] += t_star ./ mass[particle]
            end     
            # @show t_star
            # @show v_rel
            # @show wave_speed
            # @show traction
            

        end   
    end

    return _acceleration,F_total_mold
end

@inline function viscous_stress3d!(system,vis,dt,fixed, semi)
    (; deformation_grad, young_modulus, poisson_ratio) = system
    #println("viscous_Regime")
    
    v_vis = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2)
                    ,size(deformation_grad,3))
    F,b,d = calc_deformation_grad3d!(deformation_grad, system,dt,fixed, semi)
    K = young_modulus/(3-6*poisson_ratio)
    wave_speed = K

    @threaded semi for particle in eachparticle(system)
        det_F = max(det(F[:,:,particle]), 1e-6)
        J= sqrt(det_F)
        dev_d = zeros(eltype(system), size(d,1), size(d,2) , size(d,3))
        dev_d = d[:,:,particle] - 1/3* (tr(d[:,:,particle]))*I
        for j in 1:ndims(system), i in 1:ndims(system)
            # Precompute PK1 / rho^2 to avoid repeated divisions in the interaction loop
            @inbounds v_vis[i, j, particle] = K*log(J)*I+2*vis[particle]*dev_d[i, j]
        end
    end

    return v_vis, wave_speed
end

@inline function elastic_stress3d!(system,ys, hard, vis, dt, _alpha ,fixed, semi)
    (; deformation_grad,young_modulus,poisson_ratio,temp, tmelt, hardening, material_density, cp, temp) = system

    #println("size deformation grad: ",size(deformation_grad))
    v_elas = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2)
                    ,size(deformation_grad,3))
    
    F,b,d = calc_deformation_grad3d!(deformation_grad, system,dt, fixed, semi)
    
    Fp = zeros(eltype(system), size(F,1), size(F,2) ,size(F,3))

    mu = young_modulus/(2+2*poisson_ratio)
    K = young_modulus/(3-6*poisson_ratio)

    @threaded semi for particle in eachparticle(system)
        Fp[:,:,particle] .= Matrix{Float64}(I, 3, 3)
        detF = det(F[:,:,particle])

        J = max(detF, 1e-6)

        dev_b = zeros(eltype(system), size(b,1), size(b,2))
        dev_b = b[:,:,particle] - 1/3* (tr(b[:,:,particle]))*I       

        @inbounds v_elas[:, :, particle] .= K*log(J)*I+mu*dev_b

        yf = sqrt(1.5)*sqrt(sum((v_elas[:, :, particle]- 1/3* (tr(v_elas[:, :, particle]))*I).^ 2))- (ys[particle]-hard[particle])
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
            #println("plastic_Regime")
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
            
            dev_v_elas = v_elas[:, :, particle] - 1/3* (tr(v_elas[:, :, particle]))*I
            norm_dev = sqrt(sum(dev_v_elas .* dev_v_elas))
            n = dev_v_elas/norm_dev
            dFp = strain_rate * n * dt
            Fp_old = Fp[:,:,particle]
            Fp[:,:,particle] += dFp * Fp[:,:,particle]
            Fe = F[:,:,particle] * inv(Fp[:,:,particle])
            J = det(Fe)
            be = Fe * Fe'   # left Cauchy-Green
            dev_be = be - 1/3*tr(be)*I
            v_elas[:, :, particle] .= K*log(J)*I + mu*dev_be 
            alpha_dot = strain_rate
            _alpha[particle] += alpha_dot  

            depsilon = 0.5 * ((Fp[:,:,particle] - Fp_old) * inv(Fp_old) + ((Fp[:,:,particle] - Fp_old) * inv(Fp_old))')
            plastic_work = sum(v_elas[:,:,particle] .* depsilon)
            beta = 0.9
            temp[particle] += beta * plastic_work / (material_density[particle]*cp) * dt
        else
            #println("Elastic_Regime")
        end
    end
    return v_elas, _alpha
end
    

@inline function calc_deformation_grad3d!(deformation_grad, system, dt,fixed, semi)
    (;mass,material_density,temp,temp_ref) = system

    velocity = system.initial_condition.velocity
    # Reset deformation gradient
    b = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2) ,size(deformation_grad,3)) 
    d = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2) ,size(deformation_grad,3))
    for i in 1:length(temp)
        deformation_grad[:,:,i] .= Matrix{Float64}(I, 3, 3)
        b[:,:,i] .= Matrix{Float64}(I, 3, 3)
        d[:,:,i] .= Matrix{Float64}(I, 3, 3)
    end
    velocity_grad = zeros(eltype(system), size(deformation_grad,1), size(deformation_grad,2),size(deformation_grad,3),size(deformation_grad,4))

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    initial_coords = initial_coordinates(system)
    foreach_point_neighbor(system, system, initial_coords, initial_coords,
                           semi) do particle, neighbor, pos_diff, initial_distance
        # Only consider particles with a distance > 0.
        # See `src/general/smoothing_kernels.jl` for more details.
        initial_distance^2 < eps(initial_smoothing_length(system)^2) && return

        volume = @inbounds mass[neighbor] / material_density[neighbor]
        pos_diff_ = @inbounds current_coords(system, particle) -
                              current_coords(system, neighbor)
        # On GPUs, convert `Float64` coordinates to `Float32` after computing the difference
        pos_diff = convert.(eltype(system), pos_diff_)
        vel_diff = velocity[particle] - velocity[neighbor]

        #r_norm = norm(pos_diff_) + 1e-12  # epsilon in denominator

        grad_kernel = smoothing_kernel_grad(system, pos_diff,
                                            initial_distance, particle)

        # Multiply by L_{0a}
        L = @inbounds correction_matrix(system, particle)

        result = volume * pos_diff * grad_kernel' * L'
        result_v = volume * vel_diff * grad_kernel' * L'

        for j in 1:ndims(system), i in 1:ndims(system)
            @inbounds velocity_grad[i,j,particle] -= dt * result_v[i,j]   
            @inbounds deformation_grad[i, j,particle] -= dt * result[i,j]        
        end

        # Skip too-close particles (optional)
        # if r_norm < 1e-6
        #     return
        # end

        #I = Matrix{Float64}(I, ndims(system), ndims(system))
        #@inbounds deformation_grad_init[:, :, particle] = I + alpha * (temp[particle]-temp_ref[particle]) .* I
        #@inbounds deformation_grad[:, :, particle] = deformation_grad[:, :, particle] * deformation_grad_init[:, :, particle]

        F = deformation_grad[:,:,particle]
        b[:,:,particle] = F * F'

        _L = velocity_grad[:,:,particle]
        d[:,:,particle] = 0.5 * (_L + _L')

    end

    return deformation_grad,b,d
end
