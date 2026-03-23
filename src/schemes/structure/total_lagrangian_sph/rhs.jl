# Structure-structure interaction (includes cross-system volumetric contact)
function interact!(dv, v_particle_system, u_particle_system, dv_neighbor,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::TotalLagrangianSPHSystem,
                   semi;
                   integrate_tlsph=semi.integrate_tlsph[])
    
    integrate_tlsph || return dv

    if particle_system === neighbor_system
        # Self-interaction: internal stress, penalty force, viscosity
        interact_structure_structure!(dv, v_particle_system, particle_system, semi)
    else
        # Cross-system: volumetric contact between cylinder and floor
        # Pass both dv arrays so we can apply equal-and-opposite forces efficiently
        interact_Reimann!(dv, dv_neighbor, v_particle_system, u_particle_system,
                            v_neighbor_system, u_neighbor_system,
                            particle_system, neighbor_system, semi)
    end

    return dv
end

# ==========================================================================================
# Volumetric contact between two TLSPH systems
# ==========================================================================================
# ==========================================================================================
# CSF-based contact surface detection
# Eq. 24-26 from Tang et al.
# ==========================================================================================
function compute_contact_surface!(system::TotalLagrangianSPHSystem,
                                  neighbor_system::TotalLagrangianSPHSystem,
                                  u_system, u_neighbor, semi)

    system_coords   = current_coordinates(u_system, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)
    h               = initial_smoothing_length(system)
    threshold       = 0.01 / h

    # n_contact_particles x ndims surface normal vector per particle
    n_particles = nparticles(system)
    n_surf      = zeros(eltype(system_coords), ndims(system), n_particles)
    is_contact  = falses(n_particles)

    # Compute kernel gradient imbalance (eq 24)
    for particle in each_integrated_particle(system)
        r_a = system_coords[:, particle]

        for neighbor in 1:nparticles(neighbor_system)
            r_b      = neighbor_coords[:, neighbor]
            pos_diff = r_a - r_b
            distance = norm(pos_diff)

            distance < eps(h) && continue
            distance >= h      && continue

            m_b   = neighbor_system.mass[neighbor]
            rho_b = neighbor_system.material_density[neighbor]

            # Kernel gradient
            grad_W = smoothing_kernel_grad(system, pos_diff, distance, particle)

            # Eq. 24: n*_i += -(m_j/rho_j) * grad_W_ij
            for d in 1:ndims(system)
                n_surf[d, particle] -= (m_b / rho_b) * grad_W[d]
            end
        end

        # Eq. 25: delta*_i = ||n*_i||
        delta_i = norm(n_surf[:, particle])

        # Eq. 26: classify as contact surface particle
        is_contact[particle] = delta_i > threshold
    end

    return n_surf, is_contact
end

# ==========================================================================================
# Volumetric contact with CSF detection + Pressure Neumann BC
# Eq. 42-43 from Tang et al. adapted for solid-solid contact
# ==========================================================================================
function interact_volumetric_contact!(dv,
                                      v_particle_system, u_particle_system,
                                      v_neighbor_system, u_neighbor_system,
                                      particle_system::TotalLagrangianSPHSystem,
                                      neighbor_system::TotalLagrangianSPHSystem,
                                      semi)

    system_coords   = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Only apply when particle_system is above neighbor_system (cylinder above floor)
    z_sys_mean = sum(system_coords[3, :]) / size(system_coords, 2)
    z_nbr_mean = sum(neighbor_coords[3, :]) / size(neighbor_coords, 2)
    z_sys_mean > z_nbr_mean || return dv

    h      = initial_smoothing_length(particle_system)
    E_cyl  = young_modulus(particle_system, 1)
    nu_cyl = poisson_ratio(particle_system, 1)
    E_star = E_cyl / (1 - nu_cyl^2)
    R_eff  = 0.006  # cylinder radius

    # Hertz-derived stiffness
    K_contact = (4.0/3.0) * E_star * sqrt(R_eff)

    # ==========================
    # Step 1: CSF surface detection
    # Identify cylinder particles at contact surface
    # ==========================
    n_surf, is_contact = compute_contact_surface!(particle_system,
                                                   neighbor_system,
                                                   u_particle_system,
                                                   u_neighbor_system,
                                                   semi)

    println("CSF contact surface particles: ", sum(is_contact),
            " / ", nparticles(particle_system))

    # Precompute normalisation sum over contact particles only
    norm_sum = 0.0
    for particle in each_integrated_particle(particle_system)
        is_contact[particle] || continue  # only surface particles
        r_a = system_coords[:, particle]

        for neighbor in 1:nparticles(neighbor_system)
            r_b      = neighbor_coords[:, neighbor]
            distance = norm(r_a - r_b)
            distance < eps(h) && continue
            distance >= h      && continue

            V_b       = neighbor_system.mass[neighbor] / neighbor_system.material_density[neighbor]
            W         = TrixiParticles.kernel(particle_system.smoothing_kernel, distance, h)
            norm_sum += V_b * W
        end
    end
    norm_sum = max(norm_sum, eps(Float64))

    # ==========================
    # Step 2: Pressure Neumann BC contact force
    # Eq. 42-43 adapted for solid stress
    # Non-penetration enforced via normal stress continuity
    # ==========================
    for particle in each_integrated_particle(particle_system)
        is_contact[particle] || continue  # only CSF-detected surface particles

        r_a    = system_coords[:, particle]
        n_i    = n_surf[:, particle]
        n_norm = norm(n_i)
        n_norm < eps(Float64) && continue
        n_hat  = n_i / n_norm  # unit outward normal at contact surface

        # Normal stress at cylinder particle (from PK1/Cauchy)
        # Use sigma_33 proxy — negative = compressive
        p_i = -particle_system.material_density[particle] *
              dot(pk1_rho2(particle_system, particle) *
                  particle_system.material_density[particle],
                  n_hat)  # normal traction

        for neighbor in 1:nparticles(neighbor_system)
            r_b      = neighbor_coords[:, neighbor]
            pos_diff = r_a - r_b
            distance = norm(pos_diff)

            distance < eps(h) && continue
            distance >= h      && continue

            normal      = pos_diff / distance
            penetration = h - distance
            penetration <= 0 && continue

            V_b   = neighbor_system.mass[neighbor] / neighbor_system.material_density[neighbor]
            W     = TrixiParticles.kernel(particle_system.smoothing_kernel, distance, h)
            gamma = V_b * W / norm_sum

            # Normal stress at floor neighbor
            p_j = -neighbor_system.material_density[neighbor] *
                  dot(pk1_rho2(neighbor_system, neighbor) *
                      neighbor_system.material_density[neighbor],
                      n_hat)

            # Eq. 43 adapted — weak Neumann BC:
            # enforce n · sum_j (m_j/rho_j)(p_j - p_i) * grad_W = 0
            # Contact force from stress difference (pressure jump at interface)
            grad_W    = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)
            neumann_f = dot(n_hat, (p_j - p_i) * grad_W) * V_b

            # Hertz volumetric elastic force (geometric penalty)
            F_elastic = K_contact * gamma * penetration^1.5

            # Riemann damping
            v_i   = current_velocity(v_particle_system, particle_system, particle)
            v_j   = current_velocity(v_neighbor_system, neighbor_system, neighbor)
            v_rel = dot(v_i - v_j, normal)

            rho_i     = particle_system.material_density[particle]
            c_i       = sqrt(E_cyl / rho_i)
            Z_i       = rho_i * c_i
            F_damping = gamma * Z_i * max(0.0, -v_rel)

            # Total — Hertz penalty + Neumann stress correction + damping
            F_total = F_elastic + F_damping

            m_a = particle_system.mass[particle]
            m_b = neighbor_system.mass[neighbor]

            # Apply Neumann correction + elastic force to cylinder
            for d in 1:ndims(particle_system)
                dv[d, particle] += (F_total * normal[d] + neumann_f * n_hat[d]) / m_a
            end

            # Reaction on floor
            for d in 1:ndims(neighbor_system)
                dv[d, neighbor] -= (F_total * normal[d] + neumann_f * n_hat[d]) / m_b
            end
        end
    end

    return dv
end

# Function barrier without dispatch for unit testing
@inline function interact_structure_structure!(dv, v_system, system, semi)
    (; penalty_force) = system

    # Everything here is done in the initial coordinates
    system_coords = initial_coordinates(system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    # For structure-structure interaction, this has to happen in the initial coordinates.
    foreach_point_neighbor(system, system, system_coords, system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       initial_pos_diff,
                                                                       initial_distance
        # Only consider particles with a distance > 0.
        # See `src/general/smoothing_kernels.jl` for more details.
        initial_distance^2 < eps(initial_smoothing_length(system)^2) && return

        rho_a = @inbounds system.material_density[particle]
        rho_b = @inbounds system.material_density[neighbor]

        grad_kernel = smoothing_kernel_grad(system, initial_pos_diff,
                                            initial_distance, particle)

        m_a = @inbounds system.mass[particle]
        m_b = @inbounds system.mass[neighbor]

        # PK1 / rho^2
        pk1_rho2_a = @inbounds pk1_rho2(system, particle)
        pk1_rho2_b = @inbounds pk1_rho2(system, neighbor)

        current_pos_diff_ = @inbounds current_coords(system, particle) -
                                      current_coords(system, neighbor)
        # On GPUs, convert `Float64` coordinates to `Float32` after computing the difference
        current_pos_diff = convert.(eltype(system), current_pos_diff_)
        current_distance = norm(current_pos_diff)

        dv_stress = m_b * (pk1_rho2_a + pk1_rho2_b) * grad_kernel

        dv_penalty_force_ = @inbounds dv_penalty_force(penalty_force, particle, neighbor,
                                                       initial_pos_diff, initial_distance,
                                                       current_pos_diff, current_distance,
                                                       system, m_a, m_b, rho_a, rho_b)

        dv_viscosity = @inbounds dv_viscosity_tlsph(system, v_system, particle, neighbor,
                                                    current_pos_diff, current_distance,
                                                    m_a, m_b, rho_a, rho_b, grad_kernel)

        dv_particle = dv_stress + dv_penalty_force_ + dv_viscosity

        for i in 1:ndims(system)
            @inbounds dv[i, particle] += dv_particle[i]
        end

        # TODO continuity equation for boundary model with `ContinuityDensity`?
    end

    return dv
end

# Structure-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::AbstractFluidSystem, semi;
                   integrate_tlsph=semi.integrate_tlsph[])
    # Skip interaction if TLSPH systems are integrated separately
    integrate_tlsph || return dv

    sound_speed = system_sound_speed(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # Only consider particles with a distance > 0. See `src/general/smoothing_kernels.jl` for more details.
        distance^2 < eps(initial_smoothing_length(particle_system)^2) && return

        # Apply the same force to the structure particle
        # that the fluid particle experiences due to the structure particle.
        # Note that the same arguments are passed here as in fluid-structure interact!,
        # except that pos_diff has a flipped sign.
        #
        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        # Use kernel from the fluid system in order to get the same force here in
        # structure-fluid interaction as for fluid-structure interaction.
        # TODO this will not use corrections if the fluid uses corrections.
        grad_kernel = smoothing_kernel_grad(neighbor_system, pos_diff, distance, particle)

        # In fluid-structure interaction, use the "hydrodynamic pressure" of the structure particles
        # corresponding to the chosen boundary model.
        p_a = current_pressure(v_particle_system, particle_system, particle)
        p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # Particle and neighbor (and corresponding systems and all corresponding quantities)
        # are switched in the following two calls.
        # This way, we obtain the exact same force as for the fluid-structure interaction,
        # but with a flipped sign (because `pos_diff` is flipped compared to fluid-structure).
        dv_boundary = pressure_acceleration(neighbor_system, particle_system,
                                            neighbor, particle,
                                            m_b, m_a, p_b, p_a, rho_b, rho_a, pos_diff,
                                            distance, grad_kernel,
                                            neighbor_system.correction)

        dv_viscosity_ = dv_viscosity(neighbor_system, particle_system,
                                     v_neighbor_system, v_particle_system,
                                     neighbor, particle, pos_diff, distance,
                                     sound_speed, m_b, m_a, rho_a, rho_b, grad_kernel)

        dv_particle = dv_boundary + dv_viscosity_

        for i in 1:ndims(particle_system)
            # Multiply `dv` (acceleration on fluid particle b) by the mass of
            # particle b to obtain the same force as for the fluid-structure interaction.
            # Divide by the material mass of particle a to obtain the acceleration
            # of structure particle a.
            dv[i, particle] += dv_particle[i] * m_b / particle_system.mass[particle]
        end

        continuity_equation!(dv, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             m_b, rho_a, rho_b,
                             particle_system, neighbor_system, grad_kernel)
    end

    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::TotalLagrangianSPHSystem,
                                      neighbor_system::AbstractFluidSystem,
                                      grad_kernel)
    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                      neighbor_system::AbstractFluidSystem,
                                      grad_kernel)
    fluid_density_calculator = neighbor_system.density_calculator

    v_diff = current_velocity(v_particle_system, particle_system, particle) -
             current_velocity(v_neighbor_system, neighbor_system, neighbor)

    # Call the dummy BC version of the continuity equation
    continuity_equation!(dv, fluid_density_calculator, m_b, rho_a, rho_b, v_diff,
                         grad_kernel, particle)
end

function interact_Reimann!(dv, dv_neighbor, v_particle_system, u_particle_system,
                           v_neighbor_system, u_neighbor_system,
                           particle_system::TotalLagrangianSPHSystem,
                           neighbor_system::TotalLagrangianSPHSystem,
                           semi; integrate_tlsph=semi.integrate_tlsph[])

    # Self-interaction — handled by interact_structure_structure!
    if particle_system === neighbor_system
        interact_structure_structure!(dv, v_particle_system, particle_system, semi)
        return dv
    end

    integrate_tlsph || return dv

    (; mass, material_density) = particle_system

    system_coords   = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # # Only apply when particle_system is above neighbor_system (cylinder above floor)
    # z_sys_mean = sum(system_coords[3, :]) / size(system_coords, 2)
    # z_nbr_mean = sum(neighbor_coords[3, :]) / size(neighbor_coords, 2)
    # z_sys_mean > z_nbr_mean || return dv

    # ==========================
    # Parameters — all derived from material/geometry, no free params
    # ==========================
    h_global = initial_smoothing_length(particle_system)
    ps       = h_global / 1.5        # particle spacing (factor=1.5 from setup)
    z_wall   = maximum(neighbor_coords[3, :])  # flat plate top surface z
    r_i      = ps / 2                # effective particle radius ≈ Δp/2
    A_eff    = ps^2                  # effective contact area per particle (eq. 10)
    α        = 0.1                   # stabiliser threshold factor
    δ_tol    = α * ps               # penetration tolerance (eq. 8)

    # # Wall normal — flat plate, pointing upward into cylinder
    # n_w1 = zero(eltype(system_coords))
    # n_w2 = zero(eltype(system_coords))
    # n_w3 = one(eltype(system_coords))


    z_sys_mean = sum(system_coords[3, :]) / size(system_coords, 2)
    z_nbr_mean = sum(neighbor_coords[3, :]) / size(neighbor_coords, 2)

    # Determine contact direction — neighbor below (floor) or above (mold)
    neighbor_is_below = z_nbr_mean < z_sys_mean
    neighbor_is_above = z_nbr_mean > z_sys_mean

    # Skip if systems at same level (no contact)
    (neighbor_is_below || neighbor_is_above) || return dv

    # Wall reference z and normal direction
    # Floor below: z_wall = top of floor, normal points UP (+z)
    # Mold above:  z_wall = bottom of mold, normal points DOWN (-z)
    z_wall = neighbor_is_below ? maximum(neighbor_coords[3, :]) :
                                 minimum(neighbor_coords[3, :])
    n_w3   = neighbor_is_below ? one(eltype(system_coords)) :
                                 -one(eltype(system_coords))
    n_w1   = zero(eltype(system_coords))
    n_w2   = zero(eltype(system_coords))
    r_i    = ps / 2

    # ==========================
    # Contact traction per particle — O(N), no neighbor loop
    # ==========================
    for particle in each_integrated_particle(particle_system)

        # # --- Geometry: signed gap (eq. 3) ---
        # z_i = system_coords[3, particle]
        # g_n = (z_i - z_wall) - r_i    # signed gap: negative = penetrating
        # δ   = max(zero(g_n), -g_n)    # penetration depth δ ≥ 0

        z_i = system_coords[3, particle]
        # Gap: positive = separated, negative = penetrating
        # Floor: g_n = z_particle - z_floor_top - r_i
        # Mold:  g_n = z_mold_bottom - z_particle - r_i  (flipped sign)
        g_n = neighbor_is_below ? (z_i - z_wall) - r_i :
                                  (z_wall - z_i) - r_i
        δ   = max(zero(g_n), -g_n)

        δ <= 0 && continue             # no contact — skip

        # --- Material properties ---
        E     = young_modulus(particle_system, particle)
        rho_i = material_density[particle]
        c_i   = sqrt(E / rho_i)        # wave speed
        Z_i   = rho_i * c_i            # acoustic impedance
        m_i   = mass[particle]

        # --- Velocity: use relative normal velocity between systems ---
        v_i = current_velocity(v_particle_system, particle_system, particle)

        # Compute mean neighbor velocity (approximate wall/mold velocity at contact)
        v_neighbor_mean = zero(v_i)
        n_neigh = 0
        for nb in each_integrated_particle(neighbor_system)
            v_neighbor_mean += current_velocity(v_neighbor_system, neighbor_system, nb)
            n_neigh += 1
        end
        if n_neigh > 0
            v_neighbor_mean = v_neighbor_mean / n_neigh
        end

        # Wall normal vector (only z-component non-zero) and relative normal velocity
        n_w = SVector(n_w1, n_w2, n_w3)
        v_rel = dot(v_i - v_neighbor_mean, n_w)

        # --- Riemann contact traction (eq. 7) ---
        # Use relative normal velocity between the two systems. Cap rapid approaches
        v_rel_capped = sign(-v_rel) * min(abs(v_rel), 0.01 * sqrt(E / rho_i))
        # For a semi-infinite neighbor the impedance combination reduces appropriately
        t_riemann = Z_i * max(zero(v_rel_capped), -v_rel_capped)

        # --- Stabilising penalty (eq. 8) ---
        # Activates only for deep penetration δ > δ_tol
        k_n    = E / ps                            # stiffness from material, no free param
        c_n    = 2 * sqrt(k_n * m_i / A_eff)       # critical damping

        t_stab = δ > δ_tol ? k_n * (δ - δ_tol) : zero(δ)
        t_damp = (δ > 0 && v_rel < 0) ? c_n * (-v_rel) : zero(v_rel)
        t_corr = t_stab + t_damp

        # --- Total traction (eq. 9) ---
        t_n = t_riemann + t_corr       # always ≥ 0

        # --- Force (N) and acceleration (m/s^2) on system particle ---
        F_n = t_n * A_eff
        a_mag = F_n / m_i

        @inbounds dv[1, particle] += a_mag * n_w1
        @inbounds dv[2, particle] += a_mag * n_w2
        @inbounds dv[3, particle] += a_mag * n_w3

        # --- Reaction: find nearest neighbor and apply equal-and-opposite force ---
        min_dist = Inf
        min_idx = 0
        p_pos = system_coords[:, particle]
        
        # Check only particles within smoothing distance for efficiency
        for nb in each_integrated_particle(neighbor_system)
            nb_pos = neighbor_coords[:, nb]
            d_sq = sum((p_pos .- nb_pos).^2)
            if d_sq < min_dist && d_sq < (smoothing_length(particle_system))^2
                min_dist = d_sq
                min_idx = nb
            end
        end

        if min_idx > 0
            m_j = neighbor_system.mass[min_idx]
            a_neighbor_mag = -F_n / m_j
            @inbounds dv_neighbor[1, min_idx] += a_neighbor_mag * n_w1
            @inbounds dv_neighbor[2, min_idx] += a_neighbor_mag * n_w2
            @inbounds dv_neighbor[3, min_idx] += a_neighbor_mag * n_w3
        end
    end

    return dv
end


# Structure-boundary (solid wall) interaction via Riemann contact + penalty
# function interact!(dv, v_particle_system, u_particle_system,
#                    v_neighbor_system, u_neighbor_system,
#                    particle_system::TotalLagrangianSPHSystem,
#                    neighbor_system::Union{WallBoundarySystem, OpenBoundarySystem},
#                    semi; integrate_tlsph=semi.integrate_tlsph[])
                   
#     integrate_tlsph || return dv

#     (; mass, material_density) = particle_system

#     system_coords   = current_coordinates(u_particle_system, particle_system)
#     neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

#     foreach_point_neighbor(particle_system, neighbor_system,
#                            system_coords, neighbor_coords, semi;
#                            points=each_integrated_particle(particle_system)) do particle,
#                                                                                 neighbor,
#                                                                                 pos_diff,
#                                                                                 distance
#         h = smoothing_length(particle_system, particle)/1.5

#         distance >= h && return
#         #distance <= 0.1 * h && return  # skip if too close — already overlapping badly
#         # Normal vector pointing from mold toward polymer particle
#         normal = pos_diff / (distance + eps(distance))
#         # Material properties
#         E     = young_modulus(particle_system, particle)
#         rho_i = material_density[particle]
#         # Wave speed and impedances
#         c_i = sqrt(E / rho_i)
#         Z_i = rho_i * c_i
#         Z_j = Z_i   # rigid wall assumption
#         # Velocities
#         v_i = current_velocity(v_particle_system, particle_system, particle)
#         v_j = current_velocity(v_neighbor_system, neighbor_system, neighbor)
#         # Relative normal velocity (negative = approaching)
#         v_rel = dot(v_i - v_j, normal)
#         # Only apply force during compression
#         #v_rel < 0 || return
#         overlap = h - distance
#         overlap <= 0 && return

#         nu_i   = poisson_ratio(particle_system, particle)
#         E_star = E / (1 - nu_i^2)
#         #R_eff  = 0.006
#         # # Hertz contact FORCE (Newtons) — fixed dimensionally
#         # n_active_contacts = 0
#         # for p in each_integrated_particle(particle_system)
#         #     # Check if particle is near the floor (simple Z-check is fastest)
#         #     # floor_z_max is the top of your boundary
#         #     z_pos = system_coords[3, p]
#         #     if z_pos < (minimum(neighbor_coords[3, :]) + h)
#         #         n_active_contacts += 1
#         #     end
#         # end
        
#         # Safety: avoid division by zero
#         # n_contact = max(1, n_active_contacts)
#         # F_hertz = (4.0/3.0) * E_star * sqrt(R_eff) * overlap^1.5 / n_contact
        
#         # m_i     = mass[particle]
#         # k_hertz = (4.0/3.0) * E_star * sqrt(R_eff * overlap) / n_contact
#         # c_damp  = 2.0 * sqrt(m_i * k_hertz)
#         # F_damp  = c_damp * abs(v_rel)
#         # # All forces → acceleration = F/m, traction/rho already has right units
#         # acc = (F_hertz / m_i + F_damp / m_i + t_riemann / rho_i) * normal
#         # for d in 1:ndims(particle_system)
#         #     dv[d, particle] += acc[d]
#         # end
#         k= 1*(0.001/h)
#         # Pure Riemann contact — no Hertz, no separate damping
#         t_riemann = (Z_i * Z_j) / (Z_i + Z_j) * max(0.0, -v_rel)
#         p_elastic = k * (E_star / h) * overlap

#         acc = (p_elastic / rho_i + t_riemann / rho_i) * normal

#         for d in 1:ndims(particle_system)
#             dv[d, particle] += acc[d]
#         end
#     end
#     #@show maximum(abs.(dv[:, 1:nparticles(particle_system)]))

#     return dv
# end

# function interact_Reimann!(dv, v_particle_system, u_particle_system,
#                            v_neighbor_system, u_neighbor_system,
#                            particle_system::TotalLagrangianSPHSystem,
#                            neighbor_system::TotalLagrangianSPHSystem,
#                            semi; integrate_tlsph=semi.integrate_tlsph[])

#     # Self-interaction — handled by interact_structure_structure!
#     if particle_system === neighbor_system
#         interact_structure_structure!(dv, v_particle_system, particle_system, semi)
#         return dv
#     end

#     integrate_tlsph || return dv

#     (; mass, material_density) = particle_system

#     system_coords   = current_coordinates(u_particle_system, particle_system)
#     neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

#     # Only apply when particle_system is above neighbor_system (cylinder above floor)
#     z_sys_mean = sum(system_coords[3, :]) / size(system_coords, 2)
#     z_nbr_mean = sum(neighbor_coords[3, :]) / size(neighbor_coords, 2)
#     z_sys_mean > z_nbr_mean || return dv

#     h_global = initial_smoothing_length(particle_system)

#     # ==========================
#     # Step 1: Geometric contact surface detection
#     # Contact = cylinder particles within h of floor top surface
#     # ==========================
#     n_particles = nparticles(particle_system)
#     is_contact  = falses(n_particles)
#     n_surf      = zeros(eltype(system_coords), ndims(particle_system), n_particles)

#     z_floor_top = maximum(neighbor_coords[3, :])
#     for particle in each_integrated_particle(particle_system)
#         if system_coords[3, particle] < (z_floor_top + h_global)
#             is_contact[particle]    = true
#             n_surf[3, particle]     = 1.0  # upward unit normal for bottom surface
#         end
#     end

#     # println("Geometric contact: n=", sum(is_contact),
#     #         " z_floor_top=",  round(z_floor_top*1e3,                           digits=3), "mm",
#     #         " threshold_z=",  round((z_floor_top + h_global)*1e3,              digits=3), "mm",
#     #         " gap=",          round((minimum(system_coords[3,:]) - z_floor_top)*1e3, digits=3), "mm")

#     # ==========================
#     # Step 2: Contact force — Riemann + elastic penalty + Neumann BC
#     # Brute force loop over current coordinates
#     # ==========================
#     n_contact_pairs = Ref(0)
#     total_F_z       = Ref(0.0)

#     for particle in each_integrated_particle(particle_system)
#         is_contact[particle] || continue

#         r_a   = system_coords[:, particle]
#         n_hat = n_surf[:, particle]  # upward unit normal [0, 0, 1]

#         E      = young_modulus(particle_system, particle)
#         rho_i  = material_density[particle]
#         nu_i   = poisson_ratio(particle_system, particle)
#         E_star = E / (1 - nu_i^2)
#         c_i    = sqrt(E / rho_i)
#         Z_i    = rho_i * c_i
#         m_a    = mass[particle]

#         for neighbor in 1:nparticles(neighbor_system)
#             r_b      = neighbor_coords[:, neighbor]
#             pos_diff = r_a - r_b
#             distance = norm(pos_diff)

#             distance < eps(h_global) && continue
#             distance >= h_global      && continue

#             normal  = pos_diff / distance  # points from floor to cylinder (upward)
#             overlap = h_global - distance
#             overlap <= 0 && continue

#             rho_b = neighbor_system.material_density[neighbor]
#             E_b   = young_modulus(neighbor_system, neighbor)
#             c_b   = sqrt(E_b / rho_b)
#             Z_j   = rho_b * c_b
#             m_b   = neighbor_system.mass[neighbor]

#             # Velocities
#             v_i   = current_velocity(v_particle_system, particle_system, particle)
#             v_j   = current_velocity(v_neighbor_system, neighbor_system, neighbor)
#             v_rel = dot(v_i - v_j, normal)  # positive = separating

#             # Kernel weight for volumetric force distribution
#             W     = TrixiParticles.kernel(particle_system.smoothing_kernel, distance, h_global)
#             V_j   = m_b / rho_b
#             gamma = V_j * W / norm_sum

#             # Hertz-consistent elastic force (no free parameter)
#             R_eff     = 0.006  # cylinder radius
#             F_elastic = (4.0/3.0) * E_star * sqrt(R_eff) * gamma * overlap^1.5

#             # Riemann traction — only on approach
#             F_riemann = (Z_i * Z_j) / (Z_i + Z_j) * max(0.0, -v_rel) * gamma

#             # Neumann BC stress correction (eq 43 from Tang et al.)
#             pk1_a   = pk1_rho2(particle_system, particle) * rho_i^2
#             pk1_b   = pk1_rho2(neighbor_system, neighbor) * rho_b^2
#             p_i     = -dot(pk1_a * n_hat, n_hat)
#             p_j     = -dot(pk1_b * n_hat, n_hat)
#             grad_W  = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)
#             F_neumann = dot(n_hat, (p_j - p_i) * grad_W) * V_j
#             F_neumann = clamp(F_neumann, -abs(F_elastic), abs(F_elastic))

#             # Total acceleration = F / m
#             acc = ((F_elastic + F_riemann) / m_a) .* normal .+
#                   (F_neumann / m_a) .* n_hat

#             # Apply to cylinder
#             for d in 1:ndims(particle_system)
#                 dv[d, particle] += acc[d]
#             end

#             n_contact_pairs[] += 1
#             total_F_z[]        += acc[3] * m_a

#             # Diagnostic: first 3 pairs for particle 1
#             # if particle == 1 && n_contact_pairs[] <= 3
#             #     println("  pair #",      n_contact_pairs[],
#             #             " dist=",        round(distance,    digits=6),
#             #             " overlap=",     round(overlap,     digits=6),
#             #             " gamma=",       round(gamma,       digits=8),
#             #             " F_elastic=",   round(F_elastic,   digits=6),
#             #             " F_riemann=",   round(F_riemann,   digits=6),
#             #             " F_neumann=",   round(F_neumann,   digits=6),
#             #             " acc_z=",       round(acc[3],      digits=4),
#             #             " normal_z=",    round(normal[3],   digits=4))
#             # end
#         end
#     end

#     # Summary
#     # weight = sum(mass) * 9.81
#     # println("Contact pairs=",   n_contact_pairs[],
#     #         " total_F_z=",      round(total_F_z[], digits=6), "N",
#     #         " weight=",         round(weight,      digits=6), "N",
#     #         " F/W=",            round(total_F_z[] / max(weight, eps()), digits=3))

#     return dv
# end

