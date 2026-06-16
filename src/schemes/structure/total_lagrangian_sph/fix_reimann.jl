# ------------------------------------------------------------------------------
# Cross-system TLSPH<->TLSPH penalty contact override.
#
# Why we override:
#   - The package's `update_nhs!` for the (TLSPH, TLSPH) cross-system pair is
#     a deliberate no-op (see src/general/neighborhood_search.jl ~L290), so
#     `foreach_point_neighbor` returns no pairs once bodies move apart.
#   - The package's flat-plate `interact_Reimann!` (rhs.jl) only works for a
#     true flat plate (single z_wall, n_w = ±ẑ). Our mold is a shaped trapezoid
#     so its z_min is far below the cavity walls, and the flat-plate δ blows
#     up unphysically.
#
# Formulation (pair-wise; mirrors rhs.jl `interact_Reimann!` with the two
# minimal modifications required for an implicit Newton solver):
#
#   - Brute-force pair sweep (≲ 15k tests per RHS for our 60 / 215 / 235 systems).
#   - Stabilising penalty (same form as rhs.jl):
#         δ = max(-g_n, 0),   δ_tol = 0.1·ps
#         τ_stab = k_n · max(δ - δ_tol, 0),   k_n = E / ps
#         τ_dmp  = c_n · max(-v_rel, 0),      c_n = 0.05·2·sqrt(k_n·m_i/A_eff)
#   - Riemann acoustic traction (smoothed):
#         τ_riem = 0.01·Z·v_ref·tanh(-v_rel / v_ref),  v_ref = 0.01·c
#     This is the rhs.jl term `0.01·Z·max(0, sign(-v_rel)·min(|v_rel|,v_ref))`
#     with `sign·min` replaced by `v_ref·tanh(·/v_ref)`. Identical for
#     |v_rel| ≪ v_ref, saturates at ±v_ref, but C∞ smooth.
#   - F_n = (τ_riem + τ_stab + τ_dmp) · A_eff,  A_eff = ps².
#   - Repulsive only; mild overlap saturation at −0.9·ps for rejected steps.
#
# Differences from rhs.jl `interact_Reimann!`:
#   * Pair-wise normal from the current particle-pair geometry instead of a flat wall normal.
#   * `tanh` instead of `sign+min` in the Riemann term (smooth ⇒ NLNewton converges).
# ------------------------------------------------------------------------------

# Tool–charge Reimann penalty: k_n = active_contact_e_scale() * E / ps.
# Compression (molten charge): use melt-scale stiffness (K_melt/E), not solid composite E.
# Retraction (solid springback): softer scale via TP_CLIP_RETRACTION_CONTACT_E_SCALE.
const compression_contact_e_scale = Ref(1.0)
const retraction_contact_e_scale = Ref(1.0)

@inline function active_contact_e_scale()
    retraction_contact_e_scale[] < 1.0 - 1.0e-12 &&
        return retraction_contact_e_scale[]
    return compression_contact_e_scale[]
end

@inline function _reimann_pair_force!(dv,
                                      particle::Int,
                                      dx::Float64, dy::Float64, dz::Float64,
                                      distance::Float64,
                                      ps::Float64,
                                      E::Float64, rho_i::Float64, m_i::Float64,
                                      vix::Float64, viy::Float64, viz::Float64,
                                      vjx::Float64, vjy::Float64, vjz::Float64)
    distance < eps(ps) && return
    g_n = distance - ps
    g_n >= 0.0 && return

    inv_d = 1.0 / distance
    nx = dx * inv_d
    ny = dy * inv_d
    nz = dz * inv_d

    # Pair-relative normal velocity (negative => approaching).
    v_rel = (vix - vjx) * nx + (viy - vjy) * ny + (viz - vjz) * nz

    δ = max(0.0, -g_n)
    δ <= 0.0 && return

    A_eff = ps * ps
    α = 0.1
    δ_tol = α * ps

    # --- Stabilising penalty (same form as rhs.jl eq. 8) ------------------
    k_n = active_contact_e_scale() * E / ps
    c_n = 0.05 * 2.0 * sqrt(k_n * m_i / A_eff)

    t_stab = δ > δ_tol ? k_n * (δ - δ_tol) : zero(δ)
    t_damp = (δ > 0.0 && v_rel < 0.0) ? c_n * (-v_rel) : zero(v_rel)
    t_corr = t_stab + t_damp

    # --- Riemann acoustic traction (DISABLED for implicit-solver stability) -
    # The rhs.jl form `0.01 * Z * max(0, sign(-v_rel) * min(|v_rel|, 0.01*c))`
    # -- even when smoothed with tanh -- adds a large velocity-coupling block
    # to the Newton Jacobian (slope at v_rel=0 is 0.01*Z ~ 5e4 Pa*s/m,
    # multiplied by hundreds of active pairs).  This dominates the structural
    # Jacobian and causes Newton iterations to diverge -> dt collapses to
    # ~1e-20.  The Kelvin-Voigt damping above already supplies the equivalent
    # physical dissipation smoothly via the linear c_pt*v_rel term.
    # Keep the computation commented for reference:
            c_sound = sqrt(E / rho_i); Z_i = rho_i * c_sound; v_ref = 0.01 * c_sound
            arg     = -v_rel / v_ref
            t_riemann = 0.1 * (arg > 0.0 ? Z_i * v_ref * tanh(arg) : zero(v_rel))

        # --- Total traction / force (same form as rhs.jl eq. 9) ---------------
        t_n = t_riemann + t_corr
        F_n = t_n * A_eff
        F_n <= 0.0 && return                        # repulsive only

    a_mag = F_n / m_i
    dv[1, particle] += a_mag * nx
    dv[2, particle] += a_mag * ny
    dv[3, particle] += a_mag * nz
    return
end

function TrixiParticles.interact_Reimann!(dv, v_particle_system, u_particle_system,
                           v_neighbor_system, u_neighbor_system,
                           particle_system::TrixiParticles.TotalLagrangianSPHSystem,
                           neighbor_system::TrixiParticles.TotalLagrangianSPHSystem,
                           semi; integrate_tlsph=semi.integrate_tlsph[])
    if particle_system === neighbor_system
        TrixiParticles.interact_structure_structure!(dv, v_particle_system,
                                                     particle_system, semi)
        return dv
    end
    integrate_tlsph || return dv

    system_coords   = TrixiParticles.current_coordinates(u_particle_system,
                                                         particle_system)
    neighbor_coords = TrixiParticles.current_coordinates(u_neighbor_system,
                                                         neighbor_system)

    ps             = TrixiParticles.initial_smoothing_length(particle_system) / 1.2
    contact_radius = 1.5 * ps
    radius2        = contact_radius * contact_radius

    n_q = TrixiParticles.nparticles(neighbor_system)

    @inbounds for particle in TrixiParticles.each_integrated_particle(particle_system)
        xi1 = system_coords[1, particle]
        xi2 = system_coords[2, particle]
        xi3 = system_coords[3, particle]

        v_i = TrixiParticles.current_velocity(v_particle_system,
                                              particle_system, particle)
        vix = v_i[1]; viy = v_i[2]; viz = v_i[3]

        E   = TrixiParticles.young_modulus(particle_system, particle)
        m_i = particle_system.mass[particle]
        rho_i = particle_system.material_density[particle]

        for neighbor in 1:n_q
            dx = xi1 - neighbor_coords[1, neighbor]
            dy = xi2 - neighbor_coords[2, neighbor]
            dz = xi3 - neighbor_coords[3, neighbor]
            d2 = dx * dx + dy * dy + dz * dz
            d2 > radius2 && continue

            distance = sqrt(d2)

            v_j = TrixiParticles.current_velocity(v_neighbor_system,
                                                  neighbor_system, neighbor)
            vjx = v_j[1]; vjy = v_j[2]; vjz = v_j[3]

            _reimann_pair_force!(dv, particle,
                                 dx, dy, dz, distance,
                                 ps, E, rho_i, m_i,
                                 vix, viy, viz,
                                 vjx, vjy, vjz)
        end
    end

    return dv
end