# ------------------------------------------------------------------------------
# Cross-system TLSPH<->TLSPH penalty contact override.
#
# Molten L-bend + corner stability:
#   - Wider penalty deadband (contact_penalty_alpha, default 0.25·ps).
#   - tanh saturation on effective overlap so multi-tool-particle corners do not
#     stack linear penalty forces without bound.
#   - Melt-aware stiffness: k_n_eff = k_n * (sf + lf * melt_contact_softness).
#   - Liquid-fraction-weighted per-particle |a| cap after summing tool pairs:
#       a_cap = a_cap_solid + lf * (a_cap_molten - a_cap_solid).
# ------------------------------------------------------------------------------

# Tool–charge Reimann penalty: k_n = active_contact_e_scale() * E / ps.
const compression_contact_e_scale = Ref(1.0)
const retraction_contact_e_scale = Ref(1.0)
# Penalty deadband δ_tol = alpha * ps (was hard-coded 0.1).
const contact_penalty_alpha = Ref(0.25)
# Saturate penalty overlap: δ_eff = δ_sat * tanh((δ-δ_tol)/δ_sat), δ_sat = frac * ps.
const contact_overlap_sat_frac = Ref(0.5)
# Legacy single cap (used when liquid-fraction buffer is not wired).
const contact_total_accel_cap = Ref(1.0e4)
# Melt-aware contact caps and stiffness (set from driver via ENV).
const contact_accel_cap_solid = Ref(1.0e4)
const contact_accel_cap_molten = Ref(8.0e4)
const melt_contact_softness = Ref(0.15)
# Deep tool–charge overlap (δ = ps − distance): apply a stricter |a| cap so stacked
# soft-contact neighbors from ~1·ps penetration cannot dump huge contact kicks.
# δ_deep = contact_deep_overlap_frac * ps; |a|_cap = contact_deep_accel_cap (0 = off).
const contact_deep_overlap_frac = Ref(0.5)
const contact_deep_accel_cap = Ref(5.0e3)
# Rate-dependent contact dashpot (fraction of critical): c_n = ζ · 2√(k_n m / A).
# Deep-overlap uses contact_dashpot_zeta_deep when δ > δ_deep (0 = inherit ζ).
const contact_dashpot_zeta = Ref(0.05)
const contact_dashpot_zeta_deep = Ref(0.0)
# Charge liquid-fraction buffer (Ref to driver-owned vector).
const contact_liquid_fraction_buf = Ref{Union{Nothing, Vector{Float64}}}(nothing)
# Fixed blankholder system + resting-shelf mask (driver sets after packing).
const blankholder_neighbor_system = Ref{Any}(nothing)
const charge_resting_on_shelf_buf = Ref{Union{Nothing, Vector{Bool}}}(nothing)

@inline function skip_blankholder_contact_for_particle(particle::Int,
                                                      neighbor_system)
    bh = blankholder_neighbor_system[]
    bh === nothing && return false
    neighbor_system !== bh && return false
    buf = charge_resting_on_shelf_buf[]
    buf === nothing && return false
    particle > length(buf) && return true
    return !buf[particle]
end

@inline function contact_stiffness_melt_scale(lf::Float64)
    lf = clamp(lf, 0.0, 1.0)
    sf = 1.0 - lf
    return sf + lf * melt_contact_softness[]
end

@inline function contact_accel_cap_for_particle(lf::Float64)
    lf = clamp(lf, 0.0, 1.0)
    a_s = contact_accel_cap_solid[]
    a_m = contact_accel_cap_molten[]
    # molten cap ≤ 0 means "disabled" — do not interpolate toward 0 (that uncapped lf≈1).
    a_m <= 0.0 && return a_s
    return a_s + lf * (a_m - a_s)
end

@inline function contact_pair_overlap(distance::Float64, ps::Float64)
    distance < eps(ps) && return ps
    return max(0.0, ps - distance)
end

# Resolve |a| cap after summing tool-pair contact on one charge particle.
@inline function resolve_contact_accel_cap(lf::Float64, δ_max::Float64, ps::Float64,
                                           use_melt_contact::Bool)
    cap = Inf
    if use_melt_contact
        c = contact_accel_cap_for_particle(lf)
        c > 0.0 && (cap = min(cap, c))
    else
        c = contact_total_accel_cap[]
        c > 0.0 && (cap = min(cap, c))
    end
    deep_cap = contact_deep_accel_cap[]
    if deep_cap > 0.0 && δ_max > contact_deep_overlap_frac[] * ps
        cap = min(cap, deep_cap)
    end
    return cap
end

@inline function apply_resolved_contact_accel_cap(ax::Float64, ay::Float64, az::Float64,
                                                  cap::Float64)
    !isfinite(cap) && return ax, ay, az
    cap <= 0.0 && return ax, ay, az
    a_mag = sqrt(ax * ax + ay * ay + az * az)
    a_mag <= cap && return ax, ay, az
    s = cap / a_mag
    return ax * s, ay * s, az * s
end

# retraction_contact_e_scale=1.0 means inherit compression/hold stiffness at restart.
@inline function active_contact_e_scale()
    retraction_contact_e_scale[] < 1.0 - 1.0e-12 &&
        return retraction_contact_e_scale[]
    return compression_contact_e_scale[]
end

@inline function _reimann_pair_accel(dx::Float64, dy::Float64, dz::Float64,
                                     distance::Float64,
                                     ps::Float64,
                                     E::Float64, rho_i::Float64, m_i::Float64,
                                     vix::Float64, viy::Float64, viz::Float64,
                                     vjx::Float64, vjy::Float64, vjz::Float64,
                                     melt_k_scale::Float64=1.0)
    distance < eps(ps) && return 0.0, 0.0, 0.0
    g_n = distance - ps
    g_n >= 0.0 && return 0.0, 0.0, 0.0

    inv_d = 1.0 / distance
    nx = dx * inv_d
    ny = dy * inv_d
    nz = dz * inv_d

    v_rel = (vix - vjx) * nx + (viy - vjy) * ny + (viz - vjz) * nz

    δ = max(0.0, -g_n)
    δ <= 0.0 && return 0.0, 0.0, 0.0

    A_eff = ps * ps
    δ_tol = contact_penalty_alpha[] * ps
    δ_sat = max(contact_overlap_sat_frac[] * ps, δ_tol + eps(ps))

    k_n = active_contact_e_scale() * E / ps * melt_k_scale
    δ_deep = contact_deep_overlap_frac[] * ps
    ζ = contact_dashpot_zeta[]
    ζ_deep = contact_dashpot_zeta_deep[]
    if ζ_deep > 0.0 && δ > δ_deep
        ζ = ζ_deep
    end
    ζ = max(ζ, 0.0)
    c_n = ζ * 2.0 * sqrt(max(k_n * m_i / A_eff, 0.0))

    # Smooth penalty: linear near δ_tol, saturates at δ_sat (corners / deep overlap).
    δ_excess = max(δ - δ_tol, 0.0)
    δ_pen = δ_sat * tanh(δ_excess / δ_sat)
    t_stab = k_n * δ_pen
    # Viscous dashpot while overlapping: oppose relative normal velocity (approach and
    # rebound). Previously only approaching (v_rel<0) at ζ=0.05 — too weak for demold.
    t_damp = (δ > 0.0 && c_n > 0.0) ? c_n * (-v_rel) : zero(v_rel)
    t_corr = t_stab + t_damp

    # Riemann acoustic term disabled (see prior comment in git history).
    t_n = t_corr
    F_n = t_n * A_eff
    F_n <= 0.0 && return 0.0, 0.0, 0.0

    a_mag = F_n / m_i
    return a_mag * nx, a_mag * ny, a_mag * nz
end

function _clip_interact_Reimann_override!(dv, v_particle_system, u_particle_system,
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
    charge_sys = semi.systems[1]
    on_charge = particle_system === charge_sys
    lf_buf = contact_liquid_fraction_buf[]
    use_melt_contact = on_charge && lf_buf !== nothing
    # Always allow capping on the charge when any cap is configured (incl. deep-overlap).
    cap_total = on_charge && (contact_deep_accel_cap[] > 0.0 ||
        (use_melt_contact ?
            (contact_accel_cap_molten[] > 0.0 || contact_accel_cap_solid[] > 0.0) :
            contact_total_accel_cap[] > 0.0))

    if isdefined(Main, :CUDA) && dv isa Main.CUDA.CuArray
        dv_host = Array(dv)
        system_coords_host = Array(system_coords)
        neighbor_coords_host = Array(neighbor_coords)
        v_particle_host = Array(v_particle_system)
        v_neighbor_host = Array(v_neighbor_system)
        mass_host = Array(particle_system.mass)
        rho_host = Array(particle_system.material_density)

        @inbounds for particle in TrixiParticles.each_integrated_particle(particle_system)
            skip_blankholder_contact_for_particle(particle, neighbor_system) && continue

            xi1 = system_coords_host[1, particle]
            xi2 = system_coords_host[2, particle]
            xi3 = system_coords_host[3, particle]

            vix = v_particle_host[1, particle]
            viy = v_particle_host[2, particle]
            viz = v_particle_host[3, particle]

            E = TrixiParticles.young_modulus(particle_system, particle)
            m_i = mass_host[particle]
            rho_i = rho_host[particle]

            lf = 1.0
            if use_melt_contact && particle <= length(lf_buf)
                lf = lf_buf[particle]
            end
            melt_k_scale = use_melt_contact ? contact_stiffness_melt_scale(lf) : 1.0

            ax = 0.0
            ay = 0.0
            az = 0.0
            δ_max = 0.0

            for neighbor in 1:n_q
                dx = xi1 - neighbor_coords_host[1, neighbor]
                dy = xi2 - neighbor_coords_host[2, neighbor]
                dz = xi3 - neighbor_coords_host[3, neighbor]
                d2 = dx * dx + dy * dy + dz * dz
                d2 > radius2 && continue

                distance = sqrt(d2)
                δ_max = max(δ_max, contact_pair_overlap(distance, ps))

                vjx = v_neighbor_host[1, neighbor]
                vjy = v_neighbor_host[2, neighbor]
                vjz = v_neighbor_host[3, neighbor]

                dax, day, daz = _reimann_pair_accel(dx, dy, dz, distance,
                                                    ps, E, rho_i, m_i,
                                                    vix, viy, viz,
                                                    vjx, vjy, vjz,
                                                    melt_k_scale)
                ax += dax
                ay += day
                az += daz
            end

            if cap_total
                cap = resolve_contact_accel_cap(lf, δ_max, ps, use_melt_contact)
                ax, ay, az = apply_resolved_contact_accel_cap(ax, ay, az, cap)
            end

            dv_host[1, particle] += ax
            dv_host[2, particle] += ay
            dv_host[3, particle] += az
        end

        copyto!(dv, dv_host)
        return dv
    end

    @inbounds for particle in TrixiParticles.each_integrated_particle(particle_system)
        skip_blankholder_contact_for_particle(particle, neighbor_system) && continue

        xi1 = system_coords[1, particle]
        xi2 = system_coords[2, particle]
        xi3 = system_coords[3, particle]

        v_i = TrixiParticles.current_velocity(v_particle_system,
                                              particle_system, particle)
        vix = v_i[1]; viy = v_i[2]; viz = v_i[3]

        E   = TrixiParticles.young_modulus(particle_system, particle)
        m_i = particle_system.mass[particle]
        rho_i = particle_system.material_density[particle]

        lf = 1.0
        if use_melt_contact && particle <= length(lf_buf)
            lf = lf_buf[particle]
        end
        melt_k_scale = use_melt_contact ? contact_stiffness_melt_scale(lf) : 1.0

        ax = 0.0
        ay = 0.0
        az = 0.0
        δ_max = 0.0

        for neighbor in 1:n_q
            dx = xi1 - neighbor_coords[1, neighbor]
            dy = xi2 - neighbor_coords[2, neighbor]
            dz = xi3 - neighbor_coords[3, neighbor]
            d2 = dx * dx + dy * dy + dz * dz
            d2 > radius2 && continue

            distance = sqrt(d2)
            δ_max = max(δ_max, contact_pair_overlap(distance, ps))

            v_j = TrixiParticles.current_velocity(v_neighbor_system,
                                                  neighbor_system, neighbor)
            vjx = v_j[1]; vjy = v_j[2]; vjz = v_j[3]

            dax, day, daz = _reimann_pair_accel(dx, dy, dz, distance,
                                                ps, E, rho_i, m_i,
                                                vix, viy, viz,
                                                vjx, vjy, vjz,
                                                melt_k_scale)
            ax += dax
            ay += day
            az += daz
        end

        if cap_total
            cap = resolve_contact_accel_cap(lf, δ_max, ps, use_melt_contact)
            ax, ay, az = apply_resolved_contact_accel_cap(ax, ay, az, cap)
        end

        dv[1, particle] += ax
        dv[2, particle] += ay
        dv[3, particle] += az
    end

    return dv
end

@eval TrixiParticles begin
    function interact_Reimann!(dv, v_particle_system, u_particle_system,
                               v_neighbor_system, u_neighbor_system,
                               particle_system::TotalLagrangianSPHSystem,
                               neighbor_system::TotalLagrangianSPHSystem,
                               semi; integrate_tlsph=semi.integrate_tlsph[])
        return Main._clip_interact_Reimann_override!(dv, v_particle_system, u_particle_system,
                                                     v_neighbor_system, u_neighbor_system,
                                                     particle_system, neighbor_system,
                                                     semi; integrate_tlsph=integrate_tlsph)
    end

    function initialize_save_cb!(solution_callback::SolutionSavingCallback, u, t, integrator)
        restore_active = (get(ENV, "TP_CLIP_RESTORE_CHECKPOINT", "0") == "1") || 
                         (get(ENV, "TP_CLIP_SIM_PHASE", "") == "retraction")
        
        if restore_active
            max_idx = -1
            out_dir = solution_callback.output_directory
            prefix = solution_callback.prefix
            if isdir(out_dir)
                for f in readdir(out_dir)
                    if startswith(f, prefix) && endswith(f, ".vtu")
                        idx_underscore = findlast('_', f)
                        if idx_underscore !== nothing
                            idx_dot = findlast('.', f)
                            if idx_dot !== nothing && idx_dot > idx_underscore + 1
                                idx_str = f[idx_underscore+1:idx_dot-1]
                                val = tryparse(Int, idx_str)
                                if val !== nothing
                                    max_idx = max(max_idx, val)
                                end
                            end
                        end
                    end
                end
            end
            if max_idx >= 0
                solution_callback.latest_saved_iter = max_idx
                println(">>> Restored VTU output sequence starting from index ", max_idx + 1)
                flush(stdout)
            else
                solution_callback.latest_saved_iter = -1
            end
        else
            solution_callback.latest_saved_iter = -1
        end

        solution_callback.git_hash[] = compute_git_hash()
        write_meta_data(solution_callback, integrator)

        if solution_callback.save_initial_solution
            solution_callback(integrator; from_initialize=true)
        end
    end
end
