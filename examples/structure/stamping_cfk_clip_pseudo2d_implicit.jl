# ==========================================================================================
# Pseudo-2D Thermomechanical Forming of a CFK Clip-Style Part (Implicit)
#
# Same physics stack as `stamping_cfk_clip_3d_implicit.jl`, but the domain is a
# thin slab — 1 particle thick along y — so it runs much faster for testing.
# Everything else (dual-phase constitutive model, viscous/elastic stress cache,
# plasticity history, flow-orientation kinetics, contact heat flux, implicit
# TRBDF2 solver) is unchanged.
#
# Tooling is a deep-draw clip geometry:
#   - Lower tool (female die): wide flat base + two tall side blocks leaving a
#     deep central cavity.
#   - Upper tool (male punch): full-width holder + long central protrusion that
#     mates exactly into the cavity (with one-particle clearance per side).
# ==========================================================================================
using TrixiParticles
using OrdinaryDiffEq
using ADTypes
using LinearSolve
using RecursiveFactorization
using KrylovKit
using IncompleteLU
using PointNeighbors
using Plots
using Base.Threads
using Statistics
using LinearAlgebra
using StaticArrays
using SparseArrays
using Logging
using Serialization
using Printf

include("../../src/schemes/structure/total_lagrangian_sph/fix_reimann.jl")

function env_float(name, default)
    value = get(ENV, name, string(default))
    parsed = tryparse(Float64, value)
    parsed === nothing && error("Invalid Float64 for $(name): $(value)")
    return parsed
end

function env_int(name, default)
    value = get(ENV, name, string(default))
    parsed = tryparse(Int, value)
    parsed === nothing && error("Invalid Int for $(name): $(value)")
    return parsed
end

# Implicit Jacobian mode for TRBDF2.
#   finite  — AutoFiniteDiff (default; compatible with kick_implicit_visible! Float64 scratch buffers)
#   forward — AutoForwardDiff (experimental: fails if RHS writes Dual into Vector{Float64})
function clip_jacobian_mode()
    mode = lowercase(get(ENV, "TP_CLIP_JACOBIAN_MODE", "finite"))
    mode in ("finite", "forward") ||
        error("TP_CLIP_JACOBIAN_MODE must be 'finite' or 'forward' (got $(mode)). " *
              "Sparse Jacobian AD is not supported for DynamicalODEProblem in this script.")
    mode == "forward" &&
        @warn "TP_CLIP_JACOBIAN_MODE=forward is experimental for this script: " *
              "kick_implicit_visible! uses Float64 work buffers and may throw " *
              "MethodError(convert, (Float64, Dual(...))) during Jacobian assembly."
    return mode
end

function clip_jacobian_autodiff(mode::String)
    if mode == "forward"
        return AutoForwardDiff(; chunksize=12)
    end
    return AutoFiniteDiff()
end

# Linear solver for TRBDF2 Newton steps (W = I - γJ).
#   dense / rfluf / lu — RFLUFactorization (default; best for small n_dof)
#   krylov / gmres    — KrylovKit GMRES (only worthwhile when n_dof ≫ 500)
function clip_linsolve_solver(n_dof::Int; default_mode::String="dense")
    mode = lowercase(get(ENV, "TP_CLIP_LINSOLVE", default_mode))
    mode in ("dense", "rfluf", "lu", "krylov", "gmres") ||
        error("TP_CLIP_LINSOLVE must be dense|rfluf|lu|krylov|gmres (got $(mode))")
    if mode in ("krylov", "gmres")
        # Frequent GMRES restarts with small krylovdim trigger KrylovKit bugs when the
        # ODE state mixes ArrayPartition with ThreadedBroadcastArray (full molding runs).
        # Retraction-only uses plain Vector{Float64} states (n_dof≈147), so GMRES runs,
        # but dense LU is still faster at that size because TRBDF2 builds a full dense W.
        n_dof < 500 &&
            @warn "TP_CLIP_LINSOLVE=krylov with n_dof=$n_dof: implicit step time is " *
                  "RHS-dominated; dense LU is typically faster. Krylov helps mainly when " *
                  "n_dof is large (full 3D molding)."
        # stamping_cfrp_3d_2_implicit.jl uses krylovdim=40, maxiter=200.
        krylovdim = env_int("TP_CLIP_KRYLOV_DIM", 40)
        return KrylovKitJL_GMRES(; krylovdim=krylovdim,
                                   atol=env_float("TP_CLIP_KRYLOV_ATOL", 1.0e-6),
                                   rtol=env_float("TP_CLIP_KRYLOV_RTOL", 5.0e-2),
                                   maxiter=env_int("TP_CLIP_KRYLOV_MAXITER", 200),
                                   verbosity=0)
    end
    return RFLUFactorization()
end

# Simulation phase:
#   full       — compression + hold + retraction (default)
#   hold       — compression + hold, save checkpoint, stop before retraction
#   retraction — load checkpoint, run retraction only
const CHECKPOINT_VERSION = 2
const sim_phase = lowercase(get(ENV, "TP_CLIP_SIM_PHASE", "full"))
sim_phase in ("full", "hold", "retraction") ||
    error("TP_CLIP_SIM_PHASE must be full, hold, or retraction (got $(sim_phase))")
# Run tag from particle spacing [mm]: 1.8 -> "18", 2.6 -> "26" (used in VTU prefix + checkpoint name).
const ps_mm_nominal = env_float("TP_CLIP_PS_MM", 1.8)
@inline function ps_mm_run_tag(ps_mm::Real)
    return string(round(Int, round(Float64(ps_mm) * 10)))
end
const ps_run_tag = ps_mm_run_tag(ps_mm_nominal)
const save_checkpoint_at_hold = env_int("TP_CLIP_SAVE_CHECKPOINT", 1) != 0
const checkpoint_path = get(ENV, "TP_CLIP_CHECKPOINT",
                            joinpath("out",
                                     "molding_cfk_pseudo2d_hold_checkpoint_" * ps_run_tag * ".dat"))
# TLSPH equilibration after solidification, before checkpoint (WCSPH off during tail).
# Stress relaxation at hold temperature while the die stays closed (thermoplastic creep /
# contact redistribution). Default 0.15 s — increase if checkpoint still shows tool overlap.
const hold_tlsph_tail_s = env_float("TP_CLIP_HOLD_TLSPH_TAIL_S", 0.15)
# Hold end: Spencer fibre reference n_ref ← F⁻¹·ê so I4=|F·n_ref|²=1 at formed F (stress-free fibres at demold).
const hold_end_spencer_align = env_int("TP_CLIP_HOLD_END_SPENCER_ALIGN", 1) != 0
# Strategy E: volumetric plasticity J_p with isochoric F_p. det(F) = J_e_vol * J_p; bulk uses J_e_vol vs 1.
const use_volumetric_plasticity = env_int("TP_CLIP_VOLUMETRIC_PLASTICITY", 1) != 0
const hold_end_fp_reset = env_int("TP_CLIP_HOLD_END_FP_RESET", use_volumetric_plasticity ? 0 : 1) != 0
const volumetric_plastic_relax_rate = env_float("TP_CLIP_JP_RELAX_RATE", 1.0)
const elastic_bulk_J_ref = 1.0
# Charge cohesion during springback: Monaghan tensile stabilisation (psi<0); optional gap-column Vf floor.
const charge_tensile_psi = env_float("TP_CLIP_CHARGE_TENSILE_PSI",
                                     sim_phase == "retraction" ? -0.1 : 0.0)
const charge_tensile_exponent = max(2, env_int("TP_CLIP_CHARGE_TENSILE_EXPONENT", 4))
const gap_vf_override = haskey(ENV, "TP_CLIP_GAP_VF") ? env_float("TP_CLIP_GAP_VF", 0.02) : nothing
# Do not release charge kinematics faster than elastic σ is ramped on (prevents tow-band delamination).
const retraction_kin_cap_by_elastic = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_KIN_CAP_ELASTIC", 1) != 0
# After punch lift: continue mechanics until springback equilibrates (mold gap-based contact already off).
const retraction_springback_dwell_s = env_float("TP_CLIP_RETRACTION_SPRINGBACK_DWELL_S", 0.05)
const retraction_springback_equil_max_speed = env_float("TP_CLIP_RETRACTION_SPRINGBACK_EQUIL_MAX_SPEED", 0.05)
const retraction_springback_equil_span_rate_mm_s =
    env_float("TP_CLIP_RETRACTION_SPRINGBACK_EQUIL_SPAN_RATE_MM_S", 0.02)
const retraction_springback_equil_patience =
    max(1, env_int("TP_CLIP_RETRACTION_SPRINGBACK_EQUIL_PATIENCE", 8))
# Hold kinematics: hybrid (default) = frozen during solidification hold, active during TLSPH tail;
# frozen = no motion entire closed-die hold; active = creep motion throughout hold.
function clip_hold_kinematics_mode()
    mode = lowercase(get(ENV, "TP_CLIP_HOLD_KINEMATICS", "hybrid"))
    if !haskey(ENV, "TP_CLIP_HOLD_KINEMATICS") && haskey(ENV, "TP_CLIP_HOLD_FREEZE_KINEMATICS")
        return env_int("TP_CLIP_HOLD_FREEZE_KINEMATICS", 0) != 0 ? :frozen : :active
    end
    mode in ("hybrid", "frozen", "active") ||
        error("TP_CLIP_HOLD_KINEMATICS must be 'hybrid', 'frozen', or 'active' (got $(mode))")
    return Symbol(mode)
end
const hold_kinematics_mode = (sim_phase == "full" || sim_phase == "hold") ?
    clip_hold_kinematics_mode() : :active
const hold_checkpoint_early = Ref{Union{Nothing, Dict{String, Any}}}(nothing)
# Retraction phase: TLSPH elastoplastic only (no WCSPH). Restores from .dat checkpoint.
# Default ON: springback uses solid TLSPH only (WCSPH / melt paths disabled during retraction).
const retraction_tlsph_only = env_int("TP_CLIP_RETRACTION_TLSPH_ONLY", 1) != 0
# Match stamping_cfrp_3d_2_implicit.jl when enabled. Default OFF for clip: gradual σ/contact/
# kinematics ramps during punch release are the physically intended springback path.
const retraction_cfrp_style = sim_phase == "retraction" && retraction_tlsph_only &&
    env_int("TP_CLIP_RETRACTION_CFRP_STYLE", 0) != 0
const disable_wcsph = env_int("TP_CLIP_DISABLE_WCSPH", 0) != 0

@inline function retraction_cfrp_style_active()
    return retraction_cfrp_style && retraction_started[]
end

function clip_retraction_integrator()
    sim_phase != "retraction" && return :implicit
    mode = lowercase(get(ENV, "TP_CLIP_RETRACTION_INTEGRATOR", "implicit"))
    mode in ("implicit", "explicit") ||
        error("TP_CLIP_RETRACTION_INTEGRATOR must be 'implicit' or 'explicit' (got $(mode))")
    return Symbol(mode)
end

function clip_retraction_explicit_algorithm()
    # VelocityVerlet requires drift du ≡ v exactly; kinematics ramp scales du in
    # drift_implicit_visible!, so only symplectic Euler is valid here.
    scheme = lowercase(get(ENV, "TP_CLIP_RETRACTION_EXPLICIT_SCHEME", "euler"))
    if scheme in ("euler",)
        return Euler(), "Euler"
    elseif scheme in ("velocity_verlet", "velocityverlet", "verlet")
        error("TP_CLIP_RETRACTION_EXPLICIT_SCHEME=velocity_verlet is incompatible with the ",
              "retraction kinematics ramp (drift du is scaled). Use 'euler'.")
    else
        error("TP_CLIP_RETRACTION_EXPLICIT_SCHEME must be 'euler' (got $(scheme))")
    end
end

const retraction_integrator = clip_retraction_integrator()
const retraction_use_explicit = retraction_integrator == :explicit
# Explicit springback: freeze T (no conduction / HTC in kick) — mechanics-only is much cheaper.
const retraction_explicit_freeze_t = retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_EXPLICIT_FREEZE_T", 1) != 0
const retraction_explicit_dt = env_float("TP_CLIP_RETRACTION_EXPLICIT_DT", 5.0e-7)
# After the explicit mech-ramp freeze, springback needs a smaller dt (5e-7 blows up at release).
const retraction_explicit_dt_springback = retraction_use_explicit ?
    env_float("TP_CLIP_RETRACTION_EXPLICIT_DT_SPRINGBACK", 1.0e-7) :
    retraction_explicit_dt
# SciMLBase default unstable_check aborts on any NaN/Inf in u; explicit contact can spike
# briefly before the accepted-step repair runs — keep going and let the callback fix state.
const retraction_explicit_ignore_unstable = retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_EXPLICIT_IGNORE_UNSTABLE", 1) != 0

@inline function retraction_explicit_active()
    return retraction_use_explicit
end

# Explicit retraction uses the same blended σ/contact/kinematics ramp as implicit.
# Explicit Euler also needs springback dt from t0 and matched kick/drift kin scaling.
function enforce_explicit_retraction_dt!(integrator)
    if !(retraction_use_explicit && retraction_started[])
        return nothing
    end
    sb_dt = retraction_explicit_dt_springback
    if integrator.dt != sb_dt
        integrator.dt = sb_dt
    end
    return nothing
end

@inline function apply_velocity_safety_clamps()
    return enable_velocity_safety_clamps || retraction_accel_clamp
end

@inline function apply_coordinate_safety_clamps()
    return enable_safety_clamps
end

println("--- SIMULATION STARTING (pseudo-2D) ---")
println("Threads available: ", nthreads())
println("Particle spacing: ", ps_mm_nominal, " mm (run tag=", ps_run_tag, ")")
println("Simulation phase: ", sim_phase,
        " (checkpoint: ", abspath(checkpoint_path), ")")
(sim_phase == "full" || sim_phase == "hold") && hold_tlsph_tail_s > 0.0 &&
    println("Hold: TLSPH equilibration tail ", hold_tlsph_tail_s,
            " s after solidification (WCSPH off",
            hold_kinematics_mode == :frozen ? ", kinematics frozen" :
            hold_kinematics_mode == :hybrid ? ", kinematics active in tail only" :
            ", kinematics active (full hold creep)",
            ") before checkpoint")
(sim_phase == "full" || sim_phase == "hold") && hold_end_spencer_align &&
    println("Hold end: Spencer fibre reference aligned to hold F (I4=1 at demold)")
use_volumetric_plasticity &&
    println("Volumetric plasticity (Strategy E): J_p + isochoric F_p; bulk ln(J_e_vol) vs ",
            elastic_bulk_J_ref, "; hold-end Fp reset=", hold_end_fp_reset)
(sim_phase == "full" || sim_phase == "hold") &&
    println("Hold kinematics mode: ", hold_kinematics_mode,
            hold_kinematics_mode == :hybrid ?
                " (solidification hold frozen; TLSPH tail creep + contact)" :
            hold_kinematics_mode == :frozen ?
                " (thermal-only + plastic commit on fixed F throughout)" :
                " (TLSPH/contact while mold closed)")
sim_phase == "retraction" && retraction_cfrp_style &&
    println("Retraction: CFRP-style implicit (full system_interaction!, elastic trial σ, ",
            "GMRES/TRBDF2; no σ/contact/kin ramps)")
sim_phase == "retraction" && retraction_tlsph_only && !retraction_cfrp_style &&
    println("Retraction: TLSPH elastoplastic + charge conduction + tool–charge HTC; ",
            "WCSPH, Nakamura, molten drag, and flow-orientation kinetics are OFF")
sim_phase == "retraction" && retraction_use_explicit && begin
    _, explicit_scheme_name = clip_retraction_explicit_algorithm()
    println("Retraction integrator: explicit ", explicit_scheme_name,
            " (dt=", retraction_explicit_dt_springback,
            " s; same σ/contact ramp as implicit; matched kick/drift kin scaling; freeze T=",
            retraction_explicit_freeze_t,
            "; ignore SciML unstable abort=", retraction_explicit_ignore_unstable, ")")
end
println("---------------------------------------")

function checkpoint_meta(; particle_spacing, n_charge, n_floor, n_mold,
                         t_compress, z_shift_down_end, mold_velocity_state,
                         compression_thickness_ratio=missing,
                         hold_tlsph_tail_s=missing,
                         hold_tlsph_equilibrated=missing)
    meta = Dict{String, Any}(
        "version" => CHECKPOINT_VERSION,
        "particle_spacing" => particle_spacing,
        "n_charge" => n_charge,
        "n_floor" => n_floor,
        "n_mold" => n_mold,
        "t_compress" => t_compress,
        "z_shift_down_end" => z_shift_down_end,
        "mold_velocity_state" => mold_velocity_state,
    )
    compression_thickness_ratio !== missing &&
        (meta["compression_thickness_ratio"] = compression_thickness_ratio)
    hold_tlsph_tail_s !== missing && (meta["hold_tlsph_tail_s"] = hold_tlsph_tail_s)
    hold_tlsph_equilibrated !== missing &&
        (meta["hold_tlsph_equilibrated"] = hold_tlsph_equilibrated)
    return meta
end

function validate_checkpoint_meta!(ck, meta_now; motion_from_checkpoint=false)
    ck_meta = ck["meta"]
    layout_keys = ("particle_spacing", "n_charge", "n_floor", "n_mold")
    motion_keys = ("t_compress", "z_shift_down_end", "mold_velocity_state")
    keys_to_check = motion_from_checkpoint ? layout_keys : (layout_keys..., motion_keys...)
    for key in keys_to_check
        if !haskey(ck_meta, key) || ck_meta[key] != meta_now[key]
            hint = key == "n_mold" ?
                   " Try TP_CLIP_COMPRESSION_RATIO=0.85 (or re-run hold with current geometry)." :
                   ""
            error("Checkpoint meta mismatch on '$(key)': saved=$(get(ck_meta, key, nothing)), " *
                  "current=$(meta_now[key]). Use the same geometry/env as the hold run." * hint)
        end
    end
    return nothing
end

# Estimate male-punch particle count for a trial compression ratio (must match build_punch_particles).
function estimate_mold_particle_count(compression_ratio;
                                      particle_spacing, charge_bottom_clearance,
                                      charge_length, charge_thickness_discrete,
                                      cyl_z_max, initial_gap_upper, cavity_depth,
                                      tool_length, punch_side_clearance, holder_thickness,
                                      cavity_internal_width, z_base_top=0.0)
    ps = particle_spacing
    final_thickness = compression_ratio * charge_thickness_discrete
    charge_top_outer = cyl_z_max + 0.5 * ps
    z_punch_bot = charge_top_outer + initial_gap_upper
    z_punch_top = z_punch_bot + cavity_depth
    initial_stroke = z_punch_bot - (z_base_top + final_thickness)
    z_blocks_top = z_base_top + cavity_depth
    cavity_top_half_w = 0.5 * cavity_internal_width
    cavity_bot_half_w = 0.5 * (charge_length + 0.003)
    nx_tool = ceil(Int, tool_length / ps)
    ny_tool = 1
    tool_origin_x = -0.5 * tool_length
    holder_layers = max(1, ceil(Int, holder_thickness / ps))
    nz_body = ceil(Int, (z_punch_top - z_punch_bot) / ps)
    n_mold = 0
    for k in 1:nz_body
        z = z_punch_bot + (k - 0.5) * ps
        z_seated = clamp(z - initial_stroke, z_base_top, z_blocks_top)
        frac = (z_seated - z_base_top) / (z_blocks_top - z_base_top)
        frac = clamp(frac, 0.0, 1.0)
        half_w_die = cavity_bot_half_w + frac * (cavity_top_half_w - cavity_bot_half_w)
        half_w = max(0.5 * punch_side_clearance, half_w_die - punch_side_clearance)
        for i in 1:nx_tool
            x = tool_origin_x + (i - 0.5) * ps
            abs(x) < half_w && (n_mold += 1)
        end
    end
    return n_mold + nx_tool * ny_tool * holder_layers
end

function infer_compression_ratio_for_hold_meta(ck_meta;
                                             particle_spacing, charge_bottom_clearance,
                                             charge_length, charge_thickness_discrete,
                                             cyl_z_max, initial_gap_upper, cavity_depth,
                                             tool_length, punch_side_clearance, holder_thickness,
                                             cavity_internal_width,
                                             preferred=0.9)
    target_n_mold = ck_meta["n_mold"]
    best_ratio = preferred
    best_gap = Inf
    for cr in 0.50:0.001:1.0
        n_mold = estimate_mold_particle_count(cr;
            particle_spacing=particle_spacing,
            charge_bottom_clearance=charge_bottom_clearance,
            charge_length=charge_length,
            charge_thickness_discrete=charge_thickness_discrete,
            cyl_z_max=cyl_z_max, initial_gap_upper=initial_gap_upper,
            cavity_depth=cavity_depth, tool_length=tool_length,
            punch_side_clearance=punch_side_clearance,
            holder_thickness=holder_thickness,
            cavity_internal_width=cavity_internal_width)
        if n_mold == target_n_mold
            gap = abs(cr - preferred)
            if gap < best_gap
                best_gap = gap
                best_ratio = cr
            end
        end
    end
    best_gap < Inf ||
        error("Could not infer TP_CLIP_COMPRESSION_RATIO for checkpoint n_mold=$(target_n_mold). " *
              "Set TP_CLIP_COMPRESSION_RATIO explicitly to match the hold run.")
    return best_ratio
end

function capture_hold_checkpoint_state(integrator, semi_local)
    cyl_sys = semi_local.systems[1]
    floor_sys = semi_local.systems[2]
    mold_sys = semi_local.systems[3]
    return Dict{String, Any}(
        "version" => CHECKPOINT_VERSION,
        "t" => integrator.t,
        "meta" => checkpoint_meta(; particle_spacing=particle_spacing,
                                  n_charge=n_cylinder_particles,
                                  n_floor=TrixiParticles.nparticles(floor_sys),
                                  n_mold=TrixiParticles.nparticles(mold_sys),
                                  t_compress=t_compress,
                                  z_shift_down_end=z_shift_down_end,
                                  mold_velocity_state=mold_velocity_state,
                                  compression_thickness_ratio=compression_thickness_ratio,
                                  hold_tlsph_tail_s=hold_tlsph_tail_s,
                                  hold_tlsph_equilibrated=hold_tlsph_tail_s > 0.0),
        "v_ode" => copy(integrator.u.x[1]),
        "u_ode" => copy(integrator.u.x[2]),
        "deformation_grad" => copy(cyl_sys.deformation_grad),
        "charge_temp" => copy(cyl_sys.temp),
        "floor_temp" => copy(floor_sys.temp),
        "mold_temp" => copy(mold_sys.temp),
        "Fp_state" => copy(Fp_state[]),
        "Fp_committed" => copy(Fp_committed[]),
        "alpha_committed" => copy(alpha_committed[]),
        "J_ref_particle_buf" => copy(J_ref_particle_buf),
        "J_p_particle_buf" => copy(J_p_particle_buf),
        "crystallinity_particle_buf" => copy(crystallinity_particle_buf),
        "liquid_fraction_particle_buf" => copy(liquid_fraction_particle_buf),
        "solid_fraction_particle_buf" => copy(solid_fraction_particle_buf),
        "ys_particle_buf" => copy(ys_particle_buf),
        "hard_particle_buf" => copy(hard_particle_buf),
        "vis_particle_buf" => copy(vis_particle_buf),
        "thermal_softening_particle_buf" => copy(thermal_softening_particle_buf),
        "orientation_tensor" => copy(orientation_tensor_state[]),
        "fiber_direction" => copy(fiber_direction),
        "vel_grad_buf" => copy(vel_grad_buf),
        "nhs_updated_at_t" => nhs_updated_at_t[],
        "wcsph_ramp_scale" => wcsph_ramp_scale[],
        "solidification_reached_time" => solidification_reached_time[],
        "solidification_complete" => solidification_complete[],
        "solidification_hold_announced" => solidification_hold_announced[],
        "charge_hold_dtmax_active" => charge_hold_dtmax_active[],
        "solver_dtmax_before_hold" => solver_dtmax_before_hold[],
    )
end

function save_hold_checkpoint!(path, ck)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        serialize(io, ck)
    end
    println(">>> Saved hold checkpoint: ", abspath(path), " at t=", round(ck["t"]; digits=6), " s")
    flush(stdout)
    return nothing
end

function maybe_save_periodic_checkpoint!(integrator, semi_local)
    isempty(periodic_checkpoint_save_times) && return
    retraction_started[] && return
    t = integrator.t
    idx = periodic_checkpoint_index[]
    while idx + 1 <= length(periodic_checkpoint_save_times) &&
          t + 1.0e-12 >= periodic_checkpoint_save_times[idx + 1]
        idx += 1
        periodic_checkpoint_index[] = idx
        ck = capture_hold_checkpoint_state(integrator, semi_local)
        save_hold_checkpoint!(checkpoint_path, ck)
    end
    return nothing
end

function load_hold_checkpoint(path)
    isfile(path) || error("Hold checkpoint not found: $(abspath(path))")
    ck = open(path, "r") do io
        deserialize(io)
    end
    ck["version"] in (1, CHECKPOINT_VERSION) ||
        error("Unsupported checkpoint version $(ck["version"]) (expected $(CHECKPOINT_VERSION))")
    return ck
end

function restore_hold_checkpoint_state!(ck, semi_local)
    cyl_sys = semi_local.systems[1]
    floor_sys = semi_local.systems[2]
    mold_sys = semi_local.systems[3]

    cyl_sys.deformation_grad .= ck["deformation_grad"]
    cyl_sys.temp .= ck["charge_temp"]
    floor_sys.temp .= ck["floor_temp"]
    mold_sys.temp .= ck["mold_temp"]

    v_ck = ck["v_ode"]
    u_ck = ck["u_ode"]
    v_cyl = TrixiParticles.wrap_v(v_ck, cyl_sys, semi_local)
    u_cyl = TrixiParticles.wrap_u(u_ck, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        for d in 1:NDIMS_CYL
            cyl_sys.current_coordinates[d, particle] = u_cyl[d, particle]
            cyl_sys.initial_condition.velocity[d, particle] = v_cyl[d, particle]
            v_cyl[NDIMS_CYL + 1, particle] = cyl_sys.temp[particle]
        end
    end

    Fp_state[] .= ck["Fp_state"]
    Fp_committed[] .= ck["Fp_committed"]
    alpha_committed[] .= ck["alpha_committed"]
    J_ref_particle_buf .= ck["J_ref_particle_buf"]
    if haskey(ck, "J_p_particle_buf")
        J_p_particle_buf .= ck["J_p_particle_buf"]
    else
        fill!(J_p_particle_buf, 1.0)
    end
    crystallinity_particle_buf .= ck["crystallinity_particle_buf"]
    liquid_fraction_particle_buf .= ck["liquid_fraction_particle_buf"]
    solid_fraction_particle_buf .= ck["solid_fraction_particle_buf"]
    ys_particle_buf .= ck["ys_particle_buf"]
    hard_particle_buf .= ck["hard_particle_buf"]
    vis_particle_buf .= ck["vis_particle_buf"]
    thermal_softening_particle_buf .= ck["thermal_softening_particle_buf"]
    orientation_tensor_state[] .= ck["orientation_tensor"]
    fiber_direction .= ck["fiber_direction"]
    vel_grad_buf .= ck["vel_grad_buf"]
    nhs_updated_at_t[] = ck["nhs_updated_at_t"]
    wcsph_ramp_scale[] = ck["wcsph_ramp_scale"]
    solidification_reached_time[] = ck["solidification_reached_time"]
    solidification_complete[] = ck["solidification_complete"]
    solidification_hold_announced[] = ck["solidification_hold_announced"]
    charge_hold_dtmax_active[] = ck["charge_hold_dtmax_active"]
    solver_dtmax_before_hold[] = ck["solver_dtmax_before_hold"]

    retraction_started[] = false
    retraction_complete[] = false
    mold_retraction_complete[] = false
    springback_equil_streak[] = 0
    springback_equil_prev_snap[] = nothing
    retraction_start_time[] = Inf
    mold_retraction_complete_time[] = Inf
    retraction_state_prepared[] = false
    retraction_reference_span_x[] = 0.0
    retraction_reference_span_z[] = 0.0
    charge_mechanics_release_announced[] = false
    retraction_elastic_ramp_announced[] = false
    retraction_kinematics_release_announced[] = false
    TrixiParticles.initialize_neighborhood_searches!(semi_local)
    TrixiParticles.foreach_system(semi_local) do system
        TrixiParticles.initialize!(system, semi_local)
        v_sys = TrixiParticles.wrap_v(v_ck, system, semi_local)
        u_sys = TrixiParticles.wrap_u(u_ck, system, semi_local)
        TrixiParticles.update_positions!(system, v_sys, u_sys, v_ck, u_ck, semi_local, ck["t"])
    end

    ck_meta = get(ck, "meta", Dict{String, Any}())
    equilibrated = get(ck_meta, "hold_tlsph_equilibrated", false)
    println(">>> Restored hold checkpoint at t=", round(ck["t"]; digits=6),
            " s (solidification_complete=", solidification_complete[],
            ", TLSPH tail equilibrated=", equilibrated, ")")
    sim_phase == "retraction" && !equilibrated && hold_tlsph_tail_s > 0.0 &&
        @warn "Checkpoint predates TLSPH hold tail — re-run hold with TP_CLIP_HOLD_TLSPH_TAIL_S>0 " *
              "for a self-consistent solid TLSPH state before retraction."
    flush(stdout)
    return nothing
end

# Solidified springback: elastoplastic TLSPH + tool contact only.
@inline function retraction_solid_mechanics_only()
    retraction_tlsph_only || return false
    sim_phase == "retraction" && return true
    return retraction_started[]
end

@inline function wcsph_mechanics_active()
    disable_wcsph && return false
    retraction_solid_mechanics_only() && return false
    hold_tlsph_equilibration_now[] && return false
    return true
end

# Force fully-solid phase buffers for TLSPH-only paths (hold tail + retraction).
function refresh_retraction_solid_phase_buffers!()
    (retraction_solid_mechanics_only() || hold_tlsph_equilibration_now[]) || return nothing
    crystallinity_particle_buf .= 1.0
    solid_fraction_particle_buf .= 1.0
    liquid_fraction_particle_buf .= 0.0
    return nothing
end

# ==========================================================================================
# STEP 1: Build CFK raw-charge geometry (flat preform blank, 1-particle thick in y)
# ==========================================================================================

particle_spacing = ps_mm_nominal * 1e-3

# Mass scaling kept for consistency with the 3D file (stays at 1.0 here).
mass_scaling = 1.0

# Composite feedstock (carbon-fibre reinforced thermoplastic, CF-PP)
matrix_density = 1180.0 * mass_scaling
glass_density = 1750.0 * mass_scaling  # CF (T700-class PAN), vs glass 2550 kg/m³
matrix_E = 3.2e9
glass_E = 230.0e9                       # CF standard modulus (T700/T300), vs glass 72 GPa
# Bulk modulus of the polymer matrix in the melt state.
# Thermoplastic melts (PA6, PP, PPS) are nearly incompressible: K_melt ~ 1.0-1.5 GPa
# (Zoller & Walsh 1995, PVT data).  Physical value ~1.5e9 Pa, but using a reduced
# numerical bulk modulus here: large enough to keep density variations < ~1% under
# typical stamping pressures (~1 MPa), small enough that the TRBDF2 Jacobian is
# well-conditioned.  Revert to 1.5e9 if accurate compressibility modelling is needed.
# Numerically reduced from physical ~1.5 GPa: keeps TRBDF2 Jacobian well-conditioned.
# EOS blends K_melt (near-solidification) → K_liq (hot melt).
matrix_K_melt = 1e8
matrix_cp = 1800.0
glass_cp = 710.0           # CF specific heat [J/(kg·K)], vs glass 840 J/(kg·K) (Chung 2010)
matrix_k = 0.22
glass_k = 1.0              # CF transverse thermal conductivity [W/(m·K)], standard modulus PAN-CF (T700)
                           # 0/90 lay-up: all fibres are in-plane, so z (through-thickness, primary cooling
                           # direction) is always transverse to fibres → k_transverse ≈ 1.0 W/(m·K).
                           # Axial CF k (~10 W/(m·K)) is irrelevant for through-thickness solidification.
# Physical through-thickness k is too low for coarse 0.55 mm SPH to cool the compressed
# slab on a ~30 s hold (L²/α ~ 190 s at physical k). Scale k so centre T and Nakamura
# solidification match validation 06 / Ijaz (2007) GFPP hold timeline (≈29 s to α≥0.92).
# Override: TP_CLIP_THERMAL_CONDUCTIVITY_SCALE (1.0 = physical k only).
thermal_conductivity_scale = env_float("TP_CLIP_THERMAL_CONDUCTIVITY_SCALE", 4.5)
matrix_tmelt = 398.0      # K (125°C) — iPP crystallisation onset on cooling, Brucato 2002 / Pantani 2005
glass_tmelt = 1700.0
matrix_temp_liq = 378.0   # K (105°C) — iPP crystallisation completion on cooling, Brucato 2002
glass_temp_liq = 1500.0
# Enthalpy of fusion for iPP: 165 J/g (literature ΔHf for 100% crystalline = 209 J/g × ~79% crystallinity).
# Source: Wunderlich 1990, Brucato 2002. Replaces previous 110 kJ/kg DSC-melting value.
matrix_latent_heat = 1.65e5
# ==========================================================================================
# Nakamura non-isothermal crystallization kinetics (Nakamura 1972; Pantani et al. 2005)
# Model:  dα/dt = n·K_N(T)·(1−α)·[−ln(1−α)]^((n−1)/n)
#         K_N(T) = K_0 · exp(−U*/(R·(T−T∞))) · exp(−Kg/(T·ΔT))
# Parameters fitted to iPP DSC data (Brucato 2002, Pantani 2005):
#   n  = 3     (Avrami exponent, 3-D spherulitic growth)
#   T_m0 = 460 K   (equilibrium melting point, Hoffman-Lauritzen)
#   T_∞  = 223 K   (T_glass − 30 K; mobility freezes below this)
#   U*   = 6284 J/mol  (transport activation energy, WLF universal constant)
#   Kg   = 3.5e5 K²   (Lauritzen-Hoffman nucleation constant, α-iPP)
#   K_0, Kg, T_m0 calibrated vs Brucato (2002) non-isothermal DSC onset/peak T at Φ=2.5–40 K/s
# ==========================================================================================
nakamura_n      = 3.0          # Avrami exponent
nakamura_K0     = env_float("TP_CLIP_NAKAMURA_K0", 891251.0) # pre-exponential [1/s]
nakamura_U_star = 6284.0       # J/mol
nakamura_T_inf  = 223.0        # K
nakamura_Kg     = 245471.0     # K² — Lauritzen–Hoffman nucleation constant
nakamura_T_m0   = 453.0        # K  (equilibrium melting point)
nakamura_R      = 8.31446      # J/(mol·K)
nakamura_seed_crystallinity = env_float("TP_CLIP_NAKAMURA_SEED", 1.0e-6)
# Flag: when true the Nakamura ODE drives latent-heat release and phase fractions;
# the temperature-based apparent-cp fallback is disabled to avoid double counting.
use_nakamura_kinetics = true
thermal_softening_reference_temp = 270.0
matrix_viscosity = env_float("TP_CLIP_MATRIX_VISCOSITY_PA_S", 1.86e8) # Pa·s at T_liq — Cross–WLF η_ref vs Laun (1986) iPP
matrix_viscosity_ref_temp = matrix_temp_liq
# Previous hot-flow form kept below as commented reference:
#   eta_0(T) = eta_ref * exp(E_a / R * (1 / T - 1 / T_ref))
#   eta(T, gamma_dot) = eta_0 / (1 + (lambda * gamma_dot)^(1 - n))
# matrix_viscosity_activation_energy = 6.0e4
# viscosity_cross_time_constant = 1.0e-2
# viscosity_cross_power_law_index = 0.35
# Active hot-flow form: zero-pressure Cross-WLF, anchored so eta_0(T_ref) = matrix_viscosity.
# A1=15 with η_ref=8.0e6 gives η₀ ≈ 1000 Pa·s at 453 K (180°C), which is
# appropriate for a CF-reinforced thermoplastic charge after fiber concentration
# effects. The earlier η_ref=8e4 gave ~10 Pa·s, too liquid-like for punch contact.
cross_wlf_A1 = 15.0
cross_wlf_A2 = 50.0
cross_wlf_transition_temp = matrix_viscosity_ref_temp
# Critical stress τ* sets the onset of shear-thinning: γ̇* = τ*/η₀.
# 2e4 Pa from Laun (1986, Rheol. Acta 25:447) iPP capillary data — gives
# γ̇* ≈ 1–4 s⁻¹ at 180–220°C so shear-thinning is active in stamp-forming (1–100 s⁻¹).
# (2e7 Pa was 1000× too large and made the melt effectively Newtonian.)
cross_wlf_critical_stress = 21000.0
cross_wlf_power_law_index = 0.28
fiber_max_packing_fraction = 0.64
fiber_intrinsic_viscosity = 2.5
# Legacy placeholder from the earlier rule-of-mixtures viscosity model.
# The live hot-flow law below keeps viscosity matrix-dominated instead.
glass_viscosity = 1.0e16
matrix_yield_stress = 9.0e7
glass_yield_stress = 4.9e9   # CF tensile strength (T700, Toray datasheet), vs glass 2.5 GPa
matrix_hardening = 8.0e8
glass_hardening = 1.5e10     # stiffer fibre — CF post-yield tangent stiffness estimate
matrix_h_contact = 2.5e4  # W/m²K — tool–part contact HTC, Bernet 1999 / Ye 2005 (range 2–5×10⁴)
glass_h_contact = 6.0e4   # W/m²K — CF surface vs metal tool (similar range to GF)
mu_friction = env_float("TP_CLIP_MU_FRICTION", 0.3)  # Coulomb friction coefficient, tool–part interface
                                                      # 0.3 is representative for steel vs CFRTP (Guzman 2018)

# Coefficient of Thermal Expansion (CTE) for thermal residual stress / springback.
# CF T700: α∥ ≈ −0.5 ppm/K (longitudinal), α⊥ ≈ 22 ppm/K (transverse).
# PP matrix: α ≈ 80 ppm/K.  Refs: Chawla (2012), Sideridis (1994).
fiber_cte_axial      = env_float("TP_CLIP_FIBER_CTE_AXIAL",  -0.5e-6)  # CF axial CTE, 1/K
fiber_cte_transverse = env_float("TP_CLIP_FIBER_CTE_TRANS",   22.0e-6) # CF transverse CTE, 1/K
matrix_cte           = env_float("TP_CLIP_MATRIX_CTE",        80.0e-6) # PP matrix CTE, 1/K
# Stress-free reference temperature: the point at which the solid first bears load.
# Below T_stress_free, cooling generates locked-in thermal residual stress — the
# primary driver of springback on mold release.
T_stress_free = matrix_tmelt

# Flow-orientation kinetics (Jeffery + RSC slowdown + ARD anisotropic diffusion)
fiber_aspect_ratio = 10000.0
xi_ft = (fiber_aspect_ratio^2 - 1.0) / (fiber_aspect_ratio^2 + 1.0)
ci_ft = 0.0
kappa_rsc = 0.35
ci_ard_parallel = 0.02
ci_ard_perp = 0.005
orientation_coupling_gain = 0.30

# Meso-architecture — alternating 0/90 plies with tow bands.
n_plies = 7          # with only ~3 layers through thickness, keep 2 plies (0, 90)
fiber_vf_in_tow = 0.58
fiber_vf_resin_rich = 0.02
tow_width = 0.0040
tow_gap = 0.0015

# Physical setup dimensions held fixed when particle spacing changes.
# These values match the current pseudo-2D baseline at particle_spacing = 0.8 mm.
charge_thickness_target = 0.0066
charge_bottom_clearance = 3.0 * particle_spacing
initial_gap_upper_target = 0.0033
base_thickness = 0.0033
holder_thickness = 0.0033
punch_side_clearance = 0.0011
# Thermal contact is evaluated from nearest particle-center distance in the x-z plane.
# In the drafted pseudo-2D tool geometry, a half-spacing gate is too tight and can
# miss real wall/floor contact on the discrete lattice, leaving temperature unchanged.
contact_heat_gap_threshold = particle_spacing
safety_margin = 0.0066
base_layers = max(1, ceil(Int, base_thickness / particle_spacing))

# Raw charge dimensions. By default the charge is one particle thick in y
# (legacy pseudo-2D slab), but a thin 3D slab with multiple y-layers can be
# enabled to avoid hard out-of-plane projection artifacts.
# Compression-molding layout: charge is PRE-PLACED INSIDE the cavity, sitting
# on the cavity floor. Keep it small enough to fit in the narrow (bottom) part
# of the drafted trapezoidal cavity with margin > smoothing length per side,
# so contact-kernel interaction with the side walls stays zero at t = 0.
charge_length = 0.018             # 16 mm (doubled from 8 mm; cavity floor ≈ 16.9 mm wide)
charge_layers_y = max(1, env_int("TP_CLIP_CHARGE_LAYERS_Y", 1))
charge_width = charge_layers_y * particle_spacing
charge_thickness = charge_thickness_target

n_charge = (ceil(Int, charge_length / particle_spacing),
            charge_layers_y,
            max(2, ceil(Int, charge_thickness / particle_spacing)))

# Recenter the discrete particle lattice on the geometric mid-plane.
# When `ceil` adds an extra x-column, anchoring the charge at `-0.5 * charge_length`
# shifts the actual particle cloud off center and breaks left-right symmetry.
charge_length_discrete = n_charge[1] * particle_spacing
charge_thickness_discrete = n_charge[3] * particle_spacing
charge_top_target = 0.00165 + charge_thickness_target
charge_top_actual = charge_bottom_clearance + charge_thickness_discrete
# Target compressed charge height as a fraction of the discrete lattice thickness (outer envelope).
compression_thickness_ratio = env_float("TP_CLIP_COMPRESSION_RATIO", 0.9)
compression_thickness_ratio > 0.0 && compression_thickness_ratio <= 1.0 ||
    error("TP_CLIP_COMPRESSION_RATIO must be in (0, 1] (got $(compression_thickness_ratio))")
final_thickness_target = compression_thickness_ratio * charge_thickness_discrete

# ==========================================================================================
# DEEP-DRAW TOOLING LAYOUT (x–z cross-section, 1 particle thick in y)
#
# COMPRESSION-MOLDING layout matching the clip-mold reference figure:
#   - Charge is PRE-PLACED INSIDE the cavity, resting on the cavity floor.
#   - Female die has a trapezoidal cavity with draft-angle side walls.
#   - Male punch is a mating trapezoid that descends and compresses the charge.
#   - Reference frame chosen so the cavity FLOOR is at z = 0 (charge rests here).
#
#   z = cavity_depth                    ← top face of die (side blocks / land)
#   z = 0                               ← cavity floor = charge bottom
#   z = -base_layers * ps               ← bottom of die base plate
#
# Cavity (open, no particles):
#   |x| < cavity_half_width_at(z),  z ∈ [0, cavity_depth]
#
# Male punch:
#   - Trapezoidal body of length `cavity_depth` that mates the cavity 1:1
#     (inset by one particle-spacing per side for clearance).
#   - Full-width holder plate above the body.
#   - No ejector pin.
# ==========================================================================================
tool_length = 0.040                       # 40 mm wide (charge is 12 mm; ~14 mm margin each side)
cavity_internal_width = 0.026             # 26 mm top width — accepts 12 mm charge with room
cavity_depth = 0.014                      # 14 mm deep
nz_cavity_wall = ceil(Int, cavity_depth / particle_spacing)
holder_layers = max(1, ceil(Int, holder_thickness / particle_spacing))
# Keep the actual charge placement spacing-dependent; the punch stroke then
# follows the current discretized charge top.
charge_origin = (-0.5 * charge_length_discrete, -0.5 * charge_width,
                 charge_bottom_clearance)
# ==========================================================================================

function build_glass_fiber_charge(particle_spacing, n_charge, charge_origin;
                                  matrix_density, glass_density,
                                  n_plies, fiber_vf_in_tow, fiber_vf_resin_rich,
                                  tow_width, tow_gap,
                                  gap_vf_override=nothing)
    nx, ny, nz = n_charge
    n_total = nx * ny * nz

    coordinates = Matrix{Float64}(undef, 3, n_total)
    velocity = zeros(3, n_total)
    mass = Vector{Float64}(undef, n_total)
    density = Vector{Float64}(undef, n_total)

    fiber_volume_fraction = Vector{Float64}(undef, n_total)
    fiber_direction = Matrix{Float64}(undef, 3, n_total)
    idx_fiber_rich = Int[]
    idx_matrix_rich = Int[]

    x0, y0, z0 = charge_origin
    ply_thickness = nz > 0 ? (nz * particle_spacing) / n_plies : particle_spacing
    tow_pitch = tow_width + tow_gap
    x_center = x0 + 0.5 * nx * particle_spacing
    tow_cols_pseudo2d = max(1, round(Int, tow_width / particle_spacing))
    gap_cols_pseudo2d = max(1, round(Int, tow_gap / particle_spacing))
    tow_width_pseudo2d = tow_cols_pseudo2d * particle_spacing
    tow_pitch_pseudo2d = (tow_cols_pseudo2d + gap_cols_pseudo2d) * particle_spacing
    tow_pitch_cols_pseudo2d = tow_cols_pseudo2d + gap_cols_pseudo2d

    p = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        p += 1

        x = x0 + (i - 0.5) * particle_spacing
        y = y0 + (j - 0.5) * particle_spacing
        z = z0 + (k - 0.5) * particle_spacing

        coordinates[1, p] = x
        coordinates[2, p] = y
        coordinates[3, p] = z

        z_rel = max(0.0, z - z0)
        ply_id = clamp(floor(Int, z_rel / ply_thickness) + 1, 1, n_plies)
        ply_aligned_x = isodd(ply_id)

        if ny == 1
            # Snap the pseudo-2D tow pattern to whole particle columns and center it on the
            # charge so the cross-section stays mirror-symmetric on the discrete lattice.
            phase_shift = ply_aligned_x ? 0.5 * tow_pitch_pseudo2d : 0.0
            distance_to_tow_center = mod((x - x_center) + phase_shift +
                                         0.5 * tow_pitch_pseudo2d,
                                         tow_pitch_pseudo2d) - 0.5 * tow_pitch_pseudo2d
            in_tow = abs(distance_to_tow_center) < 0.5 * tow_width_pseudo2d

            # Use a column-index selector tied to the charge mid-plane so mirrored columns
            # always receive the same classification after `ceil` changes the particle count.
            center_offset_twice = 2 * i - nx - 1
            phase_shift_twice = ply_aligned_x ? tow_pitch_cols_pseudo2d : 0
            wrapped_offset_twice = mod(center_offset_twice + phase_shift_twice +
                                       tow_pitch_cols_pseudo2d,
                                       2 * tow_pitch_cols_pseudo2d) - tow_pitch_cols_pseudo2d
            in_tow = abs(wrapped_offset_twice) <= tow_cols_pseudo2d - 1
        else
            selector = ply_aligned_x ? (y - y0) : (x - x0)
            in_tow = mod(selector, tow_pitch) <= tow_width
        end

        vf_local = in_tow ? fiber_vf_in_tow :
                   (gap_vf_override === nothing ? fiber_vf_resin_rich : gap_vf_override)
        fiber_volume_fraction[p] = vf_local

        if ply_aligned_x
            fiber_direction[1, p] = 1.0
            fiber_direction[2, p] = 0.0
            fiber_direction[3, p] = 0.0
        else
            fiber_direction[1, p] = 0.0
            fiber_direction[2, p] = 1.0
            fiber_direction[3, p] = 0.0
        end

        rho_local = vf_local * glass_density + (1.0 - vf_local) * matrix_density
        density[p] = rho_local
        mass[p] = rho_local * particle_spacing^3

        if vf_local >= 0.5 * fiber_vf_in_tow
            push!(idx_fiber_rich, p)
        else
            push!(idx_matrix_rich, p)
        end
    end

    ic = InitialCondition(; coordinates, velocity, mass, density)

    return ic, fiber_volume_fraction, fiber_direction, idx_fiber_rich, idx_matrix_rich
end

polymer, fiber_volume_fraction, fiber_direction,
idx_fiber_rich, idx_matrix_rich =
    build_glass_fiber_charge(particle_spacing, n_charge, charge_origin;
                             matrix_density=matrix_density,
                             glass_density=glass_density,
                             n_plies=n_plies,
                             fiber_vf_in_tow=fiber_vf_in_tow,
                             fiber_vf_resin_rich=fiber_vf_resin_rich,
                             tow_width=tow_width,
                             tow_gap=tow_gap,
                             gap_vf_override=gap_vf_override)

density_cylinder = mean(polymer.density)

vf_mean = mean(fiber_volume_fraction)
fiber_stiffness_efficiency = 0.35
E_composite_eff = matrix_E * (1.0 - vf_mean) + glass_E * vf_mean * fiber_stiffness_efficiency
cp_composite_eff = matrix_cp * (1.0 - vf_mean) + glass_cp * vf_mean
k_composite_eff = thermal_conductivity_scale *
                  (matrix_k * (1.0 - vf_mean) + glass_k * vf_mean)

phase_vf_particle = copy(fiber_volume_fraction)
cp_particle = matrix_cp .* (1.0 .- phase_vf_particle) .+ glass_cp .* phase_vf_particle
k_particle = thermal_conductivity_scale .* (matrix_k .* (1.0 .- phase_vf_particle) .+
                                            glass_k .* phase_vf_particle)
# Matrix melt/solidification controls the rheology switch.
# Do not blend with glass-fiber temperatures: fibers do not melt in this process,
# and blending pushes tmelt above 1000 K in tow-rich particles, incorrectly
# forcing lf≈0 at 400-450 K (stiff latent regime and dt collapse).
temp_liq_particle = fill(matrix_temp_liq, length(phase_vf_particle))
tmelt_particle = fill(matrix_tmelt, length(phase_vf_particle))
latent_heat_particle = matrix_latent_heat .* (1.0 .- phase_vf_particle)
local_regime_threshold_particle = fill(matrix_temp_liq, length(phase_vf_particle))

# Halpin–Tsai transversely isotropic elastic moduli (Halpin & Tsai 1969).
# E1: longitudinal (rule of mixtures, fiber direction).
# E2: transverse (Halpin–Tsai ξ = 2 for circular CF cross-section).
# k_fiber_particle: Spencer (1984) anisotropy constant ≈ (E1 − E2) / 4.
# Using E2 as the Neo-Hookean isotropic base gives the correct through-thickness (z)
# stiffness (fibers always transverse to z → E_z = E2); the extra fiber-direction
# stiffness is recovered by the additive Spencer term in the elastic stress functions.
let r = glass_E / matrix_E
    global halpin_tsai_eta_E2 = (r - 1.0) / (r + 2.0)
end
E1_particle = glass_E .* phase_vf_particle .+ matrix_E .* (1.0 .- phase_vf_particle)
E2_particle = matrix_E .* (1.0 .+ 2.0 .* halpin_tsai_eta_E2 .* phase_vf_particle) ./
              max.(1.0 .- halpin_tsai_eta_E2 .* phase_vf_particle, eps(Float64))
k_fiber_particle = max.(0.0, (E1_particle .- E2_particle) ./ 4.0)

# Effective isotropic CTE per particle (volumetric average, transversely isotropic → isotropic).
# α₁ (fiber direction): E-weighted Voigt rule of mixtures.
# α₂ (transverse):      simple rule of mixtures.
# α_eff = (α₁ + 2·α₂) / 3  ← isotropic volumetric average.
# Resulting thermal Kirchhoff stress is purely hydrostatic → J2-neutral:
# no effect on yield surface, plastic flow, or Taylor-Quinney heating.
alpha1_particle = (glass_E .* fiber_cte_axial .* phase_vf_particle .+
                   matrix_E .* matrix_cte .* (1.0 .- phase_vf_particle)) ./ E1_particle
alpha2_particle = fiber_cte_transverse .* phase_vf_particle .+
                  matrix_cte .* (1.0 .- phase_vf_particle)
alpha_eff_particle = (alpha1_particle .+ 2.0 .* alpha2_particle) ./ 3.0
yield_base_particle = matrix_yield_stress .* (1.0 .- phase_vf_particle) .+
                      glass_yield_stress .* phase_vf_particle
hardening_base_particle = matrix_hardening .* (1.0 .- phase_vf_particle) .+
                         glass_hardening .* phase_vf_particle
# Keep the melt viscosity matrix-dominated. The live update below now uses a
# matrix Arrhenius law directly, so this array only carries the reference value
# for system initialization and diagnostics.
viscosity_base_particle = fill(matrix_viscosity, length(phase_vf_particle))
h_contact_particle = matrix_h_contact .* (1.0 .- phase_vf_particle) .+
                     glass_h_contact .* phase_vf_particle

orientation_scalar_particle = ones(length(phase_vf_particle))

viscosity_min_clip = 1.0e2
viscosity_max_clip = 5.0e7
temp_min_clip = 200.0
temp_max_clip = 2000.0

factor = 1.4
smoothing_kernel = SchoenbergQuinticSplineKernel{3}()
#smoothing_kernel = CubicSplineKernel{3}()
smoothing_length = factor * particle_spacing

println("✓ CFK raw charge initialized (pseudo-2D slab):")
println("  - size [mm] = ", round.(StaticArrays.SVector(charge_length_discrete, charge_width,
                                              charge_thickness_discrete) .* 1e3,
                                      digits=2))
println("  - particles = ", nparticles(polymer))
gap_vf_override !== nothing &&
    println("  - gap column Vf override = ", round(gap_vf_override, digits=4))
println("  - architecture = ", n_plies, " plies (alternating), tow width=", tow_width * 1e3,
        " mm, tow gap=", tow_gap * 1e3, " mm")
println("  - mean fibre volume fraction Vf = ", round(vf_mean, digits=4))
println("  - phase counts (fibre-rich/matrix-rich) = ", length(idx_fiber_rich), " / ",
        length(idx_matrix_rich))
println("  - homogenized rho = ", round(density_cylinder, digits=2),
        " kg/m^3, E = ", round(E_composite_eff / 1e9, digits=3), " GPa")
println("  - effective conductivity scale = ", thermal_conductivity_scale,
    "x (pseudo-2D accelerated cooling)")
println("  - phase-specific tmelt matrix/glass = ", matrix_tmelt, " / ", glass_tmelt, " K")
println("  - matrix latent heat = ", matrix_latent_heat, " J/kg")

# ==========================================================================================
# STEP 3: Build mating deep-draw tooling (female die + male punch), 1 particle thick in y
# ==========================================================================================
floor_density = density_cylinder

function merge_rect_shapes(shapes)
    all_coords = reduce(hcat, map(s -> s.coordinates, shapes))
    all_vel = reduce(hcat, map(s -> s.velocity, shapes))
    all_mass = reduce(vcat, map(s -> s.mass, shapes))
    all_dens = reduce(vcat, map(s -> s.density, shapes))
    return InitialCondition(; coordinates=all_coords, velocity=all_vel,
                mass=all_mass, density=all_dens)
end

cyl_z_min = minimum(polymer.coordinates[3, :])
cyl_z_max = maximum(polymer.coordinates[3, :])

# Option 1 — open the initial gap to ~2 * smoothing_length so the mold-motion
# ramp completes *before* first contact. This prevents the stiff t=0 contact +
# heat-flux transient that collapses TRBDF2's dt to ~1e-7.
initial_gap       = 2 * smoothing_length
initial_gap_upper = max(initial_gap_upper_target, 2 * smoothing_length)

if sim_phase == "retraction"
    isfile(checkpoint_path) ||
        error("Hold checkpoint not found for retraction: $(abspath(checkpoint_path))")
    hold_checkpoint_early[] = load_hold_checkpoint(checkpoint_path)
    ck_meta = hold_checkpoint_early[]["meta"]
    if !haskey(ENV, "TP_CLIP_COMPRESSION_RATIO")
        if haskey(ck_meta, "compression_thickness_ratio")
            compression_thickness_ratio = ck_meta["compression_thickness_ratio"]
            println(">>> Retraction: using compression_thickness_ratio=",
                    compression_thickness_ratio, " from checkpoint meta")
        else
            compression_thickness_ratio = infer_compression_ratio_for_hold_meta(ck_meta;
                particle_spacing=particle_spacing,
                charge_bottom_clearance=charge_bottom_clearance,
                charge_length=charge_length,
                charge_thickness_discrete=charge_thickness_discrete,
                cyl_z_max=cyl_z_max, initial_gap_upper=initial_gap_upper,
                cavity_depth=cavity_depth, tool_length=tool_length,
                punch_side_clearance=punch_side_clearance,
                holder_thickness=holder_thickness,
                cavity_internal_width=cavity_internal_width)
            println(">>> Retraction: inferred TP_CLIP_COMPRESSION_RATIO=",
                    round(compression_thickness_ratio; digits=3),
                    " from checkpoint n_mold=$(ck_meta["n_mold"])")
        end
        final_thickness_target = compression_thickness_ratio * charge_thickness_discrete
    end
    flush(stdout)
end

# Reference frame: cavity FLOOR at z = 0 (this is where the charge sits).
# The charge is built at z0 = 0 (see charge_origin), so its bottom face
# automatically rests on the cavity floor.
#   - z = 0               → cavity floor = top of base plate
#   - z = cavity_depth    → top face of die (land/shoulder, outside the cavity)
#   - z = -base_layers*ps → bottom of die base plate
z_base_top   = 0.0
z_blocks_top = z_base_top + cavity_depth
z_base_bot   = z_base_top - base_layers * particle_spacing

# -------- Trapezoidal die + mating trapezoidal punch (angled walls / draft) --------
# Both die cavity and punch share the SAME master grid so they mate particle-for-
# particle, with a one-ps clearance per side. The cavity is widest at the top
# (where the charge enters) and narrows toward the floor — classic draft geometry
# as in the reference figure. The punch mirrors the cavity shape exactly, offset
# inward by one particle spacing per side.
nx_tool = ceil(Int, tool_length / particle_spacing)
ny_tool = charge_layers_y
tool_origin_y = -0.5 * ny_tool * particle_spacing
tool_origin_x = -0.5 * tool_length
use_pseudo2d_projection = charge_layers_y == 1 && ny_tool == 1
# Thin-3D mode (charge_layers_y > 1) uses the same plane-strain constraints as pseudo-2D
# apply_y_plane_constraint: gates the F22=1 plane-strain fix for both pseudo-2D and thin-3D.
# Kernel support in y is always poor for thin slabs (1–3 y-neighbours), so F22 must be
# corrected in both cases.
#
# Velocity / displacement constraints differ by mode:
#   pseudo-2D  → hard zeros (v_y=0, du_y=0, dv_y=0): all particles collapse to y=0.
#   thin-3D    → stiff spring-damper in dv_y: particles stay near their initial y-planes
#                but TRBDF2 sees a non-zero Jacobian column (∂(dv_y)/∂v_y = -y_constraint_damp)
#                so Newton converges without the force artifacts hard-zeroing produces.
apply_y_plane_constraint = use_pseudo2d_projection || charge_layers_y > 1

# Spring-damper parameters for the thin-3D y-plane constraint.
# Elastic wave speed → sets the constraint stiffness so TRBDF2 (L-stable) damps y-motion
# within 1–2 timesteps while the Jacobian remains well-conditioned.
let c_el = sqrt(matrix_E / matrix_density)                  # elastic wave speed [m/s]
    global y_constraint_damp   = 20.0 * c_el / particle_spacing  # [s^-1] damping coefficient
    global y_constraint_spring = (0.5 * y_constraint_damp)^2     # [s^-2] spring = omega_n^2
end
@info "Y-layer mode" charge_layers_y use_pseudo2d_projection apply_y_plane_constraint charge_width_mm=(charge_width*1e3) particle_spacing_mm=(particle_spacing*1e3)
charge_layers_y > 1 && @info "Thin-3D mode active: for similar runtime to 1-layer, set TP_CLIP_PS_MM=$(round(0.8*sqrt(Float64(charge_layers_y)), digits=2))"

# Redefined geometry: set bottom width directly, derive draft angle.
cavity_top_half_w = 0.5 * cavity_internal_width   # widest (at z_blocks_top)
cavity_bot_half_w = 0.5 * (charge_length + 0.003) # 1.5mm clearance per side
wall_inset = cavity_top_half_w - cavity_bot_half_w
draft_angle = atan(wall_inset / cavity_depth)
@info "Derived draft angle for wider cavity" angle_deg=rad2deg(draft_angle)

# Cavity half-width as a function of z ∈ [z_base_top, z_blocks_top].
@inline function cavity_half_width_at(z)
    frac = (z - z_base_top) / (z_blocks_top - z_base_top)
    frac = clamp(frac, 0.0, 1.0)
    return cavity_bot_half_w + frac * (cavity_top_half_w - cavity_bot_half_w)
end

# Build die particles by filtering a rectangular grid:
#   - base plate: full rectangle from z_base_bot to z_base_top
#   - wall region: z ∈ [z_base_top, z_blocks_top], keep if |x| > half_w(z)
function build_die_particles(ps, x_min, x_max, y_origin, ny,
                             z_base_bot, z_base_top, z_blocks_top,
                             half_width_fn, density)
    coords = Float64[]
    nx = ceil(Int, (x_max - x_min) / ps)
    # Base plate layers
    nz_base = ceil(Int, (z_base_top - z_base_bot) / ps)
    for k in 1:nz_base
        z = z_base_bot + (k - 0.5) * ps
        for j in 1:ny, i in 1:nx
            x = x_min + (i - 0.5) * ps
            y = y_origin + (j - 0.5) * ps
            push!(coords, x, y, z)
        end
    end
    # Cavity-wall region (with trapezoidal hole)
    nz_walls = ceil(Int, (z_blocks_top - z_base_top) / ps)
    for k in 1:nz_walls
        z = z_base_top + (k - 0.5) * ps
        half_w = half_width_fn(z)
        for j in 1:ny, i in 1:nx
            x = x_min + (i - 0.5) * ps
            y = y_origin + (j - 0.5) * ps
            if abs(x) > half_w
                push!(coords, x, y, z)
            end
        end
    end
    n = length(coords) ÷ 3
    coords_m = reshape(coords, 3, n)
    mass = fill(density * ps^3, n)
    dens = fill(density, n)
    vel = zeros(3, n)
    return InitialCondition(; coordinates=coords_m, velocity=vel, mass=mass, density=dens)
end

floor_particles = build_die_particles(particle_spacing,
                                      tool_origin_x, tool_origin_x + nx_tool * particle_spacing,
                                      tool_origin_y, ny_tool,
                                      z_base_bot, z_base_top, z_blocks_top,
                                      cavity_half_width_at,
                                      floor_density)

# -------- Male punch: trapezoidal body that truly MATES the cavity --------
# Construction strategy:
#   1. Define the SEATED position of the punch body (as if fully stroked down).
#      Seated z-range: [z_base_top + final_thickness, z_blocks_top + final_thickness]
#      where `final_thickness` is the compressed charge outer height on the cavity floor.
#   2. At each seated absolute z, the punch half-width equals the cavity
#      half-width at that z minus one ps per side (true mating profile — the
#      punch exactly fills the draft-walled cavity with 1 ps clearance).
#   3. Translate the body upward by `initial_stroke` so that at t=0 its bottom
#      sits `initial_gap_upper` above the actual charge top outer face.
final_thickness = final_thickness_target                # compressed charge outer height at cavity floor
charge_top_outer = cyl_z_max + 0.5 * particle_spacing
z_punch_bot     = charge_top_outer + initial_gap_upper  # start-position punch bottom
z_punch_top     = z_punch_bot + cavity_depth            # start-position punch top
initial_stroke  = z_punch_bot - (z_base_top + final_thickness)  # descent required to seat

@inline function punch_half_width_at_start(z_start)
    # Map starting z back to its seated z, then use the cavity profile.
    z_seated    = z_start - initial_stroke
    z_seated_cl = clamp(z_seated, z_base_top, z_blocks_top)
    half_w_die  = cavity_half_width_at(z_seated_cl)
    return max(0.5 * punch_side_clearance, half_w_die - punch_side_clearance)
end

function build_punch_particles(ps, x_min, x_max, y_origin, ny,
                               z_punch_bot, z_punch_top,
                               half_width_fn_at_start,
                               z_holder_bot, holder_layers, density)
    coords = Float64[]
    nx = ceil(Int, (x_max - x_min) / ps)

    # Trapezoidal punch body — profile from seated absolute z (via the map
    # inside `half_width_fn_at_start`).
    nz_body = ceil(Int, (z_punch_top - z_punch_bot) / ps)
    for k in 1:nz_body
        z = z_punch_bot + (k - 0.5) * ps
        half_w = half_width_fn_at_start(z)
        for j in 1:ny, i in 1:nx
            x = x_min + (i - 0.5) * ps
            y = y_origin + (j - 0.5) * ps
            if abs(x) < half_w
                push!(coords, x, y, z)
            end
        end
    end

    # Full-width holder plate above the punch body.
    for k in 1:holder_layers
        z = z_holder_bot + (k - 0.5) * ps
        for j in 1:ny, i in 1:nx
            x = x_min + (i - 0.5) * ps
            y = y_origin + (j - 0.5) * ps
            push!(coords, x, y, z)
        end
    end

    n = length(coords) ÷ 3
    coords_m = reshape(coords, 3, n)
    mass = fill(density * ps^3, n)
    dens = fill(density, n)
    vel = zeros(3, n)
    return InitialCondition(; coordinates=coords_m, velocity=vel, mass=mass, density=dens)
end

z_holder_bot = z_punch_top
mold_particles = build_punch_particles(particle_spacing,
                                       tool_origin_x, tool_origin_x + nx_tool * particle_spacing,
                                       tool_origin_y, ny_tool,
                                       z_punch_bot, z_punch_top,
                                       punch_half_width_at_start,
                                       z_holder_bot, holder_layers, floor_density)

floor_z_max = maximum(floor_particles.coordinates[3,:])
mold_z_min  = minimum(mold_particles.coordinates[3,:])
println("✓ Compression-mold tooling built (pseudo-2D, trapezoidal / drafted walls):")
println("  - tool_length = ", tool_length * 1e3, " mm, slab thickness y = ",
        particle_spacing * 1e3, " mm")
println("  - cavity: top width = ", cavity_internal_width * 1e3,
        " mm, bot width = ", round(2 * cavity_bot_half_w * 1e3; digits=2),
        " mm, depth = ", cavity_depth * 1e3, " mm, draft = ",
        round(rad2deg(draft_angle); digits=1), " deg")
println("  - charge inside cavity: length = ", charge_length * 1e3,
        " mm, thickness = ", charge_thickness * 1e3,
    " mm (initialized slightly above cavity floor for startup stability)")
println("  - male punch: mating trapezoid (1 ps clearance per side, no ejector)")
println("  - gap (mold bot ↔ charge top) = ", round(mold_z_min - cyl_z_max; digits=6),
        " m, initial stroke to seat = ", round(initial_stroke * 1e3; digits=3), " mm")
println("  - h = ", factor * particle_spacing, " m")
println("  - Contact at t=0? ",
        (cyl_z_min - (z_base_top - 0.5 * particle_spacing)) < factor * particle_spacing,
        " (floor), ",
        (mold_z_min - cyl_z_max) < factor * particle_spacing, " (mold)")
println("  - Tool particles (floor/mold) = ", nparticles(floor_particles), " / ",
        nparticles(mold_particles))

# Numerical safety rails for pseudo-2D debug runs.
# With all fundamental issues fixed (dt sync, 2D kernel correction, and plane-strain),
# these artificial limiters are no longer needed and just ruin Newton convergence.
enable_safety_clamps = false
enable_velocity_safety_clamps = false
# F/Fp repair runs in the accepted-step callback only. Doing it inside kick_implicit!
# corrupts the finite-difference Jacobian (each column perturbation can flip the
# J-threshold check and write a different F into shared buffers).
enable_emergency_repair_clamp = false
x_safety_bound = 0.5 * tool_length + safety_margin
z_safety_min = z_base_bot - safety_margin
z_safety_max = z_punch_top + safety_margin
max_cylinder_speed = 2.0      # m/s
max_cylinder_accel = 1.0e5    # m/s^2
# Per-particle acceleration cap during retraction (explicit springback stability).
const retraction_accel_clamp = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_ACCEL_CLAMP", 1) != 0
const retraction_max_accel = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_MAX_ACCEL", 5.0e2) : max_cylinder_accel
# Targeted limiter for liquid/mushy mechanics (WCSPH branch). This is applied
# even when global safety clamps are disabled, since the dt-collapse trigger is
# liquid-force acceleration spikes in the latent regime.
max_liquid_accel = 2.5e5      # m/s^2

# ==========================================================================================
# THERMOMECHANICAL STATE INITIALIZATION
# ==========================================================================================
n_cylinder_particles = nparticles(polymer)

Fp_initial = zeros(3, 3, n_cylinder_particles)
for i in 1:n_cylinder_particles
    Fp_initial[:, :, i] .= Matrix{Float64}(I, 3, 3)
end
Fp_state = Ref(Fp_initial)

alpha_committed = Ref(zeros(n_cylinder_particles))
Fp_committed    = Ref(copy(Fp_initial))

F_total_mold_state = Ref(0.0)

# Press speed. Higher than the 3D run so the pseudo-2D slab finishes quickly.
# Press speed: real stamp forming of CFRTP clips is 0.01–0.05 m/s (Wakeman 2006, Hou 1998).
# 0.05 m/s is the upper end of the physical range; faster rates give less time for
# crystallisation kinetics to develop during forming and overpredict residual stress.
mold_velocity_state = -env_float("TP_CLIP_MOLD_SPEED_MPS", 0.05)  # m/s
t_ramp_mold = env_float("TP_CLIP_MOLD_RAMP_S", 1.0e-3)

# Mild velocity-proportional damping for the pseudo-2D slab.
# Keep baseline damping low to avoid stiffening the Newton residual; add extra
# damping only close to tool contact.
# NOTE: damping set to zero — artificial damping suppresses viscous velocity gradients
# and causes the charge to shrink. Physical viscosity (η = 8e4 Pa·s) handles dissipation.
velocity_damping_base = 0.0      # 1/s
velocity_damping_contact = 0.0   # 1/s
molten_wall_drag_factor = env_float("TP_CLIP_MOLTEN_WALL_DRAG_FACTOR", 0.05)
molten_wall_drag_max_rate = env_float("TP_CLIP_MOLTEN_WALL_DRAG_MAX_RATE", 1.0e5) # 1/s

# Target stroke: drive the punch from its start position (z_punch_bot) down until
# the charge is compressed to `final_thickness` (= compression_thickness_ratio × initial).
target_stroke = z_punch_bot - (z_base_top + final_thickness)
t_full = (target_stroke / abs(mold_velocity_state)) + 0.5 * t_ramp_mold
t_down_end = t_full
z_shift_down_end = if t_down_end < t_ramp_mold
    0.5 * mold_velocity_state / t_ramp_mold * t_down_end^2
else
    mold_velocity_state * (t_down_end - 0.5 * t_ramp_mold)
end
t_compress = t_down_end
retraction_motion_from_checkpoint = false
if sim_phase == "retraction" && hold_checkpoint_early[] !== nothing
    ck_meta = hold_checkpoint_early[]["meta"]
    t_compress = ck_meta["t_compress"]
    z_shift_down_end = ck_meta["z_shift_down_end"]
    target_stroke = abs(z_shift_down_end)
    t_down_end = t_compress
    retraction_motion_from_checkpoint = true
    println(">>> Retraction: restored press timeline from checkpoint ",
            "(t_compress=", round(t_compress; digits=6),
            " s, z_shift_down_end=", round(z_shift_down_end * 1e3; digits=3), " mm)")
    flush(stdout)
end
# Hold until solidification_fraction_target (default 100% of particles) meet sf threshold.
# Default: no time cap (Inf). Set TP_CLIP_SOLIDIFICATION_HOLD_MAX_S to a finite value to
# allow early exit before full solidification (legacy/debug behaviour).
solidification_hold_max = env_float("TP_CLIP_SOLIDIFICATION_HOLD_MAX_S", Inf)
# ODE tspan upper-bound when hold is unlimited (hold/hold-tail terminate earlier).
solidification_hold_plan_s = env_float("TP_CLIP_SOLIDIFICATION_HOLD_PLAN_S", 120.0)
solidification_hold_budget = isfinite(solidification_hold_max) ?
    solidification_hold_max : solidification_hold_plan_s
# Per-particle Nakamura sf to count as crystallized (0.92 = elastic-mechanics onset).
solidification_fraction_threshold = env_float("TP_CLIP_SOLIDIFICATION_THRESHOLD", 0.92)
# Fraction of charge particles that must meet the threshold before mold opens (1.0 = all).
solidification_fraction_target = env_float("TP_CLIP_SOLIDIFICATION_TARGET", 1.0)
const vtu_save_dt = env_float("TP_CLIP_VTU_SAVE_DT", 1.0e-1)
# Compression VTU output: 2× denser than hold by default (0.1 s -> 0.05 s).
const compression_vtu_save_scale = env_float("TP_CLIP_COMPRESSION_VTU_SAVE_DT_SCALE", 2.0)
const compression_vtu_save_dt = env_float("TP_CLIP_COMPRESSION_VTU_SAVE_DT",
                                          vtu_save_dt / compression_vtu_save_scale)
# Retraction VTU output: 4× denser than compress/hold by default (0.1 s -> 0.025 s).
const retraction_vtu_save_scale = env_float("TP_CLIP_RETRACTION_VTU_SAVE_DT_SCALE", 4.0)
const retraction_vtu_save_dt = env_float("TP_CLIP_RETRACTION_VTU_SAVE_DT",
                                         vtu_save_dt / retraction_vtu_save_scale)
@inline function active_vtu_save_dt(; t=nothing, in_retraction::Bool=false)
    if sim_phase == "retraction" || in_retraction
        return retraction_vtu_save_dt
    elseif t !== nothing && t < t_compress
        return compression_vtu_save_dt
    else
        return vtu_save_dt
    end
end

function build_compress_hold_vtu_save_times(t_final)
    times = Float64[]
    t = 0.0
    while true
        dt_save = t < t_compress ? compression_vtu_save_dt : vtu_save_dt
        t += dt_save
        t > t_final + 1.0e-10 && break
        push!(times, t)
    end
    return times
end
post_solidification_dwell = env_float("TP_CLIP_POST_SOLID_DWELL_S", 2.0e-3)
retraction_duration_max = abs(z_shift_down_end) / abs(mold_velocity_state)
t_total = t_compress + solidification_hold_budget + post_solidification_dwell +
      hold_tlsph_tail_s + retraction_duration_max + retraction_springback_dwell_s
println("  - compression target = ", round(100 * compression_thickness_ratio; digits=0),
        "% of initial charge thickness (", round(charge_thickness_discrete * 1e3; digits=2),
        " mm -> ", round(final_thickness_target * 1e3; digits=2), " mm)")
println("  - press speed = ", abs(mold_velocity_state), " m/s, ramp = ", t_ramp_mold,
        " s, target stroke = ", round(target_stroke * 1e3; digits=2),
        ", t_down_end = ", round(t_down_end; digits=6),
        " s, t_motion = ", round(t_compress; digits=6),
        " s, solidification hold ",
        isfinite(solidification_hold_max) ?
            ("<= " * string(round(solidification_hold_max; digits=6)) * " s") :
            "until 100% particles solidify (no time cap; plan " *
            string(round(solidification_hold_plan_s; digits=6)) * " s)",
        " (target ", round(100 * solidification_fraction_target; digits=0),
        "% particles sf>=", solidification_fraction_threshold, ")",
    ", post-solidification dwell = ", round(post_solidification_dwell; digits=6),
    " s, TLSPH equilibration tail = ", round(hold_tlsph_tail_s; digits=6),
    " s, retract <= ", round(retraction_duration_max; digits=6),
    " s, springback dwell <= ", round(retraction_springback_dwell_s; digits=6),
    " s, VTU save dt = ", vtu_save_dt, " s (compression ",
    compression_vtu_save_dt, " s)")

@inline function mold_z_shift_at_time(t)
    z_shift = if t < t_ramp_mold
        0.5 * mold_velocity_state / t_ramp_mold * t^2
    elseif t < t_down_end
        mold_velocity_state * (t - 0.5 * t_ramp_mold)
    else
        z_shift_down_end
    end
    return clamp(z_shift, -target_stroke, 0.0)
end

@inline function mold_z_velocity_at_time(t)
    if t < t_ramp_mold
        return mold_velocity_state / t_ramp_mold * t
    elseif t < t_down_end
        return mold_velocity_state
    elseif retraction_started[]
        dt_ret = t - retraction_start_time[]
        if dt_ret < retraction_ramp_s
            return abs(mold_velocity_state) * dt_ret / retraction_ramp_s
        else
            return abs(mold_velocity_state)
        end
    else
        return 0.0
    end
end

z_shift_motion_end = mold_z_shift_at_time(t_compress)
retraction_started = Ref(false)
retraction_complete = Ref(false)
mold_retraction_complete = Ref(false)
springback_equil_streak = Ref(0)
springback_equil_prev_snap = Ref{Any}(nothing)
retraction_start_time = Ref(Inf)
mold_retraction_complete_time = Ref(Inf)
solidification_reached_time = Ref(Inf)
retraction_state_prepared = Ref(false)
# Solidification hold: freeze charge u/v and skip mechanics RHS; thermal + Nakamura continue.
const charge_hold_phase_dtmax = env_float("TP_CLIP_HOLD_DTMAX", 5.0e-3)
const retraction_ramp_s = env_float("TP_CLIP_RETRACTION_RAMP_S", 0.02)
# Tool traction / molten-drag / mechanical-dv blend window (v5 used 0.03 s).
const retraction_hold_release_s = env_float("TP_CLIP_RETRACTION_HOLD_RELEASE_S", 0.1)
# Spencer + matrix elastic stress ramp — keep long enough to avoid dt collapse.
const retraction_elastic_ramp_s = env_float("TP_CLIP_RETRACTION_ELASTIC_RAMP_S", 0.2)
# Kinematics release (default ≥ elastic ramp so σ is on before charge moves freely).
const retraction_kinematics_release_s = env_float("TP_CLIP_RETRACTION_KINEMATICS_RELEASE_S", 0.22)
const retraction_contact_gap_off_ps = env_float("TP_CLIP_RETRACTION_CONTACT_GAP_OFF_PS", 2.5)
const retraction_contact_gap_on_ps = env_float("TP_CLIP_RETRACTION_CONTACT_GAP_ON_PS", 0.75)
# Smooth mold-contact release when span grows past hold reference (not a step to zero).
const retraction_span_contact_band = env_float("TP_CLIP_RETRACTION_SPAN_CONTACT_BAND", 0.05)
# Retraction tool penalty: default 0.1·E (softer than full E for tractable explicit springback).
const retraction_contact_e_scale_value = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_CONTACT_E_SCALE", 0.1) : 1.0
const retraction_contact_e_ramp = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_CONTACT_E_RAMP", 0) != 0
const retraction_contact_e_start = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_CONTACT_E_START", 1.0) : 1.0
const retraction_contact_e_ramp_s = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_CONTACT_E_RAMP_S", 0.05) : 0.0
# Keep floor/mold Reimann contact during springback (do not fade with span growth).
const retraction_contact_on_springback = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_CONTACT_ON_SPRINGBACK", 1) != 0
sim_phase == "retraction" && (retraction_contact_e_scale[] = retraction_contact_e_ramp ?
    retraction_contact_e_start : retraction_contact_e_scale_value)
sim_phase == "retraction" && retraction_kin_cap_by_elastic &&
    println("Retraction kinematics capped by elastic σ ramp (tow-band cohesion)")
sim_phase == "retraction" && println(
    "Retraction stability fixes: implicit default; hold/elastic/kin ramps=",
    retraction_hold_release_s, "/", retraction_elastic_ramp_s, "/",
    retraction_kinematics_release_s, " s; ",
    retraction_contact_on_springback ?
        "floor+mold contact stay on during springback; " :
        "floor contact off on span growth; smooth mold span release (band=" *
        string(retraction_span_contact_band) * "); ",
    "springback dwell=", retraction_springback_dwell_s, " s; ",
    "contact E scale=", retraction_contact_e_scale_value,
    retraction_contact_e_ramp ?
        "; contact E ramp " * string(retraction_contact_e_start) * "→" *
        string(retraction_contact_e_scale_value) * " over " *
        string(retraction_contact_e_ramp_s) * " s" : "",
    retraction_accel_clamp ? "; accel clamp=" * string(retraction_max_accel) * " m/s²" : "",
    "; F update during mech ramp")
const wcsph_span_tolerance = env_float("TP_CLIP_WCSPH_SPAN_TOLERANCE", 0.02)
const wcsph_ramp_scale = Ref(1.0)
retraction_reference_span_x = Ref(0.0)
retraction_reference_span_z = Ref(0.0)
solver_dtmax_before_hold = Ref(1.0e-3)
charge_hold_dtmax_active = Ref(false)
charge_mechanics_release_announced = Ref(false)
retraction_elastic_ramp_announced = Ref(false)
retraction_kinematics_release_announced = Ref(false)
# Retraction fail-fast (opt-in): terminate only when dt is tiny *and* sim time stalls.
# Default OFF — micro-Δt creep during ramped retraction is slow but can still advance t.
# Enable with TP_CLIP_RETRACTION_FAIL_FAST_DT=1e-5 to catch true blow-ups (t frozen).
const retraction_fail_fast_dt = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_FAIL_FAST_DT", 0.0) : Inf
const retraction_fail_fast_max_steps = env_int("TP_CLIP_RETRACTION_FAIL_FAST_STEPS", 500)
const retraction_fail_fast_min_t_adv = env_float("TP_CLIP_RETRACTION_FAIL_FAST_MIN_T_ADV", 1.0e-10)
const retraction_fail_fast_stall_counter = Ref(0)
retraction_fail_fast_t_prev = Ref(-Inf)
const retraction_info_interval = sim_phase == "retraction" ?
    max(1, env_int("TP_CLIP_RETRACTION_INFO_INTERVAL", 1)) : 2
# Per-step blow-up diagnostics (max |dv|, gaps, spans, J) during retraction-only runs.
const retraction_diag_enabled = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_DIAG", 1) != 0
const retraction_diag_interval = max(1, env_int("TP_CLIP_RETRACTION_DIAG_INTERVAL", 500))
const retraction_diag_log_path = get(ENV, "TP_CLIP_RETRACTION_DIAG_LOG",
                                     joinpath("out", "retraction_diag.csv"))
const retraction_diag_spike_dv = env_float("TP_CLIP_RETRACTION_DIAG_SPIKE_DV", 1.0e5)
const retraction_diag_spike_speed = env_float("TP_CLIP_RETRACTION_DIAG_SPIKE_SPEED", 1.0e3)
# Short retraction window for fast smoke test (e.g. 0.02 s covers explicit blow-up at ~0.014 s).
const retraction_probe_s = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_PROBE_S", 0.0) : 0.0
# Early abort: stop as soon as pre-blow-up signatures appear (minutes, not hours).
const retraction_early_abort = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_EARLY_ABORT", 0) != 0
const retraction_early_abort_kick_dv = env_float("TP_CLIP_RETRACTION_EARLY_ABORT_KICK_DV", 5.0e3)
const retraction_early_abort_speed = env_float("TP_CLIP_RETRACTION_EARLY_ABORT_SPEED", 50.0)
const retraction_early_abort_span_ratio = env_float("TP_CLIP_RETRACTION_EARLY_ABORT_SPAN_RATIO", 1.08)
const retraction_early_abort_z_mm = env_float("TP_CLIP_RETRACTION_EARLY_ABORT_Z_MM", 4.5)
const retraction_early_abort_z_rate = env_float("TP_CLIP_RETRACTION_EARLY_ABORT_Z_RATE_MM_S", 200.0)
retraction_diag_step_count = Ref(0)
retraction_diag_prev_snapshot = Ref{Any}(nothing)
retraction_diag_kick_max_dv = Ref(0.0)
retraction_diag_instability_announced = Ref(false)
# Hold-only: plastic-commit substeps while mold closed (creep / stress relaxation).
const hold_relax_substeps = max(1, env_int("TP_CLIP_HOLD_RELAX_SUBSTEPS", 6))
hold_tlsph_tail_announced = Ref(false)
hold_tlsph_equilibration_now = Ref(false)

@inline function hold_tlsph_tail_start_t()
    return solidification_reached_time[] + post_solidification_dwell
end

@inline function hold_tlsph_tail_active(t)
    sim_phase == "retraction" && return false
    hold_tlsph_tail_s <= 0.0 && return false
    solidification_complete[] || return false
    elapsed = t - hold_tlsph_tail_start_t()
    return elapsed >= 0.0 && elapsed < hold_tlsph_tail_s
end

@inline function hold_tlsph_checkpoint_ready(t)
    sim_phase == "retraction" && return false
    solidification_complete[] || return false
    dwell_elapsed = t - solidification_reached_time[]
    return dwell_elapsed >= post_solidification_dwell + hold_tlsph_tail_s
end

# Mold closed, charge v/u frozen — thermal-only hold OR TLSPH equilibration tail.
@inline function hold_kinematics_frozen(t)
    sim_phase == "retraction" && return false
    t < t_compress && return false
    retraction_started[] && return false
    if hold_kinematics_mode == :active
        return false
    elseif hold_kinematics_mode == :hybrid
        return !hold_tlsph_tail_active(t)
    else
        return true
    end
end

# Closed-die hold (compression end → retraction): plastic commit substeps + optional skip_heat.
@inline function hold_plastic_relax_active(t)
    sim_phase == "retraction" && return false
    t < t_compress && return false
    return !retraction_started[]
end

@inline function charge_hold_thermal_only(t)
    hold_kinematics_frozen(t) || return false
    return !hold_tlsph_tail_active(t)
end

# Hold only: zero charge v/du and skip elastic RHS (thermal + crystallization path).
@inline charge_hold_mechanics_frozen(t) = charge_hold_thermal_only(t)

@inline function smoothstep01(x)
    y = clamp(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)
end

@inline function hold_release_factor(t)
    if !retraction_started[]
        return 1.0
    end
    elapsed = t - retraction_start_time[]
    return smoothstep01(elapsed / retraction_hold_release_s)
end

@inline function retraction_elastic_scale(t)
    if !retraction_started[]
        return 1.0
    end
    elapsed = t - retraction_start_time[]
    return smoothstep01(elapsed / retraction_elastic_ramp_s)
end

@inline function retraction_kinematics_scale(t)
    if !retraction_started[]
        return 1.0
    end
    elapsed = t - retraction_start_time[]
    kin = smoothstep01(elapsed / retraction_kinematics_release_s)
    if retraction_kin_cap_by_elastic
        kin = min(kin, retraction_elastic_scale(t))
    end
    return kin
end

# Optional: ramp tool–charge penalty stiffness over a dedicated short window (not kin ramp).
@inline function update_retraction_contact_e_scale!(t)
    if !retraction_contact_e_ramp || !retraction_started[]
        return
    end
    elapsed = t - retraction_start_time[]
    ramp = smoothstep01(elapsed / retraction_contact_e_ramp_s)
    retraction_contact_e_scale[] = retraction_contact_e_start +
        (retraction_contact_e_scale_value - retraction_contact_e_start) * ramp
end

# Mechanical dv blend: contact ramp × elastic σ ramp (kinematics uses separate du scale).
@inline function retraction_dv_mech_scale(t)
    if !retraction_started[]
        return 1.0
    end
    return hold_release_factor(t) * retraction_elastic_scale(t)
end

@inline function charge_partial_mechanics_ramp(hold_release)
    return hold_release < 1.0 - 1.0e-8
end

@inline function retraction_dv_mech_ramping(t)
    retraction_started[] && retraction_dv_mech_scale(t) < 1.0 - 1.0e-8
end

@inline function retraction_kinematics_ramping(t)
    retraction_started[] && retraction_kinematics_scale(t) < 1.0 - 1.0e-8
end

# Hold only: du ≡ 0. Retraction uses gradual kinematics scale in drift instead.
@inline function charge_kinematics_frozen(t)
    return charge_hold_thermal_only(t)
end

# Molten wall drag + Coulomb friction: ON during compression and TLSPH hold tail;
# during retraction they follow the hold-release ramp. Off during thermal-only hold.
@inline function tool_wall_coupling_active(t)
    charge_hold_thermal_only(t) && return false
    if retraction_started[]
        return hold_release_factor(t) > 1.0e-10
    end
    return true
end

@inline function charge_mechanics_fully_released(hold_release)
    return !charge_partial_mechanics_ramp(hold_release)
end

@inline function charge_kinematics_fully_released(t)
    return !retraction_kinematics_ramping(t)
end

function charge_span_ratio(cyl_sys)
    if retraction_reference_span_x[] <= 0.0 || retraction_reference_span_z[] <= 0.0
        return 1.0
    end
    xs = @view cyl_sys.current_coordinates[1, :]
    zs = @view cyl_sys.current_coordinates[3, :]
    span_x = maximum(xs) - minimum(xs)
    span_z = maximum(zs) - minimum(zs)
    return max(span_x / retraction_reference_span_x[],
               span_z / retraction_reference_span_z[])
end

@inline function charge_span_within_reference(cyl_sys)
    return charge_span_ratio(cyl_sys) <= 1.0 + wcsph_span_tolerance
end

# Ramp mold contact off over a span band (legacy springback path when contact-on-springback is off).
function mold_span_contact_scale(cyl_sys)
    retraction_contact_on_springback && return 1.0
    ratio = charge_span_ratio(cyl_sys)
    span_gate = 1.0 + wcsph_span_tolerance
    ratio <= span_gate && return 1.0
    band = max(retraction_span_contact_band, eps(Float64))
    return clamp(1.0 - smoothstep01((ratio - span_gate) / band), 0.0, 1.0)
end

function floor_reimann_activation_factor(t, cyl_sys)
    if !retraction_started[]
        return 1.0
    end
    hold = hold_release_factor(t)
    retraction_contact_on_springback && return hold
    # Legacy: floor contact ramps off as the charge expands (springback).
    span_rel = charge_span_ratio(cyl_sys)
    span_off = span_rel > 1.0 + 1.0e-8 ?
               smoothstep01((span_rel - 1.0) / max(wcsph_span_tolerance, eps(Float64))) :
               0.0
    return hold * (1.0 - span_off)
end

function mold_reimann_activation_factor(cyl_sys, mold_sys, t)
    if !retraction_started[]
        return 1.0
    end
    hold_release_factor(t) < 1.0 - 1.0e-8 && return 0.0
    min_gap = Inf
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        x_i = cyl_sys.current_coordinates[1, particle]
        z_i = cyl_sys.current_coordinates[3, particle]
        min_gap = min(min_gap,
                      tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper))
    end
    gap_off = retraction_contact_gap_off_ps * particle_spacing
    gap_on = retraction_contact_gap_on_ps * particle_spacing
    gap_w = 1.0 - smoothstep01((min_gap - gap_on) / max(gap_off - gap_on, eps(Float64)))
    return clamp(gap_w * mold_span_contact_scale(cyl_sys), 0.0, 1.0)
end

@inline function charge_particle_jacobian(cyl_sys, particle)
    F11 = cyl_sys.deformation_grad[1, 1, particle]
    F12 = cyl_sys.deformation_grad[1, 2, particle]
    F13 = cyl_sys.deformation_grad[1, 3, particle]
    F21 = cyl_sys.deformation_grad[2, 1, particle]
    F22 = cyl_sys.deformation_grad[2, 2, particle]
    F23 = cyl_sys.deformation_grad[2, 3, particle]
    F31 = cyl_sys.deformation_grad[3, 1, particle]
    F32 = cyl_sys.deformation_grad[3, 2, particle]
    F33 = cyl_sys.deformation_grad[3, 3, particle]
    return F11 * (F22 * F33 - F23 * F32) -
           F12 * (F21 * F33 - F23 * F31) +
           F13 * (F21 * F32 - F22 * F31)
end

function charge_jacobian_extremes(cyl_sys)
    J_min = Inf
    J_max = -Inf
    J_min_particle = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        J = charge_particle_jacobian(cyl_sys, particle)
        if J < J_min
            J_min = J
            J_min_particle = particle
        end
        if J > J_max
            J_max = J
        end
    end
    return J_min, J_max, J_min_particle
end

function charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    min_mold_gap = Inf
    min_floor_gap = Inf
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        x_i = cyl_sys.current_coordinates[1, particle]
        z_i = cyl_sys.current_coordinates[3, particle]
        min_mold_gap = min(min_mold_gap,
                           tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper))
        min_floor_gap = min(min_floor_gap,
                            tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower))
    end
    return min_mold_gap, min_floor_gap
end

function max_charge_kick_dv(dv_ode, cyl_sys, semi_local)
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    max_dv = 0.0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        for d in 1:TrixiParticles.ndims(cyl_sys)
            max_dv = max(max_dv, abs(dv_cyl[d, particle]))
        end
    end
    return max_dv
end

function count_charge_nonfinite_state(cyl_sys, v_cyl)
    n_bad = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        x = cyl_sys.current_coordinates[1, particle]
        z = cyl_sys.current_coordinates[3, particle]
        vx = v_cyl[1, particle]
        vz = v_cyl[3, particle]
        if !isfinite(x) || !isfinite(z) || !isfinite(vx) || !isfinite(vz)
            n_bad += 1
        end
    end
    return n_bad
end

function charge_max_sorted_x_gap_mm(cyl_sys)
    xs = Float64[]
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        push!(xs, cyl_sys.current_coordinates[1, particle])
    end
    length(xs) < 2 && return 0.0
    sort!(xs)
    max_gap = 0.0
    @inbounds for i in 2:length(xs)
        max_gap = max(max_gap, xs[i] - xs[i - 1])
    end
    return max_gap * 1e3
end

function build_retraction_diag_snapshot(integrator, cyl_sys, floor_sys, mold_sys, v_cyl)
    xs = @view cyl_sys.current_coordinates[1, :]
    zs = @view cyl_sys.current_coordinates[3, :]
    span_x = maximum(xs) - minimum(xs)
    span_z = maximum(zs) - minimum(zs)
    J_min, J_max, J_min_particle = charge_jacobian_extremes(cyl_sys)
    min_mold_gap, min_floor_gap = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    max_speed = 0.0
    max_abs_vz = 0.0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        vx = v_cyl[1, particle]
        vz = v_cyl[3, particle]
        max_speed = max(max_speed, hypot(vx, vz))
        max_abs_vz = max(max_abs_vz, abs(vz))
    end
    t_now = integrator.t
    return (
        step = retraction_diag_step_count[],
        t = t_now,
        dt = integrator.dt,
        kick_max_dv = retraction_diag_kick_max_dv[],
        max_speed = max_speed,
        max_abs_vz = max_abs_vz,
        span_x_mm = span_x * 1e3,
        span_z_mm = span_z * 1e3,
        max_x_gap_mm = charge_max_sorted_x_gap_mm(cyl_sys),
        span_ratio = charge_span_ratio(cyl_sys),
        J_min = J_min,
        J_max = J_max,
        J_min_particle = J_min_particle,
        min_mold_gap_mm = min_mold_gap * 1e3,
        min_floor_gap_mm = min_floor_gap * 1e3,
        hold_rel = hold_release_factor(t_now),
        elas_scale = retraction_elastic_scale(t_now),
        kin_scale = retraction_kinematics_scale(t_now),
        mech_dv_scale = retraction_dv_mech_scale(t_now),
        floor_act = floor_reimann_activation_factor(t_now, cyl_sys),
        mold_act = mold_reimann_activation_factor(cyl_sys, mold_sys, t_now),
        nonfinite_n = count_charge_nonfinite_state(cyl_sys, v_cyl),
    )
end

function print_retraction_diag_snapshot(label, snap)
    println(">>> RETRACTION-DIAG [", label, "] step=", snap.step,
            " t=", round(snap.t; digits=9), " s dt=", snap.dt,
            " kick_max|dv|=", round(snap.kick_max_dv; sigdigits=4),
            " m/s² max_speed=", round(snap.max_speed; sigdigits=4),
            " m/s max|vz|=", round(snap.max_abs_vz; sigdigits=4), " m/s")
    println("    span_x=", round(snap.span_x_mm; digits=4), " mm span_z=",
            round(snap.span_z_mm; digits=4), " mm max_x_gap=",
            round(snap.max_x_gap_mm; digits=4), " mm span_ratio=",
            round(snap.span_ratio; digits=5),
            " J_min=", round(snap.J_min; sigdigits=4),
            " (p=", snap.J_min_particle, ") J_max=", round(snap.J_max; sigdigits=4))
    println("    min_mold_gap=", round(snap.min_mold_gap_mm; digits=4), " mm",
            " min_floor_gap=", round(snap.min_floor_gap_mm; digits=4), " mm",
            " nonfinite_n=", snap.nonfinite_n)
    println("    ramps: hold_rel=", round(snap.hold_rel; digits=4),
            " elas=", round(snap.elas_scale; digits=4),
            " kin=", round(snap.kin_scale; digits=4),
            " mech_dv=", round(snap.mech_dv_scale; digits=4),
            " floor_act=", round(snap.floor_act; digits=4),
            " mold_act=", round(snap.mold_act; digits=4))
    flush(stdout)
    return nothing
end

function write_retraction_diag_csv_row(io, snap)
    println(io, snap.step, ",", snap.t, ",", snap.dt, ",", snap.kick_max_dv, ",",
            snap.max_speed, ",", snap.max_abs_vz, ",", snap.span_x_mm, ",",
            snap.span_z_mm, ",", snap.max_x_gap_mm, ",", snap.span_ratio, ",", snap.J_min, ",",
            snap.J_max, ",", snap.J_min_particle, ",", snap.min_mold_gap_mm, ",",
            snap.min_floor_gap_mm, ",", snap.hold_rel, ",", snap.elas_scale, ",",
            snap.kin_scale, ",", snap.mech_dv_scale, ",", snap.floor_act, ",",
            snap.mold_act, ",", snap.nonfinite_n)
    flush(io)
    return nothing
end

function retraction_early_abort_reason(snap, prev_snap)
    retraction_early_abort || return nothing
    if snap.nonfinite_n > 0
        return "nonfinite_n=$(snap.nonfinite_n)"
    end
    if isfinite(snap.kick_max_dv) && snap.kick_max_dv >= retraction_early_abort_kick_dv
        return "kick_max|dv|=$(round(snap.kick_max_dv; sigdigits=4)) m/s² >= " *
               string(retraction_early_abort_kick_dv)
    end
    if isfinite(snap.max_speed) && snap.max_speed >= retraction_early_abort_speed
        return "max_speed=$(round(snap.max_speed; sigdigits=4)) m/s >= " *
               string(retraction_early_abort_speed)
    end
    if snap.span_ratio > retraction_early_abort_span_ratio
        return "span_ratio=$(round(snap.span_ratio; digits=5)) > " *
               string(retraction_early_abort_span_ratio)
    end
    if snap.span_z_mm > retraction_early_abort_z_mm
        return "span_z=$(round(snap.span_z_mm; digits=4)) mm > " *
               string(retraction_early_abort_z_mm)
    end
    if prev_snap !== nothing
        dz = snap.span_z_mm - prev_snap.span_z_mm
        dt_sim = snap.t - prev_snap.t
        if dt_sim > 0.0 && dz / dt_sim > retraction_early_abort_z_rate
            return "span_z rate=$(round(dz / dt_sim; digits=2)) mm/s > " *
                   string(retraction_early_abort_z_rate)
        end
    end
    return nothing
end

function capture_retraction_reference_span!(cyl_sys)
    xs = @view cyl_sys.current_coordinates[1, :]
    zs = @view cyl_sys.current_coordinates[3, :]
    retraction_reference_span_x[] = maximum(xs) - minimum(xs)
    retraction_reference_span_z[] = maximum(zs) - minimum(zs)
    return nothing
end

# Spencer uses Fn = F·n_ref and I4 = |Fn|². Set n_ref ← F⁻¹·ê with ê = F·n_ref/|F·n_ref| so I4=1 at hold F.
function align_spencer_fibre_reference!(cyl_sys, fiber_dir)
    I3 = StaticArrays.SMatrix{3, 3, Float64}(I)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        det_F = det(F)
        F_reg = if !isfinite(det_F) || abs(det_F) < 1e-10
            I3
        else
            StaticArrays.SMatrix{3, 3, Float64}(F)
        end
        n_ref = StaticArrays.SVector{3, Float64}(
            fiber_dir[1, particle], fiber_dir[2, particle], fiber_dir[3, particle])
        Fn = F_reg * n_ref
        len_fn = sqrt(Fn[1]^2 + Fn[2]^2 + Fn[3]^2)
        len_fn < 1e-14 && continue
        e_sp = Fn / len_fn
        F_inv = inv(F_reg)
        n_new = F_inv * e_sp
        fiber_dir[1, particle] = n_new[1]
        fiber_dir[2, particle] = n_new[2]
        fiber_dir[3, particle] = n_new[3]
    end
    return nothing
end

function prepare_charge_state_for_retraction!(cyl_sys, semi_local, v_ode, u_ode,
                                              Fp_buf, J_ref_buf, J_p_buf, vel_grad_buf,
                                              dv_marker_buf; align_spencer::Bool=hold_end_spencer_align,
                                              reset_fp::Bool=hold_end_fp_reset)
    TrixiParticles.update_nhs!(semi_local, u_ode)

    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    freeze_charge_mechanical_state!(v_cyl, cyl_sys)

    # Keep the hold-end F from compression/hold. Recalculating F from SPH neighborhoods on
    # frozen charge geometry can yield det(F) ≪ 1 and GPa-scale bulk stress at retraction.
    apply_charge_plane_strain_F_fix!(cyl_sys)

    I3 = StaticArrays.SMatrix{3, 3, Float64}(I)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        det_F = det(F)
        if !isfinite(det_F) || abs(det_F) < 1e-10
            F = I3
            for jj in 1:3, ii in 1:3
                cyl_sys.deformation_grad[ii, jj, particle] = F[ii, jj]
            end
            for jj in 1:3, ii in 1:3
                Fp_buf[ii, jj, particle] = F[ii, jj]
            end
            J_ref_buf[particle] = use_volumetric_plasticity ? elastic_bulk_J_ref : 1.0
            J_p_buf[particle] = 1.0
        else
            if use_volumetric_plasticity
                J_p_buf[particle] = max(det_F, 1e-6)
                J_ref_buf[particle] = elastic_bulk_J_ref
                if reset_fp
                    F_iso = F / cbrt(det_F)
                    for jj in 1:3, ii in 1:3
                        Fp_buf[ii, jj, particle] = F_iso[ii, jj]
                    end
                else
                    d_fp = det(StaticArrays.SMatrix{3, 3}(@view Fp_buf[:, :, particle]))
                    if isfinite(d_fp) && abs(d_fp) > 1e-14
                        for jj in 1:3, ii in 1:3
                            Fp_buf[ii, jj, particle] /= cbrt(d_fp)
                        end
                    end
                end
            else
                F_iso = F / cbrt(det_F)
                for jj in 1:3, ii in 1:3
                    Fp_buf[ii, jj, particle] = F_iso[ii, jj]
                end
                J_ref_buf[particle] = det_F
            end
        end
        for jj in 1:3, ii in 1:3
            vel_grad_buf[ii, jj, particle] = 0.0
        end
        dv_marker_buf[1, particle] = 0.0
        dv_marker_buf[2, particle] = 0.0
        dv_marker_buf[3, particle] = 0.0
    end

    align_spencer && align_spencer_fibre_reference!(cyl_sys, fiber_direction)

    return nothing
end

function snapshot_charge_dv!(marker, dv_ode, cyl_sys, semi_local)
    dv = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        marker[1, particle] = dv[1, particle]
        marker[2, particle] = dv[2, particle]
        marker[3, particle] = dv[3, particle]
    end
    return nothing
end

function blend_charge_dv_from_marker!(dv_ode, cyl_sys, semi_local, marker, scale)
    scale >= 1.0 - 1.0e-12 && return nothing
    dv = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        dv[1, particle] = marker[1, particle] +
                          scale * (dv[1, particle] - marker[1, particle])
        dv[2, particle] = marker[2, particle] +
                          scale * (dv[2, particle] - marker[2, particle])
        dv[3, particle] = marker[3, particle] +
                          scale * (dv[3, particle] - marker[3, particle])
    end
    return nothing
end

function charge_wcsph_ramp_scale_value(cyl_sys)
    if !retraction_started[]
        return 1.0
    end
    return charge_span_within_reference(cyl_sys) ? 1.0 : 0.0
end

const retraction_vtu_suffix = get(ENV, "TP_CLIP_RETRACTION_VTU_SUFFIX", "")
const vtu_output_prefix = sim_phase == "retraction" ?
                          "molding_cfk_pseudo2d_elastic_" * ps_run_tag * "_ret" *
                          retraction_vtu_suffix * "_solution" :
                          "molding_cfk_pseudo2d_elastic_" * ps_run_tag * "_solution"
sim_phase == "retraction" &&
    println(">>> VTU prefix (retraction): ", vtu_output_prefix)
sim_phase == "retraction" &&
    println(">>> VTU save dt (retraction): ", retraction_vtu_save_dt,
            " s (compress/hold default ", vtu_save_dt, " s)")

function vtu_fiber_vf_quantity(sys, data, t)
    return TrixiParticles.nparticles(sys) == length(phase_vf_particle) ?
           phase_vf_particle : fill(NaN, TrixiParticles.nparticles(sys))
end

function save_vtu_snapshot!(integrator; iter_override=nothing,
                            save_dt=nothing, time_origin=nothing)
    dt_vtu = save_dt === nothing ?
        active_vtu_save_dt(t=integrator.t, in_retraction=retraction_started[]) : save_dt
    t0 = time_origin === nothing ? first(tspan) : time_origin
    iter = if iter_override === nothing
        Int(div(integrator.t - t0, dt_vtu, RoundNearest))
    else
        iter_override
    end
    dvdu_ode = try
        get_du(integrator)
    catch
        zero(integrator.u)
    end
    TrixiParticles.trixi2vtk(dvdu_ode, integrator.u, integrator.p, integrator.t;
                             iter=iter,
                             prefix=vtu_output_prefix,
                             max_coordinates=Inf,
                             fiber_vf=vtu_fiber_vf_quantity)
    return nothing
end

function complete_mold_retraction!(integrator)
    mold_retraction_complete[] = true
    mold_retraction_complete_time[] = integrator.t
    save_vtu_snapshot!(integrator)
    if retraction_springback_dwell_s <= 0.0
        return complete_retraction_simulation!(integrator; reason="mold stroke complete (dwell=0)")
    end
    cyl_sys = integrator.p.systems[1]
    v_cyl = TrixiParticles.wrap_v(integrator.u.x[1], cyl_sys, integrator.p)
    springback_equil_streak[] = 0
    springback_equil_prev_snap[] = build_retraction_diag_snapshot(integrator, cyl_sys,
        integrator.p.systems[2], integrator.p.systems[3], v_cyl)
    println(">>> Mold stroke complete at t=", round(integrator.t; digits=6),
            " s; springback equilibration dwell (max ",
            round(retraction_springback_dwell_s; digits=4), " s)")
    flush(stdout)
    return nothing
end

function complete_retraction_simulation!(integrator; reason="springback equilibrated")
    retraction_complete[] = true
    save_vtu_snapshot!(integrator)
    println(">>> Retraction finished (", reason, ") at t=", round(integrator.t; digits=6), " s")
    flush(stdout)
    terminate!(integrator)
    return nothing
end

function springback_equilibration_met(snap, prev_snap)
    prev_snap === nothing && return false
    dt_sim = snap.t - prev_snap.t
    dt_sim <= 0.0 && return false
    snap.max_speed > retraction_springback_equil_max_speed && return false
    dz = snap.span_z_mm - prev_snap.span_z_mm
    abs(dz / dt_sim) > retraction_springback_equil_span_rate_mm_s && return false
    return true
end

function freeze_charge_mechanical_state!(v_cyl, cyl_sys)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        v_cyl[1, particle] = 0.0
        v_cyl[2, particle] = 0.0
        v_cyl[3, particle] = 0.0
    end
    return nothing
end

function apply_charge_hold_thermal_kick!(dv_ode, v_ode, semi_local, t)
    cyl_sys = semi_local.systems[1]
    floor_sys = semi_local.systems[2]
    mold_sys = semi_local.systems[3]
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)

    freeze_charge_mechanical_state!(v_cyl, cyl_sys)

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        t_trial = v_cyl[NDIMS_CYL + 1, particle]
        if !isfinite(t_trial)
            t_trial = cyl_sys.temp[particle]
        end
        cyl_sys.temp[particle] = clamp(t_trial, temp_min_clip, temp_max_clip)
    end

    contact_heat_flux = compute_contact_heat_flux!(contact_heat_flux_buf, cyl_sys,
                                                   floor_sys, mold_sys, particle_spacing)
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    TrixiParticles.thermal_rhs_sph3d!(cyl_sys, dv_cyl, v_cyl, 0.0,
                                      particle_spacing, bound_coordinate_thermal, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        T_i = cyl_sys.temp[particle]
        cp_eff = effective_heat_capacity_particle(T_i, particle)
        dv_cyl[NDIMS_CYL + 1, particle] *= cyl_sys.cp / cp_eff
        if use_nakamura_kinetics
            α_i = crystallinity_particle_buf[particle]
            dα_dt = nakamura_dalpha_dt(clamp(T_i, temp_min_clip, temp_max_clip), α_i)
            L_i = latent_heat_particle[particle]
            cp_i = cp_particle[particle]
            dv_cyl[NDIMS_CYL + 1, particle] += L_i * dα_dt / cp_i
        end
        if contact_heat_flux[particle] != 0.0
            rho_i = cyl_sys.material_density[particle]
            dv_cyl[NDIMS_CYL + 1, particle] -= contact_heat_flux[particle] /
                                              (rho_i * cp_eff * particle_spacing)
        end
        dv_cyl[1, particle] = 0.0
        dv_cyl[2, particle] = 0.0
        dv_cyl[3, particle] = 0.0
    end

    return dv_ode
end

function apply_charge_hold_rhs!(dv_ode, v_ode, u_ode, semi_local, t)
    apply_charge_hold_thermal_kick!(dv_ode, v_ode, semi_local, t)
    # Frozen-geometry stress evaluation (no contact / momentum): keeps σ and Fp commit
    # consistent through solidification without reintroducing stiff tool interaction.
    enable_elastic_stress || return dv_ode
    cyl_sys = semi_local.systems[1]
    if t != nhs_updated_at_t[]
        TrixiParticles.update_nhs!(semi_local, u_ode)
        nhs_updated_at_t[] = t
    end
    TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(cyl_sys)
    apply_charge_plane_strain_F_fix!(cyl_sys)
    update_implicit_stress_cache!(semi_local, v_ode, t)
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        dv_cyl[1, particle] = 0.0
        dv_cyl[2, particle] = 0.0
        dv_cyl[3, particle] = 0.0
    end
    return dv_ode
end

function apply_charge_plane_strain_F_fix!(cyl_sys)
    apply_y_plane_constraint || return nothing
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        cyl_sys.deformation_grad[2, 1, particle] = 0.0
        cyl_sys.deformation_grad[2, 2, particle] = 1.0
        cyl_sys.deformation_grad[2, 3, particle] = 0.0
        cyl_sys.deformation_grad[1, 2, particle] = 0.0
        cyl_sys.deformation_grad[3, 2, particle] = 0.0
    end
    return nothing
end

# Ḟ = L·F on accepted steps during retraction (position-based F skipped in NL iterates).
function advance_F_from_vel_grad!(cyl_sys, vel_grad_cache, dt)
    I3 = StaticArrays.SMatrix{3, 3, Float64}(I)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        L_corr = TrixiParticles.extract_smatrix(cyl_sys.correction_matrix, cyl_sys, particle)
        L = StaticArrays.SMatrix{3, 3, Float64}(vel_grad_cache[:, :, particle]) * L_corr'
        F_old = TrixiParticles.deformation_gradient(cyl_sys, particle)
        F_new = (I3 + dt * L) * F_old
        for jj in 1:3, ii in 1:3
            cyl_sys.deformation_grad[ii, jj, particle] = F_new[ii, jj]
        end
    end
    apply_charge_plane_strain_F_fix!(cyl_sys)
    return nothing
end

function apply_charge_interact_skip_penalty_only!(dv_ode, v_ode, semi_local)
    cyl_sys = semi_local.systems[1]
    dv = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)

    _cached = TrixiParticles.STRESS_TENSOR_CACHE[]
    _use_cache = _cached !== nothing && _cached[1] == objectid(cyl_sys)
    _cache_stress = _use_cache ? _cached[2] : nothing
    _tensile_cache = TrixiParticles.tensile_stress_cache_tlsph(cyl_sys.tensile_stress, cyl_sys,
                                                               _use_cache, _cache_stress)

    system_coords = TrixiParticles.initial_coordinates(cyl_sys)
    nhs = TrixiParticles.get_neighborhood_search(cyl_sys, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)

    PointNeighbors.foreach_point_neighbor(
        system_coords, system_coords, nhs;
        parallelization_backend=TrixiParticles.PolyesterBackend(),
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, initial_pos_diff, initial_distance
        initial_distance^2 < eps(TrixiParticles.initial_smoothing_length(cyl_sys)^2) && return

        rho_a = @inbounds cyl_sys.material_density[particle]
        rho_b = @inbounds cyl_sys.material_density[neighbor]
        grad_kernel = TrixiParticles.smoothing_kernel_grad(cyl_sys, initial_pos_diff,
                                                           initial_distance, particle)
        m_a = @inbounds cyl_sys.mass[particle]
        m_b = @inbounds cyl_sys.mass[neighbor]

        pk1_rho2_a = _use_cache ?
            TrixiParticles.extract_smatrix(_cache_stress, cyl_sys, particle) / (rho_a * rho_a) :
            @inbounds TrixiParticles.pk1_rho2(cyl_sys, particle)
        pk1_rho2_b = _use_cache ?
            TrixiParticles.extract_smatrix(_cache_stress, cyl_sys, neighbor) / (rho_b * rho_b) :
            @inbounds TrixiParticles.pk1_rho2(cyl_sys, neighbor)

        current_pos_diff_ = @inbounds TrixiParticles.current_coords(cyl_sys, particle) -
                                      TrixiParticles.current_coords(cyl_sys, neighbor)
        current_pos_diff = convert.(eltype(cyl_sys), current_pos_diff_)
        current_distance = norm(current_pos_diff)

        dv_stress = m_b * (pk1_rho2_a + pk1_rho2_b) * grad_kernel
        dv_viscosity = @inbounds TrixiParticles.dv_viscosity_tlsph(cyl_sys, v_cyl, particle,
                                                                   neighbor, current_pos_diff,
                                                                   current_distance,
                                                                   m_a, m_b, rho_a, rho_b,
                                                                   grad_kernel)
        dv_tensile = @inbounds TrixiParticles.dv_tensile_stress_tlsph(cyl_sys.tensile_stress,
                                                                        cyl_sys, particle,
                                                                        neighbor,
                                                                        initial_distance,
                                                                        grad_kernel, m_b,
                                                                        _tensile_cache)
        dv_particle = dv_stress + dv_viscosity + dv_tensile
        for i in 1:NDIMS_CYL
            @inbounds dv[i, particle] += dv_particle[i]
        end
    end
    return dv_ode
end

function system_interaction_ramped!(dv_ode, v_ode, u_ode, semi_local,
                                    floor_activation, mold_activation)
    if floor_activation >= 1.0 - 1.0e-8 && mold_activation >= 1.0 - 1.0e-8
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
        return dv_ode
    end

    apply_charge_interact_skip_penalty_only!(dv_ode, v_ode, semi_local)

    cyl_sys = semi_local.systems[1]
    floor_sys = semi_local.systems[2]
    mold_sys = semi_local.systems[3]

    if floor_activation > 1.0e-10
        for (system, neighbor) in ((cyl_sys, floor_sys), (floor_sys, cyl_sys))
            TrixiParticles.interact!(dv_ode, v_ode, u_ode, system, neighbor, semi_local)
        end
    end

    if mold_activation > 1.0e-10
        for (system, neighbor) in ((cyl_sys, mold_sys), (mold_sys, cyl_sys))
            TrixiParticles.interact!(dv_ode, v_ode, u_ode, system, neighbor, semi_local)
        end
    end

    return dv_ode
end

mold_motion = PrescribedMotion(
    (x, t) -> begin
        z_shift = if t <= t_compress
            mold_z_shift_at_time(t)
        elseif retraction_started[]
            dt_ret = t - retraction_start_time[]
            dt_lift = min(dt_ret, retraction_duration_max)
            if dt_lift < retraction_ramp_s
                z_shift_motion_end +
                    0.5 * abs(mold_velocity_state) / retraction_ramp_s * dt_lift^2
            else
                z_shift_motion_end +
                    abs(mold_velocity_state) * (dt_lift - 0.5 * retraction_ramp_s)
            end
        else
            z_shift_motion_end
        end
        z_shift = clamp(z_shift, z_shift_motion_end, 0.0)
        x + StaticArrays.SVector(0.0, 0.0, z_shift)
    end,
    t -> true)

bound_coordinate_thermal = (3, minimum(floor_particles.coordinates[3, :]))

println("✓ Thermomechanical regime configured:")
println("  - Viscous regime: activated when T > matrix_temp_liq = ", matrix_temp_liq, " K")
println("  - Composite feedstock: glass fibres embedded in thermoplastic matrix")

# ==========================================================================================
# STEP 5: Build the TLSPH systems
# ==========================================================================================
material_polymer = (density=density_cylinder, E=E_composite_eff, nu=0.3, beta=0.000,
                    temp=thermal_softening_reference_temp,
                    temp_ref=thermal_softening_reference_temp,
                    cp=mean(cp_particle), k=mean(k_particle),
                    temp_liq=mean(temp_liq_particle),
                    h=mean(h_contact_particle),
                    hardening=mean(hardening_base_particle),
                    tmelt=mean(tmelt_particle),
                    viscosity=mean(viscosity_base_particle),
                    yield_stress=mean(yield_base_particle))
const local_regime_threshold = 0.5 * material_polymer.tmelt

# Contact penalty stiffness for clamped tools.
# A factor of 0.01 is still ~300 MPa — far stiffer than any real die clearance.
# Using the full composite E (~30 GPa) here drives TRBDF2's Newton solver to
# Δt ~ 1e-7 s under contact because the penalty residual becomes extremely stiff.
E_boundary = 0.01 * material_polymer.E

# Molten compression: Reimann k_n = (E_scale * E)/ps with E_scale ≈ K_melt/E (~1 MPa per 1% ps overlap).
const compression_contact_e_scale_default = matrix_K_melt / material_polymer.E
compression_contact_e_scale[] = env_float("TP_CLIP_COMPRESSION_CONTACT_E_SCALE",
                                        compression_contact_e_scale_default)
println("  - compression Reimann contact E scale = ",
        round(compression_contact_e_scale[], digits=4),
        " (default K_melt/E = ",
        round(compression_contact_e_scale_default, digits=4), ")")

import PointNeighbors: DictionaryCellList

nhs_template = PrecomputedNeighborhoodSearch{3}(; max_neighbors=400)

floor_system = TotalLagrangianSPHSystem(floor_particles,
                                        smoothing_kernel,
                                        smoothing_length,
                                        E_boundary,
                                        material_polymer.nu,
                                        material_polymer.beta,
                                        material_polymer.temp,
                                        material_polymer.temp_ref,
                                        material_polymer.cp,
                                        material_polymer.k,
                                        material_polymer.temp_liq,
                                        material_polymer.h,
                                        material_polymer.hardening,
                                        material_polymer.tmelt,
                                        material_polymer.yield_stress;
                                        clamped_particles=collect(1:nparticles(floor_particles)),
                                        acceleration=(0.0, 0.0, 0.0))

mold_system = TotalLagrangianSPHSystem(mold_particles,
                                       smoothing_kernel,
                                       smoothing_length,
                                       E_boundary,
                                       material_polymer.nu,
                                       material_polymer.beta,
                                       material_polymer.temp,
                                       material_polymer.temp_ref,
                                       material_polymer.cp,
                                       material_polymer.k,
                                       material_polymer.temp_liq,
                                       material_polymer.h,
                                       material_polymer.hardening,
                                       material_polymer.tmelt,
                                       material_polymer.yield_stress;
                                       clamped_particles=collect(1:nparticles(mold_particles)),
                                       clamped_particles_motion=mold_motion,
                                       acceleration=(0.0, 0.0, 0.0),
                                       self_interaction_nhs=nhs_template)

const charge_tensile_stress = charge_tensile_psi < 0.0 ?
    TensileArtificialStressMonaghan(psi=charge_tensile_psi,
                                    exponent=charge_tensile_exponent) : nothing
charge_tensile_stress !== nothing &&
    println("Charge tensile stabilisation: Monaghan psi=", charge_tensile_psi,
            " exponent=", charge_tensile_exponent)

cylinder_system = TotalLagrangianSPHSystem(polymer,
                                           smoothing_kernel,
                                           smoothing_length,
                                           material_polymer.E,
                                           material_polymer.nu,
                                           material_polymer.beta,
                                           material_polymer.temp,
                                           material_polymer.temp_ref,
                                           material_polymer.cp,
                                           material_polymer.k,
                                           material_polymer.temp_liq,
                                           material_polymer.h,
                                           material_polymer.hardening,
                                           material_polymer.tmelt,
                                           material_polymer.yield_stress;
                                           acceleration=(0.0, 0.0, 0.0),
                                           penalty_force=PenaltyForceGanzenmueller(
                                               alpha=env_float("TP_CLIP_PENALTY_ALPHA", 0.05)),
                                           tensile_stress=charge_tensile_stress,
                                           viscosity=ArtificialViscosityMonaghan(alpha=0.5,
                                                                                   beta=4.0),
                                           self_interaction_nhs=nhs_template)

# ==========================================================================================
# STEP 6: Semidiscretization and solve
# ==========================================================================================
semi = Semidiscretization(cylinder_system, floor_system, mold_system;
                          neighborhood_search=GridNeighborhoodSearch{3}(;
                              cell_list=DictionaryCellList{3}(),
                              search_radius=smoothing_length))

# Preheat: charge above solidification window (matrix_tmelt=398 K → 180°C = 453 K → fully molten).
preheating_target_temp = 453.0
tool_preheat_temp = 320.0

println("\n=== PREHEATING PHASE ===")
semi.systems[1].temp .= preheating_target_temp
semi.systems[2].temp .= tool_preheat_temp
semi.systems[3].temp .= tool_preheat_temp
println("✓ Cylinder preheated: T = $preheating_target_temp K")
println("✓ Floor preheated:    T = $tool_preheat_temp K")
println("✓ Mold  preheated:    T = $tool_preheat_temp K")
println("=== PREHEATING COMPLETE ===\n")

hold_checkpoint_restore = nothing
if sim_phase == "retraction"
    hold_checkpoint_restore = hold_checkpoint_early[]
    meta_now = checkpoint_meta(; particle_spacing=particle_spacing,
                               n_charge=n_cylinder_particles,
                               n_floor=nparticles(floor_particles),
                               n_mold=nparticles(mold_particles),
                               t_compress=t_compress,
                               z_shift_down_end=z_shift_down_end,
                               mold_velocity_state=mold_velocity_state,
                               compression_thickness_ratio=compression_thickness_ratio)
    validate_checkpoint_meta!(hold_checkpoint_restore, meta_now;
                              motion_from_checkpoint=retraction_motion_from_checkpoint)
    t_ret0 = hold_checkpoint_restore["t"]
    t_ret1 = t_ret0 + retraction_duration_max + retraction_springback_dwell_s
    if retraction_probe_s > 0.0
        t_ret1 = min(t_ret1, t_ret0 + retraction_probe_s)
        println(">>> Retraction PROBE: tspan capped to ", retraction_probe_s,
                " s (full retract ", round(retraction_duration_max; digits=4),
                " s + dwell ", round(retraction_springback_dwell_s; digits=4), " s)")
    end
    tspan = (t_ret0, t_ret1)
    retraction_early_abort &&
        println(">>> Retraction early-abort ON: kick_dv>=", retraction_early_abort_kick_dv,
                " speed>=", retraction_early_abort_speed,
                " span_ratio>", retraction_early_abort_span_ratio,
                " span_z>", retraction_early_abort_z_mm, " mm")
    println(">>> Retraction-only run: tspan=", tspan)
    flush(stdout)
elseif sim_phase == "hold"
    tspan = (0.0, t_total)
    println(">>> Hold-only run: will save checkpoint and stop before retraction")
    flush(stdout)
else
    tspan = (0.0, t_total)
end

# VTU save schedule (compress/hold/full). Retraction-only uses fixed dt instead.
const compress_hold_vtu_save_times = sim_phase == "retraction" ?
    Float64[] : build_compress_hold_vtu_save_times(t_total)
# Overwrite the hold checkpoint .dat on the same cadence as VTU output (t=0 + each save time).
const periodic_checkpoint_with_vtu = save_checkpoint_at_hold &&
    env_int("TP_CLIP_CHECKPOINT_SAVE_WITH_VTU", 1) != 0
const periodic_checkpoint_save_times = periodic_checkpoint_with_vtu && sim_phase != "retraction" ?
    vcat([0.0], compress_hold_vtu_save_times) : Float64[]
const periodic_checkpoint_index = Ref(0)
if periodic_checkpoint_with_vtu && sim_phase != "retraction"
    println(">>> Periodic checkpoint (overwrite): ", abspath(checkpoint_path),
            " at VTU cadence (t=0 + compress ", compression_vtu_save_dt,
            " s / hold ", vtu_save_dt, " s)")
    flush(stdout)
end

ode_base = semidiscretize(semi, tspan)

# ==========================================================================================
# PSEUDO-2D TLSPH FIX: Compute the 2x2 Gradient Correction Matrix
# The default 3D correction_matrix_inversion_step! sees det(L) = 0 because all particles
# are coplanar in y, so it falls back to the identity matrix L_inv = I. This completely
# disables kernel correction, ruining TLSPH stability.
# We manually re-compute the x-z components of L, invert the 2x2 block, and set y=1.
# ==========================================================================================
println(">>> Applying pseudo-2D kernel correction matrix fix...")
TrixiParticles.foreach_system(semi) do system
    TrixiParticles.set_zero!(system.correction_matrix)
    initial_coords = TrixiParticles.initial_coordinates(system)
    
    TrixiParticles.foreach_point_neighbor(system, system, initial_coords, initial_coords,
                                          semi) do particle, neighbor, pos_diff, distance
        grad_kernel = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance, particle)
        iszero(grad_kernel) && return
        volume = system.mass[neighbor] / system.material_density[neighbor]
        result = volume * grad_kernel * pos_diff'
        for j in 1:3, i in 1:3
            system.correction_matrix[i, j, particle] -= result[i, j]
        end
    end
    for particle in TrixiParticles.eachparticle(system)
        L = TrixiParticles.extract_smatrix(system.correction_matrix, system, particle)
        Lx = L[1, 1]; Lxz = L[1, 3]
        Lzx = L[3, 1]; Lz = L[3, 3]
        detL2d = Lx * Lz - Lxz * Lzx
        L_inv = Matrix(1.0I, 3, 3)
        
        # Regularize the determinant to avoid catastrophic jumps when detL2d < 1e-9
        # that drop kernel correction entirely and result in massive unphysical forces.
        detL2d_reg = sign(detL2d) * max(abs(detL2d), 1e-6)
        # Fall back to identity ONLY if the whole matrix is nearly zero to prevent NaNs
        if abs(Lx) + abs(Lz) + abs(Lxz) + abs(Lzx) > 1e-9
            L_inv[1, 1] = Lz / detL2d_reg
            L_inv[1, 3] = -Lxz / detL2d_reg
            L_inv[3, 1] = -Lzx / detL2d_reg
            L_inv[3, 3] = Lx / detL2d_reg
        end
        for j in 1:3, i in 1:3
            system.correction_matrix[i, j, particle] = L_inv[i, j]
        end
    end
end
println(">>> Pseudo-2D kernel correction applied.")
# ==========================================================================================

v_stress_buf          = zeros(3, 3, n_cylinder_particles)
v_stress_elastic_buf  = zeros(3, 3, n_cylinder_particles)
v_stress_viscous_buf  = zeros(3, 3, n_cylinder_particles)
vel_grad_buf          = zeros(3, 3, n_cylinder_particles)
wcsph_visc_lagged_dv_buf = zeros(3, n_cylinder_particles)
charge_dv_mech_marker_buf = zeros(3, n_cylinder_particles)
wcsph_visc_scratch_dv_ode = zeros(length(ode_base.u0.x[1]))
wcsph_liquid_density_ref_buf = zeros(n_cylinder_particles)
wcsph_liquid_density_curr_buf = zeros(n_cylinder_particles)
wcsph_pressure_buf = zeros(n_cylinder_particles)
wcsph_L_curr_buf = zeros(2, 2, n_cylinder_particles)
wcsph_tau_vis_buf = zeros(2, 2, n_cylinder_particles)
# Use systems from `semi` (TLSPH copies inside Semidiscretization differ from local constructors).
const wcsph_nhs_search_radius = TrixiParticles.compact_support(semi.systems[1], semi.systems[1])
const wcsph_current_nhs = PointNeighbors.copy_neighborhood_search(
    TrixiParticles.get_neighborhood_search(semi.systems[1], semi.systems[2], semi),
    wcsph_nhs_search_radius,
    n_cylinder_particles)
const wcsph_liquid_density_ref_initialized = Ref(false)

nhs_updated_at_t = Ref(-Inf)
contact_diag_interval = 5000

enable_contact_diag = false
contact_heat_flux_buf = zeros(n_cylinder_particles)
ys_particle_buf = zeros(n_cylinder_particles)
hard_particle_buf = zeros(n_cylinder_particles)
vis_particle_buf = zeros(n_cylinder_particles)
thermal_softening_particle_buf = zeros(n_cylinder_particles)
liquid_fraction_particle_buf = zeros(n_cylinder_particles)
solid_fraction_particle_buf  = zeros(n_cylinder_particles)
# J_ref: volumetric reference Jacobian set to det(F) while a particle is liquid.
# At re-solidification J_ref = J_at_solidification, so K·log(J_e/J_ref) = K·log(1) = 0,
# giving zero volumetric elastic stress at the moment of re-solidification and
# eliminating the sudden 1–2 GPa pressure spike that collapses TRBDF2 Δt.
J_ref_particle_buf = ones(n_cylinder_particles)
# Volumetric plastic Jacobian J_p (Strategy E). Stress-free bulk: J_e_vol = det(F)/J_p = 1.
J_p_particle_buf = ones(n_cylinder_particles)
# Nakamura crystallinity state: α_c ∈ [0, 1].  0 = fully liquid, 1 = fully crystallized.
# Initialised to 0 because the charge starts fully molten at ~430 K > T_melt.
crystallinity_particle_buf = zeros(n_cylinder_particles)
# The charge is preheated above matrix_tmelt, so initialize phase buffers as molten.
# The Nakamura seed is applied only when cooling enters the crystallization range.
fill!(liquid_fraction_particle_buf, 1.0)
fill!(solid_fraction_particle_buf, 0.0)
fill!(thermal_softening_particle_buf, 0.0)
vis_particle_buf .= viscosity_base_particle
orientation_tensor_state = Ref(zeros(3, 3, n_cylinder_particles))
solidification_hold_announced = Ref(false)
solidification_complete = Ref(false)

# Solidification-capture: track which particles were liquid on the previous RHS call.
first_viscous_regime    = Ref(true)
first_elastic_regime    = Ref(true)

# Fix C — match dt_cap with dtmax. The trial_dt_state is used in elastic_stress3d_trial!
# to scale the stress computation; if it doesn't match the actual dt being tried by the
# solver (especially before the first accepted step), the stress scaling is wrong, causing
# huge forces and particle ejection. Initialize dt_cap to match dtmax (set later at line ~1172).
dt_cap = 1.0e-5
trial_dt_state = Ref(dt_cap)

rhs_time_pos_ns = Ref(0)
rhs_time_nhs_ns = Ref(0)
rhs_time_quant_ns = Ref(0)
rhs_time_implicit_ns = Ref(0)
rhs_time_pressure_ns = Ref(0)
rhs_time_boundary_ns = Ref(0)
rhs_time_final_ns = Ref(0)
rhs_time_stress_interact_ns = Ref(0)
rhs_time_thermal_ns = Ref(0)

rhs_time_pos_last_ns = Ref(0)
rhs_time_nhs_last_ns = Ref(0)
rhs_time_quant_last_ns = Ref(0)
rhs_time_implicit_last_ns = Ref(0)
rhs_time_pressure_last_ns = Ref(0)
rhs_time_boundary_last_ns = Ref(0)
rhs_time_final_last_ns = Ref(0)
rhs_time_stress_interact_last_ns = Ref(0)
rhs_time_thermal_last_ns = Ref(0)

# Fine-grained timing for constitutive model functions (within update_implicit_stress_cache!)
rhs_time_property_update_ns = Ref(0)         # phase fractions, yield, hardening, thermal softening
rhs_time_viscosity_update_ns = Ref(0)        # Cross–WLF + concentration viscosity law
rhs_time_elastic_trial_ns = Ref(0)           # elastic_stress3d_trial_pseudo2d!
rhs_time_viscous_stress_ns = Ref(0)          # viscous_stress_from_cached_grad!
rhs_time_stress_blend_ns = Ref(0)            # stress blending + NaN checking
rhs_time_stress_cache_ns = Ref(0)            # update_implicit_stress_cache! wrapper
rhs_time_system_interaction_ns = Ref(0)      # TrixiParticles.system_interaction! (TLSPH forces)
rhs_time_source_terms_ns = Ref(0)            # TrixiParticles.add_source_terms!
rhs_time_wcsph_pressure_ns = Ref(0)          # apply_wcsph_liquid_pressure!
rhs_time_wcsph_viscosity_ns = Ref(0)         # apply_wcsph_liquid_viscosity! / lag apply
rhs_time_contact_heat_ns = Ref(0)            # compute_contact_heat_flux!
rhs_time_thermal_sph_ns = Ref(0)             # thermal_rhs_sph3d! (inside thermal section)
rhs_time_nakamura_rhs_ns = Ref(0)            # Nakamura latent-heat term in thermal RHS loop
rhs_time_nakamura_accepted_ns = Ref(0)       # step_nakamura_crystallinity! (accepted step)
rhs_time_plastic_commit_ns = Ref(0)          # commit_plastic_history_and_heat! (accepted step)

# Callback timing
rhs_time_callback_vel_grad_ns = Ref(0)       # refresh_velocity_gradient_cache! (callback)
rhs_time_callback_orientation_ns = Ref(0)    # update_flow_orientation_kinetics! (callback)

const formulation_profile_enabled = env_int("TP_CLIP_FORMULATION_PROFILE", 0) != 0
const formulation_profile_interval = max(1, env_int("TP_CLIP_FORMULATION_PROFILE_INTERVAL", 1000))
const formulation_profile_path = get(ENV, "TP_CLIP_FORMULATION_PROFILE_LOG",
                                     joinpath("out", "formulation_profile.log"))
const formulation_profile_step_count = Ref(0)
const rhs_eval_count = Ref(0)
const formulation_profile_last_t = Ref(0.0)
const formulation_profile_io = Ref{Union{IO, Nothing}}(nothing)

function formulation_profile_reset_counters!()
    for ref in (rhs_time_pos_ns, rhs_time_nhs_ns, rhs_time_quant_ns, rhs_time_implicit_ns,
                rhs_time_pressure_ns, rhs_time_boundary_ns, rhs_time_final_ns,
                rhs_time_stress_interact_ns, rhs_time_thermal_ns,
                rhs_time_property_update_ns, rhs_time_viscosity_update_ns,
                rhs_time_elastic_trial_ns, rhs_time_viscous_stress_ns, rhs_time_stress_blend_ns,
                rhs_time_stress_cache_ns, rhs_time_system_interaction_ns,
                rhs_time_source_terms_ns, rhs_time_wcsph_pressure_ns, rhs_time_wcsph_viscosity_ns,
                rhs_time_contact_heat_ns, rhs_time_thermal_sph_ns, rhs_time_nakamura_rhs_ns,
                rhs_time_nakamura_accepted_ns, rhs_time_plastic_commit_ns,
                rhs_time_callback_vel_grad_ns, rhs_time_callback_orientation_ns)
        ref[] = 0
    end
    rhs_eval_count[] = 0
    return nothing
end

function print_formulation_timing_report_final!(t; label="final")
    wcsph_ns = rhs_time_wcsph_pressure_ns[] + rhs_time_wcsph_viscosity_ns[]
    thermal_ns = rhs_time_contact_heat_ns[] + rhs_time_thermal_sph_ns[] + rhs_time_nakamura_rhs_ns[]
    crystallization_ns = rhs_time_nakamura_accepted_ns[] + rhs_time_nakamura_rhs_ns[]
    particle_interaction_ns = rhs_time_system_interaction_ns[] + rhs_time_source_terms_ns[]
    rhs_evals = max(rhs_eval_count[], 1)
    accepted = max(formulation_profile_step_count[], 1)
    rhs_total_ns = rhs_time_pos_ns[] + rhs_time_nhs_ns[] + rhs_time_quant_ns[] +
                   rhs_time_implicit_ns[] + rhs_time_pressure_ns[] + rhs_time_boundary_ns[] +
                   rhs_time_final_ns[] + rhs_time_stress_interact_ns[] + rhs_time_thermal_ns[] +
                   rhs_time_callback_vel_grad_ns[] + rhs_time_callback_orientation_ns[] +
                   rhs_time_nakamura_accepted_ns[] + rhs_time_plastic_commit_ns[]
    rhs_total_ns = max(rhs_total_ns, 1)
    ns_to_ms(ns) = 1.0e-6 * ns
    report = join([
        "=== Formulation timing [$label] t=$(round(t; digits=6)) s ===",
        "RHS evaluations: $(rhs_eval_count[]) over $accepted accepted steps",
        @sprintf("  WCSPH                         %8.3f ms/eval (%5.1f%%)",
                 ns_to_ms(wcsph_ns / rhs_evals), 100 * wcsph_ns / rhs_total_ns),
        @sprintf("  Particle interaction (TLSPH)    %8.3f ms/eval (%5.1f%%)",
                 ns_to_ms(particle_interaction_ns / rhs_evals), 100 * particle_interaction_ns / rhs_total_ns),
        @sprintf("  Property + viscosity + stress %8.3f ms/eval",
                 ns_to_ms((rhs_time_property_update_ns[] + rhs_time_viscosity_update_ns[] +
                           rhs_time_elastic_trial_ns[] + rhs_time_viscous_stress_ns[] +
                           rhs_time_stress_blend_ns[]) / rhs_evals)),
        @sprintf("  Thermal + crystallization     %8.3f ms/eval",
                 ns_to_ms((thermal_ns + crystallization_ns) / rhs_evals)),
        "Note: TRBDF2 Jacobian + dense LU not included.",
    ], '\n')
    println(report)
    flush(stdout)
    if formulation_profile_io[] !== nothing
        println(formulation_profile_io[], report)
        println(formulation_profile_io[])
        flush(formulation_profile_io[])
    end
    return nothing
end

function print_formulation_timing_report!(integrator; label="snapshot")
    formulation_profile_last_t[] = integrator.t
    ns_to_ms(ns) = 1.0e-6 * ns
    rhs_evals = max(rhs_eval_count[], 1)
    accepted = max(formulation_profile_step_count[], 1)

    wcsph_ns = rhs_time_wcsph_pressure_ns[] + rhs_time_wcsph_viscosity_ns[]
    thermal_ns = rhs_time_contact_heat_ns[] + rhs_time_thermal_sph_ns[] + rhs_time_nakamura_rhs_ns[]
    crystallization_ns = rhs_time_nakamura_accepted_ns[] + rhs_time_nakamura_rhs_ns[]
    particle_interaction_ns = rhs_time_system_interaction_ns[] + rhs_time_source_terms_ns[]
    constitutive_ns = rhs_time_property_update_ns[] + rhs_time_viscosity_update_ns[] +
                      rhs_time_elastic_trial_ns[] + rhs_time_viscous_stress_ns[] +
                      rhs_time_stress_blend_ns[]
    rhs_total_ns = rhs_time_pos_ns[] + rhs_time_nhs_ns[] + rhs_time_quant_ns[] +
                   rhs_time_implicit_ns[] + rhs_time_pressure_ns[] + rhs_time_boundary_ns[] +
                   rhs_time_final_ns[] + rhs_time_stress_interact_ns[] + rhs_time_thermal_ns[] +
                   rhs_time_callback_vel_grad_ns[] + rhs_time_callback_orientation_ns[] +
                   rhs_time_nakamura_accepted_ns[] + rhs_time_plastic_commit_ns[]
    rhs_total_ns = max(rhs_total_ns, 1)

    lines = String[
        "=== Formulation timing [$label] t=$(round(integrator.t; digits=6)) s ===",
        "RHS evaluations: $(rhs_eval_count[]) over $accepted accepted steps",
        "Per RHS eval (ms):",
        @sprintf("  WCSPH (pressure+viscosity)     %8.3f  (%5.1f%%)",
                 ns_to_ms(wcsph_ns / rhs_evals), 100 * wcsph_ns / rhs_total_ns),
        @sprintf("  Particle interaction (TLSPH)   %8.3f  (%5.1f%%)",
                 ns_to_ms(particle_interaction_ns / rhs_evals), 100 * particle_interaction_ns / rhs_total_ns),
        @sprintf("  Property update (phase/yield)  %8.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_property_update_ns[] / rhs_evals),
                 100 * rhs_time_property_update_ns[] / rhs_total_ns),
        @sprintf("  Viscosity evolution (Cross-WLF)%7.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_viscosity_update_ns[] / rhs_evals),
                 100 * rhs_time_viscosity_update_ns[] / rhs_total_ns),
        @sprintf("  Elastic trial stress           %8.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_elastic_trial_ns[] / rhs_evals),
                 100 * rhs_time_elastic_trial_ns[] / rhs_total_ns),
        @sprintf("  Viscous TLSPH stress           %8.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_viscous_stress_ns[] / rhs_evals),
                 100 * rhs_time_viscous_stress_ns[] / rhs_total_ns),
        @sprintf("  Stress blend / regime gate     %8.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_stress_blend_ns[] / rhs_evals),
                 100 * rhs_time_stress_blend_ns[] / rhs_total_ns),
        @sprintf("  Thermal (cond.+tool HTC)       %8.3f  (%5.1f%%)",
                 ns_to_ms((rhs_time_contact_heat_ns[] + rhs_time_thermal_sph_ns[]) / rhs_evals),
                 100 * (rhs_time_contact_heat_ns[] + rhs_time_thermal_sph_ns[]) / rhs_total_ns),
        @sprintf("  Crystallization (Nakamura)     %8.3f  (%5.1f%%)",
                 ns_to_ms(crystallization_ns / rhs_evals), 100 * crystallization_ns / rhs_total_ns),
        @sprintf("  Fibre orientation (callback)   %8.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_callback_orientation_ns[] / accepted),
                 100 * rhs_time_callback_orientation_ns[] / rhs_total_ns),
        @sprintf("  Velocity gradient (callback)   %8.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_callback_vel_grad_ns[] / accepted),
                 100 * rhs_time_callback_vel_grad_ns[] / rhs_total_ns),
        @sprintf("  Plastic commit (callback)      %8.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_plastic_commit_ns[] / accepted),
                 100 * rhs_time_plastic_commit_ns[] / rhs_total_ns),
        @sprintf("  Neighborhood search (NHS)      %8.3f  (%5.1f%%)",
                 ns_to_ms(rhs_time_nhs_ns[] / rhs_evals), 100 * rhs_time_nhs_ns[] / rhs_total_ns),
        @sprintf("  SPH infrastructure (pos/quant) %8.3f  (%5.1f%%)",
                 ns_to_ms((rhs_time_pos_ns[] + rhs_time_quant_ns[] + rhs_time_implicit_ns[] +
                           rhs_time_pressure_ns[] + rhs_time_boundary_ns[] +
                           rhs_time_final_ns[]) / rhs_evals),
                 100 * (rhs_time_pos_ns[] + rhs_time_quant_ns[] + rhs_time_implicit_ns[] +
                        rhs_time_pressure_ns[] + rhs_time_boundary_ns[] +
                        rhs_time_final_ns[]) / rhs_total_ns),
        "Note: TRBDF2 Jacobian + dense LU linear solves are NOT in RHS timers (dominant during micro-dt).",
    ]
    report = join(lines, '\n')
    println(report)
    flush(stdout)
    if formulation_profile_io[] !== nothing
        println(formulation_profile_io[], report)
        println(formulation_profile_io[])
        flush(formulation_profile_io[])
    end
    formulation_profile_reset_counters!()
    return nothing
end

if formulation_profile_enabled
    mkpath(dirname(abspath(formulation_profile_path)))
    formulation_profile_io[] = open(formulation_profile_path, "w")
    println(">>> Formulation profile ON: log=", abspath(formulation_profile_path),
            " interval=", formulation_profile_interval, " accepted steps")
    flush(stdout)
end

# Previous-cumulative TRBDF2 stat counters for per-step delta reporting.
const diag_prev_nf          = Ref(0)
const diag_prev_njacs       = Ref(0)
const diag_prev_nsolve      = Ref(0)
const diag_prev_nnonliniter = Ref(0)
const diag_prev_nreject     = Ref(0)

# Plastic-heat-injection diagnostic: at most N prints across the whole run,
# triggered when a single commit step would inject more than `diag_heat_dT_K`.
const diag_heat_remaining = Ref(50)
const diag_heat_dT_K      = 50.0

# RHS stiffness diagnostic: fires once per stall event.
# A stall is a new event when t > diag_stiff_last_t + 1e-4 (0.1 ms gap).
# Prints the dominant stress term so the root cause is identifiable.
const diag_stiff_last_t = Ref(-1.0)   # sim time of last diagnostic print
const diag_force_stage_last_t = Ref(-1.0)
const diag_force_stage_start_t = env_float("TP_CLIP_FORCE_STAGE_START_T", 0.0)
const diag_force_stage_dt = env_float("TP_CLIP_FORCE_STAGE_DT", 1.0e-4)

# function contact_condition(cyl_sys, floor_sys, mold_sys, i, contact_dist)
#     z_i         = cyl_sys.current_coordinates[3, i]
#     z_floor_top = maximum(floor_sys.current_coordinates[3, :])
#     z_mold_bot  = minimum(mold_sys.current_coordinates[3, :])
#     return (z_i - z_floor_top) <= contact_dist || (z_mold_bot - z_i) <= contact_dist
# end

@inline soft_limit(x, lim) = lim * tanh(x / max(lim, eps(Float64)))

@inline function retraction_charge_accel_cap(t)
    if retraction_started[] && retraction_accel_clamp
        kin = retraction_kinematics_ramping(t) ? retraction_kinematics_scale(t) : 1.0
        return retraction_max_accel * max(kin, 0.1)
    elseif retraction_started[] && retraction_kinematics_ramping(t)
        return max_cylinder_accel * max(retraction_kinematics_scale(t), 0.1)
    else
        return max_cylinder_accel
    end
end

function apply_retraction_charge_accel_caps!(dv_cyl, cyl_sys, t)
    apply_velocity_safety_clamps() || return nothing
    accel_cap = retraction_charge_accel_cap(t)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        dv_cyl[1, particle] = soft_limit(dv_cyl[1, particle], accel_cap)
        dv_cyl[3, particle] = soft_limit(dv_cyl[3, particle], accel_cap)
    end
    return nothing
end

function force_stage_maxima(prev_dv, dv_ode, cyl_sys, semi_local)
    dv_now = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    nd = TrixiParticles.ndims(cyl_sys)

    max_abs = 0.0
    pid_abs = 0
    max_delta = 0.0
    pid_delta = 0

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        ax = dv_now[1, particle]
        az = dv_now[nd, particle]
        abs_norm = sqrt(ax * ax + az * az)
        if abs_norm > max_abs
            max_abs = abs_norm
            pid_abs = particle
        end

        dax = dv_now[1, particle] - prev_dv[1, particle]
        daz = dv_now[nd, particle] - prev_dv[nd, particle]
        delta_norm = sqrt(dax * dax + daz * daz)
        if delta_norm > max_delta
            max_delta = delta_norm
            pid_delta = particle
        end
    end

    return max_abs, pid_abs, max_delta, pid_delta
end

function print_force_stage_delta!(label, prev_dv, dv_ode, cyl_sys, semi_local, t)
    # Diagnostics disabled.
    # max_abs, pid_abs, max_delta, pid_delta =
    #     force_stage_maxima(prev_dv, dv_ode, cyl_sys, semi_local)
    # dv_now = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    # prev_dv .= dv_now
    #
    # pid = pid_delta > 0 ? pid_delta : pid_abs
    # if pid > 0
    #     println(stderr,
    #             "  FORCE-STAGE ", label,
    #             "  |dv|max=", round(max_abs; digits=4),
    #             " p=", pid_abs,
    #             "  |Δdv|=", round(max_delta; digits=4),
    #             " pΔ=", pid_delta,
    #             "  T=", round(cyl_sys.temp[pid]; digits=2),
    #             "  lf=", round(liquid_fraction_particle_buf[pid]; digits=3),
    #             "  sf=", round(solid_fraction_particle_buf[pid]; digits=3),
    #             "  pos=(",
    #             round(cyl_sys.current_coordinates[1, pid] * 1e3; digits=2),
    #             ", ",
    #             round(cyl_sys.current_coordinates[3, pid] * 1e3; digits=2),
    #             ") mm")
    # else
    #     println(stderr,
    #             "  FORCE-STAGE ", label,
    #             "  |dv|max=", round(max_abs; digits=4),
    #             "  |Δdv|=", round(max_delta; digits=4))
    # end
    return nothing
end

# @inline function soft_box(x, lo, hi)
#     c = 0.5 * (lo + hi)
#     r = max(0.5 * (hi - lo), eps(Float64))
#     return c + r * tanh((x - c) / r)
# end

function local_tool_surface_center_z(system_tool, x_target, z_target; side)
    best_dx = Inf
    best_z = side === :lower ? -Inf : Inf
    found = false

    @inbounds for p in 1:nparticles(system_tool)
        x_p = system_tool.current_coordinates[1, p]
        z_p = system_tool.current_coordinates[3, p]
        is_candidate = side === :lower ? (z_p <= z_target) : (z_p >= z_target)
        is_candidate || continue

        dx = abs(x_p - x_target)
        if dx < best_dx - eps(Float64)
            best_dx = dx
            best_z = z_p
            found = true
        elseif abs(dx - best_dx) <= eps(Float64)
            best_z = side === :lower ? max(best_z, z_p) : min(best_z, z_p)
            found = true
        end
    end

    if found
        return best_z
    end

    @inbounds for p in 1:nparticles(system_tool)
        x_p = system_tool.current_coordinates[1, p]
        z_p = system_tool.current_coordinates[3, p]
        dx = abs(x_p - x_target)
        if dx < best_dx - eps(Float64)
            best_dx = dx
            best_z = z_p
        elseif abs(dx - best_dx) <= eps(Float64)
            best_z = side === :lower ? max(best_z, z_p) : min(best_z, z_p)
        end
    end

    return best_z
end

@inline function tool_surface_gap(system_tool, x_target, z_target, particle_spacing; side)
    z_tool_center = local_tool_surface_center_z(system_tool, x_target, z_target; side=side)
    center_gap = side === :lower ? (z_target - z_tool_center) : (z_tool_center - z_target)
    return center_gap - particle_spacing
end

@inline function tool_thermal_gap(system_tool, x_target, z_target, particle_spacing)
    best_distance2 = Inf

    @inbounds for p in 1:nparticles(system_tool)
        dx = x_target - system_tool.current_coordinates[1, p]
        dz = z_target - system_tool.current_coordinates[3, p]
        distance2 = dx * dx + dz * dz
        if distance2 < best_distance2
            best_distance2 = distance2
        end
    end

    return sqrt(best_distance2) - particle_spacing
end

function compute_contact_heat_flux!(ext_heat_per_particle, system_cyl, system_floor,
                                    system_mold, particle_spacing)
    fill!(ext_heat_per_particle, 0.0)

    T_floor = mean(system_floor.temp)
    T_mold = mean(system_mold.temp)

    h_contact = system_cyl.h
    contact_threshold = contact_heat_gap_threshold
    T_smooth_band = 5.0

    @inline smoothstep01(x) = begin
        y = clamp(x, 0.0, 1.0)
        y * y * (3.0 - 2.0 * y)
    end
    # C∞ gap activation: tanh logistic instead of smoothstep01 (which is only C¹
    # at the endpoints).  Centered at 0.4*threshold, with transition width
    # 0.15*threshold.  Restored from previous session because removing the C¹
    # kinks at the contact boundary recovered Δt by 4-6 orders of magnitude.
    @inline gap_activation(gap) = 0.5 * (1.0 - tanh((gap - 0.4 * contact_threshold) /
                                                    (0.15 * contact_threshold)))
    # C∞ dT activation: 1 - exp(-(dT/band)²) is smooth at dT=0 and saturates to 1.
    @inline dT_activation(dT) = 1.0 - exp(-(dT / T_smooth_band)^2)

    @inbounds for i in 1:nparticles(system_cyl)
        x_i = system_cyl.current_coordinates[1, i]
        z_i = system_cyl.current_coordinates[3, i]
        T_i = system_cyl.temp[i]
        h_contact_i = h_contact_particle[i]

        gap_floor = tool_thermal_gap(system_floor, x_i, z_i, particle_spacing)
        if gap_floor <= contact_threshold
            gap_floor_eff = clamp(gap_floor, 0.0, contact_threshold)
            dT = T_i - T_floor
            flux_floor = h_contact_i * dT * gap_activation(gap_floor_eff) * dT_activation(dT)
            if flux_floor != 0.0
                ext_heat_per_particle[i] += flux_floor
            end
        end

        gap_mold = tool_thermal_gap(system_mold, x_i, z_i, particle_spacing)
        if gap_mold <= contact_threshold
            gap_mold_eff = clamp(gap_mold, 0.0, contact_threshold)
            dT = T_i - T_mold
            flux_mold = h_contact_i * dT * gap_activation(gap_mold_eff) * dT_activation(dT)
            if flux_mold != 0.0
                ext_heat_per_particle[i] += flux_mold
            end
        end
    end

    return ext_heat_per_particle
end

@inline function effective_heat_capacity_particle(temp_i, particle)
    # When Nakamura kinetics are active, latent heat is supplied as an explicit
    # source term (L·dα/dt) in the RHS, so the apparent-cp boost is disabled here
    # to prevent double-counting.
    cp_eff = cp_particle[particle]
    if !use_nakamura_kinetics
        latent_i = latent_heat_particle[particle]
        if latent_i > 0.0
            # C∞ Gaussian latent-heat release centered between T_liq and T_melt.
            # Replaces the hard window `if T_liq <= T <= T_melt` which had two
            # discontinuities in dcp/dT; those corrupted the implicit Jacobian
            # whenever a particle crossed either edge.  Total integrated
            # enthalpy release ≈ latent_i (±<0.3% truncation outside ±4σ).
            T_center    = 0.5 * (matrix_tmelt + matrix_temp_liq)
            sigma       = max(0.25 * (matrix_tmelt - matrix_temp_liq), eps(Float64))
            norm_factor = 1.0 / (sigma * sqrt(2 * pi))
            x           = (temp_i - T_center) / sigma
            cp_eff += latent_i * norm_factor * exp(-0.5 * x * x)
        end
    end
    return cp_eff
end

@inline function liquid_fraction_particle(temp_i, particle)
    temp_liq_i = temp_liq_particle[particle]
    temp_melt_i = tmelt_particle[particle]
    transition_span = max(temp_melt_i - temp_liq_i, eps(Float64))
    return clamp((temp_i - temp_liq_i) / transition_span, 0.0, 1.0)
end

# ------------------------------------------------------------------------------------------
# Nakamura non-isothermal crystallization kinetics
# ------------------------------------------------------------------------------------------

"""
    nakamura_KN(T)

Hoffman-Lauritzen rate constant K_N(T) [1/s] for iPP.
Zero above T_melt (no crystallization in the melt) and below T_∞ (mobility frozen).
"""
@inline function nakamura_KN(T)
    T >= matrix_tmelt       && return 0.0
    T <= nakamura_T_inf     && return 0.0
    ΔT = nakamura_T_m0 - T
    ΔT <= 0.0               && return 0.0
    transport  = nakamura_U_star / (nakamura_R * (T - nakamura_T_inf))
    nucleation = nakamura_Kg  / (T * ΔT)
    K_N = nakamura_K0 * exp(-transport - nucleation)
    # Smooth onset over 20 K below matrix_tmelt: recalescence bursts are ~2.5 K wide,
    # so a 5 K window was too narrow (burst traversed ~60% of the window per step,
    # causing large Jacobian errors with stale Δnjac=1 and dt collapse).
    # 20 K window keeps |dK_N/dT| 4× smaller so the error estimator stays bounded.
    # At T = matrix_tmelt: ramp = 0 (K_N = 0). At T = matrix_tmelt − 20: ramp = 1 (full K_N).
    s = clamp((matrix_tmelt - T) / 20.0, 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)   # C¹ smoothstep
    return K_N * s
end

"""
    nakamura_dalpha_dt(T, α_c)

Nakamura crystallization rate dα/dt [1/s] at temperature T and current
crystallinity α_c ∈ [0, 1].
"""
@inline function nakamura_dalpha_dt(T, α_c)
    α_c >= 1.0 - 1e-10 && return 0.0
    K_N = nakamura_KN(T)
    K_N <= 0.0          && return 0.0
    # Seed the Avrami/Nakamura law once the temperature is in the crystallization
    # range. Without this, α_c = 0 gives -log(1-α_c)=0 and dα/dt stays exactly zero.
    α_eff = clamp(max(α_c, nakamura_seed_crystallinity), 0.0, 1.0 - 1e-10)
    xi = -log(max(1.0 - α_eff, 1e-14))
    n  = nakamura_n
    return n * K_N * (1.0 - α_eff) * xi^((n - 1.0) / n)
end

"""
    step_nakamura_crystallinity!(crystallinity_buf, temp, dt)

Forward-Euler integration of the Nakamura ODE for all particles.
Also updates `liquid_fraction_particle_buf` and `solid_fraction_particle_buf`
in-place from the new crystallinity values.
"""
function step_nakamura_crystallinity!(crystallinity_buf, temp,
                                      lf_buf, sf_buf, dt)
    @inbounds for i in eachindex(temp)
        T_i = clamp(temp[i], temp_min_clip, temp_max_clip)
        α_i = crystallinity_buf[i]
        α_start = nakamura_KN(T_i) > 0.0 ? max(α_i, nakamura_seed_crystallinity) : α_i
        dα  = nakamura_dalpha_dt(T_i, α_start) * dt
        α_new = clamp(α_start + dα, 0.0, 1.0)
        crystallinity_buf[i] = α_new
        # Solid fraction from Nakamura; liquid fraction is the complement
        sf_buf[i] = α_new
        lf_buf[i] = 1.0 - α_new
    end
end

function project_orientation_tensor!(A)
    A_sym = 0.5 .* (A .+ A')
    eig = eigen(Symmetric(A_sym))
    vals = clamp.(eig.values, 1.0e-8, 1.0)
    vals ./= sum(vals)
    A_proj = eig.vectors * Diagonal(vals) * eig.vectors'
    A .= 0.5 .* (A_proj .+ A_proj')
    return A
end

function initialize_orientation_tensor_state(fiber_direction; a0=0.85)
    n = size(fiber_direction, 2)
    A_state = zeros(3, 3, n)
    I3 = Matrix{Float64}(I, 3, 3)

    @inbounds for i in 1:n
        p_col = fiber_direction[:, i]
        nrm = norm(p_col)
        if nrm < eps(Float64)
            p = [1.0, 0.0, 0.0]
        else
            p = p_col / nrm
        end
        P = p * p'
        A_state[:, :, i] .= a0 .* P .+ 0.5 .* (1.0 - a0) .* (I3 .- P)
        project_orientation_tensor!(view(A_state, :, :, i))
    end

    return A_state
end

orientation_tensor_state[] = initialize_orientation_tensor_state(fiber_direction)

function update_flow_orientation_kinetics!(A_state, orientation_scalar, vel_grad, dt,
                                           fiber_direction;
                                           xi_ft, ci_ft, kappa_rsc,
                                           ci_ard_parallel, ci_ard_perp)
    I3 = Matrix{Float64}(I, 3, 3)
    dt_eff = max(dt, 1.0e-12)

    @inbounds for i in 1:size(A_state, 3)
        A = view(A_state, :, :, i)
        L = view(vel_grad, :, :, i)

        D = 0.5 .* (L .+ L')
        W = 0.5 .* (L .- L')
        gamma_dot = sqrt(2.0 * sum(D .* D))

        trAD = sum(A .* D)
        jeffery = W * A - A * W +
                  kappa_rsc * xi_ft * (D * A + A * D - 2.0 * trAD .* A)

        p_col = fiber_direction[:, i]
        pnorm = norm(p_col)
        if pnorm < eps(Float64)
            p = [1.0, 0.0, 0.0]
        else
            p = p_col / pnorm
        end

        D_r = ci_ard_perp .* I3 .+ (ci_ard_parallel - ci_ard_perp) .* (p * p')
        diffusion = 2.0 * ci_ft * gamma_dot .* (I3 .- 3.0 .* A)
        ard = 2.0 * gamma_dot .* (D_r .- tr(D_r) .* A)

        A .+= dt_eff .* (jeffery .+ diffusion .+ ard)
        project_orientation_tensor!(A)

        align = clamp(dot(p, A * p), 1.0 / 3.0, 1.0)
        align_norm = (align - 1.0 / 3.0) / (2.0 / 3.0)
        orientation_scalar[i] = 0.9 + 0.3 * align_norm
    end

    return nothing
end

function refresh_velocity_gradient_cache!(system, v, semi, vel_grad_cache)
    fill!(vel_grad_cache, 0.0)

    initial_coords = TrixiParticles.initial_coordinates(system)
    nhs = TrixiParticles.get_neighborhood_search(system, system, semi)
    PointNeighbors.foreach_point_neighbor(
        initial_coords, initial_coords, nhs;
        parallelization_backend=TrixiParticles.PolyesterBackend()
    ) do particle, neighbor, pos_diff_initial, initial_distance
        initial_distance^2 < eps(TrixiParticles.initial_smoothing_length(system)^2) && return

        volume = @inbounds system.mass[neighbor] / system.material_density[neighbor]
        vel_diff = v[:, particle] - v[:, neighbor]
        grad_kernel = TrixiParticles.smoothing_kernel_grad(system, pos_diff_initial,
                                                           initial_distance, particle)
        result_v = volume * vel_diff * grad_kernel'

        for j in 1:TrixiParticles.ndims(system), i in 1:TrixiParticles.ndims(system)
            @inbounds vel_grad_cache[i, j, particle] -= result_v[i, j]
        end
    end

    return vel_grad_cache
end

function viscous_stress_from_cached_grad!(system, vis;
                                          v_vis_buf=nothing,
                                          vel_grad_buf,
                                          include_bulk_term=true,
                                          solid_fraction_buf=nothing,
                                          skip_solid_fraction=1.0)
    (; young_modulus, poisson_ratio) = system

    n_particles = size(vel_grad_buf, 3)
    v_vis = v_vis_buf !== nothing ? v_vis_buf : zeros(Float64, 3, 3, n_particles)
    K = young_modulus / (3 - 6 * poisson_ratio)

    Threads.@threads for particle in 1:n_particles
        if solid_fraction_buf !== nothing && solid_fraction_buf[particle] >= skip_solid_fraction
            @inbounds v_vis[:, :, particle] .= 0.0
            continue
        end

        F = TrixiParticles.deformation_gradient(system, particle)
        L_corr = @inbounds TrixiParticles.correction_matrix(system, particle)
        J = max(det(F), 1e-6)

        _L = vel_grad_buf[:, :, particle] * L_corr'
        d = 0.5 * (_L + _L')
        dev_d = d - 1 / 3 * tr(d) * I

        tau = 2 * vis[particle] * dev_d
        # Clip deviatoric viscous stress norm to matrix_yield_stress to prevent
        # unphysically large σ_v (caused by huge vel-grad when a particle is being
        # compressed against the mold wall) from driving Δt to femtoseconds.
        # This is equivalent to a Bingham-like yield cap: viscous flow cannot exceed
        # the solid-state yield stress regardless of strain rate.
        tau_norm = sqrt(sum(tau .^ 2))
        if tau_norm > matrix_yield_stress
            tau = tau * (matrix_yield_stress / tau_norm)
        end
        if include_bulk_term
            # Solid-skeleton bulk for semi-solid particles. Use matrix_K_melt (≈0.1 GPa)
            # instead of the composite K (≈22 GPa): above matrix_tmelt the crystalline
            # stiffness is gone; using K_composite here produces O(100 MPa) pressure
            # from sub-1% overlaps and collapses Δt. The fiber-composite stiffness is
            # handled by the elastic path (active below tmelt via matrix_blend).
            lf = solid_fraction_buf !== nothing ? (1.0 - solid_fraction_buf[particle]) : 1.0
            p_vol = (1.0 - lf) * matrix_K_melt * log(J)
            tau += p_vol * I
        end

        # Clamp singular values of F before computing FinvT. For liquid/semi-liquid
        # particles that have flowed far from their reference position, F becomes
        # highly anisotropic and pinv(F) has large eigenvalues that amplify even a
        # physically-capped Kirchhoff stress tau by 10–100× in PK space.
        # This is a TLSPH discretisation artifact (reference-config framework breaking
        # down under large deformation), not a real stress amplification.
        # Clamping singular values to [s_min, s_max] is the standard FEM large-deformation
        # stabilisation: preserves accuracy for moderate deformations while preventing
        # singularity. Physically: the viscous SPH kernel gradient contribution
        # saturates once the particle has moved beyond ~2× its reference spacing.
        # Blend out TLSPH viscous contribution for liquid/semi-liquid particles.
        # Above matrix_tmelt the reference-config framework (F, L_corr) is no longer
        # valid: particles have rearranged far from reference, so FinvT and L_corr
        # become ill-conditioned and amplify stress by 10–100×. This is a discretisation
        # artefact, not physics. The elastic path uses the same 5K blend to zero out
        # its contribution; we do the same here so both paths are consistent.
        # For T > matrix_tmelt+2.5K: visc_blend=0 → no TLSPH viscous term.
        # For T < matrix_tmelt-2.5K: visc_blend=1 → full TLSPH viscous term.
        # Liquid-phase behaviour is handled by apply_wcsph_liquid_pressure!.
        # NOTE: do NOT use a nonzero floor here — even 5% TLSPH viscous on a liquid
        # particle with ill-conditioned F causes thermal runaway (T→2000K) via
        # viscous dissipation far exceeding conduction.
        temp_p = system.temp[particle]
        visc_blend = clamp((matrix_tmelt + 2.5 - temp_p) / 5.0, 0.0, 1.0)
        if visc_blend < 1e-6
            v_vis[:, :, particle] .= 0.0
            continue
        end

        FinvT = pinv(F)'
        v_vis[:, :, particle] .= (tau * FinvT) * L_corr * visc_blend
    end

    return v_vis
end

@inline function particle_J_total(F_reg)
    return max(det(F_reg), 1e-6)
end

@inline function elastic_volumetric_J(F_reg, particle, Fp_local)
    if use_volumetric_plasticity
        J_p = max(J_p_particle_buf[particle], 1e-6)
        return max(particle_J_total(F_reg) / J_p, 1e-6)
    end
    Fp_inv = inv(Fp_local)
    return max(det(F_reg * Fp_inv), 1e-6)
end

@inline function matrix_bulk_log_J(F_reg, particle, Fp_local, matrix_blend, J_ref_legacy)
    if use_volumetric_plasticity
        matrix_blend < 1.0 - 1e-6 && return 0.0
        J_e_vol = elastic_volumetric_J(F_reg, particle, Fp_local)
        return log(max(J_e_vol / elastic_bulk_J_ref, 1e-6))
    end
    if matrix_blend < 1.0 - 1e-6
        J_e = elastic_volumetric_J(F_reg, particle, Fp_local)
        return log(max(J_e / J_e, 1e-6))
    end
    J_e = elastic_volumetric_J(F_reg, particle, Fp_local)
    return log(max(J_e / max(J_ref_legacy, 1e-6), 1e-6))
end

function commit_volumetric_plastic_relax!(F_reg, particle; active, commit_blend, matrix_blend)
    use_volumetric_plasticity || return
    active || return
    commit_blend < 1e-6 && return
    J_tot = particle_J_total(F_reg)
    if matrix_blend < 1.0 - 1e-6
        J_p_particle_buf[particle] = J_tot
        return
    end
    J_p = max(J_p_particle_buf[particle], 1e-6)
    rate = clamp(volumetric_plastic_relax_rate, 0.0, 1.0)
    J_p_new = J_p + rate * (J_tot - J_p)
    J_p_particle_buf[particle] = clamp(J_p_new, 1e-6, max(J_tot, 1e-6))
    return nothing
end

function elastic_stress3d_trial_skip_liquid!(system, ys, hard, vis, dt, _alpha, _Fp, semi;
                                             solid_fraction_buf,
                                             v_elas_buf=nothing,
                                             skip_solid_fraction=0.0)
    (; deformation_grad, young_modulus, poisson_ratio, temp, tmelt, hardening) = system

    n_particles = size(deformation_grad, 3)
    v_elas = v_elas_buf !== nothing ? v_elas_buf : zeros(eltype(young_modulus), 3, 3, n_particles)

    mu = young_modulus / (2 + 2 * poisson_ratio)
    K  = young_modulus / (3 - 6 * poisson_ratio)
    H_theta = 1.0 / (tmelt - system.temp_ref[1])

    Threads.@threads for particle in 1:n_particles
        if solid_fraction_buf[particle] <= skip_solid_fraction
            @inbounds v_elas[:, :, particle] .= 0.0
            # Fp and J_ref tracking for liquid particles is done in the accepted-step
            # callback (commit_plastic_history_and_heat!) — NOT here. Writing _Fp or
            # J_ref_particle_buf inside the RHS corrupts the finite-difference Jacobian.
            continue
        end

        F = TrixiParticles.deformation_gradient(system, particle)
        Fp_local = StaticArrays.SMatrix{3, 3}(@view _Fp[:, :, particle])
        L_corr = @inbounds TrixiParticles.correction_matrix(system, particle)

        det_F = det(F)
        F_reg = if !isfinite(det_F) || abs(det_F) < 1e-10
            F + 1e-4 * one(StaticArrays.SMatrix{3, 3, eltype(F)})
        else
            F
        end

        d_fp = det(Fp_local)
        if isfinite(d_fp) && abs(d_fp) > 1e-14
            Fp_local = Fp_local / cbrt(d_fp)
        end

        Fp_inv = inv(Fp_local)
        Fe = F_reg * Fp_inv
        J_e = max(det(Fe), 1e-6)
        be = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - 1 / 3 * tr(be_bar) * I
        # Per-particle Lamé constants: Halpin–Tsai transverse E2 as isotropic base.
        mu = E2_particle[particle] / (2 + 2 * poisson_ratio)
        K  = E2_particle[particle] / (3 - 6 * poisson_ratio)

        # Spencer fiber direction (always active — fibers remain solid in molten matrix).
        n_ref_p = StaticArrays.SVector{3}(fiber_direction[1, particle],
                                          fiber_direction[2, particle],
                                          fiber_direction[3, particle])
        Fn_p = F_reg * n_ref_p
        I4_p = dot(Fn_p, Fn_p)
        spencer_tau_p = (2 * k_fiber_particle[particle] * (I4_p - 1.0)) .* (Fn_p * Fn_p')

        # Smooth blend of matrix solid fraction over a 20 K window centred on matrix_tmelt.
        matrix_blend = clamp((matrix_tmelt + matrix_blend_half_K - temp[particle]) /
                             (2 * matrix_blend_half_K), 0.0, 1.0)

        J_ref_p = max(J_ref_particle_buf[particle], 1e-6)
        if !use_volumetric_plasticity && matrix_blend < 1.0 - 1e-6
            J_ref_p = J_e
        end
        bulk_log = matrix_bulk_log_J(F_reg, particle, Fp_local, matrix_blend, J_ref_p)
        tau_solid = K * bulk_log * I + mu * dev_be + spencer_tau_p

        dev_tau = tau_solid - 1 / 3 * tr(tau_solid) * I
        yf = sqrt(1.5) * sqrt(sum(dev_tau .^ 2)) - (ys[particle] + hard[particle])

        if matrix_blend > 1e-6 && yf > 1e-6
            H_alpha_theta = hardening * (1 - H_theta * (temp[particle] - system.temp_ref[1]))
            if temp[particle] < 0.5 * tmelt
                delta_gamma = yf / (3 * mu + H_alpha_theta)
            else
                delta_gamma = yf * dt / max(vis[particle], 1e-8)
            end

            norm_dev = sqrt(sum(dev_tau .* dev_tau))
            if norm_dev > 1e-14 && isfinite(norm_dev)
                n_dir = dev_tau / norm_dev
                c = delta_gamma * sqrt(1.5)
                A = c * n_dir
                A2 = A * A
                sc = abs(c) < 1e-14 ? one(c) : sinh(c) / c
                cc = abs(c) < 1e-14 ? one(c) : (cosh(c) - 1) / (c * c)
                exp_A = one(StaticArrays.SMatrix{3, 3, eltype(n_dir)}) + sc * A + cc * A2
                Fp_local = exp_A * Fp_local
                Fp_local = Fp_local / cbrt(det(Fp_local))

                Fp_inv = inv(Fp_local)
                Fe = F_reg * Fp_inv
                J_e = max(det(Fe), 1e-6)
                be = Fe * Fe'
                be_bar = be / (J_e^(2 / 3))
                dev_be = be_bar - 1 / 3 * tr(be_bar) * I
                bulk_log = matrix_bulk_log_J(F_reg, particle, Fp_local, matrix_blend, J_ref_p)
                tau_solid = K * bulk_log * I + mu * dev_be + spencer_tau_p
            end
        end

        # Thermal Kirchhoff stress: τ_th = −K·3·α_eff·ΔT·I  (volumetric, J2-neutral).
        delta_T_th = temp[particle] - T_stress_free
        tau_solid += (-K * 3.0 * alpha_eff_particle[particle] * delta_T_th *
                      solid_fraction_buf[particle]) * I

        # Blend: above melt the matrix+Spencer contribution fades to zero continuously.
        tau = matrix_blend * tau_solid

        FinvT = inv(F_reg)'
        v_local = (tau * FinvT) * L_corr
        # Stop NaN at the source: any non-finite component (degenerate Fp on
        # particles whose sf became 0 mid-step while still passing the stale
        # skip check) is zeroed before leaving this function.  The
        # post-acceptance "NaN guard" above still resets Fp/F as before; this
        # guard prevents the NaN from poisoning the temperature RHS via the
        # momentum equation during the implicit Newton iterations between
        # accepted steps (the actual cause of the T=2000 K runaway).
        if !all(isfinite, v_local)
            @inbounds v_elas[:, :, particle] .= 0.0
        else
            v_elas[:, :, particle] .= v_local
        end
    end

    return v_elas
end

function update_dual_phase_properties!(ys, hard, vis, temp, alpha,
                                       thermal_softening_particle_buf,
                                       liquid_fraction_particle_buf,
                                       solid_fraction_particle_buf;
                                       vel_grad_buf=nothing)
    gas_constant = 8.31446261815324
    t_prop_start = time_ns()
    @inbounds for i in eachindex(temp)
        temp_i = temp[i]
        if !isfinite(temp_i)
            temp_i = matrix_temp_liq
        end
        temp_i = clamp(temp_i, temp_min_clip, temp_max_clip)

        liquid_fraction = use_nakamura_kinetics ?
                  liquid_fraction_particle_buf[i] :
                  liquid_fraction_particle(temp_i, i)
        solid_fraction = 1.0 - liquid_fraction
        liquid_fraction_particle_buf[i] = liquid_fraction
        solid_fraction_particle_buf[i] = solid_fraction

        h_theta = 1.0 / max(matrix_tmelt - thermal_softening_reference_temp, eps(Float64))
        thermal_softening = clamp(1.0 - h_theta * (temp_i - thermal_softening_reference_temp),
                                  0.0, 1.0)
        thermal_softening_particle_buf[i] = thermal_softening

        melt_softening = clamp((temp_i - 0.8 * matrix_tmelt) /
                       (0.4 * matrix_tmelt + eps(Float64)), 0.0, 1.0)
        solid_factor = 1.0 - melt_softening

        orient_scale = 1.0 + phase_vf_particle[i] * orientation_coupling_gain *
                             (orientation_scalar_particle[i] - 1.0)

        ys[i] = max(1.0e3,
                (yield_base_particle[i] + hardening_base_particle[i] * alpha[i]) *
                thermal_softening)
        ys[i] *= orient_scale
        hard[i] = hardening_base_particle[i] * thermal_softening * orient_scale
    end
    rhs_time_property_update_ns[] += time_ns() - t_prop_start

    t_visc_start = time_ns()
    @inbounds for i in eachindex(temp)
        temp_i = temp[i]
        if !isfinite(temp_i)
            temp_i = matrix_temp_liq
        end
        temp_i = clamp(temp_i, temp_min_clip, temp_max_clip)
        liquid_fraction = liquid_fraction_particle_buf[i]
        melt_softening = clamp((temp_i - 0.8 * matrix_tmelt) /
                       (0.4 * matrix_tmelt + eps(Float64)), 0.0, 1.0)
        solid_factor = 1.0 - melt_softening
        orient_scale = 1.0 + phase_vf_particle[i] * orientation_coupling_gain *
                             (orientation_scalar_particle[i] - 1.0)
        temp_shift = temp_i - cross_wlf_transition_temp
        wlf_denom = cross_wlf_A2 + temp_shift
        if abs(wlf_denom) < 1.0e-8
            wlf_denom = sign(wlf_denom) * 1.0e-8
            if wlf_denom == 0.0
                wlf_denom = 1.0e-8
            end
        end
        eta_zero_shear = matrix_viscosity * exp(-cross_wlf_A1 * temp_shift / wlf_denom)
        if vel_grad_buf === nothing
            vis[i] = eta_zero_shear
        else
            L = @view vel_grad_buf[:, :, i]
            D = 0.5 * (L + L')
            shear_rate = sqrt(max(2.0 * sum(abs2, D), 0.0))
            # Previous Cross denominator retained here for comparison:
            # cross_denominator = 1.0 + (viscosity_cross_time_constant * shear_rate)^
            #                       (1.0 - viscosity_cross_power_law_index)
            cross_argument = eta_zero_shear * shear_rate / cross_wlf_critical_stress
            cross_denominator = 1.0 + cross_argument^
                                  (1.0 - cross_wlf_power_law_index)
            concentration_base = max(1.0 - phase_vf_particle[i] / fiber_max_packing_fraction,
                                     eps(Float64))
            concentration_factor = concentration_base^
                                   (-fiber_intrinsic_viscosity * fiber_max_packing_fraction)
            vis[i] = eta_zero_shear / cross_denominator * concentration_factor
        end
        # vis[i] = viscosity_base_particle[i] * (0.2 + 0.8 * solid_factor) * orient_scale
        vis[i] = clamp(vis[i], viscosity_min_clip, viscosity_max_clip)
    end
    rhs_time_viscosity_update_ns[] += time_ns() - t_visc_start
    return nothing
end

# -----------------------------------------------------------------------------
# Fix B — WARM-UP REGIME LOCK
# While the punch is still ramping up to press speed (t < t_warmup_end) the
# charge has no mechanical driver: freezing it in the pure-viscous branch
# avoids the spurious elastic↔viscous switching that collapses TRBDF2's dt.
# After t_warmup_end the normal local temperature threshold takes over.
# -----------------------------------------------------------------------------
const t_warmup_end = t_ramp_mold

# -----------------------------------------------------------------------------
# Option E — global elastic-stress switch.
# When `enable_elastic_stress = false`, every charge particle uses the viscous
# constitutive model for the entire run. When true, the charge uses the
# elastoplastic branch with paper-style thermal softening for yield stress and
# hardening, i.e. `1 - H_theta * (theta - theta0)` with
# `H_theta = 1 / (T_melt - theta0)`.
# -----------------------------------------------------------------------------
const enable_elastic_stress = true
const solid_mechanics_solid_fraction_start = 0.92
const solid_mechanics_solid_fraction_full  = 0.999
const solid_mechanics_solid_fraction_min   = solid_mechanics_solid_fraction_full
const elastic_skip_solid_fraction          = solid_mechanics_solid_fraction_start
# Disable TLSPH viscous stress path in the mixed/liquid regime; liquid mechanics
# there is handled by WCSPH pressure+viscosity (current-configuration).
const viscous_skip_solid_fraction = 0.0
# Temperature half-window (K) for the elastic-stress matrix_blend in the RHS and the
# J_ref tracking in the accepted-step callback.  Declared here as a module-level const
# so both locations use the same value — previously the RHS had 10.0 K and the callback
# had a hardcoded 2.5 K, causing an inconsistent blend window that left J_ref stale for
# particles in the (tmelt-10 K, tmelt-2.5 K) range.
const matrix_blend_half_K = 10.0   # K  (total blend window = 2 × 10 K = 20 K)

@inline function solid_mechanics_weight(sf)
    x = clamp((sf - solid_mechanics_solid_fraction_start) /
              max(solid_mechanics_solid_fraction_full -
                  solid_mechanics_solid_fraction_start, eps(Float64)),
              0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
end

function update_implicit_stress_cache!(semi_local, v_ode, t)
    cyl_sys   = semi_local.systems[1]
    floor_sys = semi_local.systems[2]
    mold_sys  = semi_local.systems[3]

    # ===== Properties + constitutive stress cache =====
    update_dual_phase_properties!(ys_particle_buf, hard_particle_buf, vis_particle_buf,
                                  cyl_sys.temp, alpha_committed[],
                                  thermal_softening_particle_buf,
                                  liquid_fraction_particle_buf,
                                  solid_fraction_particle_buf;
                                  vel_grad_buf=vel_grad_buf)
    refresh_retraction_solid_phase_buffers!()

    # Fp / alpha / J_ref updates for liquid and solidifying particles happen only in
    # commit_plastic_history_and_heat! (accepted-step callback). No history-variable
    # writes here — the RHS must be a pure function of ODE state + committed buffers.

    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)

    # Unified stress calculation with smooth blending
    fill!(v_stress_viscous_buf, 0)
    fill!(v_stress_elastic_buf, 0)

    # ===== TIMING: Viscous Stress =====
    t_visc_start = time_ns()
    if retraction_solid_mechanics_only()
        fill!(v_stress_viscous_buf, 0.0)
        stress_visc = v_stress_viscous_buf
    else
        stress_visc = viscous_stress_from_cached_grad!(cyl_sys, vis_particle_buf;
                                                       v_vis_buf=v_stress_viscous_buf,
                                                       vel_grad_buf=vel_grad_buf,
                                                       include_bulk_term=enable_elastic_stress,
                                                       solid_fraction_buf=solid_fraction_particle_buf,
                                                       skip_solid_fraction=viscous_skip_solid_fraction)
    end
    rhs_time_viscous_stress_ns[] += time_ns() - t_visc_start

    if !enable_elastic_stress
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] = stress_visc[i, j, particle]
            end
        end
        TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), v_stress_buf)
        return
    end

    # ===== TIMING: Elastic Trial =====
    t_elas_start = time_ns()
    stress_elas = elastic_stress3d_trial_skip_liquid!(
        cyl_sys, ys_particle_buf, hard_particle_buf, vis_particle_buf,
        trial_dt_state[], alpha_committed[], Fp_committed[], semi_local;
        solid_fraction_buf=solid_fraction_particle_buf,
        skip_solid_fraction=elastic_skip_solid_fraction,
        v_elas_buf=v_stress_elastic_buf)
    rhs_time_elastic_trial_ns[] += time_ns() - t_elas_start

    # ===== TIMING: Stress Blending =====
    t_blend_start = time_ns()
    n_bad_elas = 0
    first_bad = -1
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        elas_ok = true
        for j in 1:3, i in 1:3
            if !isfinite(stress_elas[i, j, particle])
                elas_ok = false
                break
            end
        end

        if elas_ok
            mech_w = retraction_solid_mechanics_only() ? 1.0 :
                     solid_mechanics_weight(solid_fraction_particle_buf[particle])
            if mech_w <= 1.0e-10
                # Mushy/latent/liquid: mechanics is liquid-only (WCSPH branch).
                # Keep TLSPH constitutive stress fully off to avoid mixed-state
                # history coupling (Fp/J_ref/plasticity) that collapses TRBDF2 dt.
                for j in 1:3, i in 1:3
                    v_stress_buf[i, j, particle] = 0.0
                end
            elseif mech_w >= 1.0 - 1.0e-10
                # Fully solid: use solid constitutive stress.
                for j in 1:3, i in 1:3
                    v_stress_buf[i, j, particle] = stress_elas[i, j, particle]
                end
            else
                # Smooth handover in the semi-solid window to avoid stiffness
                # spikes when particles first re-enter solid mechanics.
                for j in 1:3, i in 1:3
                    v_stress_buf[i, j, particle] = mech_w * stress_elas[i, j, particle]
                end
            end
            # # Pressure floor: only for fully softened (viscous) particles — molten material
            # # cannot sustain net tension. Skip for elastic particles (thermal_blend > 0.1)
            # # to avoid disturbing the elastic restoring pressure during Newton iterations.
            # p_mean = (v_stress_buf[1, 1, particle] + v_stress_buf[2, 2, particle] +
            #           v_stress_buf[3, 3, particle]) / 3
            # if p_mean > 0.0 && thermal_softening_particle_buf[particle] < 0.1
            #     v_stress_buf[1, 1, particle] -= p_mean
            #     v_stress_buf[2, 2, particle] -= p_mean
            #     v_stress_buf[3, 3, particle] -= p_mean
            # end
        else
            n_bad_elas += 1
            if first_bad < 0
                first_bad = particle
            end
            # Zero stress only — do NOT reset Fp/F/coordinates here. State repair is
            # deferred to the accepted-step callback to keep the RHS Jacobian-consistent.
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] = 0.0
            end
        end
    end
    elas_ramp = retraction_elastic_scale(t)
    if retraction_started[] && !retraction_solid_mechanics_only() &&
       elas_ramp < 1.0 - 1.0e-8
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            for j in 1:3, i in 1:3
                tau_v = v_stress_viscous_buf[i, j, particle]
                tau_e = v_stress_buf[i, j, particle] - tau_v
                v_stress_buf[i, j, particle] = tau_v + elas_ramp * tau_e
            end
        end
    elseif retraction_started[] && retraction_solid_mechanics_only() &&
           elas_ramp < 1.0 - 1.0e-8
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            for j in 1:3, i in 1:3
                v_stress_buf[i, j, particle] *= elas_ramp
            end
        end
    end
    rhs_time_stress_blend_ns[] += time_ns() - t_blend_start

    # if n_bad_elas > 0
    #     println(">>> NaN guard: ", n_bad_elas, " particles had non-finite elastic stress at t=",
    #             round(t, digits=8), " (first idx=", first_bad,
    #             ", thermal_softening=",
    #             round(thermal_softening_particle_buf[first_bad], digits=4),
    #             ", temp=", round(cyl_sys.temp[first_bad], digits=2), " K)")
    #     flush(stdout)
    # end

    TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), v_stress_buf)
end

# CFRP-style stress cache (stamping_cfrp_3d_2_implicit.jl): update_properties! + elastic trial.
function update_implicit_stress_cache_cfrp!(semi_local, v_ode, t)
    cyl_sys = semi_local.systems[1]
    ys, hard, vis = TrixiParticles.update_properties!(cyl_sys, alpha_committed[], semi_local,
                                                      material_polymer.viscosity)
    fill!(v_stress_buf, 0)
    stress_elas = TrixiParticles.elastic_stress3d_trial!(
        cyl_sys, ys, hard, vis, trial_dt_state[], alpha_committed[], Fp_committed[], semi_local;
        v_elas_buf=v_stress_buf)
    TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), stress_elas)
    return nothing
end

function commit_plastic_history_cfrp!(cyl_sys, semi_local, dt, v_ode)
    n_cool = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        n_cool += cyl_sys.temp[particle] <= local_regime_threshold
    end
    n_cool == 0 && return nothing

    ys_c, hard_c, vis_c = TrixiParticles.update_properties!(
        cyl_sys, alpha_committed[], semi_local, material_polymer.viscosity)
    trial_dt_state[] = dt
    TrixiParticles.elastic_stress3d_fast!(
        cyl_sys, ys_c, hard_c, vis_c, dt,
        alpha_committed[], Fp_committed[], semi_local;
        v_elas_buf=v_stress_buf)

    v_wrap = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        v_wrap[NDIMS_CYL + 1, particle] = cyl_sys.temp[particle]
    end
    return nothing
end

function kick_implicit_cfrp_retraction!(dv_ode, v_ode, u_ode, semi_local, t)
    hold_tlsph_equilibration_now[] = false
    cyl_sys = semi_local.systems[1]
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)

    if use_pseudo2d_projection
        TrixiParticles.foreach_system(semi_local) do system
            v_sys = TrixiParticles.wrap_v(v_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                v_sys[2, particle] = 0.0
            end
        end
    end

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        cyl_sys.temp[particle] = v_cyl[NDIMS_CYL + 1, particle]
    end

    try
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        if t != nhs_updated_at_t[]
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end

        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(cyl_sys)

        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        apply_charge_plane_strain_F_fix!(cyl_sys)

        TrixiParticles.update_implicit_sph!(semi_local, v_ode, u_ode, t)

        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_pressure!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_boundary_interpolation!(system, v, u, v_ode, u_ode,
                                                          semi_local, t)
        end

        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_final!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        update_implicit_stress_cache_cfrp!(semi_local, v_ode, t)
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi_local, t)
    finally
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = nothing
        TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
    end

    floor_sys = semi_local.systems[2]
    mold_sys  = semi_local.systems[3]
    contact_heat_flux = compute_contact_heat_flux!(contact_heat_flux_buf, cyl_sys,
                                                   floor_sys, mold_sys, particle_spacing)
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    dx = particle_spacing
    rho_cyl = cyl_sys.material_density[1]
    cp_cyl = cyl_sys.cp
    TrixiParticles.thermal_rhs_sph3d!(cyl_sys, dv_cyl, v_cyl, 0.0,
                                      particle_spacing, bound_coordinate_thermal, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        if contact_heat_flux[particle] > 0.0
            dv_cyl[NDIMS_CYL + 1, particle] -= contact_heat_flux[particle] /
                                                (rho_cyl * cp_cyl * dx)
        end
    end

    if use_pseudo2d_projection
        TrixiParticles.foreach_system(semi_local) do system
            dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                dv_sys[2, particle] = 0.0
            end
        end
    end

    apply_retraction_charge_accel_caps!(dv_cyl, cyl_sys, t)

    if retraction_diag_enabled && retraction_started[]
        retraction_diag_kick_max_dv[] = max_charge_kick_dv(dv_ode, cyl_sys, semi_local)
    end

    return dv_ode
end

function commit_plastic_history_and_heat!(system, ys, hard, vis, dt, _alpha, _Fp;
                                          skip_heat::Bool=false, hold_creep::Bool=false)
    (; deformation_grad, young_modulus, poisson_ratio, temp, tmelt, hardening,
       material_density, cp, temp_ref) = system

    mu = young_modulus / (2 + 2 * poisson_ratio)
    K  = young_modulus / (3 - 6 * poisson_ratio)
    H_theta = 1.0 / (tmelt - temp_ref[1])

    @inbounds for particle in TrixiParticles.eachparticle(system)
        # Mirror the early-skip used in elastic_stress3d_trial_skip_liquid!:
        # liquid particles (sf ≤ 0) contribute no elastic stress to the physics
        # RHS, so they must also produce no plastic dissipation here.
        # CRUCIALLY: while a particle is liquid, Fp must track F so that when it
        # re-solidifies Fe = F*Fp_inv ≈ I (stress-free). Without this, Fp stays
        # at its last solid value while F keeps compressing → J_e→0.675 → K*log(J_e)
        # produces GPa-scale elastic stress the instant the particle re-solidifies,
        # collapsing Δt to femtoseconds.
        sf_particle = solid_fraction_particle_buf[particle]
        if sf_particle < solid_mechanics_solid_fraction_start
            # Reset Fp ← F so elastic strain is zeroed for liquid particles.
            F_liq = TrixiParticles.deformation_gradient(system, particle)
            det_F_liq = det(F_liq)
            if isfinite(det_F_liq) && abs(det_F_liq) > 1e-10
                F_liq_iso = F_liq / cbrt(det_F_liq)  # volume-preserving part
                for jj in 1:3, ii in 1:3
                    _Fp[ii, jj, particle] = F_liq_iso[ii, jj]
                end
                use_volumetric_plasticity &&
                    (J_p_particle_buf[particle] = max(det_F_liq, 1e-6))
            end
            continue
        end

        F = TrixiParticles.deformation_gradient(system, particle)
        Fp_sm = StaticArrays.SMatrix{3,3}(@view _Fp[:, :, particle])

        det_F = det(F)
        F_reg = if !isfinite(det_F) || abs(det_F) < 1e-10
            F + 1e-4 * one(StaticArrays.SMatrix{3,3,eltype(F)})
        else
            F
        end

        d_fp = det(Fp_sm)
        if isfinite(d_fp) && abs(d_fp) > 1e-14
            Fp_sm = Fp_sm / cbrt(d_fp)
        end

        Fp_inv = inv(Fp_sm)
        Fe = F_reg * Fp_inv
        J_e = max(det(Fe), 1e-6)
        J_e_vol = elastic_volumetric_J(F_reg, particle, Fp_sm)
        # Skip particles with implausibly large elastic volumetric strain.
        J_e_check = use_volumetric_plasticity ? J_e_vol : J_e
        if J_e_check < 0.7 || J_e_check > 2.0
            det_Freg = det(F_reg)
            if isfinite(det_Freg) && abs(det_Freg) > 1e-10
                F_iso = F_reg / cbrt(det_Freg)
                for jj in 1:3, ii in 1:3
                    _Fp[ii, jj, particle] = F_iso[ii, jj]
                end
                use_volumetric_plasticity &&
                    (J_p_particle_buf[particle] = max(det_Freg, 1e-6))
            end
            continue
        end
        be = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - 1 / 3 * tr(be_bar) * I
        # Per-particle Lamé constants: Halpin–Tsai transverse E2 as isotropic base.
        # (Matches elastic_stress3d_trial_skip_liquid!; the sf protection comes
        # from the early-skip above, not from scaling.)
        mu = E2_particle[particle] / (2 + 2 * poisson_ratio)
        K  = E2_particle[particle] / (3 - 6 * poisson_ratio)
        # While semi-solid (commit_blend_vol < 1), WCSPH carries volumetric load:
        # track J_ref = J_e so elastic bulk pressure stays zero until fully solid.
        # Uses matrix_blend_half_K (module-level const, same as the RHS blend window)
        # so the two blend windows are consistent.
        commit_blend_vol = clamp((matrix_tmelt + matrix_blend_half_K - temp[particle]) /
                                 (2 * matrix_blend_half_K), 0.0, 1.0)
        matrix_blend = clamp((matrix_tmelt + matrix_blend_half_K - temp[particle]) /
                               (2 * matrix_blend_half_K), 0.0, 1.0)
        J_ref_p = max(J_ref_particle_buf[particle], 1e-6)
        if use_volumetric_plasticity
            commit_blend_vol < 1.0 - 1e-6 &&
                (J_p_particle_buf[particle] = particle_J_total(F_reg))
        elseif commit_blend_vol < 1.0 - 1e-6
            J_ref_p = J_e
            J_ref_particle_buf[particle] = J_e
        end
        bulk_log = matrix_bulk_log_J(F_reg, particle, Fp_sm, matrix_blend, J_ref_p)
        tau = K * bulk_log * I + mu * dev_be

        # Spencer (1984) transversely isotropic fiber reinforcement.
        n_ref_p = StaticArrays.SVector{3}(fiber_direction[1, particle],
                                          fiber_direction[2, particle],
                                          fiber_direction[3, particle])
        Fn_p = F_reg * n_ref_p
        I4_p = dot(Fn_p, Fn_p)
        spencer_tau_p = (2 * k_fiber_particle[particle] * (I4_p - 1.0)) .* (Fn_p * Fn_p')
        tau += spencer_tau_p

        # Guard: skip return mapping and heating if stress is non-finite
        # (e.g. Spencer term overflow from near-singular F at corners).
        if !all(isfinite, tau)
            continue
        end

        dev_tau = tau - 1 / 3 * tr(tau) * I
        yf = sqrt(1.5) * sqrt(sum(dev_tau .^ 2)) - (ys[particle] + hard[particle])

        # Gate plasticity on matrix temperature with a 5 K smooth blend (same window as
        # elastic_stress3d_trial_skip_liquid!) to keep ∂RHS/∂T continuous across the
        # melt transition.  A hard cutoff at matrix_tmelt creates a C⁰ kink that the
        # TRBDF2 local-error estimator sees as a large f″ jump, collapsing Δt.
        #   blend = 1 → fully solid  (T ≤ matrix_tmelt - 2.5 K)
        #   blend = 0 → fully molten (T ≥ matrix_tmelt + 2.5 K) → yf set negative
        commit_blend = clamp((matrix_tmelt + 2.5 - temp[particle]) / 5.0, 0.0, 1.0)
        if commit_blend < 1e-6
            yf = -1.0  # fully molten: no plastic work injection
        end

        vol_relax = hold_creep || yf > 1e-6
        commit_volumetric_plastic_relax!(F_reg, particle;
            active=vol_relax, commit_blend=commit_blend, matrix_blend=matrix_blend)

        if yf > 1e-6
            H_alpha_theta = hardening * (1 - H_theta * (temp[particle] - temp_ref[1]))
            if temp[particle] < 0.5 * tmelt
                delta_gamma = yf / (3 * mu + H_alpha_theta)
            else
                delta_gamma = yf * dt / max(vis[particle], 1e-8)
            end
            # Cap delta_gamma: c = delta_gamma*sqrt(1.5) feeds into sinh(c)/c in
            # the exponential map. For c > 20, sinh overflows to Inf → Fp = NaN.
            delta_gamma = min(delta_gamma, 12.0)

            norm_dev = sqrt(sum(dev_tau .* dev_tau))
            if norm_dev > 1e-14 && isfinite(norm_dev)
                n_dir = dev_tau / norm_dev
                c = delta_gamma * sqrt(1.5)
                A = c * n_dir
                A2 = A * A
                sc = abs(c) < 1e-14 ? one(c) : sinh(c) / c
                cc = abs(c) < 1e-14 ? one(c) : (cosh(c) - 1) / (c * c)
                exp_A = one(StaticArrays.SMatrix{3,3,eltype(n_dir)}) + sc * A + cc * A2
                Fp_old = Fp_sm
                Fp_old_inv = inv(Fp_old)
                Fp_sm = exp_A * Fp_sm
                Fp_sm = Fp_sm / cbrt(det(Fp_sm))

                Fp_inv = inv(Fp_sm)
                Fe = F_reg * Fp_inv
                J_e = max(det(Fe), 1e-6)
                be = Fe * Fe'
                be_bar = be / (J_e^(2 / 3))
                dev_be = be_bar - 1 / 3 * tr(be_bar) * I
                J_ref_p2 = max(J_ref_particle_buf[particle], 1e-6)
                bulk_log = matrix_bulk_log_J(F_reg, particle, Fp_sm, matrix_blend, J_ref_p2)
                tau = K * bulk_log * I + mu * dev_be
                tau += spencer_tau_p  # restore fiber reinforcement after return mapping
                commit_volumetric_plastic_relax!(F_reg, particle;
                    active=true, commit_blend=commit_blend, matrix_blend=matrix_blend)
                _alpha[particle] += delta_gamma

                dFp_total = Fp_sm - Fp_old

                depsilon = 0.5 * (dFp_total * Fp_old_inv + (dFp_total * Fp_old_inv)')
                # Use only the neo-Hookean (matrix) part of tau for plastic heating.
                # The Spencer fiber term (spencer_tau_p) is a hyperelastic fiber reinforcement —
                # it does no dissipative work on the matrix during plastic flow and must be
                # excluded here.  Including it causes O(GPa) spurious heat injection for
                # particles where I4 >> 1 (fibres stretched far from reference), leading to
                # T → 2000 K spikes when a particle first enters the semi-solid regime.
                tau_matrix = tau - spencer_tau_p
                plastic_work = sum(tau_matrix .* depsilon)
                # Taylor-Quinney coefficient β = 0.9: 90% of plastic work converts to heat.
                # Guard against NaN/Inf from ill-conditioned Fp (e.g. near-singular F).
                # Also blend by commit_blend so that particles near the melt boundary inject
                # proportionally less heat (avoids 1000+ K spikes at T ≈ matrix_tmelt).
                if !skip_heat && isfinite(plastic_work) && plastic_work > 0.0
                    dT_inject = commit_blend * solid_mechanics_weight(sf_particle) * 0.9 * plastic_work /
                                (material_density[particle] * cp)
                    # if dT_inject > diag_heat_dT_K && diag_heat_remaining[] > 0
                    #     diag_heat_remaining[] -= 1
                    #     sf_p = solid_fraction_particle_buf[particle]
                    #     println(stderr,
                    #         ">>> HEAT-INJECT  p=", particle,
                    #         "  T=", round(temp[particle], digits=2),
                    #         "  sf=", round(sf_p, digits=4),
                    #         "  J_e=", round(J_e, digits=4),
                    #         "  I4=", round(I4_p, digits=4),
                    #         "  |dev_tau|=", round(norm_dev, sigdigits=3),
                    #         "  yf=", round(yf, sigdigits=3),
                    #         "  dg=", round(delta_gamma, sigdigits=3),
                    #         "  pw=", round(plastic_work, sigdigits=3),
                    #         "  dT=", round(dT_inject, digits=1), " K")
                    # end
                    temp[particle] += dT_inject
                end
            end
        end

        for j in 1:3, i in 1:3
            _Fp[i, j, particle] = Fp_sm[i, j]
        end
    end

    return nothing
end

# ==========================================================================================
# Smooth WCSPH EOS: C¹ at J = 1 to prevent TRBDF2 dt collapse at first mold contact.
#
# Root cause: the standard clamped-log EOS  p = −K·log(max(J,ε)),  J = clamp(ρ₀/ρ,0,1)
# has a discontinuous first derivative (kink) at J = 1.  When the punch first compresses
# a particle, ρ crosses ρ₀ for the first time; the TRBDF2 local-error estimator sees a
# large change in f″ at that kink and collapses Δt to femtoseconds.  The sim never
# recovers because each new contact particle triggers the same kink.
#
# Fix: quadratic onset in J ∈ [1−blend, 1] so dp/dJ → 0 as J → 1 (C¹ smooth).
# For J < 1−blend the log regime is used (shifted for value continuity at J = 1−blend).
# Physics is unchanged for compressions > 2%; pressure onset is simply gentler.
# ==========================================================================================
const wcsph_J_blend = 0.02   # 2% density range for smooth quadratic onset
const wcsph_latent_force_min = 0.30  # pressure-force scale at peak latent stiffness
# PEEK melt bulk modulus. Physical value ~5 GPa (Zoller & Walsh PVT data, hot melt);
# numerically reduced so WCSPH pressure waves do not collapse TRBDF2 dt during contact.
const matrix_K_liq = 2e7
const wcsph_hot_soften_span = 35.0    # K above melt window to relax liquid bulk stiffness smoothly
const wcsph_pressure_cap = 1e8        # Pa: 100 MPa, upper bound for compression-moulding pressures
const wcsph_visc_pair_accel_cap = 5.0e4 # m/s², pairwise cap before global liquid accel limiter
const enable_wcsph_liquid_viscosity = false
const lag_wcsph_liquid_viscosity = true
@inline function wcsph_pressure_smooth(J_raw, K)
    J_raw >= 1.0 && return 0.0                         # no tension
    Δ = 1.0 - J_raw
    if Δ <= wcsph_J_blend
        return K * Δ * Δ / (2 * wcsph_J_blend)         # quadratic: dp/dJ = 0 at J = 1
    else
        # log regime shifted to match quadratic value at J = 1 − blend
        return K * (wcsph_J_blend / 2 + log((1 - wcsph_J_blend) / max(J_raw, 1e-6)))
    end
end

@inline function wcsph_liquid_bulk_modulus(temp_i)
    # Keep the melt-window response close to the physical matrix bulk modulus,
    # then soften the hotter fully-liquid regime smoothly toward a lower melt
    # bulk modulus.  This is a temperature-dependent liquid compressibility model,
    # not a global stiffness reduction.
    temp_i <= matrix_tmelt + 2.5 && return matrix_K_melt
    s = clamp((temp_i - (matrix_tmelt + 2.5)) / wcsph_hot_soften_span, 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)   # C¹ smoothstep
    return matrix_K_melt + (matrix_K_liq - matrix_K_melt) * s
end

@inline function wcsph_latent_window_weight(temp_i)
    # Smooth bump in the 5 K melt window centered at matrix_tmelt:
    # b=0/1 (outside window) -> weight=0, b=0.5 (center) -> weight=1.
    # sinpi(b)^2 has zero slope at both edges, avoiding a new RHS kink.
    b = clamp((matrix_tmelt + 2.5 - temp_i) / 5.0, 0.0, 1.0)
    return sinpi(b)^2
end

# ==========================================================================================
# WCSPH LIQUID PRESSURE FORCE (current-position kernel)
#
# The TLSPH formulation evaluates kernel gradients at INITIAL (reference) positions.
# For liquid particles this means the pressure gradient ∇p is discretised in reference
# space, which becomes inaccurate for large deformations and gives near-zero lateral
# force under uniform mold compression (symmetric reference neighbourhood → ∇W₀ = 0).
#
# Fix: compute p_i via the smooth C¹ EOS for each liquid particle, then evaluate the SPH
# pressure gradient at CURRENT positions — identical to standard WCSPH (Monaghan 1994).
# Only liquid-fraction-weighted contributions are included; the solid part is already
# handled by the TLSPH elastic + reference-space bulk term.
# ==========================================================================================
function update_wcsph_current_nhs!(nhs, coords)
    PointNeighbors.initialize!(nhs, coords, coords)
    return nhs
end

function compute_wcsph_current_density!(ρ_out, cyl_sys, coords, nhs)
    fill!(ρ_out, 0.0)
    h = smoothing_length
    kern = smoothing_kernel
    h2_eps = eps(h * h)

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        ρ_out[particle] += cyl_sys.mass[particle] * TrixiParticles.kernel(kern, 0.0, h)
    end

    PointNeighbors.foreach_point_neighbor(
        coords, coords, nhs;
        parallelization_backend=TrixiParticles.PolyesterBackend(),
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, pos_diff, distance
        distance^2 < h2_eps && return
        w = TrixiParticles.kernel(kern, distance, h)
        @inbounds ρ_out[particle] += cyl_sys.mass[neighbor] * w
    end

    return ρ_out
end

function apply_wcsph_liquid_pressure!(dv_ode, cyl_sys, semi_local)
    n_p = nparticles(cyl_sys)
    dv = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    kern = smoothing_kernel
    h = smoothing_length
    h2_eps = eps(h * h)

    if !wcsph_liquid_density_ref_initialized[]
        ref_coords = TrixiParticles.initial_coordinates(cyl_sys)
        update_wcsph_current_nhs!(wcsph_current_nhs, ref_coords)
        compute_wcsph_current_density!(wcsph_liquid_density_ref_buf, cyl_sys,
                                       ref_coords, wcsph_current_nhs)
        wcsph_liquid_density_ref_initialized[] = true
    end

    # Pass 1: current-space SPH density sum (grid NHS at current coordinates).
    current_coords = cyl_sys.current_coordinates
    update_wcsph_current_nhs!(wcsph_current_nhs, current_coords)
    ρ_curr = wcsph_liquid_density_curr_buf
    compute_wcsph_current_density!(ρ_curr, cyl_sys, current_coords, wcsph_current_nhs)

    # Pass 2: WC pressure from J = ρ_ref/ρ_curr (liquid-fraction weighted).
    p_wc = wcsph_pressure_buf
    fill!(p_wc, 0.0)
    @inbounds for i in 1:n_p
        lf = liquid_fraction_particle_buf[i]
        lf < 1e-12 && continue
        ρ_ref_i = max(wcsph_liquid_density_ref_buf[i], 1e-12 * cyl_sys.material_density[i])
        ρc = max(ρ_curr[i], 1e-12 * ρ_ref_i)
        J = ρ_ref_i / ρc
        K_liq = wcsph_liquid_bulk_modulus(cyl_sys.temp[i])
        p_raw = lf * wcsph_pressure_smooth(J, K_liq)
        p_wc[i] = min(p_raw, wcsph_pressure_cap * lf)
    end

    # Pass 3: WCSPH symmetric pressure force at current positions (grid NHS).
    PointNeighbors.foreach_point_neighbor(
        current_coords, current_coords, wcsph_current_nhs;
        parallelization_backend=TrixiParticles.PolyesterBackend(),
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, pos_diff, distance
        distance^2 < h2_eps && return
        pi = p_wc[particle]
        pj = p_wc[neighbor]
        (pi <= 0.0 && pj <= 0.0) && return

        grad_W = TrixiParticles.kernel_grad(kern, pos_diff, distance, h)
        ρ0i = max(wcsph_liquid_density_ref_buf[particle],
                  1e-12 * cyl_sys.material_density[particle])
        ρ0j = max(wcsph_liquid_density_ref_buf[neighbor],
                  1e-12 * cyl_sys.material_density[neighbor])
        mj = cyl_sys.mass[neighbor]

        coeff = pi / (ρ0i * ρ0i) + pj / (ρ0j * ρ0j)
        w_i = wcsph_latent_window_weight(cyl_sys.temp[particle])
        w_j = wcsph_latent_window_weight(cyl_sys.temp[neighbor])
        latent_scale = 1.0 - (1.0 - wcsph_latent_force_min) * max(w_i, w_j)
        coeff *= latent_scale

        @inbounds begin
            dv[1, particle] -= mj * coeff * grad_W[1]
            dv[3, particle] -= mj * coeff * grad_W[3]
        end
    end

    scale = wcsph_ramp_scale[]
    if scale < 1.0 - 1.0e-12
        @inbounds for i in 1:n_p
            dv[1, i] *= scale
            dv[3, i] *= scale
        end
    end

    return ρ_curr
end

# ─── Eulerian (current-config) viscous force for liquid particles ──────────────
# For T > matrix_tmelt, particles have rearranged far from their reference positions
# and the TLSPH reference-config framework (F, FinvT, L_corr) becomes ill-conditioned.
# This function computes the viscous force entirely in the current configuration:
#
#   Step A: velocity gradient L_i = (1/ρ_i) Σ_j m_j (v_j − v_i) ⊗ ∇W_ij   (current-config)
#   Step B: τ_vis_i = 2η_i dev(sym(L_i)),  capped at matrix_yield_stress
#   Step C: symmetric SPH divergence force:
#             dv_α^i += Σ_j m_j (τ_αβ^i/ρ_i² + τ_αβ^j/ρ_j²) ∇W_ij[β]
#
# Only applied to particles with lf > liquid_eps (purely Eulerian, no F dependence).
# Particle pairs with both lf ≈ 0 contribute nothing (both τ_vis = 0).
function apply_wcsph_liquid_viscosity!(dv_ode, v_ode, cyl_sys, semi_local, ρ_curr)
    n_p = nparticles(cyl_sys)
    dv = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    v = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    kern = smoothing_kernel
    h = smoothing_length
    h2_eps = eps(h * h)
    current_coords = cyl_sys.current_coordinates
    update_wcsph_current_nhs!(wcsph_current_nhs, current_coords)

    # Step A: current-config velocity gradient L_curr (grid NHS).
    L_curr = wcsph_L_curr_buf
    fill!(L_curr, 0.0)

    PointNeighbors.foreach_point_neighbor(
        current_coords, current_coords, wcsph_current_nhs;
        parallelization_backend=TrixiParticles.PolyesterBackend(),
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, pos_diff, distance
        distance^2 < h2_eps && return
        lf_i = liquid_fraction_particle_buf[particle]
        lf_i < 1e-12 && return

        grad_W = TrixiParticles.kernel_grad(kern, pos_diff, distance, h)
        ρ_i = max(ρ_curr[particle], 1e-12)
        mj = cyl_sys.mass[neighbor]
        fac = mj / ρ_i

        dvx = v[1, neighbor] - v[1, particle]
        dvz = v[3, neighbor] - v[3, particle]

        @inbounds begin
            L_curr[1, 1, particle] += fac * dvx * grad_W[1]
            L_curr[1, 2, particle] += fac * dvx * grad_W[3]
            L_curr[2, 1, particle] += fac * dvz * grad_W[1]
            L_curr[2, 2, particle] += fac * dvz * grad_W[3]
        end
    end

    # Step B: τ_vis_i = 2η_i dev(sym(L_curr_i)) for liquid particles, with yield cap.
    tau_vis = wcsph_tau_vis_buf
    fill!(tau_vis, 0.0)
    @inbounds for i in 1:n_p
        lf_i = liquid_fraction_particle_buf[i]
        lf_i < 1e-12 && continue
        η = vis_particle_buf[i] * lf_i

        L11 = L_curr[1, 1, i];  L13 = L_curr[1, 2, i]
        L31 = L_curr[2, 1, i];  L33 = L_curr[2, 2, i]

        div_v = L11 + L33
        sym13 = 0.5 * (L13 + L31)

        dev11 = L11 - div_v / 3.0
        dev33 = L33 - div_v / 3.0

        τ11 = 2 * η * dev11
        τ13 = 2 * η * sym13
        τ33 = 2 * η * dev33

        τ_norm = sqrt(τ11 * τ11 + 2 * τ13 * τ13 + τ33 * τ33)
        if τ_norm > matrix_yield_stress
            scale = matrix_yield_stress / τ_norm
            τ11 *= scale
            τ13 *= scale
            τ33 *= scale
        end

        tau_vis[1, 1, i] = τ11
        tau_vis[1, 2, i] = τ13
        tau_vis[2, 1, i] = τ13
        tau_vis[2, 2, i] = τ33
    end

    # Step C: symmetric SPH divergence force (grid NHS; accumulate on particle only).
    PointNeighbors.foreach_point_neighbor(
        current_coords, current_coords, wcsph_current_nhs;
        parallelization_backend=TrixiParticles.PolyesterBackend(),
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, pos_diff, distance
        distance^2 < h2_eps && return
        lf_i = liquid_fraction_particle_buf[particle]
        lf_j = liquid_fraction_particle_buf[neighbor]
        (lf_i < 1e-12 && lf_j < 1e-12) && return

        grad_W = TrixiParticles.kernel_grad(kern, pos_diff, distance, h)
        ρ_i = max(ρ_curr[particle], 1e-12)
        ρ_j = max(ρ_curr[neighbor], 1e-12)
        ρi2 = ρ_i * ρ_i
        ρj2 = ρ_j * ρ_j
        mj = cyl_sys.mass[neighbor]

        cx = (tau_vis[1, 1, particle] / ρi2 + tau_vis[1, 1, neighbor] / ρj2) * grad_W[1] +
             (tau_vis[1, 2, particle] / ρi2 + tau_vis[1, 2, neighbor] / ρj2) * grad_W[3]
        cz = (tau_vis[2, 1, particle] / ρi2 + tau_vis[2, 1, neighbor] / ρj2) * grad_W[1] +
             (tau_vis[2, 2, particle] / ρi2 + tau_vis[2, 2, neighbor] / ρj2) * grad_W[3]

        ai_x = mj * cx
        ai_z = mj * cz
        ai_norm = sqrt(ai_x * ai_x + ai_z * ai_z)
        if ai_norm > wcsph_visc_pair_accel_cap
            ai_scale = wcsph_visc_pair_accel_cap / ai_norm
            ai_x *= ai_scale
            ai_z *= ai_scale
        end

        @inbounds begin
            dv[1, particle] += ai_x
            dv[3, particle] += ai_z
        end
    end

    scale = wcsph_ramp_scale[]
    if scale < 1.0 - 1.0e-12
        @inbounds for i in 1:n_p
            dv[1, i] *= scale
            dv[3, i] *= scale
        end
    end

    return nothing
end

function refresh_lagged_wcsph_liquid_viscosity!(cache, scratch_dv_ode, v_ode,
                                                cyl_sys, semi_local)
    fill!(cache, 0.0)
    fill!(scratch_dv_ode, 0.0)

    # Reuse the pressure routine only for its current-configuration density estimate.
    # Clear scratch afterwards so the cached term contains viscosity only.
    ρ_curr = apply_wcsph_liquid_pressure!(scratch_dv_ode, cyl_sys, semi_local)
    fill!(scratch_dv_ode, 0.0)
    apply_wcsph_liquid_viscosity!(scratch_dv_ode, v_ode, cyl_sys, semi_local, ρ_curr)

    dv_tmp = TrixiParticles.wrap_v(scratch_dv_ode, cyl_sys, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        cache[1, particle] = dv_tmp[1, particle]
        cache[2, particle] = dv_tmp[2, particle]
        cache[3, particle] = dv_tmp[3, particle]
    end

    return cache
end

function kick_implicit_visible!(dv_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.set_zero!(dv_ode)
    rhs_eval_count[] += 1
    hold_tlsph_equilibration_now[] = hold_tlsph_tail_active(t)

    if retraction_cfrp_style_active()
        return kick_implicit_cfrp_retraction!(dv_ode, v_ode, u_ode, semi_local, t)
    end

    cyl_sys = semi_local.systems[1]
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)

    # Pseudo-2D only: hard-zero v_y before force evaluation.
    # Thin-3D uses a spring-damper in compute_implicit_rhs! instead so that
    # AutoFiniteDiff sees ∂(dv_y)/∂v_y = -y_constraint_damp ≠ 0  (non-singular Jacobian).
    if use_pseudo2d_projection
        TrixiParticles.foreach_system(semi_local) do system
            v_sys = TrixiParticles.wrap_v(v_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                v_sys[2, particle] = 0.0
            end
        end
    end

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        if apply_velocity_safety_clamps()
            v_cyl[1, particle] = soft_limit(v_cyl[1, particle], max_cylinder_speed)
            v_cyl[3, particle] = soft_limit(v_cyl[3, particle], max_cylinder_speed)
        end
        t_trial = v_cyl[NDIMS_CYL + 1, particle]
        if !isfinite(t_trial)
            t_trial = cyl_sys.temp[particle]
        end
        cyl_sys.temp[particle] = clamp(t_trial, temp_min_clip, temp_max_clip)
    end

    if charge_hold_thermal_only(t)
        return apply_charge_hold_rhs!(dv_ode, v_ode, u_ode, semi_local, t)
    end

    floor_sys = semi_local.systems[2]
    mold_sys  = semi_local.systems[3]

    # force_stage_diag = t > diag_force_stage_start_t &&
    #                    t > diag_force_stage_last_t[] + diag_force_stage_dt
    force_stage_diag = false
    force_stage_prev = nothing
    # if force_stage_diag
    #     diag_force_stage_last_t[] = t
    #     println(stderr, "\n>>> FORCE-STAGE-DIAG t=", round(t; digits=9),
    #             " trial_dt=", trial_dt_state[],
    #             " n_particles=", nparticles(cyl_sys))
    # end

    try
        t_pos_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        # Safety clamp for charge coordinates during nonlinear iterations.
        if apply_coordinate_safety_clamps()
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                cyl_sys.current_coordinates[1, particle] = clamp(cyl_sys.current_coordinates[1, particle],
                                                                 -x_safety_bound, x_safety_bound)
                cyl_sys.current_coordinates[3, particle] = clamp(cyl_sys.current_coordinates[3, particle],
                                                                 z_safety_min, z_safety_max)
                if use_pseudo2d_projection
                    cyl_sys.current_coordinates[2, particle] = 0.0
                end
            end
        end

        rhs_time_pos_ns[] += time_ns() - t_pos_start

        hold_rel = hold_release_factor(t)
        mech_dv_scale = retraction_dv_mech_scale(t)
        floor_act = floor_reimann_activation_factor(t, cyl_sys)
        mold_act = mold_reimann_activation_factor(cyl_sys, mold_sys, t)
        if retraction_dv_mech_ramping(t)
            snapshot_charge_dv!(charge_dv_mech_marker_buf, dv_ode, cyl_sys, semi_local)
        end
        if retraction_started[] && !retraction_elastic_ramp_announced[]
            retraction_elastic_ramp_announced[] = true
            println(">>> Retraction ramp at t=", round(t; digits=6),
                    " s (contact/hold-release ", retraction_hold_release_s,
                    " s; elastic σ ramp ", retraction_elastic_ramp_s,
                    " s; kinematics ramp ", retraction_kinematics_release_s,
                    " s; solid TLSPH only)")
            flush(stdout)
        end
        wcsph_ramp_scale[] = wcsph_mechanics_active() ?
                             charge_wcsph_ramp_scale_value(cyl_sys) : 0.0

        t_nhs_start = time_ns()
        if t != nhs_updated_at_t[]
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end
        rhs_time_nhs_ns[] += time_ns() - t_nhs_start

        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = objectid(cyl_sys)

        t_quant_start = time_ns()
        for system in semi_local.systems
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        rhs_time_quant_ns[] += time_ns() - t_quant_start

        apply_charge_plane_strain_F_fix!(cyl_sys)

        # In pure-viscous mode the deformation gradient is only an auxiliary quantity
        # used by TLSPH internals; it carries no elastic history that must be preserved.
        # If a particle inverts (J <= 0) or nearly collapses, reset that particle's F to
        # plane-strain identity before the stress/contact assembly to avoid a later stall
        # from nonphysical F^{-T} / penalty-force usage.
        if enable_emergency_repair_clamp
            u_cyl = TrixiParticles.wrap_u(u_ode, cyl_sys, semi_local)
            repaired_F_particles = 0
            worst_J = Inf
            worst_J_particle = 0
            J_min = enable_elastic_stress ? 2.0e-1 : 5.0e-2
            J_max = enable_elastic_stress ? 5.0 : 10.0
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                F11 = cyl_sys.deformation_grad[1, 1, particle]
                F12 = cyl_sys.deformation_grad[1, 2, particle]
                F13 = cyl_sys.deformation_grad[1, 3, particle]
                F21 = cyl_sys.deformation_grad[2, 1, particle]
                F22 = cyl_sys.deformation_grad[2, 2, particle]
                F23 = cyl_sys.deformation_grad[2, 3, particle]
                F31 = cyl_sys.deformation_grad[3, 1, particle]
                F32 = cyl_sys.deformation_grad[3, 2, particle]
                F33 = cyl_sys.deformation_grad[3, 3, particle]
                J = F11 * (F22 * F33 - F23 * F32) -
                    F12 * (F21 * F33 - F23 * F31) +
                    F13 * (F21 * F32 - F22 * F31)
                if J < worst_J
                    worst_J = J
                    worst_J_particle = particle
                end
                if !isfinite(J) || J <= J_min || J >= J_max
                    cyl_sys.deformation_grad[1, 1, particle] = 1.0
                    cyl_sys.deformation_grad[1, 2, particle] = 0.0
                    cyl_sys.deformation_grad[1, 3, particle] = 0.0
                    cyl_sys.deformation_grad[2, 1, particle] = 0.0
                    cyl_sys.deformation_grad[2, 2, particle] = 1.0
                    cyl_sys.deformation_grad[2, 3, particle] = 0.0
                    cyl_sys.deformation_grad[3, 1, particle] = 0.0
                    cyl_sys.deformation_grad[3, 2, particle] = 0.0
                    cyl_sys.deformation_grad[3, 3, particle] = 1.0
                    if enable_elastic_stress
                        Fp_committed[][1, 1, particle] = 1.0
                        Fp_committed[][1, 2, particle] = 0.0
                        Fp_committed[][1, 3, particle] = 0.0
                        Fp_committed[][2, 1, particle] = 0.0
                        Fp_committed[][2, 2, particle] = 1.0
                        Fp_committed[][2, 3, particle] = 0.0
                        Fp_committed[][3, 1, particle] = 0.0
                        Fp_committed[][3, 2, particle] = 0.0
                        Fp_committed[][3, 3, particle] = 1.0
                        alpha_committed[][particle] = 0.0
                    end

                    u_cyl[1, particle] = clamp(u_cyl[1, particle], -x_safety_bound,
                                               x_safety_bound)
                    if use_pseudo2d_projection
                        u_cyl[2, particle] = 0.0
                    end
                    u_cyl[3, particle] = clamp(u_cyl[3, particle], z_safety_min,
                                               z_safety_max)

                    cyl_sys.current_coordinates[1, particle] = clamp(cyl_sys.current_coordinates[1, particle],
                                                                     -x_safety_bound,
                                                                     x_safety_bound)
                    if use_pseudo2d_projection
                        cyl_sys.current_coordinates[2, particle] = 0.0
                    end
                    cyl_sys.current_coordinates[3, particle] = clamp(cyl_sys.current_coordinates[3, particle],
                                                                     z_safety_min,
                                                                     z_safety_max)

                    v_cyl[1, particle] = 0.0
                    v_cyl[2, particle] = 0.0
                    v_cyl[3, particle] = 0.0

                    repaired_F_particles += 1
                end
            end
        end
        # ---- End pseudo-2D fix ----

        t_implicit_start = time_ns()
        TrixiParticles.update_implicit_sph!(semi_local, v_ode, u_ode, t)
        rhs_time_implicit_ns[] += time_ns() - t_implicit_start
        if force_stage_diag
            print_force_stage_delta!("implicit_sph", force_stage_prev, dv_ode,
                                     cyl_sys, semi_local, t)
        end

        t_pressure_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_pressure!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        rhs_time_pressure_ns[] += time_ns() - t_pressure_start
        if force_stage_diag
            print_force_stage_delta!("pressure", force_stage_prev, dv_ode,
                                     cyl_sys, semi_local, t)
        end

        t_boundary_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_boundary_interpolation!(system, v, u, v_ode, u_ode,
                                                          semi_local, t)
        end
        rhs_time_boundary_ns[] += time_ns() - t_boundary_start
        if force_stage_diag
            print_force_stage_delta!("boundary_interp", force_stage_prev, dv_ode,
                                     cyl_sys, semi_local, t)
        end

        t_final_start = time_ns()
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_final!(system, v, u, v_ode, u_ode, semi_local, t)
        end
        rhs_time_final_ns[] += time_ns() - t_final_start
        if force_stage_diag
            print_force_stage_delta!("final_quantities", force_stage_prev, dv_ode,
                                     cyl_sys, semi_local, t)
        end

        t_stress_interact_start = time_ns()

        if !retraction_dv_mech_ramping(t)
            snapshot_charge_dv!(charge_dv_mech_marker_buf, dv_ode, cyl_sys, semi_local)
        end

        t_stress_cache_start = time_ns()
        update_implicit_stress_cache!(semi_local, v_ode, t)
        rhs_time_stress_cache_ns[] += time_ns() - t_stress_cache_start

        update_retraction_contact_e_scale!(t)
        t_system_interaction_start = time_ns()
        system_interaction_ramped!(dv_ode, v_ode, u_ode, semi_local, floor_act, mold_act)
        rhs_time_system_interaction_ns[] += time_ns() - t_system_interaction_start
        if force_stage_diag
            print_force_stage_delta!("system_interaction", force_stage_prev, dv_ode,
                                     cyl_sys, semi_local, t)
        end

        t_source_terms_start = time_ns()
        TrixiParticles.add_source_terms!(dv_ode, v_ode, u_ode, semi_local, t)
        rhs_time_source_terms_ns[] += time_ns() - t_source_terms_start
        if force_stage_diag
            print_force_stage_delta!("source_terms", force_stage_prev, dv_ode,
                                     cyl_sys, semi_local, t)
        end

        # WCSPH bulk for mushy/liquid particles (skipped when TP_CLIP_DISABLE_WCSPH=1 or
        # TP_CLIP_RETRACTION_TLSPH_ONLY=1 during retraction-only runs).
        if wcsph_mechanics_active()
            t_wcsph_p_start = time_ns()
            ρ_curr_liquid = apply_wcsph_liquid_pressure!(dv_ode, cyl_sys, semi_local)
            rhs_time_wcsph_pressure_ns[] += time_ns() - t_wcsph_p_start
            if force_stage_diag
                print_force_stage_delta!("wcsph_pressure", force_stage_prev, dv_ode,
                                         cyl_sys, semi_local, t)
            end
            if enable_wcsph_liquid_viscosity && lag_wcsph_liquid_viscosity
                t_wcsph_v_start = time_ns()
                dv_cyl_lag = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
                wcsph_scale = wcsph_ramp_scale[]
                @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                    dv_cyl_lag[1, particle] += wcsph_scale * wcsph_visc_lagged_dv_buf[1, particle]
                    dv_cyl_lag[2, particle] += wcsph_scale * wcsph_visc_lagged_dv_buf[2, particle]
                    dv_cyl_lag[3, particle] += wcsph_scale * wcsph_visc_lagged_dv_buf[3, particle]
                end
                rhs_time_wcsph_viscosity_ns[] += time_ns() - t_wcsph_v_start
            elseif enable_wcsph_liquid_viscosity
                t_wcsph_v_start = time_ns()
                apply_wcsph_liquid_viscosity!(dv_ode, v_ode, cyl_sys, semi_local, ρ_curr_liquid)
                rhs_time_wcsph_viscosity_ns[] += time_ns() - t_wcsph_v_start
            end
            if force_stage_diag
                print_force_stage_delta!("wcsph_viscosity", force_stage_prev, dv_ode,
                                         cyl_sys, semi_local, t)
            end
        end

        blend_charge_dv_from_marker!(dv_ode, cyl_sys, semi_local,
                                     charge_dv_mech_marker_buf, mech_dv_scale)

        # # ── ONE-SHOT STIFFNESS DIAGNOSTIC (disabled) ────────────────────────────────
        # if t > diag_stiff_last_t[] + 1e-4
        #     ...
        # end

        rhs_time_stress_interact_ns[] += time_ns() - t_stress_interact_start
    finally
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = nothing
        TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
    end

    contact_damp_dist = 0.8 * smoothing_length
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    floor_sys = semi_local.systems[2]
    mold_sys  = semi_local.systems[3]
    if retraction_explicit_active() && retraction_explicit_freeze_t
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[NDIMS_CYL + 1, particle] = 0.0
        end
    else
        t_thermal_start = time_ns()
        # Tool–charge HTC (floor + punch faces): always on; active when particle–tool gap ≤ ps.
        t_contact_heat_start = time_ns()
        contact_heat_flux = compute_contact_heat_flux!(contact_heat_flux_buf, cyl_sys,
                                                       floor_sys, mold_sys, particle_spacing)
        rhs_time_contact_heat_ns[] += time_ns() - t_contact_heat_start

        t_thermal_sph_start = time_ns()
        TrixiParticles.thermal_rhs_sph3d!(cyl_sys, dv_cyl, v_cyl, 0.0,
                                          particle_spacing, bound_coordinate_thermal, semi_local)
        rhs_time_thermal_sph_ns[] += time_ns() - t_thermal_sph_start
        if use_nakamura_kinetics && !retraction_solid_mechanics_only()
            t_nak_rhs_start = time_ns()
        end
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            T_i    = cyl_sys.temp[particle]
            cp_eff = effective_heat_capacity_particle(T_i, particle)
            dv_cyl[NDIMS_CYL + 1, particle] *= cyl_sys.cp / cp_eff
            # Nakamura latent heat off during solid retraction (α ≈ 1 already).
            if use_nakamura_kinetics && !retraction_solid_mechanics_only()
                α_i    = crystallinity_particle_buf[particle]
                dα_dt  = nakamura_dalpha_dt(clamp(T_i, temp_min_clip, temp_max_clip), α_i)
                L_i    = latent_heat_particle[particle]
                cp_i   = cp_particle[particle]
                dv_cyl[NDIMS_CYL + 1, particle] += L_i * dα_dt / cp_i
            end
            if contact_heat_flux[particle] != 0.0
                rho_i = cyl_sys.material_density[particle]
                dv_cyl[NDIMS_CYL + 1, particle] -= contact_heat_flux[particle] /
                                                  (rho_i * cp_eff * particle_spacing)
            end
        end
        if use_nakamura_kinetics && !retraction_solid_mechanics_only()
            rhs_time_nakamura_rhs_ns[] += time_ns() - t_nak_rhs_start
        end
        if force_stage_diag
            print_force_stage_delta!("thermal_and_contact_heat", force_stage_prev, dv_ode,
                                     cyl_sys, semi_local, t)
        end
        rhs_time_thermal_ns[] += time_ns() - t_thermal_start
        dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    end

    if tool_wall_coupling_active(t)
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            x_i = cyl_sys.current_coordinates[1, particle]
            z_i = cyl_sys.current_coordinates[3, particle]
            gap_floor_drag = tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower)
            gap_mold_drag = tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper)
            near_floor = gap_floor_drag <= contact_damp_dist
            near_mold = gap_mold_drag <= contact_damp_dist

            # Molten wall no-slip drag (off during solid retraction).
            lf_wall = liquid_fraction_particle_buf[particle]
            if !retraction_solid_mechanics_only() && lf_wall > 1.0e-4 && (near_floor || near_mold)
                ρ_i = cyl_sys.material_density[particle]
                η_i = vis_particle_buf[particle]
                drag_rate = molten_wall_drag_factor * lf_wall * η_i /
                            (ρ_i * smoothing_length * smoothing_length)
                drag_rate = min(drag_rate, molten_wall_drag_max_rate)

                if near_floor
                    act = 1.0 - clamp(gap_floor_drag / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)
                    act = act * act * (3.0 - 2.0 * act)
                    dv_cyl[1, particle] -= act * drag_rate * v_cyl[1, particle]
                    dv_cyl[3, particle] -= act * drag_rate * v_cyl[3, particle]
                end

                if near_mold
                    act = 1.0 - clamp(gap_mold_drag / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)
                    act = act * act * (3.0 - 2.0 * act)
                    v_wall_z = mold_z_velocity_at_time(t)
                    dv_cyl[1, particle] -= act * drag_rate * v_cyl[1, particle]
                    dv_cyl[3, particle] -= act * drag_rate * (v_cyl[3, particle] - v_wall_z)
                end
            end

            # Coulomb friction at tool–part contact (solid-fraction weighted).
            if near_floor || near_mold
                sf = solid_fraction_particle_buf[particle]
                if sf > 1.0e-4
                    mu_eff   = mu_friction * sf
                    p_n      = max(0.0, -v_stress_buf[3, 3, particle])
                    rho_i    = cyl_sys.material_density[particle]
                    a_fric   = mu_eff * p_n / (rho_i * particle_spacing)
                    v_rel_x  = v_cyl[1, particle]
                    if abs(v_rel_x) > 1.0e-12
                        dv_cyl[1, particle] -= a_fric * sign(v_rel_x)
                    end
                end
            end
        end
    end
    if retraction_started[] && retraction_dv_mech_ramping(t)
        snapshot_charge_dv!(charge_dv_mech_marker_buf, dv_ode, cyl_sys, semi_local)
        blend_charge_dv_from_marker!(dv_ode, cyl_sys, semi_local,
                                     charge_dv_mech_marker_buf,
                                     retraction_dv_mech_scale(t))
    end

    apply_retraction_charge_accel_caps!(dv_cyl, cyl_sys, t)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        # Liquid/mushy-only acceleration guard: keep WCSPH force spikes from
        # forcing TRBDF2 into nanosecond dt around the melt transition.
        if solid_fraction_particle_buf[particle] < solid_mechanics_solid_fraction_min
            dv_cyl[1, particle] = soft_limit(dv_cyl[1, particle], max_liquid_accel)
            dv_cyl[3, particle] = soft_limit(dv_cyl[3, particle], max_liquid_accel)
        end
    end
    if force_stage_diag
        print_force_stage_delta!("friction_and_accel_caps", force_stage_prev, dv_ode,
                                 cyl_sys, semi_local, t)
    end
    if use_pseudo2d_projection
        # Hard zero: pseudo-2D constraint.
        TrixiParticles.foreach_system(semi_local) do system
            dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                dv_sys[2, particle] = 0.0
            end
        end
    elseif charge_layers_y > 1
        # Thin-3D: stiff spring-damper keeps each particle at its initial y-plane.
        # dv_y += -γ·v_y - k·dy  where dy = current_y - initial_y (displacement, not absolute coord).
        # Gives TRBDF2 a non-zero Jacobian column → Newton converges cleanly.
        # TRBDF2's L-stability damps y-motion to zero within 1–2 steps.
        TrixiParticles.foreach_system(semi_local) do system
            v_sys  = TrixiParticles.wrap_v(v_ode,   system, semi_local)
            u_sys  = TrixiParticles.wrap_u(u_ode,   system, semi_local)
            dv_sys = TrixiParticles.wrap_v(dv_ode,  system, semi_local)
            y0 = TrixiParticles.initial_coordinates(system)  # [dim, particle], reference positions
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                dy = u_sys[2, particle] - y0[2, particle]  # displacement from initial y-plane
                dv_sys[2, particle] += -y_constraint_damp   * v_sys[2, particle] -
                                        y_constraint_spring * dy
            end
        end
    end
    if force_stage_diag
        print_force_stage_delta!("final_after_y_constraint", force_stage_prev, dv_ode,
                                 cyl_sys, semi_local, t)
        flush(stderr)
    end

    # Explicit Euler: drift scales du by kin_scale; scale dv the same way so v does not
    # run ahead of x during the kinematics ramp (implicit Newton does not need this).
    if retraction_explicit_active() && retraction_started[] && retraction_kinematics_ramping(t)
        kin_scale = retraction_kinematics_scale(t)
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[1, particle] *= kin_scale
            if use_pseudo2d_projection
                dv_cyl[2, particle] *= kin_scale
            end
            dv_cyl[3, particle] *= kin_scale
        end
    end

    if hold_tlsph_tail_active(t) && hold_kinematics_mode == :frozen
        dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[1, particle] = 0.0
            dv_cyl[2, particle] = 0.0
            dv_cyl[3, particle] = 0.0
        end
    end

    if retraction_diag_enabled && retraction_started[]
        retraction_diag_kick_max_dv[] = max_charge_kick_dv(dv_ode, cyl_sys, semi_local)
    end

    return dv_ode
end

function drift_implicit_visible!(du_ode, v_ode, u_ode, semi_local, t)
    TrixiParticles.drift!(du_ode, v_ode, u_ode, semi_local, t)
    retraction_cfrp_style_active() && return du_ode

    kin_scale = if hold_kinematics_frozen(t)
        0.0
    elseif retraction_started[]
        retraction_kinematics_scale(t)
    else
        1.0
    end
    if kin_scale < 1.0 - 1.0e-8
        cyl_sys = semi_local.systems[1]
        du_cyl = TrixiParticles.wrap_u(du_ode, cyl_sys, semi_local)
        NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            for d in 1:NDIMS_CYL
                du_cyl[d, particle] *= kin_scale
            end
        end
    end

    # Pseudo-2D only: hard-zero du_y so y-displacement can never accumulate.
    # Thin-3D: drift is left physical (du_y = v_y); the spring term in dv_y pulls u_y back
    # toward 0 so particles stay near their initial y-planes without a hard override.
    if use_pseudo2d_projection
        TrixiParticles.foreach_system(semi_local) do system
            du_sys = TrixiParticles.wrap_u(du_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                du_sys[2, particle] = 0.0
            end
        end
    end

    return du_ode
end

if hold_checkpoint_restore !== nothing
    restore_hold_checkpoint_state!(hold_checkpoint_restore, semi)
    if retraction_tlsph_only
        refresh_retraction_solid_phase_buffers!()
        wcsph_ramp_scale[] = 0.0
    end
end

v0_ode = if hold_checkpoint_restore !== nothing
    Vector{Float64}(hold_checkpoint_restore["v_ode"])
else
    Vector{Float64}(ode_base.u0.x[1])
end
u0_ode = if hold_checkpoint_restore !== nothing
    Vector{Float64}(hold_checkpoint_restore["u_ode"])
else
    Vector{Float64}(ode_base.u0.x[2])
end

# Retraction-only: arm ramps before the first Newton step.  If retraction_started flips
# inside accepted_step_callback, dv_mech_scale jumps 1 → 0 and TRBDF2 collapses dt to ~1e-8
# (sim time stays at t0 → InfoCallback shows 0% for hundreds of steps).
if sim_phase == "retraction" && hold_checkpoint_restore !== nothing
    cyl_sys = semi.systems[1]
    t_ret = hold_checkpoint_restore["t"]
    retraction_started[] = true
    retraction_start_time[] = t_ret
    capture_retraction_reference_span!(cyl_sys)
    prepare_charge_state_for_retraction!(cyl_sys, semi, v0_ode, u0_ode,
                                         Fp_committed[], J_ref_particle_buf, J_p_particle_buf,
                                         vel_grad_buf, charge_dv_mech_marker_buf)
    retraction_state_prepared[] = true
    if retraction_cfrp_style
        println(">>> Retraction armed (CFRP-style) at checkpoint t=", round(t_ret; digits=6),
                " s — full system_interaction!, no σ/contact/kin ramps")
    else
        println(">>> Retraction armed at checkpoint t=", round(t_ret; digits=6),
                " s (contact/elastic/kinematics ramps active from t0; no step-1 discontinuity)")
    end
    hold_end_spencer_align &&
        println(">>> Hold-end Spencer alignment: ON (I4=1 at checkpoint F)")
    retraction_springback_dwell_s > 0.0 &&
        println(">>> Post-lift springback dwell: up to ", retraction_springback_dwell_s,
                " s (equil max_speed<", retraction_springback_equil_max_speed,
                " m/s, span_z rate<", retraction_springback_equil_span_rate_mm_s, " mm/s)")
    flush(stdout)
end

ode = DynamicalODEProblem(kick_implicit_visible!, drift_implicit_visible!,
                          v0_ode, u0_ode, tspan, semi)

t_start, t_end = tspan

# Local compatibility helper: provide a periodic callback without requiring
# DiffEqCallbacks.jl in this environment.
function PeriodicCallback(affect!, dt; save_positions=(false, false))
    next_t = Ref(dt)
    condition(u, t, integrator) = t + 1.0e-14 >= next_t[]

    function wrapped_affect!(integrator)
        affect!(integrator)
        while next_t[] <= integrator.t + 1.0e-14
            next_t[] += dt
        end
    end

    return DiscreteCallback(condition, wrapped_affect!;
                            save_positions=save_positions)
end

function print_timing_snapshot(integrator)
    # Diagnostics disabled.
    return nothing
    # ns_to_s(ns) = ns / 1.0e9
    # ...
end

# Open mold force log CSV (written every accepted step in the DiscreteCallback below)
mold_force_log_path = sim_phase == "retraction" ?
                      "mold_force_log_retraction.csv" : "mold_force_log.csv"
mold_force_io = open(mold_force_log_path, "w")
println(mold_force_io, "t_s,F_mold_z_N")
flush(mold_force_io)

retraction_diag_io = if retraction_diag_enabled
    io = open(retraction_diag_log_path, "w")
    println(io, "step,t,dt,kick_max_dv,max_speed,max_abs_vz,span_x_mm,span_z_mm,",
            "max_x_gap_mm,span_ratio,J_min,J_max,J_min_particle,min_mold_gap_mm,",
            "min_floor_gap_mm,hold_rel,elas_scale,kin_scale,mech_dv_scale,floor_act,",
            "mold_act,nonfinite_n")
    flush(io)
    println(">>> Retraction diagnostics ON: log=", abspath(retraction_diag_log_path),
            " interval=", retraction_diag_interval, " steps")
    flush(stdout)
    io
else
    nothing
end

accepted_step_callback = DiscreteCallback(
    (u, t, integrator) -> integrator.iter > 0 && mod(integrator.iter, 1) == 0,
    function(integrator)
            enforce_explicit_retraction_dt!(integrator)

            cyl_sys = integrator.p.systems[1]
            floor_sys = integrator.p.systems[2]
            mold_sys = integrator.p.systems[3]
            v_wrap_state = TrixiParticles.wrap_v(integrator.u.x[1], cyl_sys, integrator.p)

            if retraction_diag_enabled && retraction_started[] &&
               !retraction_diag_instability_announced[]
                n_nonfinite = count_charge_nonfinite_state(cyl_sys, v_wrap_state)
                if n_nonfinite > 0
                    retraction_diag_instability_announced[] = true
                    println(">>> RETRACTION INSTABILITY: nonfinite charge state at accepted step ",
                            retraction_diag_step_count[] + 1, " t=",
                            round(integrator.t; digits=9), " s (", n_nonfinite,
                            " particles)")
                    flush(stdout)
                    if retraction_diag_prev_snapshot[] !== nothing
                        print_retraction_diag_snapshot("last_good_step",
                                                       retraction_diag_prev_snapshot[])
                    end
                    snap_now = build_retraction_diag_snapshot(integrator, cyl_sys, floor_sys,
                                                              mold_sys, v_wrap_state)
                    print_retraction_diag_snapshot("first_bad_step", snap_now)
                end
            end

            if hold_kinematics_frozen(integrator.t)
                freeze_charge_mechanical_state!(v_wrap_state, cyl_sys)
            end

            if hold_tlsph_tail_active(integrator.t) && !hold_tlsph_tail_announced[]
                hold_tlsph_tail_announced[] = true
                println(">>> TLSPH equilibration tail: WCSPH off, full TLSPH+contact, ",
                        hold_kinematics_mode == :frozen ?
                            ("kinematics frozen for " * string(round(hold_tlsph_tail_s; digits=4)) * " s") :
                            ("kinematics active for " * string(round(hold_tlsph_tail_s; digits=4)) * " s"))
                flush(stdout)
            end
            t_vel_grad_start = time_ns()
            refresh_velocity_gradient_cache!(cyl_sys, v_wrap_state, integrator.p, vel_grad_buf)
            rhs_time_callback_vel_grad_ns[] += time_ns() - t_vel_grad_start

            if !charge_hold_thermal_only(integrator.t) && !retraction_solid_mechanics_only() &&
               !hold_tlsph_tail_active(integrator.t)
                t_orient_start = time_ns()
                update_flow_orientation_kinetics!(orientation_tensor_state[],
                                                  orientation_scalar_particle,
                                                  vel_grad_buf,
                                                  integrator.dt,
                                                  fiber_direction;
                                                  xi_ft=xi_ft,
                                                  ci_ft=ci_ft,
                                                  kappa_rsc=kappa_rsc,
                                                  ci_ard_parallel=ci_ard_parallel,
                                                  ci_ard_perp=ci_ard_perp)
                rhs_time_callback_orientation_ns[] += time_ns() - t_orient_start
            end

            u_wrap = TrixiParticles.wrap_u(integrator.u.x[2], cyl_sys, integrator.p)
            if enable_emergency_repair_clamp
                repaired_particles = 0
                @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                    x = u_wrap[1, particle]
                    z = u_wrap[3, particle]
                    vx = v_wrap_state[1, particle]
                    vz = v_wrap_state[3, particle]

                    F11 = cyl_sys.deformation_grad[1, 1, particle]
                    F12 = cyl_sys.deformation_grad[1, 2, particle]
                    F13 = cyl_sys.deformation_grad[1, 3, particle]
                    F21 = cyl_sys.deformation_grad[2, 1, particle]
                    F22 = cyl_sys.deformation_grad[2, 2, particle]
                    F23 = cyl_sys.deformation_grad[2, 3, particle]
                    F31 = cyl_sys.deformation_grad[3, 1, particle]
                    F32 = cyl_sys.deformation_grad[3, 2, particle]
                    F33 = cyl_sys.deformation_grad[3, 3, particle]
                    J = F11 * (F22 * F33 - F23 * F32) -
                        F12 * (F21 * F33 - F23 * F31) +
                        F13 * (F21 * F32 - F22 * F31)

                    J_min = enable_elastic_stress ? 2.0e-1 : 5.0e-2
                    J_max = enable_elastic_stress ? 5.0 : 10.0

                    invalid_state = !isfinite(x) || !isfinite(z) || !isfinite(vx) || !isfinite(vz) ||
                                    x < -x_safety_bound || x > x_safety_bound ||
                                    z < z_safety_min || z > z_safety_max ||
                                    abs(vx) > max_cylinder_speed || abs(vz) > max_cylinder_speed ||
                                    !isfinite(J) || J <= J_min || J >= J_max

                    if invalid_state
                        u_wrap[1, particle] = clamp(x, -x_safety_bound, x_safety_bound)
                        if use_pseudo2d_projection
                            u_wrap[2, particle] = 0.0
                        end
                        u_wrap[3, particle] = clamp(z, z_safety_min, z_safety_max)

                        cyl_sys.current_coordinates[1, particle] = u_wrap[1, particle]
                        if use_pseudo2d_projection
                            cyl_sys.current_coordinates[2, particle] = 0.0
                        end
                        cyl_sys.current_coordinates[3, particle] = u_wrap[3, particle]

                        v_wrap_state[1, particle] = 0.0
                        if use_pseudo2d_projection
                            v_wrap_state[2, particle] = 0.0
                        end
                        v_wrap_state[3, particle] = 0.0

                        cyl_sys.deformation_grad[1, 1, particle] = 1.0
                        cyl_sys.deformation_grad[1, 2, particle] = 0.0
                        cyl_sys.deformation_grad[1, 3, particle] = 0.0
                        cyl_sys.deformation_grad[2, 1, particle] = 0.0
                        cyl_sys.deformation_grad[2, 2, particle] = 1.0
                        cyl_sys.deformation_grad[2, 3, particle] = 0.0
                        cyl_sys.deformation_grad[3, 1, particle] = 0.0
                        cyl_sys.deformation_grad[3, 2, particle] = 0.0
                        cyl_sys.deformation_grad[3, 3, particle] = 1.0
                        if enable_elastic_stress
                            Fp_committed[][1, 1, particle] = 1.0
                            Fp_committed[][1, 2, particle] = 0.0
                            Fp_committed[][1, 3, particle] = 0.0
                            Fp_committed[][2, 1, particle] = 0.0
                            Fp_committed[][2, 2, particle] = 1.0
                            Fp_committed[][2, 3, particle] = 0.0
                            Fp_committed[][3, 1, particle] = 0.0
                            Fp_committed[][3, 2, particle] = 0.0
                            Fp_committed[][3, 3, particle] = 1.0
                            alpha_committed[][particle] = 0.0
                        end
                        repaired_particles += 1
                    elseif apply_velocity_safety_clamps()
                        v_wrap_state[1, particle] = soft_limit(vx, max_cylinder_speed)
                        if use_pseudo2d_projection
                            v_wrap_state[2, particle] = 0.0
                        end
                        v_wrap_state[3, particle] = soft_limit(vz, max_cylinder_speed)
                    end
                end
            end

            if retraction_cfrp_style_active()
                advance_F_from_vel_grad!(cyl_sys, vel_grad_buf, integrator.dt)
                enable_elastic_stress &&
                    commit_plastic_history_cfrp!(cyl_sys, integrator.p, integrator.dt,
                                                 integrator.u.x[1])
            elseif enable_elastic_stress && integrator.t >= t_warmup_end
                # ------------------------------------------------------------------
                # Nakamura crystallization ODE: advance α_c by one accepted step dt.
                # Must run before update_dual_phase_properties! so that liquid/solid
                # fraction buffers are refreshed from the new crystallinity.
                # ------------------------------------------------------------------
                if use_nakamura_kinetics && !retraction_solid_mechanics_only() &&
                   !hold_tlsph_tail_active(integrator.t)
                    t_nak_acc_start = time_ns()
                    step_nakamura_crystallinity!(crystallinity_particle_buf,
                                                 cyl_sys.temp,
                                                 liquid_fraction_particle_buf,
                                                 solid_fraction_particle_buf,
                                                 integrator.dt)
                    rhs_time_nakamura_accepted_ns[] += time_ns() - t_nak_acc_start
                end

                update_dual_phase_properties!(ys_particle_buf, hard_particle_buf, vis_particle_buf,
                                              cyl_sys.temp, alpha_committed[],
                                              thermal_softening_particle_buf,
                                              liquid_fraction_particle_buf,
                                              solid_fraction_particle_buf)
                refresh_retraction_solid_phase_buffers!()

                hold_active = hold_plastic_relax_active(integrator.t)
                skip_heat_commit = hold_active ||
                    (retraction_explicit_active() && retraction_explicit_freeze_t)
                n_commit = hold_active ? hold_relax_substeps : 1
                dt_commit = integrator.dt / n_commit
                t_plastic_start = time_ns()
                for _ in 1:n_commit
                    commit_plastic_history_and_heat!(cyl_sys,
                                                       ys_particle_buf,
                                                       hard_particle_buf,
                                                       vis_particle_buf,
                                                       dt_commit,
                                                       alpha_committed[],
                                                       Fp_committed[];
                                                       skip_heat=skip_heat_commit,
                                                       hold_creep=hold_active)
                end
                rhs_time_plastic_commit_ns[] += time_ns() - t_plastic_start

                v_wrap = TrixiParticles.wrap_v(integrator.u.x[1], cyl_sys, integrator.p)
                NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
                @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                    v_wrap[NDIMS_CYL + 1, particle] = cyl_sys.temp[particle]
                end
            end

            if retraction_cfrp_style_active()
                # F + plastic commit handled in CFRP-style branch above.
            elseif charge_hold_thermal_only(integrator.t)
                # F frozen during solidification hold.
            elseif retraction_started[]
                kin = retraction_kinematics_scale(integrator.t)
                if kin > 1.0e-10
                    advance_F_from_vel_grad!(cyl_sys, vel_grad_buf, integrator.dt * kin)
                end
            else
                advance_F_from_vel_grad!(cyl_sys, vel_grad_buf, integrator.dt)
            end

            if wcsph_mechanics_active() && enable_wcsph_liquid_viscosity &&
               lag_wcsph_liquid_viscosity
                refresh_lagged_wcsph_liquid_viscosity!(wcsph_visc_lagged_dv_buf,
                    wcsph_visc_scratch_dv_ode,
                    integrator.u.x[1],
                    cyl_sys,
                    integrator.p)
            end

            if sim_phase != "retraction" && integrator.t >= t_compress &&
               !solidification_hold_announced[]
                println(">>> Entering solidification hold at t=", round(integrator.t; digits=6),
                    " s (solid fraction threshold=",
                    round(100 * solidification_fraction_threshold; digits=1),
                    "%, target fraction=", round(100 * solidification_fraction_target; digits=1),
                        "%)")
                flush(stdout)
                solidification_hold_announced[] = true
            end

            if sim_phase != "retraction" && integrator.t >= t_compress &&
               !charge_hold_dtmax_active[]
                solver_dtmax_before_hold[] = integrator.opts.dtmax
                integrator.opts.dtmax = charge_hold_phase_dtmax
                integrator.dt = min(integrator.dt, charge_hold_phase_dtmax)
                charge_hold_dtmax_active[] = true
                println(">>> Entering closed-die hold at t=", round(integrator.t; digits=6),
                        " s (",
                        hold_kinematics_mode == :frozen ?
                            "kinematics frozen; thermal + plastic commit" :
                        hold_kinematics_mode == :hybrid ?
                            "kinematics frozen until TLSPH tail; plastic commit" :
                            "kinematics active; TLSPH/contact + plastic commit",
                        "; dtmax=", charge_hold_phase_dtmax,
                        " s; hold_relax_substeps=", hold_relax_substeps, ")")
                flush(stdout)
            end

            if sim_phase != "retraction" && integrator.t >= t_compress &&
               !solidification_complete[]
                solidified_count = count(fs >= solidification_fraction_threshold for fs in
                             solid_fraction_particle_buf)
                solidified_fraction = solidified_count / length(cyl_sys.temp)
                hold_elapsed = integrator.t - t_compress
                max_hold_reached = isfinite(solidification_hold_max) &&
                    hold_elapsed >= solidification_hold_max
                if solidified_fraction >= solidification_fraction_target || max_hold_reached
                    if solidified_fraction >= solidification_fraction_target
                        println(">>> All charge particles solidified at t=",
                                round(integrator.t; digits=6), " s (hold ",
                                round(hold_elapsed; digits=3), " s; ",
                                round(100 * solidified_fraction; digits=1),
                            "% with sf>=",
                            round(solidification_fraction_threshold; digits=3), ")")
                    else
                        println(">>> Solidification max hold reached at t=",
                                round(integrator.t; digits=6), " s (hold ",
                                round(hold_elapsed; digits=3), " s; ",
                                round(100 * solidified_fraction; digits=1),
                            "% with sf>=", round(solidification_fraction_threshold; digits=3),
                            "; target=", round(100 * solidification_fraction_target; digits=1),
                            "%)")
                    end
                    flush(stdout)
                    solidification_complete[] = true
                    solidification_reached_time[] = integrator.t
                    println(">>> Holding closed: thermal dwell ",
                            round(post_solidification_dwell; digits=6),
                            " s, then TLSPH equilibration tail ",
                            round(hold_tlsph_tail_s; digits=6),
                            " s before checkpoint/retraction")
                    flush(stdout)
                end
            end

            if solidification_complete[] && !retraction_started[]
                if hold_tlsph_checkpoint_ready(integrator.t)
                    if !retraction_state_prepared[]
                        prepare_charge_state_for_retraction!(cyl_sys, integrator.p,
                            integrator.u.x[1], integrator.u.x[2],
                            Fp_committed[], J_ref_particle_buf, J_p_particle_buf, vel_grad_buf,
                            charge_dv_mech_marker_buf)
                        retraction_state_prepared[] = true
                        println(">>> Fp/J_p/J_ref aligned after TLSPH hold tail at t=",
                                round(integrator.t; digits=6), " s",
                                use_volumetric_plasticity ?
                                    " (Strategy E: J_p=det(F), J_ref=1, Fp reset=$(hold_end_fp_reset))" :
                                    "",
                                hold_end_spencer_align ?
                                    " (Spencer fibre ref aligned, I4=1 at hold F)" : "")
                        flush(stdout)
                    end
                    if sim_phase == "hold" || (sim_phase == "full" && save_checkpoint_at_hold)
                        ck = capture_hold_checkpoint_state(integrator, integrator.p)
                        save_hold_checkpoint!(checkpoint_path, ck)
                        save_vtu_snapshot!(integrator)
                    end
                    if sim_phase == "hold"
                        println(">>> Hold phase complete; stopping before retraction.")
                        flush(stdout)
                        terminate!(integrator)
                        return
                    end
                    retraction_started[] = true
                    retraction_start_time[] = integrator.t
                    capture_retraction_reference_span!(cyl_sys)
                    if !retraction_state_prepared[]
                        prepare_charge_state_for_retraction!(cyl_sys, integrator.p,
                            integrator.u.x[1], integrator.u.x[2],
                            Fp_committed[], J_ref_particle_buf, J_p_particle_buf, vel_grad_buf,
                            charge_dv_mech_marker_buf)
                        retraction_state_prepared[] = true
                        println(">>> Charge hold-end state prepared for retraction at t=",
                                round(integrator.t; digits=6), " s (",
                                use_volumetric_plasticity ?
                                    "Strategy E: J_p=det(F), J_ref=1, Fp reset=$(hold_end_fp_reset)" :
                                    "Fp=F_iso, J_ref=det(F); stored F kept",
                                hold_end_spencer_align ?
                                    "; Spencer fibre ref aligned, I4=1 at hold F)" : ")")
                        flush(stdout)
                    end
                    println(">>> Starting mold retraction at t=",
                            round(integrator.t; digits=6),
                            " s (mold ramp=", retraction_ramp_s,
                            " s; contact ramp ", retraction_hold_release_s,
                            " s; elastic ramp ", retraction_elastic_ramp_s,
                            " s; kinematics ramp ", retraction_kinematics_release_s,
                            " s)")
                    flush(stdout)
                end
            end

            if retraction_started[] && !charge_mechanics_release_announced[] &&
               !retraction_dv_mech_ramping(integrator.t)
                charge_mechanics_release_announced[] = true
                if charge_hold_dtmax_active[]
                    integrator.opts.dtmax = solver_dtmax_before_hold[]
                    charge_hold_dtmax_active[] = false
                end
                println(">>> Retraction contact/σ-dv ramp complete at t=",
                        round(integrator.t; digits=6),
                        " s; dtmax restored to ", solver_dtmax_before_hold[], " s")
                retraction_use_explicit &&
                    println(">>> Explicit retraction: mech ramp complete at t=",
                            round(integrator.t; digits=6), " s (dt=",
                            retraction_explicit_dt_springback, " s throughout)")
                flush(stdout)
            end

            if retraction_started[] && !retraction_kinematics_release_announced[] &&
               charge_kinematics_fully_released(integrator.t)
                retraction_kinematics_release_announced[] = true
                println(">>> Charge kinematics fully released at t=",
                        round(integrator.t; digits=6),
                        " s (springback enabled)")
                flush(stdout)
            end

            if retraction_started[] && !mold_retraction_complete[]
                retract_elapsed = integrator.t - retraction_start_time[]
                if retract_elapsed >= retraction_duration_max
                    complete_mold_retraction!(integrator)
                end
            end

            if mold_retraction_complete[] && !retraction_complete[] &&
               retraction_springback_dwell_s > 0.0
                dwell_elapsed = integrator.t - mold_retraction_complete_time[]
                snap_eq = build_retraction_diag_snapshot(integrator, cyl_sys, floor_sys, mold_sys,
                                                        v_wrap_state)
                prev_eq = springback_equil_prev_snap[]
                if springback_equilibration_met(snap_eq, prev_eq)
                    springback_equil_streak[] += 1
                else
                    springback_equil_streak[] = 0
                end
                springback_equil_prev_snap[] = snap_eq
                if springback_equil_streak[] >= retraction_springback_equil_patience
                    complete_retraction_simulation!(integrator;
                        reason="springback equilibrated (max_speed=" *
                        string(round(snap_eq.max_speed; sigdigits=3)) * " m/s, span_z=" *
                        string(round(snap_eq.span_z_mm; digits=4)) * " mm)")
                elseif dwell_elapsed >= retraction_springback_dwell_s
                    complete_retraction_simulation!(integrator;
                        reason="springback dwell time reached (span_z=" *
                        string(round(snap_eq.span_z_mm; digits=4)) * " mm, max_speed=" *
                        string(round(snap_eq.max_speed; sigdigits=3)) * " m/s)")
                end
            end

            maybe_save_periodic_checkpoint!(integrator, integrator.p)

            if formulation_profile_enabled
                formulation_profile_step_count[] += 1
                if formulation_profile_step_count[] % formulation_profile_interval == 0
                    print_formulation_timing_report!(integrator;
                        label="every $(formulation_profile_interval) steps")
                end
            end

            # Mold reaction force estimate: Σ_i σ_zz(i) * V_i for all charge particles.
            # σ_zz is cached in v_stress_buf[3,3,i] (Cauchy stress, compression < 0).
            # Reaction force on the mold (upward) = -Σ σ_zz * V (Newton's 3rd law).
            F_mold_z = 0.0
            @inbounds for i in TrixiParticles.each_integrated_particle(cyl_sys)
                V_i = cyl_sys.mass[i] / cyl_sys.material_density[i]
                F_mold_z -= v_stress_buf[3, 3, i] * V_i
            end
            F_total_mold_state[] = F_mold_z
            println(mold_force_io, integrator.t, ",", F_mold_z)
            flush(mold_force_io)

            if retraction_diag_enabled && retraction_started[]
                retraction_diag_step_count[] += 1
                snap = build_retraction_diag_snapshot(integrator, cyl_sys, floor_sys, mold_sys,
                                                      v_wrap_state)
                if retraction_diag_io !== nothing &&
                   (retraction_diag_step_count[] == 1 ||
                    retraction_diag_step_count[] % retraction_diag_interval == 0)
                    write_retraction_diag_csv_row(retraction_diag_io, snap)
                end
                if !retraction_diag_instability_announced[] &&
                   (snap.kick_max_dv >= retraction_diag_spike_dv ||
                    snap.max_speed >= retraction_diag_spike_speed)
                    println(">>> RETRACTION-DIAG SPIKE at step=", snap.step, " t=",
                            round(snap.t; digits=9), " s: kick_max|dv|=",
                            round(snap.kick_max_dv; sigdigits=4),
                            " max_speed=", round(snap.max_speed; sigdigits=4), " m/s")
                    print_retraction_diag_snapshot("spike", snap)
                    flush(stdout)
                end
                abort_reason = retraction_early_abort_reason(snap,
                                                             retraction_diag_prev_snapshot[])
                if abort_reason !== nothing
                    retraction_diag_instability_announced[] = true
                    println(">>> RETRACTION EARLY ABORT at step=", snap.step, " t=",
                            round(snap.t; digits=9), " s: ", abort_reason)
                    print_retraction_diag_snapshot("early_abort", snap)
                    flush(stdout)
                    terminate!(integrator)
                    return
                end
                retraction_diag_prev_snapshot[] = snap
            end

            # ------------------------------------------------------------------
            # Solver convergence diagnostic.  Only printed when Δt collapses
            # (< 1e-7) so the log stays quiet during healthy progress.  Per-step
            # deltas of (nf, njacs, nsolve, nnonliniter, nreject) help diagnose
            # any remaining dt-collapse pathologies.
            if sim_phase == "retraction" && retraction_started[] &&
               retraction_fail_fast_dt > 0.0 && integrator.dt < retraction_fail_fast_dt
                t_adv = integrator.t - retraction_fail_fast_t_prev[]
                retraction_fail_fast_t_prev[] = integrator.t
                if t_adv < retraction_fail_fast_min_t_adv
                    retraction_fail_fast_stall_counter[] += 1
                    if retraction_fail_fast_stall_counter[] >= retraction_fail_fast_max_steps
                        println(">>> Retraction fail-fast: dt=", integrator.dt,
                                " s and sim time stalled for ",
                                retraction_fail_fast_max_steps,
                                " steps (Δt≈", t_adv, " s) — stopping.")
                        flush(stdout)
                        terminate!(integrator)
                        return
                    end
                else
                    retraction_fail_fast_stall_counter[] = 0
                end
            elseif sim_phase == "retraction"
                retraction_fail_fast_t_prev[] = integrator.t
                retraction_fail_fast_stall_counter[] = 0
            end

            # if integrator.dt < 1e-7
            #     stats = integrator.stats
            #     dnf       = stats.nf       - diag_prev_nf[]
            #     dnjacs    = stats.njacs    - diag_prev_njacs[]
            #     dnsolve   = stats.nsolve   - diag_prev_nsolve[]
            #     dnnonlin  = stats.nnonliniter - diag_prev_nnonliniter[]
            #     dnreject  = stats.nreject  - diag_prev_nreject[]
            #     println(">>> SOLVER-DIAG t=", round(integrator.t; digits=8),
            #             " dt=", integrator.dt,
            #             " Δnf=", dnf, " Δnjac=", dnjacs,
            #             " Δnsolve=", dnsolve,
            #             " Δnnonlin=", dnnonlin,
            #             " Δnreject=", dnreject)
            #     diag_prev_nf[]          = stats.nf
            #     diag_prev_njacs[]       = stats.njacs
            #     diag_prev_nsolve[]      = stats.nsolve
            #     diag_prev_nnonliniter[] = stats.nnonliniter
            #     diag_prev_nreject[]     = stats.nreject
            # end
        end)

const retraction_save_final = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_SAVE_FINAL", 1) != 0
vtu_save_callback = if sim_phase == "retraction"
    SolutionSavingCallback(dt=retraction_vtu_save_dt,
                           prefix=vtu_output_prefix,
                           save_final_solution=retraction_save_final,
                           max_coordinates=Inf,
                           fiber_vf=vtu_fiber_vf_quantity)
else
    SolutionSavingCallback(save_times=compress_hold_vtu_save_times,
                           prefix=vtu_output_prefix,
                           save_final_solution=true,
                           max_coordinates=Inf,
                           fiber_vf=vtu_fiber_vf_quantity)
end
info_callback = InfoCallback(interval=retraction_info_interval)

callbacks = if sim_phase == "retraction"
    CallbackSet(accepted_step_callback, vtu_save_callback, info_callback)
else
    CallbackSet(accepted_step_callback,
                # PeriodicCallback(print_timing_snapshot, 2.0e-2,
                #                  save_positions=(false, false)),
                vtu_save_callback,
                info_callback)
end

retraction_abstol = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_ABSTOL", retraction_cfrp_style ? 1.0e-5 : 1.0e-3) : 1.0e-3
retraction_reltol = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_RELTOL", retraction_cfrp_style ? 1.0e-3 : 1.0e-3) : 1.0e-3
retraction_dtmax = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_DTMAX", retraction_cfrp_style ? 1.0e-4 : 1.0e-3) : 1.0e-3

sol = Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Warn)) do
    if retraction_use_explicit
        explicit_alg, explicit_scheme_name = clip_retraction_explicit_algorithm()
        println("\n>>> Entering ODE solve (tspan=", tspan,
                ", retraction solver=explicit ", explicit_scheme_name,
                ", dt=", retraction_explicit_dt_springback,
                ", freeze T=", retraction_explicit_freeze_t, ")")
        flush(stdout)
        if retraction_explicit_ignore_unstable
            solve(ode, explicit_alg;
                  callback=callbacks,
                  save_everystep=false,
                  adaptive=false,
                  dt=retraction_explicit_dt_springback,
                  unstable_check=(dt, u, p, t) -> false,
                  maxiters=10_000_000)
        else
            solve(ode, explicit_alg;
                  callback=callbacks,
                  save_everystep=false,
                  adaptive=false,
                  dt=retraction_explicit_dt_springback,
                  maxiters=10_000_000)
        end
    else
        jacobian_mode = if retraction_cfrp_style
            haskey(ENV, "TP_CLIP_JACOBIAN_MODE") ? clip_jacobian_mode() : "finite"
        else
            clip_jacobian_mode()
        end
        jacobian_autodiff = retraction_cfrp_style && !haskey(ENV, "TP_CLIP_JACOBIAN_MODE") ?
            AutoFiniteDiff() : clip_jacobian_autodiff(jacobian_mode)
        n_ode_dof = length(v0_ode) + length(u0_ode)
        # CFRP uses GMRES at 3D scale; pseudo-2D retraction (n_dof≈147) stays dense LU.
        linsolve_default = (retraction_cfrp_style && n_ode_dof >= 500) ? "gmres" : "dense"
        linsolve_solver = clip_linsolve_solver(n_ode_dof; default_mode=linsolve_default)
        linsolve_name = lowercase(get(ENV, "TP_CLIP_LINSOLVE", linsolve_default))
        println("\n>>> Entering ODE solve (tspan=", tspan,
                ", implicit solver=TRBDF2, jacobian=", jacobian_mode,
                ", linsolve=", linsolve_name, ", n_dof=", n_ode_dof,
                retraction_cfrp_style ? ", style=CFRP" : "", ")")
        flush(stdout)
        nlsolve = retraction_cfrp_style ? NLNewton() :
            NLNewton(κ=1e-1, max_iter=20, fast_convergence_cutoff=0.9, always_new=false)
        solve(ode, TRBDF2(linsolve=linsolve_solver, autodiff=jacobian_autodiff,
                          nlsolve=nlsolve);
              callback=callbacks,
              save_everystep=false,
              abstol=retraction_abstol,
              reltol=retraction_reltol,
              dtmax=retraction_dtmax,
              maxiters=10_000_000)
    end
end

if formulation_profile_enabled && rhs_eval_count[] > 0
    print_formulation_timing_report_final!(sol.t; label="final partial window")
end
if formulation_profile_enabled && formulation_profile_io[] !== nothing
    close(formulation_profile_io[])
    formulation_profile_io[] = nothing
end

println("\n=== SIMULATION COMPLETE ===")
println("Solver retcode: ", sol.retcode)

if retraction_diag_enabled && sol.retcode == :Success && retraction_probe_s > 0.0
    println(">>> RETRACTION PROBE PASSED: t reached ", round(sol.t; digits=6),
            " s within probe window ", retraction_probe_s, " s")
    if retraction_diag_prev_snapshot[] !== nothing
        print_retraction_diag_snapshot("probe_pass", retraction_diag_prev_snapshot[])
    end
end
if retraction_diag_enabled && sol.retcode != :Success
    if retraction_diag_prev_snapshot[] !== nothing
        print_retraction_diag_snapshot("post_mortem_last_accepted",
                                       retraction_diag_prev_snapshot[])
    else
        println(">>> RETRACTION-DIAG post-mortem: no accepted-step snapshot recorded")
        flush(stdout)
    end
    if retraction_diag_io !== nothing
        println(">>> Retraction diagnostic CSV: ", abspath(retraction_diag_log_path))
        flush(stdout)
    end
end

# # COMPREHENSIVE TIMING DIAGNOSTICS (disabled)
# println("\n" * "="^80)
# println("COMPREHENSIVE TIMING DIAGNOSTICS")
# ...

close(mold_force_io)
retraction_diag_io !== nothing && close(retraction_diag_io)
# println(">>> Mold force log written to mold_force_log.csv")
