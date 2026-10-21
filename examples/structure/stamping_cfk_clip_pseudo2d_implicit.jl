# ==========================================================================================
# Pseudo-2D Thermomechanical Forming of an Aerospace CFRT Clip (Implicit)
#
# Product: carbon-fibre reinforced thermoplastic (CFRT) aircraft structural clip.
# Canonical molding physics stack (plane-strain / thin-3D constraints, Nakamura hold,
# retraction, WCSPH/TLSPH blend). Procedural trapezoidal cavity by default; STL /
# user L-die geometry via TP_CLIP_USE_STL=1 (see clip_stl_geometry_setup.jl).
# Legacy alternate driver (deprecated): misc/stamping_cfk_clip_3d_implicit.jl
# GPU thin-3D driver: stamping_cfk_clip_3d_l_thin3d_implicit_gpu.jl (TP_CLIP_GPU=1)
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
using SciMLBase: ArrayPartition
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
using Random

include("../../src/schemes/structure/total_lagrangian_sph/fix_reimann.jl")
include(joinpath(@__DIR__, "clip_charge_plies.jl"))

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

const celsius_to_kelvin_offset = 273.15
@inline celsius_to_kelvin(temp_c) = temp_c + celsius_to_kelvin_offset
@inline kelvin_to_celsius(temp_k) = temp_k - celsius_to_kelvin_offset

function env_temp_c(name, default_c)
    return celsius_to_kelvin(env_float(name, default_c))
end

function fmt_temp(temp_k)
    return string(round(kelvin_to_celsius(temp_k); digits=2), " °C (",
                  round(temp_k; digits=2), " K)")
end

# GPU parallelization (TP_CLIP_GPU=1 → CUDABackend + FullGridCellList NHS).
# Thin GPU driver: stamping_cfk_clip_3d_l_thin3d_implicit_gpu.jl
const clip_use_gpu = env_int("TP_CLIP_GPU", 0) != 0
if clip_use_gpu
    using CUDA
    CUDA.functional() || error("TP_CLIP_GPU=1 but CUDA is not functional on this machine")
    cuda_device = env_int("TP_CLIP_CUDA_DEVICE", 0)
    CUDA.device!(cuda_device)
    const clip_parallelization_backend = CUDABackend()
    println("GPU backend: CUDA device ", cuda_device, " (", CUDA.name(CUDA.device()), ")")
    println("  Note: CUDA RHS via CUDABackend; mold kinematics synced on CPU each step")
else
    const clip_parallelization_backend = PolyesterBackend()
end

function clip_gpu_allowscalar(f)
    if clip_use_gpu
        CUDA.allowscalar(true)
        try
            return f()
        finally
            CUDA.allowscalar(false)
        end
    end
    return f()
end

function clip_ode_vector(n::Integer; T::Type{<:AbstractFloat}=Float64)
    if clip_use_gpu
        return CUDA.zeros(T, n)
    end
    return zeros(T, n)
end

function clip_ode_state_from_checkpoint(ck)
    v = Vector{Float64}(ck["v_ode"])
    u = Vector{Float64}(ck["u_ode"])
    if clip_use_gpu
        return CUDA.CuArray(v), CUDA.CuArray(u)
    end
    return v, u
end

function clip_ode_initial_state(ode_base, hold_checkpoint_restore)
    if hold_checkpoint_restore !== nothing
        return clip_ode_state_from_checkpoint(hold_checkpoint_restore)
    end
    return ode_base.u0.x[1], ode_base.u0.x[2]
end

function clip_particle_buffer(eltype, dims...)
    if clip_use_gpu
        return CUDA.zeros(eltype, dims...)
    end
    return zeros(eltype, dims...)
end

function clip_host_copy(x)
    if clip_use_gpu
        return x isa CUDA.CuArray ? Array(x) : copy(x)
    end
    return copy(x)
end

function clip_host_assign!(dest, src)
    if clip_use_gpu
        if dest isa CUDA.CuArray && !(src isa CUDA.CuArray)
            copyto!(dest, CUDA.CuArray(src))
        elseif dest isa CUDA.CuArray && src isa CUDA.CuArray
            copyto!(dest, src)
        else
            dest .= src
        end
    else
        dest .= src
    end
    return nothing
end

function clip_nhs_max_points_per_cell(n_tool::Integer)
    if haskey(ENV, "TP_CLIP_NHS_MAX_POINTS_PER_CELL")
        return max(64, env_int("TP_CLIP_NHS_MAX_POINTS_PER_CELL", 2048))
    end
    # DictionaryCellList default (100) overflows when thin-3D tool slices exceed ~2k particles.
    return max(256, min(8192, 128 + n_tool ÷ 16))
end

function clip_build_grid_cell_list(n_points::Integer, max_points_per_cell::Integer)
    list_size = max(256, 2 * n_points)
    return SpatialHashingCellList{3}(; list_size, max_points_per_cell=max_points_per_cell)
end

function clip_build_neighborhood_search(search_radius;
                                        polymer_coords,
                                        floor_coords,
                                        mold_coords,
                                        extra_z_pad=0.02)
    n_total = size(polymer_coords, 2) + size(floor_coords, 2) + size(mold_coords, 2)
    n_tool = size(floor_coords, 2) + size(mold_coords, 2)
    max_ppc = clip_nhs_max_points_per_cell(n_tool)
    if clip_use_gpu
        coords = hcat(polymer_coords, floor_coords, mold_coords)
        pad = 4 * search_radius
        min_corner = vec(minimum(coords, dims=2)) .- pad
        max_corner = vec(maximum(coords, dims=2)) .+ pad
        max_corner[3] += extra_z_pad
        cell_list = FullGridCellList(; min_corner, max_corner, max_points_per_cell=max_ppc)
        return GridNeighborhoodSearch{3}(; cell_list, search_radius)
    end
    return GridNeighborhoodSearch{3}(;
        cell_list=clip_build_grid_cell_list(n_total, max_ppc),
        search_radius=search_radius)
end

function clip_build_charge_neighborhood_search(search_radius, coords)
    n = size(coords, 2)
    max_ppc = clip_nhs_max_points_per_cell(0)
    if clip_use_gpu
        pad = 4 * search_radius
        min_corner = vec(minimum(coords, dims=2)) .- pad
        max_corner = vec(maximum(coords, dims=2)) .+ pad
        cell_list = FullGridCellList(; min_corner, max_corner, max_points_per_cell=max_ppc)
        return GridNeighborhoodSearch{3}(; cell_list, search_radius)
    end
    return GridNeighborhoodSearch{3}(;
        cell_list=clip_build_grid_cell_list(n, max_ppc),
        search_radius=search_radius)
end

const clip_updated_lagrange = env_int("TP_CLIP_UPDATED_LAGRANGE", 0) != 0
const clip_updated_lagrange_interval = env_float("TP_CLIP_UPDATED_LAGRANGE_INTERVAL_S", 0.05)

function clip_precomputed_nhs_template(; max_neighbors::Integer, n_tool::Integer=0)
    max_ppc = clip_nhs_max_points_per_cell(n_tool)
    inner = GridNeighborhoodSearch{3}(;
        cell_list=clip_build_grid_cell_list(max(n_tool, 1), max_ppc),
        search_radius=0.0)
    return PrecomputedNeighborhoodSearch{3}(;
        update_neighborhood_search=inner, max_neighbors=max_neighbors)
end

function clip_charge_nhs_template(; max_neighbors::Integer, n_tool::Integer=0)
    return clip_precomputed_nhs_template(; max_neighbors=max_neighbors, n_tool=n_tool)
end

function clip_build_wcsph_neighborhood_search(search_radius, semi)
    PointNeighbors.copy_neighborhood_search(
        TrixiParticles.get_neighborhood_search(semi.systems[1], clip_floor_system(semi), semi),
        search_radius,
        nparticles(semi.systems[1]))
end

# WCSPH current-config neighbor loops: on GPU reuse the main simulation grid NHS
# (CuArray cell list + full-domain bbox from semidiscretize). A separate CPU-built
# FullGridCellList cannot be re-initialized on device coordinates.
@inline function clip_wcsph_nhs(semi)
    clip_use_gpu && return TrixiParticles.get_neighborhood_search(semi.systems[1],
                                                                  clip_floor_system(semi),
                                                                  semi)
    return wcsph_current_nhs
end

# Implicit Jacobian mode for TRBDF2.
#   finite — AutoFiniteDiff (default)
function clip_jacobian_mode()
    mode = lowercase(get(ENV, "TP_CLIP_JACOBIAN_MODE", "finite"))
    mode == "finite" ||
        error("TP_CLIP_JACOBIAN_MODE must be 'finite' (got $(mode)). " *
              "Sparse Jacobian AD is not supported for DynamicalODEProblem in this script.")
    return mode
end

function clip_jacobian_autodiff(mode::String)
    return AutoFiniteDiff()
end

# Linear solver for implicit Newton steps (W = I - γJ).
#   dense / rfluf / lu — RFLUFactorization (default; best for n_dof≲1500 clip runs)
#   krylov / gmres    — GMRES: KrylovJL for matrix_free JVP; KrylovKit for dense+large n_dof
#   sparse            — Sparse LU factorization (UmfpackFactorization); pairs with sparse Jacobian
#   colored Jacobian always uses dense/RFLU (W is dense even though assembly is colored)
function clip_linsolve_solver(n_dof::Int; default_mode::String="dense",
                              matrix_free::Bool=false,
                              jacobian_strategy::String="dense")
    mode = lowercase(get(ENV, "TP_CLIP_LINSOLVE", default_mode))
    if jacobian_strategy == "sparse" && !haskey(ENV, "TP_CLIP_LINSOLVE")
        mode = "sparse"
    elseif jacobian_strategy == "colored" && !haskey(ENV, "TP_CLIP_LINSOLVE")
        mode = "dense"
    end
    mode in ("dense", "rfluf", "lu", "krylov", "gmres", "sparse") ||
        error("TP_CLIP_LINSOLVE must be dense|rfluf|lu|krylov|gmres|sparse (got $(mode))")
    if mode == "sparse"
        return LinearSolve.UMFPACKFactorization()
    end
    if mode in ("krylov", "gmres")
        if matrix_free
            # matrix_free JVP: KrylovJL is broken with OrdinaryDiffEq JVPs.
            # Swapping to KrylovKitJL_GMRES to enable matrix-free 3D scaling.
            krylovdim = env_int("TP_CLIP_KRYLOV_DIM", 120)
            # Optional crude block-diagonal preconditioner for the flattened (v,u) state:
            # rescales the u block relative to v to improve conditioning of (I - γJ).
            #
            # Enable with: TP_CLIP_MF_PREC=1
            # Tune with:   TP_CLIP_MF_PREC_V_SCALE, TP_CLIP_MF_PREC_U_SCALE
            #
            # Note: This does not approximate J; it only rescales mixed-unit blocks to
            # reduce GMRES stagnation. It is safe to disable (default).
            use_prec = env_int("TP_CLIP_MF_PREC", 0) != 0
            v_scale = env_float("TP_CLIP_MF_PREC_V_SCALE", 1.0)
            u_scale = env_float("TP_CLIP_MF_PREC_U_SCALE", 1.0)
            precs = if use_prec
                function (_A, _p)
                    n = n_dof
                    nv = clip_ode_nv[]
                    nu = clip_ode_nu[]
                    if nv <= 0 || nu <= 0 || nv + nu != n
                        return (nothing, nothing)
                    end
                    s = Vector{Float64}(undef, n)
                    @inbounds for i in 1:nv
                        s[i] = v_scale
                    end
                    @inbounds for i in nv+1:n
                        s[i] = u_scale
                    end
                    P = LinearSolve.InvPreconditioner(LinearAlgebra.Diagonal(s))
                    return (P, nothing)  # left preconditioner only
                end
            else
                LinearSolve.DEFAULT_PRECS
            end
            return KrylovKitJL_GMRES(; krylovdim=krylovdim,
                                       atol=env_float("TP_CLIP_KRYLOV_ATOL", 1.0e-6),
                                       rtol=env_float("TP_CLIP_KRYLOV_RTOL", 5.0e-2),
                                       maxiter=env_int("TP_CLIP_KRYLOV_MAXITER", 400),
                                       verbosity=0,
                                       precs=precs)
        end
        # Dense Jacobian + GMRES (CFRP 3D): KrylovKit with large krylovdim.
        n_dof < 500 &&
            @warn "TP_CLIP_LINSOLVE=krylov with n_dof=$n_dof: dense LU is typically faster " *
                  "unless the Jacobian is very large."
        krylovdim = env_int("TP_CLIP_KRYLOV_DIM", 40)
        return KrylovKitJL_GMRES(; krylovdim=krylovdim,
                                   atol=env_float("TP_CLIP_KRYLOV_ATOL", 1.0e-6),
                                   rtol=env_float("TP_CLIP_KRYLOV_RTOL", 5.0e-2),
                                   maxiter=env_int("TP_CLIP_KRYLOV_MAXITER", 200),
                                   verbosity=0)
    end
    return RFLUFactorization()
end

# Implicit time integrator (OrdinaryDiffEq).
#   implicit_euler — 1 implicit stage; default for clip (pairs with matrix_free JVP)
#   reduced_implicit_euler — same as implicit_euler, Newton on [u,T] only
#   trbdf2         — 2-stage L-stable (legacy default behaviour; full [v,T,u])
#   reduced_trbdf2 — same TRBDF2 tableau, Newton on [u,T] only
#   rodas5         — stiff Rosenbrock (fallback for difficult transients)
function clip_integrator_mode()
    mode = lowercase(get(ENV, "TP_CLIP_INTEGRATOR", "trbdf2"))
    mode in ("trbdf2", "reduced_trbdf2", "implicit_euler", "reduced_implicit_euler",
             "rodas5", "explicit") ||
        error("TP_CLIP_INTEGRATOR must be trbdf2, reduced_trbdf2, implicit_euler, " *
              "reduced_implicit_euler, rodas5, or explicit (got $(mode))")
    return mode
end

@inline clip_reduced_implicit_euler_active() =
    lowercase(get(ENV, "TP_CLIP_INTEGRATOR", "trbdf2")) == "reduced_implicit_euler"

@inline clip_reduced_trbdf2_active() =
    lowercase(get(ENV, "TP_CLIP_INTEGRATOR", "trbdf2")) == "reduced_trbdf2"

@inline clip_reduced_ode_active() =
    clip_reduced_implicit_euler_active() || clip_reduced_trbdf2_active()

# Jacobian assembly for implicit Newton solves.
#   matrix_free — directional J·v via 2 RHS evals + GMRES (no dense 882×882 build)
#   dense       — AutoFiniteDiff column differencing + dense/RFLU linear solve
#   sparse      — SPH-locality coloring + sparse storage + UMFPACK
#   colored     — same coloring as sparse, dense Jacobian storage + dense/RFLU LU
function clip_normalize_jacobian_strategy(strategy::String)
    s = lowercase(strategy)
    s == "colored_dense" && return "colored"
    return s
end

function clip_jacobian_strategy()
    strategy = clip_normalize_jacobian_strategy(get(ENV, "TP_CLIP_JACOBIAN_STRATEGY", "dense"))
    strategy in ("dense", "matrix_free", "sparse", "colored") ||
        error("TP_CLIP_JACOBIAN_STRATEGY must be dense, matrix_free, sparse, or colored " *
              "(got $(strategy))")
    return strategy
end

@inline clip_uses_custom_jacobian(strategy::String) =
    strategy in ("sparse", "colored")

const clip_ode_flat = Ref(false)
const clip_ode_nv = Ref(0)
const clip_ode_nu = Ref(0)
const clip_ode_reduced = Ref(false)
const clip_reduced_prev_t = Ref(0.0)
const clip_reduced_prev_u = Ref(Float64[])
const clip_reduced_prev_v = Ref(Float64[])
const clip_reduced_mech_v_idx = Ref(Int[])
const clip_reduced_temp_v_idx = Ref(Int[])
const clip_reduced_temp_ranges = Ref(UnitRange{Int}[])
const clip_reduced_v_work = Ref(Float64[])
const clip_reduced_dv_work = Ref(Float64[])
const clip_reduced_accepted_v = Ref(Float64[])
const clip_reduced_accepted_t = Ref(-Inf)
# Reduced TRBDF2 only (stock TRBDF2 / both implicit-Euler paths ignore these).
# OrdinaryDiffEq TRBDF2 tableau: γ=2-√2, d=γ/2, ω=√2/4.
const CLIP_TRBDF2_γ = 2 - sqrt(2)
const CLIP_TRBDF2_d = 1 - sqrt(2) / 2
const CLIP_TRBDF2_ω = sqrt(2) / 4
const clip_reduced_trbdf2_an = Ref(Float64[])
const clip_reduced_trbdf2_aγ = Ref(Float64[])
const clip_reduced_trbdf2_vγ = Ref(Float64[])
const clip_reduced_trbdf2_fγ = Ref(Float64[])
const clip_reduced_trbdf2_an_t = Ref(-Inf)
const clip_reduced_trbdf2_an_h = Ref(NaN)
# Step size inferred from the trapezoid node (Δt = γ h). Do not classify stages from
# clip_integrator_dt: after an accepted step the PI controller may raise dt before the
# next RHS, so a stale clip_integrator_dt mislabels trapezoid as BDF2.
const clip_reduced_trbdf2_h_step = Ref(NaN)
const clip_reduced_trbdf2_has_γ = Ref(false)
const clip_reduced_trbdf2_freeze_stage_cache = Ref(false)
const clip_reduced_trbdf2_fsalfirst = Ref{Union{Nothing, Vector{Float64}}}(nothing)
const CURRENT_JACOBIAN_STRATEGY = Ref("dense")
const clip_jvp_work = Ref{Union{Nothing, NamedTuple{(:y_plus, :y_minus, :fp, :fm),
                                                    NTuple{4, Vector{Float64}}}}}(nothing)

const clip_jac_colors = Ref{Vector{Int}}()
const clip_jac_groups = Ref{Vector{Vector{Int}}}()
const clip_jac_coloring_n = Ref(0)
const clip_jac_sparsity_pattern = Ref{Union{Nothing,SparseMatrixCSC{Float64,Int}}}(nothing)
const clip_in_jacobian_fill = Ref(false)
const clip_jac_radius_mult = env_float("TP_CLIP_JAC_RADIUS_MULT", 3.5)
# Print progress during colored/sparse Jacobian fills (color group spam).
const clip_jac_fill_print = env_int("TP_CLIP_JAC_FILL_PRINT", 1) != 0
# Rebuild SPH-locality sparsity pattern + coloring from trial Newton state y before each
# colored Jacobian fill (colored strategy only; sparse CSC structure is fixed at setup).
const clip_jac_pattern_refresh = env_int("TP_CLIP_JAC_PATTERN_REFRESH", 0) != 0
const clip_jac_pattern_refresh_count = Ref(0)
# During colored/sparse Jacobian fill: 0 = freeze NHS at trial state (matches dense
# AutoFiniteDiff); 1 = refresh NHS on every perturbed RHS (more accurate ∂f/∂y but
# inconsistent with frozen-NHS Newton residuals → poor convergence).
const clip_jac_nhs_refresh = env_int("TP_CLIP_JAC_NHS_REFRESH", 0) != 0
# During colored/sparse Jacobian fill: 1 = freeze thermal dT/dt at trial state (matches dense
# AutoFiniteDiff NHS freeze); 0 = recompute contact + SPH thermal on every perturbed RHS.
const clip_jac_thermal_frozen = env_int("TP_CLIP_JAC_THERMAL_FROZEN", 1) != 0
# Optional: cache thermal dT/dt on first RHS at each time `t` (dense Jacobian / Newton).
# Same approximation as frozen NHS at fixed `t`; off by default.
const clip_thermal_same_t_cache = env_int("TP_CLIP_THERMAL_SAME_T_CACHE", 0) != 0
const clip_thermal_jac_cache_valid = Ref(false)
const clip_thermal_same_t_updated_at = Ref(-Inf)

# Environmental/ambient heat loss settings (convection and radiation)
const use_ambient_heat_loss = env_int("TP_CLIP_USE_AMBIENT_HEAT_LOSS", 0) != 0
const temp_ambient = env_temp_c("TP_CLIP_TEMP_AMBIENT", 25.0)       # °C input; stored as K
const h_ambient = env_float("TP_CLIP_H_AMBIENT", 15.0)             # W/(m²·K)
const emissivity_charge = env_float("TP_CLIP_EMISSIVITY", 0.85)     # Emissivity of composite surface
const stefan_boltzmann = 5.670374e-8                              # W/(m²·K⁴)

const charge_boundary_exposure = Float64[]

# During colored/sparse Jacobian fill: freeze tool–charge contact increment (Reimann +
# source terms + WCSPH) at the trial state. Off by default until validated on a run.
const clip_jac_contact_frozen = env_int("TP_CLIP_JAC_CONTACT_FROZEN", 0) != 0
# Optional: reuse first contact increment at each time `t` (dense Jacobian / Newton).
const clip_contact_same_t_cache = env_int("TP_CLIP_CONTACT_SAME_T_CACHE", 0) != 0
const clip_contact_jac_cache_valid = Ref(false)
const clip_contact_same_t_updated_at = Ref(-Inf)

# Reuse finite-difference Jacobian *entries* across OrdinaryDiffEq jac requests.
# ODE already has NLNewton(always_new=false), but stiff contact / adaptive Δt often
# still requests a fresh J every accepted step. When enabled, skip the colored FD
# fill and copy the last nzval cache for up to TP_CLIP_JAC_REUSE_STEPS requests
# (W = γΔt scaling is still rebuilt by OrdinaryDiffEq from the reused J).
const clip_jac_reuse_entries = env_int("TP_CLIP_JAC_REUSE_ENTRIES", 0) != 0
const clip_jac_reuse_max_steps = max(0, env_int("TP_CLIP_JAC_REUSE_STEPS", 10))
const clip_jac_cached_nzval = Ref{Union{Nothing, Vector{Float64}}}(nothing)
const clip_jac_reuse_age = Ref(0)
const clip_jac_reuse_hits = Ref(0)
const clip_jac_reuse_fills = Ref(0)

function clip_reset_jacobian_coloring!()
    clip_jac_coloring_n[] = 0
    clip_jac_sparsity_pattern[] = nothing
    clip_jac_cached_nzval[] = nothing
    clip_jac_reuse_age[] = 0
    clip_jac_reuse_hits[] = 0
    clip_jac_reuse_fills[] = 0
    return nothing
end

function clip_maybe_reuse_sparse_jac_entries!(J)::Bool
    clip_jac_reuse_entries || return false
    clip_jac_reuse_max_steps <= 0 && return false
    cache = clip_jac_cached_nzval[]
    cache === nothing && return false
    length(cache) != length(J.nzval) && return false
    clip_jac_reuse_age[] >= clip_jac_reuse_max_steps && return false
    copyto!(J.nzval, cache)
    clip_jac_reuse_age[] += 1
    clip_jac_reuse_hits[] += 1
    if clip_jac_fill_print
        println(">>> Reusing Sparse Jacobian entries (age=", clip_jac_reuse_age[],
                "/", clip_jac_reuse_max_steps, "; fills=", clip_jac_reuse_fills[],
                ", hits=", clip_jac_reuse_hits[], ")")
        flush(stdout)
    end
    return true
end

function clip_store_sparse_jac_entries!(J)
    clip_jac_reuse_entries || return nothing
    clip_jac_cached_nzval[] = copy(J.nzval)
    clip_jac_reuse_age[] = 0
    clip_jac_reuse_fills[] += 1
    return nothing
end

function clip_precompute_jacobian_coloring!(jac_sparsity; force::Bool=false)
    if !force && clip_jac_coloring_n[] > 0
        return clip_jac_coloring_n[]
    end
    println(">>> Precomputing custom SPH Jacobian coloring...")
    flush(stdout)
    clip_jac_colors[] = clip_compute_colors(jac_sparsity)
    n_colors = maximum(clip_jac_colors[])
    groups = [Int[] for _ in 1:n_colors]
    for col in 1:size(jac_sparsity, 2)
        push!(groups[clip_jac_colors[][col]], col)
    end
    clip_jac_groups[] = groups
    clip_jac_coloring_n[] = n_colors
    println("✓ Precomputed coloring: ", n_colors, " color groups.")
    flush(stdout)
    return n_colors
end

function clip_colored_jacobian_fill!(y, p, t, nv::Int, nu::Int, set_column!)
    groups = clip_jac_groups[]
    n_colors = length(groups)
    n_colors == 0 && error("Jacobian coloring not initialized; call clip_precompute_jacobian_coloring! first")
    prev_jac_fill = clip_in_jacobian_fill[]
    clip_in_jacobian_fill[] = true
    clip_thermal_jac_cache_valid[] = false
    clip_contact_jac_cache_valid[] = false
    try
        f0 = similar(y)
        clip_combined_rhs!(f0, y, p, t, nv, nu)
        y_pert = copy(y)
        f_pert = similar(y)
        epsilon = 1e-7
        jac_progress = max(1, env_int("TP_CLIP_SPARSE_JAC_PROGRESS", 100))
        label = CURRENT_JACOBIAN_STRATEGY[] == "sparse" ? "Sparse" : "Colored dense"

        if clip_jac_fill_print && n_colors > 0
            nhs_mode = clip_jac_nhs_refresh ? "NHS refresh ON" : "NHS frozen (dense-consistent)"
            thermal_mode = clip_jac_thermal_frozen ? "thermal frozen" : "thermal refresh ON"
            contact_mode = clip_jac_contact_frozen ? "contact frozen" : "contact refresh ON"
            println(">>> Filling ", label, " Jacobian (", n_colors, " RHS groups; ",
                    nhs_mode, "; ", thermal_mode, "; ", contact_mode, ")...")
            flush(stdout)
        end

        for c in 1:n_colors
            if clip_jac_fill_print && (c == 1 || c == n_colors || c % jac_progress == 0)
                println(">>> ", label, " Jacobian: color group ", c, "/", n_colors,
                        " (trial_dt=", trial_dt_state[], " s)")
                flush(stdout)
            end
            cols = groups[c]
            for col in cols
                y_pert[col] += epsilon
            end
            clip_combined_rhs!(f_pert, y_pert, p, t, nv, nu)
            for col in cols
                y_pert[col] = y[col]
            end
            for col in cols
                set_column!(col, f0, f_pert, epsilon)
            end
        end
        if clip_jac_fill_print
            println(">>> ", label, " Jacobian fill complete (", n_colors, " color groups).")
            flush(stdout)
        end
    finally
        clip_in_jacobian_fill[] = prev_jac_fill
    end
    return nothing
end

function clip_sparse_jacobian!(J, y, p, t)
    clip_sync_trial_dt_state!()
    if clip_maybe_reuse_sparse_jac_entries!(J)
        println(">>> Factorizing / solving Newton linear system (UMFPACK; J reused)...")
        flush(stdout)
        return nothing
    end
    nv = clip_ode_nv[]
    nu = clip_ode_nu[]
    fill!(J.nzval, 0.0)
    colptr = J.colptr
    rowval = J.rowval
    nzval = J.nzval
    function set_column_sparse!(col, f0, f_pert, epsilon)
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            nzval[idx] = (f_pert[row] - f0[row]) / epsilon
        end
    end
    clip_colored_jacobian_fill!(y, p, t, nv, nu, set_column_sparse!)
    clip_store_sparse_jac_entries!(J)
    println(">>> Factorizing / solving Newton linear system (UMFPACK; J refreshed)...")
    flush(stdout)
    return nothing
end

function clip_maybe_refresh_jacobian_pattern!(y, semi_local, nv::Int, nu::Int)
    clip_jac_pattern_refresh || return nothing
    CURRENT_JACOBIAN_STRATEGY[] == "colored" || return nothing
    u_trial = @view y[nv+1:nv+nu]
    v_trial = @view y[1:nv]
    jac_sparsity = clip_build_sparse_jacobian_prototype(semi_local, v_trial, u_trial;
                                                        quiet=true)
    old_nnz = clip_jac_sparsity_pattern[] === nothing ? 0 : nnz(clip_jac_sparsity_pattern[])
    clip_jac_sparsity_pattern[] = jac_sparsity
    clip_jac_coloring_n[] = 0
    n_colors = clip_precompute_jacobian_coloring!(jac_sparsity; force=true)
    clip_jac_pattern_refresh_count[] += 1
    n = clip_jac_pattern_refresh_count[]
    if n == 1 || n % 20 == 0
        println(">>> Jacobian pattern refreshed from trial y (#", n, "): nnz ",
                old_nnz, " -> ", nnz(jac_sparsity), ", ", n_colors, " color groups.")
        flush(stdout)
    end
    return nothing
end

function clip_colored_dense_jacobian!(J, y, p, t)
    clip_sync_trial_dt_state!()
    nv = clip_ode_nv[]
    nu = clip_ode_nu[]
    clip_maybe_refresh_jacobian_pattern!(y, p, nv, nu)
    sp = clip_jac_sparsity_pattern[]
    sp === nothing && error("Colored dense Jacobian sparsity pattern not initialized")
    fill!(J, 0.0)
    colptr = sp.colptr
    rowval = sp.rowval
    function set_column_dense_pattern!(col, f0, f_pert, epsilon)
        for idx in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[idx]
            J[row, col] = (f_pert[row] - f0[row]) / epsilon
        end
    end
    clip_colored_jacobian_fill!(y, p, t, nv, nu, set_column_dense_pattern!)
    return nothing
end

function clip_compute_colors(jac_prototype)
    n = size(jac_prototype, 2)
    colors = zeros(Int, n)
    col_ptrs = jac_prototype.colptr
    rowval = jac_prototype.rowval
    
    row_to_cols = [Int[] for _ in 1:size(jac_prototype, 1)]
    for col in 1:n
        for r_idx in col_ptrs[col]:(col_ptrs[col+1]-1)
            r = rowval[r_idx]
            push!(row_to_cols[r], col)
        end
    end
    
    for col in 1:n
        neighbor_cols = Int[]
        for r_idx in col_ptrs[col]:(col_ptrs[col+1]-1)
            r = rowval[r_idx]
            append!(neighbor_cols, row_to_cols[r])
        end
        unique!(neighbor_cols)
        
        used_colors = colors[neighbor_cols]
        c = 1
        while c in used_colors
            c += 1
        end
        colors[col] = c
    end
    return colors
end

@inline function clip_jvp_epsilon(y, vdir)
    ny = norm(y)
    nv = norm(vdir)
    nv < eps(eltype(y)) && return sqrt(eps(eltype(y)))
    return sqrt(eps(eltype(y))) * max(1.0, ny) / nv
end

function clip_ensure_jvp_work!(n::Int)
    w = clip_jvp_work[]
    if w === nothing || length(w.y_plus) != n
        clip_jvp_work[] = (y_plus=zeros(n), y_minus=zeros(n), fp=zeros(n), fm=zeros(n))
    end
    return clip_jvp_work[]
end

function clip_combined_rhs!(dy, y, semi_local, t, nv::Int, nu::Int)
    clip_sync_trial_dt_state!()
    v = @view y[1:nv]
    u = @view y[nv+1:nv+nu]
    dv = @view dy[1:nv]
    du = @view dy[nv+1:nv+nu]
    kick_implicit_visible!(dv, v, u, semi_local, t)
    drift_implicit_visible!(du, v, u, semi_local, t)
    return nothing
end

# Reduced implicit Euler state q=[u,T]. Mechanical velocity is eliminated exactly:
# v_{n+1}=(u_{n+1}-u_n)/h and
# u_{n+1}=u_n+h*v_n+h^2*a(u_{n+1},v_{n+1},T_{n+1}).
# Therefore the pseudo-ODE RHS below gives the same ImplicitEuler nonlinear residual
# as the full first-order [v,T,u] system while Newton only sees 3u+1T unknowns.
function clip_reduced_rhs!(dq, q, semi_local, t)
    nu = length(clip_reduced_prev_u[])
    nt = length(clip_reduced_temp_v_idx[])
    length(q) == nu + nt ||
        error("Reduced implicit Euler state size mismatch: $(length(q)) != $(nu + nt)")

    u = @view q[1:nu]
    T = @view q[nu+1:nu+nt]
    du = @view dq[1:nu]
    dT = @view dq[nu+1:nu+nt]
    prev_u = clip_reduced_prev_u[]
    prev_v = clip_reduced_prev_v[]
    v = clip_reduced_v_work[]
    dv = clip_reduced_dv_work[]
    mech_v_idx = clip_reduced_mech_v_idx[]
    temp_v_idx = clip_reduced_temp_v_idx[]

    # For ImplicitEuler the only stage is t_{n+1}. Using t-t_n also tracks rejected
    # step-size changes, unlike a callback-updated dt cache.
    h_stage = t - clip_reduced_prev_t[]
    at_accepted_time = !(isfinite(h_stage) && h_stage > 0.0)
    h = h_stage
    if at_accepted_time
        h = max(abs(trial_dt_state[]), eps(Float64))
    end
    clip_integrator_dt[] = h
    trial_dt_state[] = h

    copyto!(v, prev_v)
    if !at_accepted_time
        @inbounds for i in eachindex(u)
            v[mech_v_idx[i]] = (u[i] - prev_u[i]) / h
        end
    end
    @inbounds for i in eachindex(T)
        v[temp_v_idx[i]] = T[i]
    end

    kick_implicit_visible!(dv, v, u, semi_local, t)
    @inbounds for i in eachindex(u)
        vi = mech_v_idx[i]
        du[i] = prev_v[vi] + h * dv[vi]
    end
    @inbounds for i in eachindex(T)
        dT[i] = dv[temp_v_idx[i]]
    end
    return nothing
end

function clip_reduced_current_rhs!(dq, q, semi_local, t)
    if clip_reduced_trbdf2_active()
        clip_reduced_trbdf2_rhs!(dq, q, semi_local, t)
    else
        clip_reduced_rhs!(dq, q, semi_local, t)
    end
    return nothing
end

function clip_reduced_trbdf2_ensure_an!(semi_local, t_n, h)
    an = clip_reduced_trbdf2_an[]
    (clip_reduced_trbdf2_an_t[] == t_n && clip_reduced_trbdf2_an_h[] == h) && return an
    prev_u = clip_reduced_prev_u[]
    prev_v = clip_reduced_prev_v[]
    v = clip_reduced_v_work[]
    dv = clip_reduced_dv_work[]
    mech_v_idx = clip_reduced_mech_v_idx[]
    copyto!(v, prev_v)
    kick_implicit_visible!(dv, v, prev_u, semi_local, t_n)
    @inbounds for i in eachindex(an)
        an[i] = dv[mech_v_idx[i]]
    end
    clip_reduced_trbdf2_an_t[] = t_n
    clip_reduced_trbdf2_an_h[] = h
    return an
end

# Reduced TRBDF2 state q=[u,T]. Stock OrdinaryDiffEq TRBDF2 is unchanged; this
# pseudo-RHS makes its two ESDIRK stages equivalent to TRBDF2 on [v,T,u] after
# eliminating mechanical velocity. Stage-2/3 u-residuals compensate whatever
# value is currently in integrator.fsalfirst (dt-dependent FSAL).
function clip_reduced_trbdf2_rhs!(dq, q, semi_local, t)
    nu = length(clip_reduced_prev_u[])
    nt = length(clip_reduced_temp_v_idx[])
    length(q) == nu + nt ||
        error("Reduced TRBDF2 state size mismatch: $(length(q)) != $(nu + nt)")

    u = @view q[1:nu]
    T = @view q[nu+1:nu+nt]
    du = @view dq[1:nu]
    dT = @view dq[nu+1:nu+nt]
    prev_u = clip_reduced_prev_u[]
    prev_v = clip_reduced_prev_v[]
    v = clip_reduced_v_work[]
    dv = clip_reduced_dv_work[]
    mech_v_idx = clip_reduced_mech_v_idx[]
    temp_v_idx = clip_reduced_temp_v_idx[]
    t_n = clip_reduced_prev_t[]
    Δt = t - t_n
    dtab = CLIP_TRBDF2_d
    ωtab = CLIP_TRBDF2_ω
    γtab = CLIP_TRBDF2_γ

    at_accepted_time = !(isfinite(Δt) && Δt > 0.0)
    h_cached = clip_reduced_trbdf2_h_step[]
    h = if at_accepted_time
        max(abs(clip_integrator_dt[]), abs(trial_dt_state[]),
            isfinite(h_cached) ? abs(h_cached) : 0.0, eps(Float64))
    elseif !clip_reduced_trbdf2_has_γ[]
        # First implicit node of this step is always trapezoid: t_n + γ h.
        Δt / γtab
    elseif isfinite(h_cached) && h_cached > 0.0
        h_cached
    else
        Δt
    end
    if !at_accepted_time && !clip_reduced_trbdf2_has_γ[]
        clip_reduced_trbdf2_h_step[] = h
    end
    clip_integrator_dt[] = h
    trial_dt_state[] = h
    dh = dtab * h

    copyto!(v, prev_v)
    if at_accepted_time
        @inbounds for i in eachindex(T)
            v[temp_v_idx[i]] = T[i]
        end
        kick_implicit_visible!(dv, v, u, semi_local, t)
        @inbounds for i in eachindex(u)
            vi = mech_v_idx[i]
            du[i] = prev_v[vi] + dh * dv[vi]
        end
        @inbounds for i in eachindex(T)
            dT[i] = dv[temp_v_idx[i]]
        end
        return nothing
    end

    # Jacobian fills at the trapezoid node reuse this residual after has_γ is set.
    # Classify those by Δt vs this step's h, not by the possibly stale integrator dt.
    stage_γ = !clip_reduced_trbdf2_has_γ[] || (Δt / h) < (γtab + 1.0) / 2.0
    if stage_γ
        an = clip_reduced_trbdf2_ensure_an!(semi_local, t_n, h)
        @inbounds for i in eachindex(u)
            v[mech_v_idx[i]] = (u[i] - prev_u[i]) / dh - prev_v[mech_v_idx[i]]
        end
        @inbounds for i in eachindex(T)
            v[temp_v_idx[i]] = T[i]
        end
        kick_implicit_visible!(dv, v, u, semi_local, t)
        fs = clip_reduced_trbdf2_fsalfirst[]
        @inbounds for i in eachindex(u)
            vi = mech_v_idx[i]
            fn_i = if fs !== nothing && length(fs) >= i
                fs[i]
            else
                prev_v[vi] + dh * an[i]
            end
            du[i] = 2.0 * prev_v[vi] + dh * (an[i] + dv[vi]) - fn_i
        end
        @inbounds for i in eachindex(T)
            dT[i] = dv[temp_v_idx[i]]
        end
        if !clip_reduced_trbdf2_freeze_stage_cache[]
            vγ = clip_reduced_trbdf2_vγ[]
            aγ = clip_reduced_trbdf2_aγ[]
            fγ = clip_reduced_trbdf2_fγ[]
            @inbounds for i in eachindex(u)
                vi = mech_v_idx[i]
                vγ[i] = v[vi]
                aγ[i] = dv[vi]
                fγ[i] = du[i]
            end
            clip_reduced_trbdf2_has_γ[] = true
        end
        return nothing
    end

    clip_reduced_trbdf2_has_γ[] ||
        error("Reduced TRBDF2 BDF2 stage evaluated before trapezoid stage cache")
    an = clip_reduced_trbdf2_ensure_an!(semi_local, t_n, h)
    vγ = clip_reduced_trbdf2_vγ[]
    aγ = clip_reduced_trbdf2_aγ[]
    fγ = clip_reduced_trbdf2_fγ[]
    ωh = ωtab * h
    @inbounds for i in eachindex(u)
        vi = mech_v_idx[i]
        v[vi] = (u[i] - prev_u[i] - ωh * prev_v[vi] - ωh * vγ[i]) / dh
    end
    @inbounds for i in eachindex(T)
        v[temp_v_idx[i]] = T[i]
    end
    kick_implicit_visible!(dv, v, u, semi_local, t)
    fs = clip_reduced_trbdf2_fsalfirst[]
    ω_over_d = ωtab / dtab
    @inbounds for i in eachindex(u)
        vi = mech_v_idx[i]
        fn_i = if fs !== nothing && length(fs) >= i
            fs[i]
        else
            prev_v[vi] + dh * an[i]
        end
        Δu = (ωtab + dtab) * h * prev_v[vi] + ωh * vγ[i] +
             ωtab * dtab * h * h * (an[i] + aγ[i]) + dh * dh * dv[vi]
        du[i] = Δu / dh - ω_over_d * (fn_i + fγ[i])
    end
    @inbounds for i in eachindex(T)
        dT[i] = dv[temp_v_idx[i]]
    end
    return nothing
end

function clip_combined_jvp!(Jv, y, vdir, semi_local, t, nv::Int, nu::Int)
    w = clip_ensure_jvp_work!(length(y))
    ε = clip_jvp_epsilon(y, vdir)
    @. w.y_plus = y + ε * vdir
    @. w.y_minus = y - ε * vdir
    clip_combined_rhs!(w.fp, w.y_plus, semi_local, t, nv, nu)
    clip_combined_rhs!(w.fm, w.y_minus, semi_local, t, nv, nu)
    @. Jv = (w.fp - w.fm) / (2ε)
    return nothing
end

function clip_integrator_vu(integrator)
    if clip_ode_reduced[]
        q = integrator.u
        nu = length(clip_reduced_prev_u[])
        u = @view q[1:nu]
        if clip_reduced_accepted_t[] != integrator.t
            v = clip_reduced_accepted_v[]
            copyto!(v, clip_reduced_prev_v[])
            h = integrator.t - clip_reduced_prev_t[]
            if isfinite(h) && h > 0.0
                if clip_reduced_trbdf2_active() && clip_reduced_trbdf2_has_γ[]
                    dtab = CLIP_TRBDF2_d
                    ωtab = CLIP_TRBDF2_ω
                    dh = dtab * h
                    ωh = ωtab * h
                    vγ = clip_reduced_trbdf2_vγ[]
                    @inbounds for i in eachindex(u)
                        vi = clip_reduced_mech_v_idx[][i]
                        v[vi] = (u[i] - clip_reduced_prev_u[][i] -
                                 ωh * clip_reduced_prev_v[][vi] - ωh * vγ[i]) / dh
                    end
                else
                    @inbounds for i in eachindex(u)
                        v[clip_reduced_mech_v_idx[][i]] =
                            (u[i] - clip_reduced_prev_u[][i]) / h
                    end
                end
            end
            @inbounds for i in eachindex(clip_reduced_temp_v_idx[])
                v[clip_reduced_temp_v_idx[][i]] = q[nu + i]
            end
            clip_reduced_accepted_t[] = integrator.t
        end
        return clip_reduced_accepted_v[], u
    end
    if clip_ode_flat[]
        nv = clip_ode_nv[]
        nu = clip_ode_nu[]
        u = integrator.u
        return @view(u[1:nv]), @view(u[nv+1:nv+nu])
    end
    return integrator.u.x[1], integrator.u.x[2]
end

function clip_integrator_vu_partition(integrator)
    if clip_ode_reduced[]
        v, u = clip_integrator_vu(integrator)
        return ArrayPartition(v, u)
    end
    if clip_ode_flat[]
        nv = clip_ode_nv[]
        u = integrator.u
        return ArrayPartition(@view(u[1:nv]), @view(u[nv+1:end]))
    end
    return integrator.u
end

function clip_commit_reduced_accepted_state!(integrator, v_ode_state, u_ode_state)
    clip_ode_reduced[] || return nothing
    nu = length(clip_reduced_prev_u[])
    q = integrator.u
    # Accepted-step callbacks may clamp velocities or update temperature. Persist those
    # changes before the next reduced Newton solve.
    @inbounds for i in eachindex(clip_reduced_temp_v_idx[])
        q[nu + i] = v_ode_state[clip_reduced_temp_v_idx[][i]]
    end
    copyto!(clip_reduced_prev_u[], u_ode_state)
    copyto!(clip_reduced_prev_v[], v_ode_state)
    clip_reduced_prev_t[] = integrator.t
    clip_reduced_accepted_t[] = -Inf
    clip_reduced_trbdf2_has_γ[] = false
    clip_reduced_trbdf2_an_h[] = NaN
    if clip_reduced_trbdf2_active()
        clip_reduced_trbdf2_refresh_fsalfirst!(integrator)
    end
    return nothing
end

# Positions for SPH-locality Jacobian sparsity (current ODE state, not reference mesh).
function clip_jacobian_pattern_coords(semi, u0_ode)
    systems = semi.systems
    ranges_u = semi.ranges_u
    coords = Vector{Matrix{Float64}}(undef, length(systems))
    for (idx, sys) in enumerate(systems)
        n = TrixiParticles.n_integrated_particles(sys)
        nd = TrixiParticles.ndims(sys)
        range_u = ranges_u[idx]
        if n == 0 || isempty(range_u)
            coords[idx] = Matrix(@view sys.current_coordinates[1:nd, 1:n])
        else
            u_sys = TrixiParticles.wrap_u(u0_ode, sys, semi)
            mat = Matrix{Float64}(undef, nd, n)
            @inbounds for p in 1:n, d in 1:nd
                mat[d, p] = u_sys[d, p]
            end
            coords[idx] = mat
        end
    end
    return coords
end

function clip_build_sparse_jacobian_prototype(semi, v0_ode, u0_ode; D_max=0.006, quiet::Bool=false)
    nv = length(v0_ode)
    nu = length(u0_ode)
    n_total = nv + nu

    I_rows = Int[]
    J_cols = Int[]
    
    # 1. Coordinate equations: du/dt = v
    for i in 1:nu
        push!(I_rows, nv + i)
        push!(J_cols, i)
    end
    
    # 2. Velocity equations: dv/dt = f(v, u)
    systems = semi.systems
    ranges_v = semi.ranges_v
    ranges_u = semi.ranges_u
    
    coords_state = clip_jacobian_pattern_coords(semi, u0_ode)
    if !quiet
        println(">>> Jacobian sparsity pattern: neighbor graph from current ODE positions ",
                "(not initial/reference mesh).")
        flush(stdout)
    end
    
    for (A_idx, sys_A) in enumerate(systems)
        range_v_A = ranges_v[A_idx]
        range_u_A = ranges_u[A_idx]
        isempty(range_v_A) && continue
        N_A = TrixiParticles.n_integrated_particles(sys_A)
        v_stride_A = TrixiParticles.v_nvariables(sys_A)
        u_stride_A = TrixiParticles.u_nvariables(sys_A)
        h_A = TrixiParticles.initial_smoothing_length(sys_A)
        
        for i in 1:N_A
            pos_i = coords_state[A_idx][:, i]
            v_i_idxs = [range_v_A[1] + d - 1 + v_stride_A * (i - 1) for d in 1:v_stride_A]
            u_i_idxs = [nv + range_u_A[1] + d - 1 + u_stride_A * (i - 1) for d in 1:u_stride_A]
            
            # Self coupling
            for r in v_i_idxs
                for c in v_i_idxs
                    push!(I_rows, r)
                    push!(J_cols, c)
                end
                for c in u_i_idxs
                    push!(I_rows, r)
                    push!(J_cols, c)
                end
            end
            
            # Neighbors and cross coupling
            for (B_idx, sys_B) in enumerate(systems)
                range_v_B = ranges_v[B_idx]
                range_u_B = ranges_u[B_idx]
                isempty(range_v_B) && continue
                N_B = TrixiParticles.n_integrated_particles(sys_B)
                v_stride_B = TrixiParticles.v_nvariables(sys_B)
                u_stride_B = TrixiParticles.u_nvariables(sys_B)
                h_B = TrixiParticles.initial_smoothing_length(sys_B)
                h_contact = max(h_A, h_B)
                
                for j in 1:N_B
                    if A_idx == B_idx && i == j
                        continue
                    end
                    
                    pos_j = coords_state[B_idx][:, j]
                    dist = norm(pos_i - pos_j)
                    
                    if dist <= clip_jac_radius_mult * h_contact
                        v_j_idxs = [range_v_B[1] + d - 1 + v_stride_B * (j - 1) for d in 1:v_stride_B]
                        u_j_idxs = [nv + range_u_B[1] + d - 1 + u_stride_B * (j - 1) for d in 1:u_stride_B]
                        
                        for r in v_i_idxs
                            for c in v_j_idxs
                                push!(I_rows, r)
                                push!(J_cols, c)
                            end
                            for c in u_j_idxs
                                push!(I_rows, r)
                                push!(J_cols, c)
                            end
                        end
                    end
                end
            end
        end
    end
    
    if !quiet
        println(">>> Debug Sparse: nv=", nv, " nu=", nu, " n_total=", n_total,
                " jac_radius_mult=", clip_jac_radius_mult)
        println(">>> Debug Sparse: I_rows bounds=[", minimum(I_rows), ", ", maximum(I_rows), "]")
        println(">>> Debug Sparse: J_cols bounds=[", minimum(J_cols), ", ", maximum(J_cols), "]")
        flush(stdout)
    end
    return sparse(I_rows, J_cols, ones(length(I_rows)), n_total, n_total)
end

function clip_initialize_reduced_state!(semi, v0_ode, u0_ode, t0)
    nu = length(u0_ode)
    mech_v_idx = zeros(Int, nu)
    temp_v_idx = Int[]
    temp_ranges = UnitRange{Int}[]
    temp_offset = 0

    for (idx, system) in enumerate(semi.systems)
        n = TrixiParticles.n_integrated_particles(system)
        rv = semi.ranges_v[idx]
        ru = semi.ranges_u[idx]
        if n == 0 || isempty(rv) || isempty(ru)
            push!(temp_ranges, (temp_offset + 1):temp_offset)
            continue
        end
        nd = TrixiParticles.ndims(system)
        vs = TrixiParticles.v_nvariables(system)
        us = TrixiParticles.u_nvariables(system)
        us == nd || error("Reduced implicit Euler requires u_nvariables == ndims")
        vs >= nd + 1 ||
            error("Reduced implicit Euler requires a temperature row in the v block")
        first_temp = temp_offset + 1
        @inbounds for particle in 1:n
            for d in 1:nd
                ui = first(ru) + d - 1 + us * (particle - 1)
                vi = first(rv) + d - 1 + vs * (particle - 1)
                mech_v_idx[ui] = vi
            end
            push!(temp_v_idx, first(rv) + nd + vs * (particle - 1))
            temp_offset += 1
        end
        push!(temp_ranges, first_temp:temp_offset)
    end
    any(iszero, mech_v_idx) &&
        error("Reduced implicit Euler could not map every displacement DOF to velocity")

    clip_ode_reduced[] = true
    clip_ode_flat[] = false
    clip_ode_nv[] = length(v0_ode)
    clip_ode_nu[] = nu
    clip_reduced_prev_t[] = Float64(t0)
    clip_reduced_prev_u[] = Vector{Float64}(u0_ode)
    clip_reduced_prev_v[] = Vector{Float64}(v0_ode)
    clip_reduced_mech_v_idx[] = mech_v_idx
    clip_reduced_temp_v_idx[] = temp_v_idx
    clip_reduced_temp_ranges[] = temp_ranges
    clip_reduced_v_work[] = similar(clip_reduced_prev_v[])
    clip_reduced_dv_work[] = similar(clip_reduced_prev_v[])
    clip_reduced_accepted_v[] = similar(clip_reduced_prev_v[])
    clip_reduced_accepted_t[] = -Inf
    clip_reduced_trbdf2_an[] = zeros(nu)
    clip_reduced_trbdf2_aγ[] = zeros(nu)
    clip_reduced_trbdf2_vγ[] = zeros(nu)
    clip_reduced_trbdf2_fγ[] = zeros(nu)
    clip_reduced_trbdf2_an_t[] = -Inf
    clip_reduced_trbdf2_an_h[] = NaN
    clip_reduced_trbdf2_h_step[] = NaN
    clip_reduced_trbdf2_has_γ[] = false
    clip_reduced_trbdf2_freeze_stage_cache[] = false
    clip_reduced_trbdf2_fsalfirst[] = nothing

    q0 = Vector{Float64}(undef, nu + length(temp_v_idx))
    copyto!(@view(q0[1:nu]), u0_ode)
    @inbounds for i in eachindex(temp_v_idx)
        q0[nu + i] = v0_ode[temp_v_idx[i]]
    end
    return q0
end

function clip_build_reduced_sparse_jacobian_prototype(semi, u0_ode)
    nu = length(u0_ode)
    nt = length(clip_reduced_temp_v_idx[])
    nq = nu + nt
    I_rows = Int[]
    J_cols = Int[]
    coords_state = clip_jacobian_pattern_coords(semi, u0_ode)
    temp_ranges = clip_reduced_temp_ranges[]

    println(">>> Reduced Jacobian sparsity: local [u,T] neighbor graph.")
    flush(stdout)
    for (A_idx, sys_A) in enumerate(semi.systems)
        ru_A = semi.ranges_u[A_idx]
        n_A = TrixiParticles.n_integrated_particles(sys_A)
        (n_A == 0 || isempty(ru_A)) && continue
        us_A = TrixiParticles.u_nvariables(sys_A)
        h_A = TrixiParticles.initial_smoothing_length(sys_A)
        tr_A = temp_ranges[A_idx]

        for i in 1:n_A
            pos_i = coords_state[A_idx][:, i]
            rows = [first(ru_A) + d - 1 + us_A * (i - 1) for d in 1:us_A]
            push!(rows, nu + first(tr_A) + i - 1)

            for (B_idx, sys_B) in enumerate(semi.systems)
                ru_B = semi.ranges_u[B_idx]
                n_B = TrixiParticles.n_integrated_particles(sys_B)
                (n_B == 0 || isempty(ru_B)) && continue
                us_B = TrixiParticles.u_nvariables(sys_B)
                h_B = TrixiParticles.initial_smoothing_length(sys_B)
                tr_B = temp_ranges[B_idx]
                h_contact = max(h_A, h_B)
                for j in 1:n_B
                    pos_j = coords_state[B_idx][:, j]
                    if (A_idx == B_idx && i == j) ||
                       norm(pos_i - pos_j) <= clip_jac_radius_mult * h_contact
                        cols = [first(ru_B) + d - 1 + us_B * (j - 1)
                                for d in 1:us_B]
                        push!(cols, nu + first(tr_B) + j - 1)
                        for r in rows, c in cols
                            push!(I_rows, r)
                            push!(J_cols, c)
                        end
                    end
                end
            end
        end
    end
    return sparse(I_rows, J_cols, ones(length(I_rows)), nq, nq)
end

function clip_reduced_sparse_jacobian!(J, q, p, t)
    if clip_maybe_reuse_sparse_jac_entries!(J)
        println(">>> Factorizing / solving reduced Newton system (UMFPACK; J reused)...")
        flush(stdout)
        return nothing
    end
    fill!(J.nzval, 0.0)
    groups = clip_jac_groups[]
    f0 = similar(q)
    fpert = similar(q)
    qpert = copy(q)
    clip_in_jacobian_fill[] = true
    clip_thermal_jac_cache_valid[] = false
    clip_contact_jac_cache_valid[] = false
    try
        clip_reduced_current_rhs!(f0, q, p, t)
        epsilon = 1.0e-7
        clip_reduced_trbdf2_freeze_stage_cache[] = true
        for cols in groups
            @inbounds for col in cols
                qpert[col] += epsilon
            end
            clip_reduced_current_rhs!(fpert, qpert, p, t)
            @inbounds for col in cols
                qpert[col] = q[col]
                for ptr in J.colptr[col]:(J.colptr[col + 1] - 1)
                    row = J.rowval[ptr]
                    J.nzval[ptr] = (fpert[row] - f0[row]) / epsilon
                end
            end
        end
    finally
        clip_reduced_trbdf2_freeze_stage_cache[] = false
        clip_in_jacobian_fill[] = false
    end
    clip_store_sparse_jac_entries!(J)
    println(">>> Factorizing / solving reduced Newton system (UMFPACK; J refreshed)...")
    flush(stdout)
    return nothing
end

function clip_build_ode_problem(v0_ode, u0_ode, tspan, semi_local;
                                jacobian_strategy::String=clip_jacobian_strategy())
    CURRENT_JACOBIAN_STRATEGY[] = jacobian_strategy
    if clip_reduced_ode_active()
        jacobian_strategy == "sparse" ||
            error("reduced_implicit_euler / reduced_trbdf2 currently require TP_CLIP_JACOBIAN_STRATEGY=sparse")
        q0 = clip_initialize_reduced_state!(semi_local, v0_ode, u0_ode, first(tspan))
        clip_reset_jacobian_coloring!()
        jac_sparsity = clip_build_reduced_sparse_jacobian_prototype(semi_local, u0_ode)
        n_colors = clip_precompute_jacobian_coloring!(jac_sparsity)
        println("✓ Reduced sparse Jacobian size $(size(jac_sparsity)), ",
                "$(nnz(jac_sparsity)) nonzeros, ", n_colors, " color groups ",
                "(full state $(length(v0_ode) + length(u0_ode)) → reduced $(length(q0)) DOF).")
        flush(stdout)
        f = ODEFunction(clip_reduced_current_rhs!;
                        jac=clip_reduced_sparse_jacobian!,
                        jac_prototype=jac_sparsity)
        return ODEProblem(f, q0, tspan, semi_local)
    elseif jacobian_strategy == "matrix_free"
        clip_ode_reduced[] = false
        nv = length(v0_ode)
        nu = length(u0_ode)
        clip_ode_flat[] = true
        clip_ode_nv[] = nv
        clip_ode_nu[] = nu
        y0 = vcat(v0_ode, u0_ode)
        rhs_local = (dy, y, p, t) -> clip_combined_rhs!(dy, y, p, t, nv, nu)
        jvp_local = (Jv, y, vdir, p, t) -> clip_combined_jvp!(Jv, y, vdir, p, t, nv, nu)
        f = ODEFunction(rhs_local; jvp=jvp_local)
        return ODEProblem(f, y0, tspan, semi_local)
    elseif jacobian_strategy == "sparse"
        clip_ode_reduced[] = false
        nv = length(v0_ode)
        nu = length(u0_ode)
        clip_ode_flat[] = true
        clip_ode_nv[] = nv
        clip_ode_nu[] = nu
        y0 = vcat(v0_ode, u0_ode)
        rhs_local = (dy, y, p, t) -> clip_combined_rhs!(dy, y, p, t, nv, nu)
        println(">>> Building SPH locality-based Sparse Jacobian prototype...")
        flush(stdout)
        clip_reset_jacobian_coloring!()
        jac_sparsity = clip_build_sparse_jacobian_prototype(semi_local, v0_ode, u0_ode)
        n_colors = clip_precompute_jacobian_coloring!(jac_sparsity)
        println("✓ Sparse Jacobian prototype built with size $(size(jac_sparsity)) and ",
                "$(nnz(jac_sparsity)) non-zero entries (", n_colors, " color groups).")
        flush(stdout)
        f = ODEFunction(rhs_local; jac=clip_sparse_jacobian!, jac_prototype=jac_sparsity)
        return ODEProblem(f, y0, tspan, semi_local)
    elseif jacobian_strategy == "colored"
        clip_ode_reduced[] = false
        nv = length(v0_ode)
        nu = length(u0_ode)
        n_total = nv + nu
        clip_ode_flat[] = true
        clip_ode_nv[] = nv
        clip_ode_nu[] = nu
        y0 = vcat(v0_ode, u0_ode)
        rhs_local = (dy, y, p, t) -> clip_combined_rhs!(dy, y, p, t, nv, nu)
        println(">>> Building SPH locality sparsity pattern for colored dense Jacobian...")
        flush(stdout)
        clip_reset_jacobian_coloring!()
        jac_sparsity = clip_build_sparse_jacobian_prototype(semi_local, v0_ode, u0_ode)
        clip_jac_sparsity_pattern[] = jac_sparsity
        n_colors = clip_precompute_jacobian_coloring!(jac_sparsity)
        jac_prototype = zeros(Float64, n_total, n_total)
        println("✓ Colored dense Jacobian: ", n_colors, " RHS color groups, ",
                "pattern nnz=", nnz(jac_sparsity), ", radius_mult=", clip_jac_radius_mult,
                ", dense storage (", n_total, "×", n_total, "), linear solve = dense LU",
                clip_jac_pattern_refresh ?
                    ", pattern refresh ON (rebuild from trial y each fill)" : ".")
        flush(stdout)
        f = ODEFunction(rhs_local; jac=clip_colored_dense_jacobian!, jac_prototype=jac_prototype)
        return ODEProblem(f, y0, tspan, semi_local)
    end
    clip_ode_reduced[] = false
    clip_ode_flat[] = false
    return DynamicalODEProblem(kick_implicit_visible!, drift_implicit_visible!,
                               v0_ode, u0_ode, tspan, semi_local)
end

function clip_build_implicit_algorithm(integrator_mode::String;
                                       linsolve_solver,
                                       use_matrix_free::Bool,
                                       jacobian_autodiff,
                                       nlsolve=nothing)
    autodiff_arg = (use_matrix_free || jacobian_autodiff === nothing) ? false : jacobian_autodiff
    nlsolve_kw = nlsolve === nothing ? () : (; nlsolve)
    if integrator_mode in ("implicit_euler", "reduced_implicit_euler")
        return ImplicitEuler(; linsolve=linsolve_solver, autodiff=autodiff_arg, nlsolve_kw...)
    elseif integrator_mode == "rodas5"
        # Rosenbrock methods do not use Newton's method; OrdinaryDiffEq Rodas5 does not accept `nlsolve`.
        return Rodas5(; linsolve=linsolve_solver, autodiff=autodiff_arg)
    elseif integrator_mode == "explicit"
        return CarpenterKennedy2N54()
    else
        return TRBDF2(; linsolve=linsolve_solver, autodiff=autodiff_arg, nlsolve_kw...)
    end
end

# Simulation phase:
#   full       — compression + hold + retraction (default)
#   hold       — compression + hold, save checkpoint, stop before retraction
#   retraction — load checkpoint, run retraction only
const separate_charge_plies = env_int("TP_CLIP_SEPARATE_CHARGE_PLIES", 0) != 0
# Legacy packed-charge files remain v2 byte-for-byte. A future multi-charge
# checkpoint must identify the per-ply layout explicitly as v3.
const CHECKPOINT_VERSION = separate_charge_plies ? 3 : 2
const sim_phase = lowercase(get(ENV, "TP_CLIP_SIM_PHASE", "hold"))
sim_phase in ("full", "hold", "retraction") ||
    error("TP_CLIP_SIM_PHASE must be full, hold, or retraction (got $(sim_phase))")
# SPH flyaways stay in the checkpoint/ODE layout but are frozen and skipped in
# tool contact, peel gaps, span, and J-floor. Default ON for retraction.
const ignore_charge_flyaways = env_int("TP_CLIP_IGNORE_FLYAWAWAYS",
    sim_phase == "retraction" ? 1 : 0) != 0
const charge_flyaway_count = Ref(0)
if separate_charge_plies && sim_phase != "hold"
    error("TP_CLIP_SEPARATE_CHARGE_PLIES=1 is currently restricted to ongoing " *
          "CPU compression+hold integration work. TP_CLIP_SIM_PHASE=$(sim_phase) is " *
          "explicitly unsupported; separate-ply checkpoint restore and retraction " *
          "have not been converted.")
end
# Geometry: pseudo2d (1 y-layer), thin3d (few y-layers, plane constraints), full3d (resolved y).
function clip_geometry_mode()
    mode = if haskey(ENV, "TP_CLIP_GEOMETRY")
        lowercase(ENV["TP_CLIP_GEOMETRY"])
    elseif env_int("TP_CLIP_CHARGE_LAYERS_Y", 1) > 1
        "thin3d"
    else
        "pseudo2d"
    end
    mode in ("pseudo2d", "thin3d", "full3d") ||
        error("TP_CLIP_GEOMETRY must be 'pseudo2d', 'thin3d', or 'full3d' (got $(mode))")
    return Symbol(mode)
end
const geometry_mode = clip_geometry_mode()
const geometry_name_tag = geometry_mode == :full3d ? "3d" : "pseudo2d"
# Run tag from particle spacing [mm]: 1.8 -> "18", 2.6 -> "26" (used in VTU prefix + checkpoint name).
const ps_mm_nominal = env_float("TP_CLIP_PS_MM", 1.8)
@inline function ps_mm_run_tag(ps_mm::Real)
    return string(round(Int, round(Float64(ps_mm) * 10)))
end
const ps_run_tag = ps_mm_run_tag(ps_mm_nominal)
# Bulk → out/bulk, sheet → out/sheet (launchers set TP_CLIP_OUT_DIR). Default remains out/.
const clip_out_dir = get(ENV, "TP_CLIP_OUT_DIR", "out")
const save_checkpoint_at_hold = env_int("TP_CLIP_SAVE_CHECKPOINT", 1) != 0
const checkpoint_path = get(ENV, "TP_CLIP_CHECKPOINT",
                            joinpath(clip_out_dir,
                                     "molding_cfk_" * geometry_name_tag * "_hold_checkpoint_" *
                                     ps_run_tag * ".dat"))
# Permanent pre-eject snapshot (in-die cool done). Reuse this to skip creep/lift/cool.
const retraction_preeject_checkpoint_path = get(ENV, "TP_CLIP_PREEJECT_CHECKPOINT",
    replace(checkpoint_path, r"\.dat$" => "_preeject.dat"))
retraction_from_preeject = Ref(false)
# Optional in-die cool snapshot (no eject). Empty = do not write.
const retraction_cool_end_checkpoint_path = strip(get(ENV, "TP_CLIP_COOL_END_CHECKPOINT", ""))
# Checkpoint A: closed-die cool to target, punch still down.
const retraction_closed30_checkpoint_path = strip(get(ENV, "TP_CLIP_CLOSED30_CHECKPOINT", ""))
# Checkpoint B: constrained settle at target T, punch still down, shape OK.
const retraction_closed30_settle_checkpoint_path =
    strip(get(ENV, "TP_CLIP_CLOSED30_SETTLE_CHECKPOINT", ""))
const retraction_closed30_settle_s =
    max(env_float("TP_CLIP_RETRACTION_CLOSED30_SETTLE_S", 0.0), 0.0)
retraction_closed30_ckpt_saved = Ref(false)
retraction_closed30_settle_start_t = Ref(Inf)
# TLSPH equilibration after solidification, before checkpoint (WCSPH off during tail).
# Stress relaxation at hold temperature while the die stays closed (thermoplastic creep /
# contact redistribution). Default 0.15 s — increase if checkpoint still shows tool overlap.
const hold_tlsph_tail_s = env_float("TP_CLIP_HOLD_TLSPH_TAIL_S", 0.0)
# Stress relaxation rate stopping condition during closed-die hold:
# Stops hold when ||dσ/dt|| drops below threshold [Pa/s] (e.g. 1.0e5 = 0.1 MPa/s).
const hold_stress_rate_threshold = env_float("TP_CLIP_HOLD_STRESS_RATE_THRESHOLD", NaN)
const hold_tlsph_tail_min_s = env_float("TP_CLIP_HOLD_TLSPH_TAIL_MIN_S", 0.0002)
const hold_tlsph_tail_ramp_s = env_float("TP_CLIP_HOLD_TLSPH_TAIL_RAMP_S", 0.0001)
const hold_stress_rate_current = Ref(Inf)
const hold_stress_rate_converged = Ref(false)
const hold_prev_stress_mean = Ref(NaN)
const hold_prev_stress_t = Ref(NaN)
# Hold end: Spencer fibre reference n_ref ← F⁻¹·ê so I4=|F·n_ref|²=1 at formed F (stress-free fibres at demold).
const hold_end_spencer_align = env_int("TP_CLIP_HOLD_END_SPENCER_ALIGN", 1) != 0
# Strategy E: volumetric plasticity J_p with isochoric F_p. det(F) = J_e_vol * J_p; bulk uses J_e_vol vs 1.
const use_volumetric_plasticity = env_int("TP_CLIP_VOLUMETRIC_PLASTICITY", 1) != 0
const hold_end_fp_reset = env_int("TP_CLIP_HOLD_END_FP_RESET", use_volumetric_plasticity ? 0 : 1) != 0
# Floor det(F) at hold-end / retraction start so crushed J (≪1) does not inject GPa bulk stress.
# Default ON for compress/hold/full/retraction (0.15). Set TP_CLIP_HOLD_END_J_FLOOR=0 to disable.
const hold_end_J_floor = sim_phase in ("retraction", "hold", "full", "compression") ?
    env_float("TP_CLIP_HOLD_END_J_FLOOR", 0.15) : 0.0
# Mid-compression: after each accepted step with t < t_compress, raise/reset det(F)
# so J_min stays ≥ floor (inversions det≤0 → I). Default ON (0.15).
# Set TP_CLIP_COMPRESS_J_FLOOR=0 to disable. Applied only in the accepted-step callback
# (not inside RHS) so Newton FD Jacobians stay clean.
const compress_J_floor = env_float("TP_CLIP_COMPRESS_J_FLOOR", 0.15)
# Retraction: same floor on every accepted step (default = hold-end floor). Prevents
# pre-lift creep from re-crushing det(F) after hold-end prep. Set
# TP_CLIP_RETRACTION_J_FLOOR=0 to disable continuous flooring (hold-end prep still runs).
const retraction_J_floor = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_J_FLOOR", hold_end_J_floor > 0.0 ? hold_end_J_floor : 0.15) :
    0.0
const hold_end_fp_repair = env_int("TP_CLIP_HOLD_END_FP_REPAIR", 1) != 0
const checkpoint_finalize_on_save = sim_phase in ("hold", "full") &&
    env_int("TP_CLIP_CHECKPOINT_FINALIZE_ON_SAVE", 1) != 0
const checkpoint_audit_on_save = sim_phase in ("hold", "full") &&
    env_int("TP_CLIP_CHECKPOINT_AUDIT", 1) != 0
const checkpoint_audit_strict = env_int("TP_CLIP_CHECKPOINT_AUDIT_STRICT", 0) != 0
const retraction_prepare_state = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_PREP_STATE", 0) != 0
const volumetric_plastic_relax_rate = env_float("TP_CLIP_JP_RELAX_RATE", 1.0)
const elastic_bulk_J_ref = 1.0
# After punch lift: continue mechanics until springback equilibrates (mold gap-based contact already off).
const retraction_springback_dwell_s = env_float("TP_CLIP_RETRACTION_SPRINGBACK_DWELL_S", 0.05)
const retraction_springback_equil_max_speed = env_float("TP_CLIP_RETRACTION_SPRINGBACK_EQUIL_MAX_SPEED", 0.05)
const retraction_springback_equil_span_rate_mm_s =
    env_float("TP_CLIP_RETRACTION_SPRINGBACK_EQUIL_SPAN_RATE_MM_S", 0.02)
const retraction_springback_equil_patience =
    max(1, env_int("TP_CLIP_RETRACTION_SPRINGBACK_EQUIL_PATIENCE", 3))
# Free cool after eject: keep springback dwell until mean charge T ≤ this °C (NaN = off).
# Requires TP_CLIP_RETRACTION_EXPLICIT_FREEZE_T=0 and TP_CLIP_USE_AMBIENT_HEAT_LOSS=1.
# Mechanical equilibration alone will not end the run while cool-until is active.
const retraction_cool_until_temp_C = env_float("TP_CLIP_RETRACTION_COOL_UNTIL_TEMP_C", NaN)
const retraction_cool_until_active = isfinite(retraction_cool_until_temp_C)
# After eject: freeze charge mechanics (thermal-only RHS) during free cool.
# At cool target, briefly release mechanics (POST_COOL_SPRINGBACK_S) for residual springback.
const retraction_freeze_mech_after_eject = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_FREEZE_MECH_AFTER_EJECT", 0) != 0
# Freeze charge mechanics/F during the rigid +z eject translation, then release after.
const retraction_freeze_mech_during_eject = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_FREEZE_MECH_DURING_EJECT", 1) != 0
const retraction_post_cool_springback_s = env_float("TP_CLIP_RETRACTION_POST_COOL_SPRINGBACK_S", 0.25)
# Floor on boundary exposure during post-eject free cool (accelerates ambient cooling;
# physical exposure is sparse on the sheet). 0 = use true exposure only.
const retraction_free_cool_exposure_floor = clamp(
    env_float("TP_CLIP_FREE_COOL_EXPOSURE_FLOOR", 0.0), 0.0, 1.0)
# Complete demold: after punch clears, rigidly eject the workpiece out of the cavity
# (Song/FARO warpage is measured on a free demolded part, not while nested in the die).
# Default ON for retraction/full. Set TP_CLIP_RETRACTION_EJECT=0 to keep in-cavity springback.
const retraction_eject_enable = (sim_phase == "retraction" || sim_phase == "full") &&
    env_int("TP_CLIP_RETRACTION_EJECT", 1) != 0
# Punch lifts first; workpiece stays in the die and cools (convection ON, plastic OFF)
# until mean T ≤ COOL_UNTIL, then the eject stroke runs. Creep→lift unchanged.
# Closed-die cool (COOL_BEFORE_LIFT): hold_creep plastic ON until punch lift by default.
# Set TP_CLIP_RETRACTION_CLOSED_DIE_COOL_PLASTIC=0 for elastic+CTE QS cool (no plastic commit).
const retraction_closed_die_cool_plastic =
    env_int("TP_CLIP_RETRACTION_CLOSED_DIE_COOL_PLASTIC", 1) != 0
const retraction_eject_after_cool = retraction_eject_enable &&
    retraction_cool_until_active &&
    env_int("TP_CLIP_RETRACTION_EJECT_AFTER_COOL", 0) != 0
# Eject temperature can be warmer than the final free-cool target. This lets
# the part leave the floor/die after CTE restoration, then cool freely.
const retraction_eject_at_temp_C =
    env_float("TP_CLIP_RETRACTION_EJECT_AT_TEMP_C", retraction_cool_until_temp_C)
# Smooth release of the thermal contraction accumulated during frozen-mechanics free cool.
# The stored stress state is in balance at the eject temperature; the remaining T drop is
# introduced through a smoothstep over RAMP_S instead of one instantaneous step.
# 0 = legacy instantaneous release.
const retraction_post_cool_thermal_ramp_s =
    env_float("TP_CLIP_RETRACTION_POST_COOL_THERMAL_RAMP_S", 0.05)
const retraction_post_cool_thermal_scale = Ref(1.0)
const retraction_post_cool_thermal_Tref_K = retraction_eject_at_temp_C + 273.15
# After eject: apply remaining T drop to COOL_UNTIL in one shot (room-temp warpage
# without hours of Newton cooling), freeze T, then dwell mechanically.
const retraction_bang_t_after_eject = sim_phase == "retraction" &&
    retraction_cool_until_active &&
    env_int("TP_CLIP_RETRACTION_BANG_T_AFTER_EJECT", 0) != 0
const retraction_post_eject_dwell_s =
    max(env_float("TP_CLIP_RETRACTION_POST_EJECT_DWELL_S", 0.5), 0.0)
retraction_post_eject_bang_done = Ref(false)
retraction_post_eject_bang_t = Ref(Inf)
# After punch lift: set floor/mold (and blankholder) tool T for in-die cool sink.
# NaN = leave tools at TP_CLIP_TOOL_TEMP (default).
const retraction_tool_temp_after_punch_lift_C =
    env_float("TP_CLIP_RETRACTION_TOOL_TEMP_AFTER_PUNCH_LIFT_C", NaN)
const retraction_tool_temp_after_punch_lift_active =
    (sim_phase == "retraction" || sim_phase == "full") &&
    isfinite(retraction_tool_temp_after_punch_lift_C)
retraction_tool_temp_after_punch_lift_done = Ref(false)
# Physical unload: keep punch+floor closed after creep, cool by tool HTC (no ambient),
# then lift when mean charge T ≤ tool T + 5 °C (cannot go below tool T in a closed die).
# Set COOL_BEFORE_LIFT=0 for legacy time-based lift after creep only.
const retraction_cool_before_lift = (sim_phase == "retraction" || sim_phase == "full") &&
    env_int("TP_CLIP_RETRACTION_COOL_BEFORE_LIFT", 1) != 0
const _retraction_tool_temp_C = env_float("TP_CLIP_TOOL_TEMP", 140.0)
const retraction_cool_before_lift_C = env_float("TP_CLIP_RETRACTION_COOL_BEFORE_LIFT_C",
    _retraction_tool_temp_C + 5.0)
retraction_closed_die_cool_active = Ref(false)
retraction_closed_die_cool_done = Ref(false)
retraction_closed_die_cool_start_t = Ref(Inf)
# Quasi-static closed-die cooling: hold T at an isothermal stage, equilibrate
# mechanics/contact, then decrement the complete temperature field by ΔT. This
# preserves its gradients and avoids both frozen-kin stress storage and a sudden
# explicit release. The tools remain fixed until the final stage equilibrates.
const retraction_qs_cool = retraction_cool_before_lift &&
    env_int("TP_CLIP_RETRACTION_QS_COOL", 1) != 0
const retraction_qs_cool_step_C =
    max(env_float("TP_CLIP_RETRACTION_QS_COOL_STEP_C", 5.0), 0.0)
const retraction_qs_equil_min_s =
    max(env_float("TP_CLIP_RETRACTION_QS_EQUIL_MIN_S", 0.05), 0.0)
const retraction_qs_equil_max_s =
    max(env_float("TP_CLIP_RETRACTION_QS_EQUIL_MAX_S", 10.0),
        retraction_qs_equil_min_s)
const retraction_qs_sample_s =
    max(env_float("TP_CLIP_RETRACTION_QS_SAMPLE_S", 0.05), 1.0e-4)
const retraction_qs_equil_patience =
    max(1, env_int("TP_CLIP_RETRACTION_QS_EQUIL_PATIENCE",
                   retraction_springback_equil_patience))
const retraction_qs_equil_rms_speed =
    max(env_float("TP_CLIP_RETRACTION_QS_EQUIL_RMS_SPEED", 0.05), 0.0)
const retraction_qs_equil_max_speed =
    max(env_float("TP_CLIP_RETRACTION_QS_EQUIL_MAX_SPEED", 0.10), 0.0)
const retraction_qs_max_j_floor_n =
    max(0, env_int("TP_CLIP_RETRACTION_QS_MAX_J_FLOOR_N", 64))
# Abort closed-die QS if span_z exceeds this (mm). Default Inf = off (L-die ~28 mm in cavity).
# Bulk organosheet runs often set TP_CLIP_RETRACTION_QS_ABORT_SPAN_Z_MM=18–20.
const retraction_qs_abort_span_z_mm =
    env_float("TP_CLIP_RETRACTION_QS_ABORT_SPAN_Z_MM", Inf)
const retraction_qs_abort_span_z_active = isfinite(retraction_qs_abort_span_z_mm)
# Abort closed-die QS if min det(F) drops below this (default 0 = no inversion). Set to
# retraction_J_floor (e.g. 0.05) to match J-floor target; use ≤ -998 to disable J abort.
const retraction_qs_abort_j_min = env_float("TP_CLIP_RETRACTION_QS_ABORT_J_MIN", 0.0)
const retraction_qs_abort_on_j_min = retraction_qs_abort_j_min > -998.0
# Smooth kin ON at the start of closed-die cool (0 = instant).
const retraction_closed_die_kin_ramp_s =
    max(env_float("TP_CLIP_RETRACTION_CLOSED_DIE_KIN_RAMP_S", 0.5), 0.0)
# Delay kin ON until Tmean ≤ this °C. Frozen + τ_th=0 until then (PVT/pressure,
# not solid kinematics). Inf restores r10–r15 (kin ON at freeze-T).
const retraction_qs_kin_after_C =
    env_float("TP_CLIP_RETRACTION_QS_KIN_AFTER_C",
              retraction_qs_cool ? retraction_cool_before_lift_C : Inf)
# Skip solid CTE overlay until closed-die mechanics start (and after lift).
const retraction_qs_skip_cte_until_kin = retraction_qs_cool &&
    env_int("TP_CLIP_RETRACTION_QS_SKIP_CTE_UNTIL_KIN", 1) != 0
# After kin ON, ramp τ_th in QS_STEP_C-equivalent stages instead of one bang.
const retraction_qs_cte_ramp = retraction_qs_cool &&
    env_int("TP_CLIP_RETRACTION_QS_CTE_RAMP", 0) != 0
# After the frozen CTE-off drop, lift the punch before releasing charge
# kinematics. Kinematic peel owns the frozen interval until mold clearance.
const retraction_lift_first_after_frozen_cool = retraction_qs_cool &&
    env_int("TP_CLIP_RETRACTION_LIFT_FIRST_AFTER_FROZEN_COOL", 0) != 0
# Die-aware F_th: while kin is frozen, absorb dT<0 into F + floor-seated positions.
# Not r9 centroid homothety. Overlay CTE stays off for the same ΔT.
const clip_fth_die_aware = env_int("TP_CLIP_FTH_DIE_AWARE", 0) != 0
const clip_kinematic_thermal_shrinkage = env_int("TP_CLIP_KINEMATIC_THERMAL_SHRINKAGE", 1) != 0
const clip_fth_skip_post_peel_cte = clip_fth_die_aware &&
    env_int("TP_CLIP_FTH_SKIP_POST_PEEL_CTE", 1) != 0
const retraction_post_peel_cte_interval_s =
    max(env_float("TP_CLIP_RETRACTION_POST_PEEL_CTE_INTERVAL_S", 0.05), 1.0e-6)
retraction_qs_cte_scale = Ref(1.0)
retraction_post_peel_cte_next_t = Ref(Inf)
retraction_qs_stage_index = Ref(0)
retraction_qs_stage_start_t = Ref(Inf)
retraction_qs_stage_target_C = Ref(NaN)
retraction_qs_equil_streak = Ref(0)
retraction_qs_prev_snap = Ref{Any}(nothing)
retraction_qs_next_sample_t = Ref(Inf)
# Legacy r9 F_th state is retained only for checkpoint compatibility; the
# centroid homothety itself is disabled below.
thermal_Fth_initialized = Ref(false)
thermal_Fth_cum_vol_logJ = Ref(0.0)
thermal_Fth_u_modified = Ref(false)
thermal_Fth_skip_Jp = Ref(false)
@inline function reset_thermal_Fth_state!()
    thermal_Fth_initialized[] = false
    thermal_Fth_cum_vol_logJ[] = 0.0
    thermal_Fth_u_modified[] = false
    thermal_Fth_skip_Jp[] = false
    retraction_qs_cte_scale[] = 1.0
    retraction_post_peel_cte_next_t[] = Inf
    retraction_qs_stage_index[] = 0
    retraction_qs_stage_start_t[] = Inf
    retraction_qs_stage_target_C[] = NaN
    retraction_qs_equil_streak[] = 0
    retraction_qs_prev_snap[] = nothing
    retraction_qs_next_sample_t[] = Inf
    return nothing
end
const retraction_eject_s = env_float("TP_CLIP_RETRACTION_EJECT_S", 0.10)
const retraction_eject_clear_ps = env_float("TP_CLIP_RETRACTION_EJECT_CLEAR_PS", 2.0)
const retraction_eject_min_dz_mm = env_float("TP_CLIP_RETRACTION_EJECT_MIN_DZ_MM", 0.0)
# How far into the free mold↔punch gap to place the workpiece after eject:
#   0.0 = just clear the mold (deck + clearance) — default; avoids flying into punch
#   0.5 = mid-gap (old behaviour; can overshoot toward a high punch and tear the part)
#   1.0 = just below the punch underside
const retraction_eject_gap_frac = clamp(env_float("TP_CLIP_RETRACTION_EJECT_GAP_FRAC", 0.0), 0.0, 1.0)
# Optional hard cap on workpiece eject lift [mm] (0 = no extra cap).
const retraction_eject_max_dz_mm = env_float("TP_CLIP_RETRACTION_EJECT_MAX_DZ_MM", 0.0)
retraction_eject_active = Ref(false)
retraction_eject_complete = Ref(false)
retraction_eject_announced = Ref(false)
retraction_eject_start_t = Ref(Inf)
retraction_eject_target_dz = Ref(0.0)
retraction_eject_z_applied = Ref(0.0)
retraction_indie_cool_announced = Ref(false)
retraction_indie_cool_last_print_t = Ref(-Inf)
retraction_post_cool_springback_active = Ref(false)
retraction_post_cool_springback_start_t = Ref(Inf)
retraction_post_cool_springback_announced = Ref(false)
# During the rigid eject stroke: keep floor Reimann ON so deep-penetration caps
# still apply while the workpiece is lifted. Punch/mold stay off. Default ON.
const retraction_keep_floor_during_eject = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_KEEP_FLOOR_DURING_EJECT", 1) != 0
# After eject: rest on the die floor as a table (punch stays OFF). Demold/eject stroke unchanged.
const retraction_keep_floor_after_eject = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_KEEP_FLOOR_AFTER_EJECT", 0) != 0
const retraction_gravity_after_eject = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_GRAVITY_AFTER_EJECT", 0.0) : 0.0
# Extra mass-proportional damping after eject (0 = off; γΔt ≳ 1 is explicit-Euler unstable).
const retraction_rayleigh_after_eject = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_RAYLEIGH_AFTER_EJECT", 0.0) : 0.0
# Optional closed-die hold_creep before punch lift (default off — ramps run from retraction t0).
const retraction_pre_lift_creep_s = sim_phase == "retraction" || sim_phase == "full" ?
    env_float("TP_CLIP_RETRACTION_PRE_LIFT_CREEP_S", 0.0) : 0.0
const retraction_pre_lift_elastic_frac =
    env_float("TP_CLIP_RETRACTION_PRE_LIFT_ELASTIC_FRAC", 0.85)
# Mid-creep Fe_iso audit interval (0 = only end-of-creep audit).
const retraction_creep_audit_dt = (sim_phase == "retraction" || sim_phase == "full") &&
    retraction_pre_lift_creep_s > 0.0 ?
    env_float("TP_CLIP_RETRACTION_CREEP_AUDIT_DT", 0.05) : 0.0
# Optional post-creep, pre-lift settle with mechanics ON and no new thermal load.
const retraction_pre_lift_settle_s = (sim_phase == "retraction" || sim_phase == "full") &&
    retraction_pre_lift_creep_s > 0.0 ?
    max(env_float("TP_CLIP_RETRACTION_PRE_LIFT_SETTLE_S", 0.0), 0.0) : 0.0
# Post-lift charge kin freeze is non-physical (punch opens while charge is glued).
# Default OFF: charge kinematics stay at 1 from late closed-die creep through lift.
# Floor overlap must be cleared *before* lift (see CREEP_KIN_AFTER_SIGMA + settle gate).
const retraction_post_lift_kin_freeze_s = (sim_phase == "retraction" || sim_phase == "full") &&
    retraction_pre_lift_creep_s > 0.0 ?
    env_float("TP_CLIP_RETRACTION_POST_LIFT_KIN_FREEZE_S", 0.0) : 0.0
# Kinematic peel: freeze charge (dv=0) through closed-die creep + punch lift; ungate
# springback only when mold gap clears (or peel timeout). Mold still lifts.
# After ungate, σ/contact soft-restart 0→1 (PEEL_SIGMA_RAMP_S) so residual floor
# penetration pushes the charge out gradually — no particle projection.
const retraction_kinematic_peel = (sim_phase == "retraction" || sim_phase == "full") &&
    env_int("TP_CLIP_RETRACTION_KINEMATIC_PEEL", 0) != 0
const retraction_peel_gap_ps = env_float("TP_CLIP_RETRACTION_PEEL_GAP_PS", 0.0)
# Safety timeout after lift start (s). Default covers full punch stroke at typical speeds.
const retraction_peel_max_s = env_float("TP_CLIP_RETRACTION_PEEL_MAX_S", 2.0)
# Soft σ/contact re-engagement after peel ungate (0 = off).
# softE run: contact_e≈0.001 still J-inverted at peel_scale≈0.002 — same cascade.
const retraction_peel_sigma_ramp_s = retraction_kinematic_peel ?
    env_float("TP_CLIP_RETRACTION_PEEL_SIGMA_RAMP_S", 0.15) : 0.0
# Optional floor/die −z open (default OFF — does not fix L-die wall contact demold).
const retraction_peel_floor_open = retraction_kinematic_peel &&
    env_int("TP_CLIP_RETRACTION_PEEL_FLOOR_OPEN", 0) != 0
const retraction_peel_floor_open_s =
    env_float("TP_CLIP_RETRACTION_PEEL_FLOOR_OPEN_S", 0.05)
const retraction_peel_floor_open_extra_ps =
    env_float("TP_CLIP_RETRACTION_PEEL_FLOOR_OPEN_EXTRA_PS", 0.05)
# During post-peel σ ramp, temporarily soften tool contact E (absolute stiffness).
# Default 1.0 = no extra E softening (softE start 0.001 did not prevent cascade).
const retraction_peel_contact_e_start = retraction_kinematic_peel ?
    env_float("TP_CLIP_RETRACTION_PEEL_CONTACT_E_START", 1.0) : 1.0
# Diagnostic: after peel, disable floor/mold Reimann (TLSPH self only) during σ ramp.
const retraction_peel_tools_off = retraction_kinematic_peel &&
    env_int("TP_CLIP_RETRACTION_PEEL_TOOLS_OFF", 0) != 0
retraction_peel_done = Ref(false)
retraction_peel_announced = Ref(false)
retraction_peel_hold_for_eject_announced = Ref(false)
retraction_peel_release_announced = Ref(false)
retraction_peel_release_time = Ref(Inf)
retraction_floor_open_z_shift = Ref(0.0)
retraction_floor_open_target_dz = Ref(0.0)
retraction_floor_open_start_t = Ref(Inf)
retraction_floor_open_active = Ref(false)
retraction_floor_open_done = Ref(false)
retraction_floor_open_announced = Ref(false)
# After σ/contact reach 1 in closed-die creep, postpone punch lift until floor overlap
# is within this tolerance (units of particle_spacing; negative = allow soft-contact deadband).
# 0 disables the gate. Max extra closed-die settle time:
const retraction_creep_floor_settle_gap_ps =
    env_float("TP_CLIP_RETRACTION_CREEP_FLOOR_SETTLE_GAP_PS", -0.25)
# Closed-die active-kin floor settle is disabled by default (cannot clear floor while mold
# still overlaps — that path exploded). Clear floor after punch opens instead.
const retraction_creep_floor_settle_max_s = (sim_phase == "retraction" || sim_phase == "full") &&
    retraction_pre_lift_creep_s > 0.0 ?
    env_float("TP_CLIP_RETRACTION_CREEP_FLOOR_SETTLE_MAX_S", 0.0) : 0.0
# Contact-led closed-die settle during retraction (not hold):
#   Phase A: contact ON + kin ACTIVE + σ=0 until floor gap OK (or settle timeout)
#   Phase B: σ ramps 0→1 while still closed-die
#   Phase C: punch lift (existing floor-settle / lift gate)
# Required so contact kicks are not scaled by σ (mech_dv=1 while stress tensor uses elas=0).
const retraction_closed_die_contact_settle = (sim_phase == "retraction" || sim_phase == "full") &&
    retraction_pre_lift_creep_s > 0.0 &&
    env_int("TP_CLIP_RETRACTION_CLOSED_DIE_CONTACT_SETTLE", 0) != 0
const retraction_contact_settle_sigma_ramp_s = retraction_closed_die_contact_settle ?
    env_float("TP_CLIP_RETRACTION_CONTACT_SETTLE_SIGMA_RAMP_S", 0.15) : 0.0
retraction_contact_settle_done = Ref(false)
retraction_sigma_ramp_start_t = Ref(Inf)
retraction_contact_settle_announced = Ref(false)
# Floor particle projection SCRAPPED (unphysical). Env flag ignored.
const retraction_floor_project_after_mold_clear = false
const retraction_floor_project_gap_ps = -0.02
const retraction_mold_clear_gap_ps =
    env_float("TP_CLIP_RETRACTION_MOLD_CLEAR_GAP_PS", 0.0)
retraction_floor_projected = Ref(false)
# Hold kinematics: frozen (default) = no motion during solidification hold + TLSPH tail;
# hybrid = frozen until tail then active; active = creep motion throughout hold.
function clip_hold_kinematics_mode()
    mode = lowercase(get(ENV, "TP_CLIP_HOLD_KINEMATICS", "frozen"))
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

function clip_compress_integrator()
    sim_phase in ("full", "hold") || return :implicit
    mode = lowercase(get(ENV, "TP_CLIP_COMPRESS_INTEGRATOR", "implicit"))
    mode in ("implicit", "explicit") ||
        error("TP_CLIP_COMPRESS_INTEGRATOR must be implicit or explicit (got $(mode))")
    return Symbol(mode)
end

function clip_retraction_integrator()
    sim_phase != "retraction" && return :implicit
    mode = lowercase(get(ENV, "TP_CLIP_RETRACTION_INTEGRATOR", "explicit"))
    mode in ("implicit", "explicit") ||
        error("TP_CLIP_RETRACTION_INTEGRATOR must be 'implicit' or 'explicit' (got $(mode))")
    return Symbol(mode)
end

function clip_explicit_euler_algorithm()
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

clip_retraction_explicit_algorithm() = clip_explicit_euler_algorithm()

const compression_integrator = clip_compress_integrator()
const compression_use_explicit = compression_integrator == :explicit
const compression_explicit_dt = compression_use_explicit ?
    env_float("TP_CLIP_COMPRESS_EXPLICIT_DT", 2.0e-5) : 0.0
const compression_explicit_hold_dt = compression_use_explicit ?
    env_float("TP_CLIP_COMPRESS_EXPLICIT_HOLD_DT", 5.0e-3) : 0.0
const compression_explicit_ignore_unstable = compression_use_explicit &&
    env_int("TP_CLIP_COMPRESS_EXPLICIT_IGNORE_UNSTABLE", 1) != 0

const retraction_integrator = clip_retraction_integrator()
const retraction_use_explicit = retraction_integrator == :explicit
# Demold (creep→lift) must match moldClear6: FREEZE_T=1, dt=5e-3, mechanics ON.
# In-die cool after punch lift: UNFREEZE_T_AFTER_PUNCH_LIFT (or EJECT_AFTER_COOL).
# Post-eject free-cool may opt in with UNFREEZE_T_AFTER_EJECT.
const retraction_explicit_freeze_t = retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_EXPLICIT_FREEZE_T", 1) != 0
const retraction_unfreeze_t_after_eject = retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_UNFREEZE_T_AFTER_EJECT", 0) != 0
const retraction_unfreeze_t_after_punch_lift = retraction_use_explicit &&
    (env_int("TP_CLIP_RETRACTION_UNFREEZE_T_AFTER_PUNCH_LIFT", 0) != 0 ||
     retraction_eject_after_cool)
const retraction_explicit_dt = env_float("TP_CLIP_RETRACTION_EXPLICIT_DT", 5.0e-3)
# Optional larger step during pre-lift closed-die creep only (defaults to demold dt).
const retraction_explicit_dt_creep = retraction_use_explicit ?
    env_float("TP_CLIP_RETRACTION_EXPLICIT_DT_CREEP", retraction_explicit_dt) :
    retraction_explicit_dt
# Larger step whenever charge mechanics are off (creep, peel, eject stroke, thermal-only cool).
const retraction_explicit_dt_mech_off = retraction_use_explicit ?
    env_float("TP_CLIP_RETRACTION_EXPLICIT_DT_MECH_OFF", retraction_explicit_dt_creep) :
    retraction_explicit_dt_creep
# Free-cool / post-eject only. Demold (creep → lift → eject) always keeps RETRACTION_EXPLICIT_DT.
const retraction_explicit_dt_springback = retraction_use_explicit ?
    env_float("TP_CLIP_RETRACTION_EXPLICIT_DT_SPRINGBACK", 5.0e-3) :
    retraction_explicit_dt
# Final mechanics release after thermal-only cooling can require a smaller step
# than both demold and the large thermal-only free-cool step.
const retraction_explicit_dt_post_cool = retraction_use_explicit ?
    env_float("TP_CLIP_RETRACTION_EXPLICIT_DT_POST_COOL", retraction_explicit_dt) :
    retraction_explicit_dt
# Closed-die QS equilibration (mech ON, tools fixed): smaller step than demold/springback.
const retraction_explicit_dt_qs_equil = retraction_use_explicit ?
    env_float("TP_CLIP_RETRACTION_EXPLICIT_DT_QS_EQUIL", retraction_explicit_dt) :
    retraction_explicit_dt
# SciMLBase default unstable_check aborts on any NaN/Inf in u; explicit contact can spike
# briefly before the accepted-step repair runs — keep going and let the callback fix state.
const retraction_explicit_ignore_unstable = retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_EXPLICIT_IGNORE_UNSTABLE", 1) != 0

@inline function clip_use_explicit_solver()
    return compression_use_explicit || retraction_use_explicit
end

@inline function retraction_explicit_active()
    return retraction_use_explicit
end

# Post-eject free-cool window (ambient / large dt / freeze-mech). Demold never enters this.
@inline function retraction_post_eject_phase()
    retraction_eject_enable || return mold_retraction_complete[]
    return retraction_eject_complete[]
end

# True → no conduction/HTC/plastic heat. Creep→lift frozen when FREEZE_T=1 (moldClear6).
# Punch-up in-die cool and/or post-eject free-cool can unfreeze T.
@inline function retraction_thermal_frozen_now()
    # Quasi-static continuation owns T: freeze the thermal RHS through creep and
    # all isothermal equilibrium stages. It mutates T only between stages.
    if retraction_qs_cool && retraction_started[] &&
       !retraction_closed_die_cool_done[]
        return true
    end
    retraction_explicit_active() || return false
    retraction_post_cool_springback_active[] && return true
    if retraction_bang_t_after_eject && retraction_post_eject_bang_done[]
        return true
    end
    if retraction_unfreeze_t_after_punch_lift && mold_retraction_complete[]
        return false
    end
    if retraction_cool_before_lift && retraction_closed_die_cool_active[]
        return false
    end
    if retraction_unfreeze_t_after_eject && retraction_post_eject_phase()
        return false
    end
    # Demold / default: identical to moldClear6 when FREEZE_T=1.
    return retraction_explicit_freeze_t
end

@inline function retraction_eject_stroke_freeze_now()
    retraction_freeze_mech_during_eject || return false
    return retraction_eject_active[] && !retraction_eject_complete[]
end

@inline function retraction_free_cool_thermal_only_now()
    retraction_eject_stroke_freeze_now() && return true
    if retraction_freeze_mech_after_eject && !retraction_post_cool_springback_active[]
        # When freeze-mech is active (Option A: post-cool springback), freeze mechanics
        # as soon as punch peel/lift is complete, throughout any in-die cooling,
        # eject stroke, and free air cooling, until post-cool springback begins at room T.
        return retraction_peel_done[] || mold_retraction_complete[] || retraction_post_eject_phase()
    end
    return false
end

# Ambient convection/radiation: after punch lift (in-die cool) and/or after eject.
# Never during closed-die creep→lift.
@inline function ambient_heat_loss_active_now()
    use_ambient_heat_loss || return false
    retraction_post_cool_springback_active[] && return false
    if retraction_unfreeze_t_after_punch_lift || retraction_eject_after_cool
        return mold_retraction_complete[]
    end
    if retraction_unfreeze_t_after_eject || retraction_cool_until_active ||
       retraction_freeze_mech_after_eject
        return retraction_post_eject_phase()
    end
    return true
end

# After punch lift / eject: no plastic-work→heat (Taylor–Quinney was heating the free-cool).
const retraction_skip_plastic_heat_after_eject = retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_SKIP_PLASTIC_HEAT_AFTER_EJECT", 0) != 0

@inline function retraction_plastic_off_after_punch_up()
    # Closed-die cool keeps hold_creep plastic commit (same as freeze-T creep).
    # Plastic-off / shrinkage-only starts after the punch has lifted.
    return (retraction_unfreeze_t_after_punch_lift || retraction_eject_after_cool) &&
        mold_retraction_complete[]
end

@inline function retraction_skip_plastic_heat_now()
    retraction_thermal_frozen_now() && return true
    retraction_skip_plastic_heat_after_eject || return false
    retraction_plastic_off_after_punch_up() && return true
    return retraction_post_eject_phase()
end

# After punch lift / eject: CTE shrinkage with mechanics ON; plastic flow/heat OFF.
# SPH conduction stays ON (dt=5e-3 is inside the explicit thermal CFL).
const retraction_thermal_shrinkage_only_after_eject = retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_THERMAL_SHRINKAGE_ONLY_AFTER_EJECT", 0) != 0
const retraction_skip_sph_conduction_after_eject = retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_SKIP_CONDUCTION_AFTER_EJECT", 0) != 0

@inline function retraction_thermal_shrinkage_only_now()
    retraction_thermal_shrinkage_only_after_eject || return false
    retraction_post_cool_springback_active[] && return false
    retraction_plastic_off_after_punch_up() && return true
    return retraction_post_eject_phase()
end

@inline function retraction_skip_sph_conduction_now()
    return retraction_skip_sph_conduction_after_eject && retraction_post_eject_phase() &&
        !retraction_post_cool_springback_active[]
end

# Large springback/free-cool dt only after demold eject finishes (never during demold).
@inline function retraction_post_eject_springback_dt_active()
    return retraction_post_eject_phase()
end

# True when drift/RHS skips full charge mechanics (thermal-only, kin freeze, peel, eject stroke).
@inline function retraction_explicit_mech_integrator_off(t::Real)
    !retraction_started[] && return false
    charge_hold_thermal_only(t) && return true
    hold_kinematics_frozen(t) && return true
    retraction_charge_kinematics_gated(t) && return true
    return false
end

@inline function retraction_qs_equil_dt_active(t::Real)
    retraction_qs_cool || return false
    retraction_closed_die_cool_active[] || return false
    retraction_closed_die_cool_done[] && return false
    return !retraction_explicit_mech_integrator_off(t)
end

@inline function retraction_explicit_target_dt(t::Real)
    # Mech release after cool has its own stable dt; thermal-only free cool may use large dt.
    retraction_post_cool_springback_active[] && return retraction_explicit_dt_post_cool
    if retraction_post_eject_springback_dt_active()
        retraction_explicit_mech_integrator_off(t) &&
            return retraction_explicit_dt_mech_off
        return retraction_explicit_dt_springback
    end
    retraction_qs_equil_dt_active(t) && return retraction_explicit_dt_qs_equil
    retraction_explicit_mech_integrator_off(t) && return retraction_explicit_dt_mech_off
    return retraction_explicit_dt
end

function enforce_explicit_compression_dt!(integrator)
    compression_use_explicit || return nothing
    if integrator.t >= t_compress
        hold_dt = compression_explicit_hold_dt
        if integrator.dt != hold_dt
            integrator.dt = hold_dt
            if !compression_explicit_hold_dt_announced[]
                compression_explicit_hold_dt_announced[] = true
                println(">>> Explicit hold: dt increased to ", hold_dt,
                        " s (press dt was ", compression_explicit_dt, " s)")
                flush(stdout)
            end
        end
    end
    return nothing
end

# Explicit retraction uses the same blended σ/contact/kinematics ramp as implicit.
# Demold always uses RETRACTION_EXPLICIT_DT; springback/free-cool dt only after eject.
function enforce_explicit_retraction_dt!(integrator)
    if !(retraction_use_explicit && retraction_started[])
        return nothing
    end
    target_dt = retraction_explicit_target_dt(integrator.t)
    if integrator.dt != target_dt
        integrator.dt = target_dt
        # Non-adaptive Euler ignores integrator.dt changes unless proposed dt is updated.
        try
            SciMLBase.set_proposed_dt!(integrator, target_dt)
        catch
        end
    end
    if integrator.opts.dtmax != target_dt
        integrator.opts.dtmax = target_dt
    end
    return nothing
end

@inline function apply_velocity_safety_clamps()
    return enable_velocity_safety_clamps || retraction_accel_clamp
end

@inline function apply_coordinate_safety_clamps()
    return enable_safety_clamps
end

println("--- SIMULATION STARTING (", geometry_mode, ") ---")
println("Threads available: ", nthreads())
println("Geometry mode: ", geometry_mode, " (output tag=", geometry_name_tag, ")")
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
(sim_phase == "full" || sim_phase == "hold") && checkpoint_finalize_on_save &&
    println("Hold: checkpoint finalize ON (J floor=", hold_end_J_floor,
            ", Fp repair", hold_end_fp_reset ? "+reset" : "", " on every save)")
compress_J_floor > 0.0 &&
    println("Compress: J floor ON (det(F)≥", compress_J_floor,
            " after accepted steps while t < t_compress; inversions reset)")
retraction_J_floor > 0.0 &&
    println("Retraction: J floor ON (det(F)≥", retraction_J_floor,
            " after every accepted step; inversions reset)")
(sim_phase == "full" || sim_phase == "hold") && checkpoint_audit_on_save &&
    println("Hold: checkpoint audit ON (strict=", checkpoint_audit_strict, ")")
sim_phase == "retraction" && !retraction_prepare_state &&
    println("Retraction: hold-end state prep OFF (checkpoint used as saved; audit only)")
(sim_phase == "retraction" || sim_phase == "full") && retraction_pre_lift_creep_s > 0.0 &&
    println("Retraction pre-lift creep: ", retraction_pre_lift_creep_s,
            " s closed-die (",
            retraction_closed_die_contact_settle ?
                "CONTACT settle: kin ACTIVE + σ=0 until floor gap OK, then σ ramp" :
            env_int("TP_CLIP_RETRACTION_CREEP_RAMP_FREE", 0) != 0 ?
                "σ/contact/kin INSTANT from checkpoint" :
                "σ/contact/kin ramp 0→1",
            ", hold_creep; mold fixed until lift;",
            " post-lift mechanics stay at 1 — no second ramp;",
            retraction_kinematic_peel ?
                "; KINEMATIC PEEL (charge frozen until gaps≥" *
                string(retraction_peel_gap_ps) * "·ps, timeout " *
                string(retraction_peel_max_s) * " s)" :
            retraction_post_lift_kin_freeze_s > 0.0 ?
                "; post-lift kin hold " * string(retraction_post_lift_kin_freeze_s) * " s" :
                "; charge kin ON at punch lift",
            retraction_creep_floor_settle_max_s > 0.0 ?
                "; floor settle ≤" * string(retraction_creep_floor_settle_max_s) *
                " s (gap≥" * string(retraction_creep_floor_settle_gap_ps) * "·ps)" : "",
            retraction_creep_audit_dt > 0.0 ?
                "; Fe_iso audit every " * string(retraction_creep_audit_dt) * " s" : "")
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
            "WCSPH, Nakamura, and molten drag are OFF; flow-orientation kinetics stay ON")
compression_use_explicit && begin
    _, explicit_scheme_name = clip_explicit_euler_algorithm()
    println("Compression integrator: explicit ", explicit_scheme_name,
            " (press dt=", compression_explicit_dt,
            " s; hold dt=", compression_explicit_hold_dt,
            " s; ignore SciML unstable abort=", compression_explicit_ignore_unstable, ")")
end
sim_phase == "retraction" && retraction_use_explicit && begin
    _, explicit_scheme_name = clip_explicit_euler_algorithm()
    freeze_t_policy = if retraction_unfreeze_t_after_punch_lift
        "demold FREEZE_T=$(retraction_explicit_freeze_t); unfreeze after punch lift"
    elseif retraction_unfreeze_t_after_eject
        "demold FREEZE_T=$(retraction_explicit_freeze_t); unfreeze after eject only"
    elseif retraction_explicit_freeze_t
        "always (moldClear6 demold)"
    else
        "off"
    end
    mech_off_dt_note = retraction_explicit_dt_mech_off != retraction_explicit_dt ?
        "; mech-off dt=" * string(retraction_explicit_dt_mech_off) * " s" : ""
    qs_equil_dt_note = retraction_qs_cool &&
        retraction_explicit_dt_qs_equil != retraction_explicit_dt ?
        "; closed-die QS equil dt=" * string(retraction_explicit_dt_qs_equil) * " s" : ""
    println("Retraction integrator: explicit ", explicit_scheme_name,
            " (demold dt=", retraction_explicit_dt, " s",
            mech_off_dt_note,
            qs_equil_dt_note,
            "; free-cool/springback dt=", retraction_explicit_dt_springback,
            " s after eject; same σ/contact ramp as implicit; freeze T=",
            freeze_t_policy,
            "; ignore SciML unstable abort=", retraction_explicit_ignore_unstable, ")")
end
println("---------------------------------------")

# Packing / layout knobs that must match hold→retraction (else n_mold/n_floor mismatch).
function checkpoint_packing_env_snapshot()
    ps_mm = env_float("TP_CLIP_PS_MM", 1.5)
    return Dict{String, Any}(
        "ps_mm" => ps_mm,
        "tool_ps_mm" => env_float("TP_CLIP_TOOL_PS_MM", ps_mm),
        "punch_side_clearance_mm" => env_float("TP_CLIP_PUNCH_SIDE_CLEARANCE_MM", 0.0),
        "shallow_punch_extra_len_mm" => env_float("TP_CLIP_SHALLOW_PUNCH_EXTRA_LEN_MM", 0.0),
        "initial_gap_upper_mm" => env_float("TP_CLIP_INITIAL_GAP_UPPER_MM", 3.0),
        "charge_floor_gap_mm" => env_float("TP_CLIP_CHARGE_FLOOR_GAP_MM", 0.0),
    )
end

"""Apply hold packing geometry ENV from checkpoint meta (before STL/tool pack)."""
function apply_checkpoint_packing_env!(ck_meta)
    env_int("TP_CLIP_CHECKPOINT_IMPORT_PACKING", 1) == 0 && return false
    applied = String[]
    for (meta_key, env_key) in (
            ("ps_mm", "TP_CLIP_PS_MM"),
            ("tool_ps_mm", "TP_CLIP_TOOL_PS_MM"),
            ("punch_side_clearance_mm", "TP_CLIP_PUNCH_SIDE_CLEARANCE_MM"),
            ("shallow_punch_extra_len_mm", "TP_CLIP_SHALLOW_PUNCH_EXTRA_LEN_MM"),
            ("initial_gap_upper_mm", "TP_CLIP_INITIAL_GAP_UPPER_MM"),
            ("charge_floor_gap_mm", "TP_CLIP_CHARGE_FLOOR_GAP_MM"),
        )
        haskey(ck_meta, meta_key) || continue
        val = ck_meta[meta_key]
        ENV[env_key] = string(val)
        push!(applied, "$(env_key)=$(val)")
    end
    if haskey(ck_meta, "compression_thickness_ratio") &&
       !haskey(ENV, "TP_CLIP_COMPRESSION_RATIO")
        ENV["TP_CLIP_COMPRESSION_RATIO"] = string(ck_meta["compression_thickness_ratio"])
        push!(applied, "TP_CLIP_COMPRESSION_RATIO=$(ck_meta["compression_thickness_ratio"])")
    end
    isempty(applied) && return false
    println(">>> Retraction: imported packing geometry from checkpoint meta:")
    for line in applied
        println("    ", line)
    end
    flush(stdout)
    return true
end

function checkpoint_meta(; particle_spacing, n_charge, n_floor, n_mold,
                         t_compress, z_shift_down_end, mold_velocity_state,
                         compression_thickness_ratio=missing,
                         hold_tlsph_tail_s=missing,
                         hold_tlsph_equilibrated=missing,
                         packing_env=missing)
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
    pack = packing_env === missing ? checkpoint_packing_env_snapshot() : packing_env
    for (k, v) in pack
        meta[k] = v
    end
    if separate_charge_plies
        meta["charge_layout"] = "separate_plies"
        meta["n_charge_systems"] = length(charge_ply_source_indices)
        meta["charge_system_counts"] = length.(charge_ply_source_indices)
    end
    return meta
end

function validate_checkpoint_meta!(ck, meta_now; motion_from_checkpoint=false)
    ck_meta = ck["meta"]
    layout_keys = separate_charge_plies ?
                  ("particle_spacing", "n_charge", "n_floor", "n_mold",
                   "charge_layout", "n_charge_systems", "charge_system_counts") :
                  ("particle_spacing", "n_charge", "n_floor", "n_mold")
    motion_keys = ("t_compress", "z_shift_down_end", "mold_velocity_state")
    keys_to_check = motion_from_checkpoint ? layout_keys : (layout_keys..., motion_keys...)
    for key in keys_to_check
        if !haskey(ck_meta, key)
            error("Checkpoint meta missing '$(key)'")
        end
        val_ck = ck_meta[key]
        val_now = meta_now[key]
        match = (val_ck isa Number && val_now isa Number) ? isapprox(val_ck, val_now; rtol=1e-4) : (val_ck == val_now)
        if !match
            hint = key == "n_mold" ?
                   " Try TP_CLIP_COMPRESSION_RATIO=0.85 (or re-run hold with current geometry)." :
                   ""
            error("Checkpoint meta mismatch on '$(key)': saved=$(val_ck), " *
                  "current=$(val_now). Use the same geometry/env as the hold run." * hint)
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

function charge_hold_end_quality(cyl_sys, Fp_buf)
    J_min = Inf
    J_min_particle = 0
    n_bad_fp = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        det_F = det(F)
        if isfinite(det_F) && det_F < J_min
            J_min = det_F
            J_min_particle = particle
        end
        any(!isfinite, @view Fp_buf[:, :, particle]) && (n_bad_fp += 1)
    end
    return J_min, J_min_particle, n_bad_fp
end

function audit_charge_plastic_state(cyl_sys, Fp_buf, J_p_buf)
    n = 0
    J_min = Inf
    J_min_p = 0
    Fe_min = Inf
    Fe_min_p = 0
    Jevol_min = Inf
    Jevol_max = -Inf
    n_bad_fp = 0
    n_big_fp = 0
    n_Fe_crushed = 0
    n_Jevol_off = 0
    # Isochoric residual elastic metrics (Strategy E: det(Fe)=det(F) with isochoric Fp,
    # so Fe_min is NOT a demold readiness signal — use Fe_iso / dev(be_bar) instead).
    Fe_iso_I_max = 0.0
    Fe_iso_I_max_p = 0
    Fe_iso_I_sum = 0.0
    n_Fe_iso = 0
    dev_be_max = 0.0
    dev_be_max_p = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        n += 1
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        det_F = det(F)
        if isfinite(det_F) && det_F < J_min
            J_min = det_F
            J_min_p = particle
        end
        Fp_view = @view Fp_buf[:, :, particle]
        any(!isfinite, Fp_view) && (n_bad_fp += 1)
        maximum(abs.(Fp_view)) > 100.0 && (n_big_fp += 1)
        isfinite(det_F) || continue
        Fp_sm = StaticArrays.SMatrix{3, 3}(Fp_view)
        d_fp = det(Fp_sm)
        if isfinite(d_fp) && abs(d_fp) > 1e-14
            Fp_sm = Fp_sm / cbrt(d_fp)
        end
        Fe = F * inv(Fp_sm)
        det_Fe = det(Fe)
        if isfinite(det_Fe) && det_Fe < Fe_min
            Fe_min = det_Fe
            Fe_min_p = particle
        end
        isfinite(det_Fe) && det_Fe < 0.5 && (n_Fe_crushed += 1)
        if isfinite(det_Fe) && abs(det_Fe) > 1e-14
            Fe_iso = Fe / cbrt(det_Fe)
            dFeI = 0.0
            @inbounds for jj in 1:3, ii in 1:3
                d = Fe_iso[ii, jj] - (ii == jj ? 1.0 : 0.0)
                dFeI += d * d
            end
            dFeI = sqrt(dFeI)
            Fe_iso_I_sum += dFeI
            n_Fe_iso += 1
            if dFeI > Fe_iso_I_max
                Fe_iso_I_max = dFeI
                Fe_iso_I_max_p = particle
            end
            be = Fe * Fe'
            be_bar = be / (abs(det_Fe)^(2 / 3))
            dev_be = be_bar - (tr(be_bar) / 3) * I
            ndev = sqrt(sum(abs2, dev_be))
            if isfinite(ndev) && ndev > dev_be_max
                dev_be_max = ndev
                dev_be_max_p = particle
            end
        end
        if use_volumetric_plasticity
            J_p = max(J_p_buf[particle], 1e-6)
            J_e_vol = max(det_F / J_p, 1e-6)
            Jevol_min = min(Jevol_min, J_e_vol)
            Jevol_max = max(Jevol_max, J_e_vol)
            abs(log(J_e_vol)) > 0.05 && (n_Jevol_off += 1)
        end
    end
    Fe_iso_I_mean = n_Fe_iso > 0 ? Fe_iso_I_sum / n_Fe_iso : 0.0
    # Legacy pass uses det(Fe); under Strategy E prefer Fe_iso_ready for demold judgment.
    pass = n_bad_fp == 0 && (n == 0 || Fe_min >= 0.5) && n_Fe_crushed == 0
    Fe_iso_ready = n_bad_fp == 0 && (n == 0 || Fe_iso_I_max <= 0.5)
    return (;
        pass,
        Fe_iso_ready,
        n,
        J_min,
        J_min_p,
        Fe_min,
        Fe_min_p,
        Fe_iso_I_max,
        Fe_iso_I_max_p,
        Fe_iso_I_mean,
        dev_be_max,
        dev_be_max_p,
        Jevol_min,
        Jevol_max,
        n_bad_fp,
        n_big_fp,
        n_Fe_crushed,
        n_Jevol_off,
    )
end

function print_charge_plastic_audit!(audit; label::String)
    println(">>> CHECKPOINT AUDIT [", label, "]: pass=", audit.pass,
            " Fe_iso_ready=", audit.Fe_iso_ready,
            " n=", audit.n,
            " J_min=", round(audit.J_min; digits=4), " (p=", audit.J_min_p, ")",
            " Fe_min=", round(audit.Fe_min; digits=4), " (p=", audit.Fe_min_p, ")",
            " ||Fe_iso-I||_F max/mean=", round(audit.Fe_iso_I_max; digits=4), "/",
            round(audit.Fe_iso_I_mean; digits=4), " (p=", audit.Fe_iso_I_max_p, ")",
            " ||dev be_bar||_max=", round(audit.dev_be_max; digits=4),
            " (p=", audit.dev_be_max_p, ")",
            use_volumetric_plasticity ?
                " J_e_vol=[" * string(round(audit.Jevol_min; digits=4)) * "," *
                string(round(audit.Jevol_max; digits=4)) * "]" : "",
            " Fp_nonfinite=", audit.n_bad_fp,
            " Fp_|entry|>100=", audit.n_big_fp,
            " Fe_det<0.5=", audit.n_Fe_crushed,
            use_volumetric_plasticity ?
                " |log J_e_vol|>0.05=" * string(audit.n_Jevol_off) : "")
    flush(stdout)
    return nothing
end

function enforce_charge_plastic_audit!(cyl_sys, Fp_buf, J_p_buf; label::String)
    audit = audit_charge_plastic_state(cyl_sys, Fp_buf, J_p_buf)
    print_charge_plastic_audit!(audit; label=label)
    if !audit.pass && checkpoint_audit_strict
        error("Checkpoint audit failed ($label): Fe_min=$(audit.Fe_min), Fp_nonfinite=$(audit.n_bad_fp)")
    end
    return audit.pass
end

function finalize_hold_end_state_for_checkpoint!(cyl_sys, Fp_buf, J_ref_buf, J_p_buf)
    apply_charge_plane_strain_F_fix!(cyl_sys)
    n_j = 0
    n_fp = 0
    if hold_end_J_floor > 0.0
        n_j = floor_hold_end_deformation_j!(cyl_sys, J_p_buf, hold_end_J_floor)
    end
    n_fp = hold_end_fp_repair ? repair_hold_end_fp_committed!(cyl_sys, Fp_buf) : 0
    I3 = StaticArrays.SMatrix{3, 3, Float64}(I)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        det_F = det(F)
        if !isfinite(det_F) || abs(det_F) < 1e-10
            continue
        end
        if use_volumetric_plasticity
            J_p_buf[particle] = max(det_F, hold_end_J_floor > 0.0 ? hold_end_J_floor : 1e-6)
            J_ref_buf[particle] = elastic_bulk_J_ref
            if hold_end_fp_reset
                F_iso = F / cbrt(det_F)
                for jj in 1:3, ii in 1:3
                    Fp_buf[ii, jj, particle] = F_iso[ii, jj]
                end
            end
        end
    end
    J_min, J_min_p, n_bad_fp = charge_hold_end_quality(cyl_sys, Fp_buf)
    if n_j > 0 || n_fp > 0 || hold_end_fp_reset
        println(">>> Checkpoint finalize: J floor ", n_j, " particles",
                hold_end_fp_reset ? "; Fp reset from F_iso" : "",
                n_fp > 0 ? "; Fp repair $n_fp particles" : "",
                "; J_min=", round(J_min; digits=4), " (p=", J_min_p, ")",
                n_bad_fp > 0 ? "; Fp nonfinite remaining=$n_bad_fp" : "")
        flush(stdout)
    end
    enforce_charge_plastic_audit!(cyl_sys, Fp_buf, J_p_buf; label="finalize")
    return J_min, n_bad_fp
end

function capture_hold_checkpoint_state(integrator, semi_local)
    cyl_sys = semi_local.systems[1]
    floor_sys = clip_floor_system(semi_local)
    mold_sys = clip_mold_system(semi_local)
    if checkpoint_finalize_on_save && solidification_complete[] && !retraction_started[] &&
       !hold_tlsph_tail_active(integrator.t)
        finalize_hold_end_state_for_checkpoint!(cyl_sys, Fp_committed[], J_ref_particle_buf,
                                                J_p_particle_buf)
    end
    v_ck, u_ck = clip_integrator_vu(integrator)
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
        "v_ode" => clip_host_copy(v_ck),
        "u_ode" => clip_host_copy(u_ck),
        "deformation_grad" => clip_host_copy(clip_pack_charge_deformation_grad(semi_local)),
        "charge_temp" => clip_host_copy(clip_pack_charge_temp(semi_local)),
        "floor_temp" => clip_host_copy(floor_sys.temp),
        "mold_temp" => clip_host_copy(mold_sys.temp),
        "Fp_state" => copy(Fp_state[]),
        "Fp_committed" => copy(Fp_committed[]),
        "alpha_committed" => copy(alpha_committed[]),
        "J_ref_particle_buf" => clip_host_copy(J_ref_particle_buf),
        "J_p_particle_buf" => clip_host_copy(J_p_particle_buf),
        "crystallinity_particle_buf" => clip_host_copy(crystallinity_particle_buf),
        "liquid_fraction_particle_buf" => clip_host_copy(liquid_fraction_particle_buf),
        "solid_fraction_particle_buf" => clip_host_copy(solid_fraction_particle_buf),
        "ys_particle_buf" => clip_host_copy(ys_particle_buf),
        "hard_particle_buf" => clip_host_copy(hard_particle_buf),
        "vis_particle_buf" => clip_host_copy(vis_particle_buf),
        "thermal_softening_particle_buf" => clip_host_copy(thermal_softening_particle_buf),
        "orientation_tensor" => copy(orientation_tensor_state[]),
        "fiber_direction" => copy(fiber_direction),
        "vel_grad_buf" => clip_host_copy(vel_grad_buf),
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

function save_preeject_checkpoint!(integrator)
    ck = capture_hold_checkpoint_state(integrator, integrator.p)
    ck["meta"]["retraction_stage"] = "preeject"
    ck["retraction_qs_cte_scale"] = retraction_qs_cte_scale[]
    save_hold_checkpoint!(retraction_preeject_checkpoint_path, ck)
    println(">>> Pre-eject checkpoint ready (skip creep/lift/cool next time): ",
            abspath(retraction_preeject_checkpoint_path))
    flush(stdout)
    return nothing
end

function save_cool_end_checkpoint!(integrator)
    isempty(retraction_cool_end_checkpoint_path) && return nothing
    ck = capture_hold_checkpoint_state(integrator, integrator.p)
    ck["meta"]["retraction_stage"] = "indie_cool"
    ck["retraction_qs_cte_scale"] = retraction_qs_cte_scale[]
    save_hold_checkpoint!(retraction_cool_end_checkpoint_path, ck)
    cyl_sys = integrator.p.systems[1]
    xs = @view cyl_sys.current_coordinates[1, :]
    ys = @view cyl_sys.current_coordinates[2, :]
    zs = @view cyl_sys.current_coordinates[3, :]
    J_min, J_max, _ = charge_jacobian_extremes(cyl_sys)
    println(">>> In-die cool checkpoint saved: ",
            abspath(retraction_cool_end_checkpoint_path))
    println("    span_x=", round((maximum(xs) - minimum(xs)) * 1e3; digits=3),
            " mm  span_y=", round((maximum(ys) - minimum(ys)) * 1e3; digits=3),
            " mm  span_z=", round((maximum(zs) - minimum(zs)) * 1e3; digits=3),
            " mm  J_min=", round(J_min; sigdigits=4),
            "  J_max=", round(J_max; sigdigits=4))
    flush(stdout)
    return nothing
end

function save_named_retraction_checkpoint!(integrator, path, stage)
    isempty(path) && return nothing
    ck = capture_hold_checkpoint_state(integrator, integrator.p)
    ck["meta"]["retraction_stage"] = stage
    ck["retraction_qs_cte_scale"] = retraction_qs_cte_scale[]
    save_hold_checkpoint!(path, ck)
    cyl_sys = integrator.p.systems[1]
    xs = @view cyl_sys.current_coordinates[1, :]
    ys = @view cyl_sys.current_coordinates[2, :]
    zs = @view cyl_sys.current_coordinates[3, :]
    J_min, J_max, _ = charge_jacobian_extremes(cyl_sys)
    println(">>> Checkpoint ", stage, " saved: ", abspath(path))
    println("    span_x=", round((maximum(xs) - minimum(xs)) * 1e3; digits=3),
            " mm  span_y=", round((maximum(ys) - minimum(ys)) * 1e3; digits=3),
            " mm  span_z=", round((maximum(zs) - minimum(zs)) * 1e3; digits=3),
            " mm  J_min=", round(J_min; sigdigits=4),
            "  J_max=", round(J_max; sigdigits=4))
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
    supported_versions = separate_charge_plies ? (CHECKPOINT_VERSION,) : (1, CHECKPOINT_VERSION)
    ck["version"] in supported_versions ||
        error("Unsupported checkpoint version $(ck["version"]) for " *
              (separate_charge_plies ? "separate-ply layout (expected v3)" :
               "legacy packed-charge layout (expected v1 or v2)"))
    return ck
end

function restore_hold_checkpoint_state!(ck, semi_local)
    cyl_sys = semi_local.systems[1]
    floor_sys = clip_floor_system(semi_local)
    mold_sys = clip_mold_system(semi_local)

    clip_unpack_charge_deformation_grad!(semi_local, ck["deformation_grad"])
    clip_unpack_charge_temp!(semi_local, ck["charge_temp"])
    clip_host_assign!(floor_sys.temp, ck["floor_temp"])
    clip_host_assign!(mold_sys.temp, ck["mold_temp"])

    v_ck = ck["v_ode"]
    u_ck = ck["u_ode"]
    clip_foreach_charge_system(semi_local) do system, _
        v_sys = TrixiParticles.wrap_v(v_ck, system, semi_local)
        u_sys = TrixiParticles.wrap_u(u_ck, system, semi_local)
        nd = TrixiParticles.ndims(system)
        @inbounds for particle in TrixiParticles.each_integrated_particle(system)
            for d in 1:nd
                system.current_coordinates[d, particle] = u_sys[d, particle]
                system.initial_condition.velocity[d, particle] = v_sys[d, particle]
            end
            v_sys[nd + 1, particle] = system.temp[particle]
        end
    end

    Fp_state[] .= ck["Fp_state"]
    Fp_committed[] .= ck["Fp_committed"]
    alpha_committed[] .= ck["alpha_committed"]
    clip_host_assign!(J_ref_particle_buf, ck["J_ref_particle_buf"])
    if haskey(ck, "J_p_particle_buf")
        clip_host_assign!(J_p_particle_buf, ck["J_p_particle_buf"])
    else
        fill!(J_p_particle_buf, 1.0)
    end
    clip_host_assign!(crystallinity_particle_buf, ck["crystallinity_particle_buf"])
    clip_host_assign!(liquid_fraction_particle_buf, ck["liquid_fraction_particle_buf"])
    clip_host_assign!(solid_fraction_particle_buf, ck["solid_fraction_particle_buf"])
    clip_host_assign!(ys_particle_buf, ck["ys_particle_buf"])
    clip_host_assign!(hard_particle_buf, ck["hard_particle_buf"])
    clip_host_assign!(vis_particle_buf, ck["vis_particle_buf"])
    clip_host_assign!(thermal_softening_particle_buf, ck["thermal_softening_particle_buf"])
    orientation_tensor_state[] .= ck["orientation_tensor"]
    fiber_direction .= ck["fiber_direction"]
    clip_host_assign!(vel_grad_buf, ck["vel_grad_buf"])
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
    retraction_eject_active[] = false
    retraction_eject_complete[] = false
    retraction_eject_announced[] = false
    retraction_eject_start_t[] = Inf
    retraction_eject_target_dz[] = 0.0
    retraction_eject_z_applied[] = 0.0
    retraction_indie_cool_announced[] = false
    retraction_indie_cool_last_print_t[] = -Inf
    retraction_closed_die_cool_active[] = false
    retraction_closed_die_cool_done[] = false
    retraction_closed_die_cool_start_t[] = Inf
    retraction_post_eject_bang_done[] = false
    retraction_post_eject_bang_t[] = Inf
    retraction_tool_temp_after_punch_lift_done[] = false
    retraction_closed30_ckpt_saved[] = false
    retraction_closed30_settle_start_t[] = Inf
    reset_thermal_Fth_state!()
    retraction_post_cool_springback_active[] = false
    retraction_post_cool_springback_start_t[] = Inf
    retraction_post_cool_springback_announced[] = false
    retraction_start_time[] = Inf
    retraction_lift_start_time[] = Inf
    mold_retraction_complete_time[] = Inf
    retraction_state_prepared[] = false
    retraction_reference_span_x[] = 0.0
    retraction_reference_span_z[] = 0.0
    charge_mechanics_release_announced[] = false
    retraction_elastic_ramp_announced[] = false
    retraction_kinematics_release_announced[] = false
    retraction_pre_lift_creep_announced[] = false
    retraction_lift_started_announced[] = false
    retraction_creep_audit_next_t[] = Inf
    retraction_lift_planned_time[] = Inf
    retraction_creep_floor_settle_announced[] = false
    retraction_contact_settle_done[] = false
    retraction_sigma_ramp_start_t[] = Inf
    retraction_contact_settle_announced[] = false
    TrixiParticles.initialize_neighborhood_searches!(semi_local)
    clip_sync_tool_kinematics_gpu!(semi_local, ck["t"])
    TrixiParticles.foreach_system(semi_local) do system
        TrixiParticles.initialize!(system, semi_local)
        v_sys = TrixiParticles.wrap_v(v_ck, system, semi_local)
        u_sys = TrixiParticles.wrap_u(u_ck, system, semi_local)
        TrixiParticles.update_positions!(system, v_sys, u_sys, v_ck, u_ck, semi_local, ck["t"])
    end

    ck_meta = get(ck, "meta", Dict{String, Any}())
    if get(ck_meta, "retraction_stage", "") == "preeject" ||
       env_int("TP_CLIP_RETRACTION_RESUME_PREEJECT", 0) != 0
        retraction_from_preeject[] = true
        if haskey(ck, "retraction_qs_cte_scale")
            retraction_qs_cte_scale[] = Float64(ck["retraction_qs_cte_scale"])
        else
            retraction_qs_cte_scale[] = 1.0
        end
    end
    equilibrated = get(ck_meta, "hold_tlsph_equilibrated", false)
    println(">>> Restored hold checkpoint at t=", round(ck["t"]; digits=6),
            " s (solidification_complete=", solidification_complete[],
            ", TLSPH tail equilibrated=", equilibrated,
            retraction_from_preeject[] ? ", stage=preeject" : "", ")")
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

# Retraction: import hold packing layout from checkpoint *before* STL/tool pack so
# n_mold/n_floor match (PUNCH_SIDE_CLEARANCE / SHALLOW_PUNCH_EXTRA / gaps / ps).
if sim_phase == "retraction"
    isfile(checkpoint_path) ||
        error("Hold checkpoint not found for retraction: $(abspath(checkpoint_path))")
    hold_checkpoint_early[] = load_hold_checkpoint(checkpoint_path)
    apply_checkpoint_packing_env!(hold_checkpoint_early[]["meta"])
end

particle_spacing = env_float("TP_CLIP_PS_MM", ps_mm_nominal) * 1e-3
const tool_particle_spacing = haskey(ENV, "TP_CLIP_TOOL_PS_MM") ?
    env_float("TP_CLIP_TOOL_PS_MM", particle_spacing * 1e3) * 1e-3 :
    particle_spacing
# Charge packing anisotropy (opt-in). Default OFF → isotropic pack identical to before.
# Physics length scales (h, contact gaps, jac) always use `particle_spacing` = TP_CLIP_PS_MM.
# When ON: Δx=Δy=TP_CLIP_PS_XY_MM (default PS_MM), Δz=TP_CLIP_PS_Z_MM (default PS_MM).
const charge_ps_anisotropic = env_int("TP_CLIP_PS_ANISOTROPIC", 0) != 0
const particle_spacing_xy = charge_ps_anisotropic ?
    env_float("TP_CLIP_PS_XY_MM", particle_spacing * 1e3) * 1e-3 :
    particle_spacing
const particle_spacing_z = charge_ps_anisotropic ?
    env_float("TP_CLIP_PS_Z_MM", particle_spacing * 1e3) * 1e-3 :
    particle_spacing
const particle_volume = particle_spacing_xy * particle_spacing_xy * particle_spacing_z
if charge_ps_anisotropic
    println(">>> Charge packing anisotropic: Δxy=", particle_spacing_xy * 1e3,
            " mm, Δz=", particle_spacing_z * 1e3,
            " mm; physics ref TP_CLIP_PS_MM=", particle_spacing * 1e3, " mm")
end
# Flyaway envelope: through the die, outside the blank radius, or isolated from the bulk.
const flyaway_below_floor = env_float("TP_CLIP_FLYAWAY_BELOW_FLOOR_MM", 5.0) * 1e-3
const flyaway_above_cavity = env_float("TP_CLIP_FLYAWAY_ABOVE_CAVITY_MM", 80.0) * 1e-3
const flyaway_r_pad = env_float("TP_CLIP_FLYAWAY_R_PAD_MM",
                                4.0 * particle_spacing * 1e3) * 1e-3
const flyaway_isolate_dist = env_float("TP_CLIP_FLYAWAY_ISOLATE_DIST_MM",
                                       8.0 * particle_spacing * 1e3) * 1e-3
include(joinpath(@__DIR__, "clip_stl_geometry_setup.jl"))
include(joinpath(@__DIR__, "clip_blankholder.jl"))
const use_stl_geometry = clip_use_stl_geometry()

# Mass scaling kept for consistency with the 3D file (stays at 1.0 here).
mass_scaling = parse(Float64, get(ENV, "TP_CLIP_MASS_SCALING", "1.0"))

# Composite feedstock (carbon-fibre reinforced thermoplastic, CF-PP)
matrix_density = 1180.0 * mass_scaling
glass_density = 1750.0 * mass_scaling  # CF (T700-class PAN), vs glass 2550 kg/m³
matrix_E = 3.2e9
glass_E = 230.0e9                       # CF standard modulus (T700/T300), vs glass 72 GPa
# Bulk modulus of the polymer matrix in the melt state.
# Thermoplastic melts (PA6, PP, PPS) are nearly incompressible: K_melt ~ 1.0-1.5 GPa
# (Zoller & Walsh 1995, PVT data).  Wang et al. (2010) piston-die PP Tait: K = B/C
# ≈ 890 MPa (melt window) and ≈ 721 MPa @ 180°C preheat — used for all WCSPH EOS forms
# (Monaghan 1994, log, smooth C¹).  See cfk_clip_physics_validation validation 07.
matrix_K_melt = 890_166_380.0
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
matrix_tmelt = celsius_to_kelvin(398.0)      # °C input; stored as K for absolute-temperature laws.
glass_tmelt = celsius_to_kelvin(1700.0)
matrix_temp_liq = celsius_to_kelvin(378.0)   # °C input; crystallization completion on cooling.
glass_temp_liq = celsius_to_kelvin(1500.0)
# Enthalpy of fusion for iPP: 165 J/g (literature ΔHf for 100% crystalline = 209 J/g × ~79% crystallinity).
# Source: Wunderlich 1990, Brucato 2002. Replaces previous 110 kJ/kg DSC-melting value.
matrix_latent_heat = 1.65e5

# ==========================================================================================
# External Material Property Override Handler
# Current default material properties are set above. If TP_CLIP_MATERIAL_FILE or
# TP_CLIP_MATERIAL_CONFIG is set in ENV, the properties defined in that file will overwrite
# the default properties above while retaining any un-specified defaults.
# ==========================================================================================
let mat_file = get(ENV, "TP_CLIP_MATERIAL_FILE", get(ENV, "TP_CLIP_MATERIAL_CONFIG", ""))
    if !isempty(mat_file) && isfile(mat_file)
        println("✓ Loading external material properties from: ", abspath(mat_file))
        include(abspath(mat_file))
    elseif !isempty(mat_file)
        @warn "Material property file specified via TP_CLIP_MATERIAL_FILE not found: $(mat_file)"
    end
end

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
# Defaults are iPP (Brucato/Pantani). TP_CLIP_MATERIAL_FILE may set PA6 (or other) values first.
(@isdefined(nakamura_n)) || (nakamura_n = 3.0)          # Avrami exponent
(@isdefined(nakamura_K0)) || (nakamura_K0 = 891251.0)   # pre-exponential [1/s]
nakamura_K0 = env_float("TP_CLIP_NAKAMURA_K0", nakamura_K0)
(@isdefined(nakamura_U_star)) || (nakamura_U_star = 6284.0)  # J/mol
(@isdefined(nakamura_T_inf)) ||
    (nakamura_T_inf = celsius_to_kelvin(223.0))   # °C input; stored as K
(@isdefined(nakamura_Kg)) || (nakamura_Kg = 245471.0)  # K² — Lauritzen–Hoffman
(@isdefined(nakamura_T_m0)) ||
    (nakamura_T_m0 = celsius_to_kelvin(453.0))   # °C input; equilibrium melting point
nakamura_R = 8.31446      # J/(mol·K)
nakamura_seed_crystallinity = env_float("TP_CLIP_NAKAMURA_SEED", 1.0e-6)
# Flag: when true the Nakamura ODE drives latent-heat release and phase fractions;
# the temperature-based apparent-cp fallback is disabled to avoid double counting.
use_nakamura_kinetics = true
# Defaults below keep any values already set by TP_CLIP_MATERIAL_FILE (include above).
(@isdefined(thermal_softening_reference_temp)) ||
    (thermal_softening_reference_temp = celsius_to_kelvin(270.0))
(@isdefined(matrix_viscosity)) || (matrix_viscosity = 1.86e8) # Pa·s — Cross–WLF η_ref (iPP default)
matrix_viscosity = env_float("TP_CLIP_MATRIX_VISCOSITY_PA_S", matrix_viscosity)
(@isdefined(matrix_viscosity_ref_temp)) || (matrix_viscosity_ref_temp = matrix_temp_liq)
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
(@isdefined(cross_wlf_A1)) || (cross_wlf_A1 = 15.0)
(@isdefined(cross_wlf_A2)) || (cross_wlf_A2 = 50.0)
(@isdefined(cross_wlf_transition_temp)) ||
    (cross_wlf_transition_temp = matrix_viscosity_ref_temp)
# Critical stress τ* sets the onset of shear-thinning: γ̇* = τ*/η₀.
# 2e4 Pa from Laun (1986, Rheol. Acta 25:447) iPP capillary data — gives
# γ̇* ≈ 1–4 s⁻¹ at 180–220°C so shear-thinning is active in stamp-forming (1–100 s⁻¹).
# (2e7 Pa was 1000× too large and made the melt effectively Newtonian.)
(@isdefined(cross_wlf_critical_stress)) || (cross_wlf_critical_stress = 21000.0)
(@isdefined(cross_wlf_power_law_index)) || (cross_wlf_power_law_index = 0.28)
fiber_max_packing_fraction = 0.64
fiber_intrinsic_viscosity = 2.5
# Legacy placeholder from the earlier rule-of-mixtures viscosity model.
# The live hot-flow law below keeps viscosity matrix-dominated instead.
glass_viscosity = 1.0e16
(@isdefined(matrix_yield_stress)) || (matrix_yield_stress = 9.0e7)
(@isdefined(glass_yield_stress)) ||
    (glass_yield_stress = 4.9e9)   # CF tensile strength (T700, Toray datasheet)
(@isdefined(matrix_hardening)) || (matrix_hardening = 8.0e8)
(@isdefined(glass_hardening)) ||
    (glass_hardening = 1.5e10)     # stiffer fibre — CF post-yield tangent estimate
(@isdefined(matrix_h_contact)) ||
    (matrix_h_contact = 2.5e4)  # W/m²K — tool–part HTC, Bernet 1999 / Ye 2005
(@isdefined(glass_h_contact)) ||
    (glass_h_contact = 6.0e4)   # W/m²K — CF surface vs metal tool
mu_friction = env_float("TP_CLIP_MU_FRICTION", 0.3)  # Coulomb friction coefficient, tool–part interface
                                                      # 0.3 is representative for steel vs CFRTP (Guzman 2018)
# IMEX split (optional): keep stiff tool contact implicit, but apply "soft" wall coupling
# (molten drag + Coulomb friction) explicitly after each accepted step to reduce Newton/JVP load.
const imex_soft_terms = env_int("TP_CLIP_IMEX_SOFT_TERMS", 0) != 0
# Monaghan tensile artificial stress: keeps SPH neighbors connected under opening/bending tension.
# Disabled on GPU: tensile_stress_cache_tlsph allocates CPU arrays incompatible with CUDA kernels.
const _clip_tensile_stress_enabled = env_int("TP_CLIP_TENSILE_STRESS", 0) != 0
const clip_tensile_stress = (_clip_tensile_stress_enabled && !clip_use_gpu) ?
    TensileArtificialStressMonaghan(psi=env_float("TP_CLIP_TENSILE_PSI", -0.1), exponent=4) :
    nothing
clip_use_gpu && _clip_tensile_stress_enabled &&
    println("  - tensile artificial stress: OFF on GPU (use CPU driver for Monaghan ψ)")

# Coefficient of Thermal Expansion (CTE) for thermal residual stress / springback.
# CF T700: α∥ ≈ −0.5 ppm/K (longitudinal), α⊥ ≈ 22 ppm/K (transverse).
# PP matrix: α ≈ 80 ppm/K.  Refs: Chawla (2012), Sideridis (1994).
# Material file may override; ENV still wins when set.
(@isdefined(fiber_cte_axial)) || (fiber_cte_axial = -0.5e-6)
(@isdefined(fiber_cte_transverse)) || (fiber_cte_transverse = 22.0e-6)
(@isdefined(matrix_cte)) || (matrix_cte = 80.0e-6)
fiber_cte_axial      = env_float("TP_CLIP_FIBER_CTE_AXIAL",  fiber_cte_axial)
fiber_cte_transverse = env_float("TP_CLIP_FIBER_CTE_TRANS",   fiber_cte_transverse)
matrix_cte           = env_float("TP_CLIP_MATRIX_CTE",        matrix_cte)
# Stress-free reference temperature: unconstrained thermal volume is J_th=exp(3α(T-T0)).
# CTE overlay is J2-neutral. Whenever kinematics are frozen while T falls (hold
# solidification cool and closed-die cool), F_th is applied to positions+F so the
# overlay increment is matched by Δlog J instead of locking as a punch-lift kick.
T_stress_free = matrix_tmelt

# Flow-orientation kinetics (Jeffery + RSC slowdown + ARD anisotropic diffusion)
(@isdefined(fiber_aspect_ratio)) || (fiber_aspect_ratio = 10000.0)
xi_ft = (fiber_aspect_ratio^2 - 1.0) / (fiber_aspect_ratio^2 + 1.0)
ci_ft = 0.0
kappa_rsc = 0.35
ci_ard_parallel = 0.02
ci_ard_perp = 0.005
orientation_coupling_gain = 0.30

# Meso-architecture — alternating 0/90 plies with tow bands.
n_plies = max(2, env_int("TP_CLIP_N_PLIES", geometry_mode == :full3d ? 3 : 7))
const min_z_layers_per_physical_ply =
    max(1, env_int("TP_CLIP_MIN_Z_LAYERS_PER_PLY", 2))
fiber_vf_in_tow = 0.58
if haskey(ENV, "TP_CLIP_UNIFORM_FIBER_VF")
    uniform_vf = env_float("TP_CLIP_UNIFORM_FIBER_VF", fiber_vf_in_tow)
    fiber_vf_in_tow = uniform_vf
    fiber_vf_resin_rich = uniform_vf
elseif haskey(ENV, "TP_CLIP_GAP_VF")
    fiber_vf_resin_rich = env_float("TP_CLIP_GAP_VF", 0.35)
else
    fiber_vf_resin_rich = 0.02
end
tow_width = 0.0040
tow_gap = 0.0015

function clip_uniform_fiber_orientation()
    haskey(ENV, "TP_CLIP_UNIFORM_FIBER_ORIENTATION") || return nothing
    axis = lowercase(strip(ENV["TP_CLIP_UNIFORM_FIBER_ORIENTATION"]))
    if axis in ("none", "off", "ply", "qi")
        return nothing
    elseif axis in ("x", "+x", "1", "0")
        return StaticArrays.SVector(1.0, 0.0, 0.0)
    elseif axis in ("y", "+y")
        return StaticArrays.SVector(0.0, 1.0, 0.0)
    elseif axis in ("z", "+z")
        return StaticArrays.SVector(0.0, 0.0, 1.0)
    else
        error("TP_CLIP_UNIFORM_FIBER_ORIENTATION must be 'x', 'y', or 'z' (got $(axis))")
    end
end
const uniform_fiber_orientation = clip_uniform_fiber_orientation()

function apply_uniform_fiber_orientation!(fiber_dir, orient::StaticArrays.SVector{3, Float64})
    n = size(fiber_dir, 2)
    @inbounds for p in 1:n
        fiber_dir[1, p] = orient[1]
        fiber_dir[2, p] = orient[2]
        fiber_dir[3, p] = orient[3]
    end
    return nothing
end

function clip_uniform_charge_temp_k()
    if haskey(ENV, "TP_CLIP_UNIFORM_CHARGE_TEMP_C")
        return env_temp_c("TP_CLIP_UNIFORM_CHARGE_TEMP_C", 340.0)
    end
    haskey(ENV, "TP_CLIP_UNIFORM_CHARGE_TEMP_K") || return nothing
    return env_temp_c("TP_CLIP_UNIFORM_CHARGE_TEMP_K", 340.0)
end
const uniform_charge_temp_k = clip_uniform_charge_temp_k()

function apply_uniform_charge_temperature!(cyl_sys, semi_local, v_ode, temp_k::Float64)
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    t_set = clamp(temp_k, temp_min_clip, temp_max_clip)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        cyl_sys.temp[particle] = t_set
        v_cyl[NDIMS_CYL + 1, particle] = t_set
    end
    return nothing
end

# Preserve the through-thickness/in-plane temperature gradients while advancing
# a quasi-static cooling continuation by one uniform ΔT stage.
function apply_charge_temperature_delta!(cyl_sys, semi_local, v_ode, delta_T::Float64)
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        t_set = clamp(cyl_sys.temp[particle] + delta_T, temp_min_clip, temp_max_clip)
        cyl_sys.temp[particle] = t_set
        v_cyl[NDIMS_CYL + 1, particle] = t_set
    end
    return nothing
end

# Physical setup dimensions held fixed when particle spacing changes.
# These values match the current pseudo-2D baseline at particle_spacing = 0.8 mm.
charge_thickness_target = 0.0066
charge_bottom_clearance = 3.0 * particle_spacing
initial_gap_upper_target = 0.0033
base_thickness = 0.0033
holder_thickness = 0.0033
punch_side_clearance = 0.003
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
const charge_width_target = env_float("TP_CLIP_CHARGE_WIDTH_MM", 3.6) * 1e-3
if geometry_mode == :full3d
    charge_layers_y = haskey(ENV, "TP_CLIP_CHARGE_LAYERS_Y") ?
                      max(2, env_int("TP_CLIP_CHARGE_LAYERS_Y", 2)) :
                      max(2, ceil(Int, charge_width_target / particle_spacing_xy))
elseif geometry_mode == :thin3d
    charge_layers_y = max(2, env_int("TP_CLIP_CHARGE_LAYERS_Y", 3))
else
    charge_layers_y = max(1, env_int("TP_CLIP_CHARGE_LAYERS_Y", 1))
end
charge_width = charge_layers_y * particle_spacing_xy
charge_thickness = charge_thickness_target

if !use_stl_geometry
    n_charge = (ceil(Int, charge_length / particle_spacing_xy),
                charge_layers_y,
                max(2, ceil(Int, charge_thickness / particle_spacing_z)))
    # Recenter the discrete particle lattice on the geometric mid-plane.
    charge_length_discrete = n_charge[1] * particle_spacing_xy
    charge_thickness_discrete = n_charge[3] * particle_spacing_z
    charge_top_target = 0.00165 + charge_thickness_target
    charge_top_actual = charge_bottom_clearance + charge_thickness_discrete
    charge_origin = (-0.5 * charge_length_discrete, -0.5 * charge_width,
                     charge_bottom_clearance)
end
# Target compressed charge height as a fraction of the discrete lattice thickness (outer envelope).
compression_thickness_ratio = env_float("TP_CLIP_COMPRESSION_RATIO", 0.9)
compression_thickness_ratio > 0.0 && compression_thickness_ratio <= 1.0 ||
    error("TP_CLIP_COMPRESSION_RATIO must be in (0, 1] (got $(compression_thickness_ratio))")

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
# ==========================================================================================

function build_glass_fiber_charge(particle_spacing, n_charge, charge_origin;
                                  matrix_density, glass_density,
                                  n_plies, fiber_vf_in_tow, fiber_vf_resin_rich,
                                  tow_width, tow_gap,
                                  particle_spacing_xy=particle_spacing,
                                  particle_spacing_z=particle_spacing)
    nx, ny, nz = n_charge
    n_total = nx * ny * nz
    ps_xy = Float64(particle_spacing_xy)
    ps_z = Float64(particle_spacing_z)
    cell_vol = ps_xy * ps_xy * ps_z

    coordinates = Matrix{Float64}(undef, 3, n_total)
    velocity = zeros(3, n_total)
    mass = Vector{Float64}(undef, n_total)
    density = Vector{Float64}(undef, n_total)

    fiber_volume_fraction = Vector{Float64}(undef, n_total)
    fiber_direction = Matrix{Float64}(undef, 3, n_total)
    idx_fiber_rich = Int[]
    idx_matrix_rich = Int[]

    x0, y0, z0 = charge_origin
    ply_thickness = nz > 0 ? (nz * ps_z) / n_plies : ps_z
    tow_cols_pseudo2d = max(1, round(Int, tow_width / ps_xy))
    gap_cols_pseudo2d = max(1, round(Int, tow_gap / ps_xy))
    tow_pitch_cols_pseudo2d = tow_cols_pseudo2d + gap_cols_pseudo2d

    @inline function particle_in_tow_band(i, j, ply_aligned_x)
        if ny == 1
            # Single y-layer slab: tow columns indexed along x (legacy pseudo-2D).
            center_offset_twice = 2 * i - nx - 1
            phase_shift_twice = ply_aligned_x ? tow_pitch_cols_pseudo2d : 0
        elseif ply_aligned_x
            # 0° ply: fibres along x → tow/gap bands across y (mirror-symmetric in j).
            center_offset_twice = 2 * j - ny - 1
            phase_shift_twice = tow_pitch_cols_pseudo2d
        else
            # 90° ply: fibres along y → tow/gap bands across x (mirror-symmetric in i).
            center_offset_twice = 2 * i - nx - 1
            phase_shift_twice = 0
        end
        wrapped_offset_twice = mod(center_offset_twice + phase_shift_twice +
                                   tow_pitch_cols_pseudo2d,
                                   2 * tow_pitch_cols_pseudo2d) - tow_pitch_cols_pseudo2d
        return abs(wrapped_offset_twice) <= tow_cols_pseudo2d - 1
    end

    p = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        p += 1

        x = x0 + (i - 0.5) * ps_xy
        y = y0 + (j - 0.5) * ps_xy
        z = z0 + (k - 0.5) * ps_z

        coordinates[1, p] = x
        coordinates[2, p] = y
        coordinates[3, p] = z

        z_rel = max(0.0, z - z0)
        ply_id = clamp(floor(Int, z_rel / ply_thickness) + 1, 1, n_plies)
        ply_aligned_x = isodd(ply_id)

        in_tow = particle_in_tow_band(i, j, ply_aligned_x)

        vf_local = in_tow ? fiber_vf_in_tow : fiber_vf_resin_rich
        fiber_volume_fraction[p] = vf_local

        if uniform_fiber_orientation !== nothing
            fiber_direction[1, p] = uniform_fiber_orientation[1]
            fiber_direction[2, p] = uniform_fiber_orientation[2]
            fiber_direction[3, p] = uniform_fiber_orientation[3]
        elseif ply_aligned_x
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
        mass[p] = rho_local * cell_vol

        if vf_local >= 0.5 * fiber_vf_in_tow
            push!(idx_fiber_rich, p)
        else
            push!(idx_matrix_rich, p)
        end
    end

    ic = InitialCondition(; coordinates, velocity, mass, density,
                          particle_spacing=Float64(particle_spacing))

    return ic, fiber_volume_fraction, fiber_direction, idx_fiber_rich, idx_matrix_rich
end

factor = 1.1
# Kernel compact support sets the physical SPH NHS / force search radius:
#   quintic (default) → 3h ; quartic → 2.5h ; wendland_c2 → 2h
# Pair with TP_CLIP_JAC_RADIUS_MULT ≥ support/h (e.g. quartic + jac 2.5).
const _clip_smoothing_kernel_name = lowercase(get(ENV, "TP_CLIP_SMOOTHING_KERNEL", "quintic"))
smoothing_kernel = if _clip_smoothing_kernel_name in ("wendland_c2", "wendlandc2", "wendland", "c2")
    WendlandC2Kernel{3}()
elseif _clip_smoothing_kernel_name in ("quartic", "schoenberg_quartic", "spline4")
    SchoenbergQuarticSplineKernel{3}()
elseif _clip_smoothing_kernel_name in ("quintic", "schoenberg_quintic", "spline5")
    SchoenbergQuinticSplineKernel{3}()
else
    error("TP_CLIP_SMOOTHING_KERNEL must be 'quintic', 'quartic', or 'wendland_c2' " *
          "(got $(_clip_smoothing_kernel_name))")
end
smoothing_length = factor * particle_spacing
tool_smoothing_length = factor * tool_particle_spacing
_clip_kernel_support_over_h = TrixiParticles.compact_support(smoothing_kernel,
                                                            smoothing_length) /
                              max(smoothing_length, eps(Float64))
println("  - smoothing kernel: ", typeof(smoothing_kernel),
        " (compact support ≈ ", round(_clip_kernel_support_over_h; digits=2), " h)")

const stl_molding_geometry = Ref{Any}(nothing)
if use_stl_geometry
    stl_molding_geometry[] = clip_setup_stl_molding_geometry(;
        particle_spacing=particle_spacing,
        particle_spacing_xy=particle_spacing_xy,
        particle_spacing_z=particle_spacing_z,
        tool_particle_spacing=tool_particle_spacing,
        geometry_mode=geometry_mode,
        charge_layers_y=charge_layers_y,
        smoothing_kernel=smoothing_kernel,
        smoothing_length=smoothing_length,
        matrix_density=matrix_density,
        glass_density=glass_density,
        n_plies=n_plies,
        fiber_vf_in_tow=fiber_vf_in_tow,
        fiber_vf_resin_rich=fiber_vf_resin_rich,
        tow_width=tow_width,
        tow_gap=tow_gap,
        env_float=env_float)
    geo = stl_molding_geometry[]
    polymer = geo.polymer
    fiber_volume_fraction = geo.fiber_volume_fraction
    fiber_direction = geo.fiber_direction
    idx_fiber_rich = geo.idx_fiber_rich
    idx_matrix_rich = geo.idx_matrix_rich
    floor_particles = geo.floor_particles
    mold_particles = geo.mold_particles
    charge_length = geo.charge_length
    charge_width = geo.charge_width
    charge_thickness = geo.charge_thickness
    charge_length_discrete = geo.charge_length_discrete
    charge_width_discrete = geo.charge_width_discrete
    charge_thickness_discrete = geo.charge_thickness_discrete
else
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
                                 particle_spacing_xy=particle_spacing_xy,
                                 particle_spacing_z=particle_spacing_z)
end

# In-plane random fiber directions (Song wet-laid sheet). Homogeneous Vf via
# TP_CLIP_UNIFORM_FIBER_VF; this replaces 0/90 tow directions only.
if env_int("TP_CLIP_RANDOM_FIBER_ORIENTATION", 0) != 0
    Random.seed!(env_int("TP_CLIP_RANDOM_FIBER_SEED", 1))
    n_orient = size(fiber_direction, 2)
    @inbounds for p in 1:n_orient
        θ = 2.0 * π * rand()
        fiber_direction[1, p] = cos(θ)
        fiber_direction[2, p] = sin(θ)
        fiber_direction[3, p] = 0.0
    end
    println("  - fiber orientation: in-plane random (TP_CLIP_RANDOM_FIBER_ORIENTATION=1, seed=",
            env_int("TP_CLIP_RANDOM_FIBER_SEED", 1), ")")
end

# Opt-in physical ply ownership. Keep the packed `polymer` and all packed
# material arrays unchanged as the canonical global particle ordering; each
# source index below maps a ply-local particle back to that ordering.
charge_initial_conditions, charge_ply_source_indices =
    if separate_charge_plies
        plies, source_indices =
            clip_split_charge_initial_condition(polymer, n_plies;
                                                layer_tolerance=0.25 *
                                                                particle_spacing_z,
                                                min_layers_per_ply=
                                                    min_z_layers_per_physical_ply)
        println("  - physically separate charge plies: ", length(plies),
                " TLSPH systems; particle counts = ", join(length.(source_indices), ", "))
        println("  - resolved z-layer requirement: ≥",
                min_z_layers_per_physical_ply, " layers per physical ply")
        println("  - physical ply mass check: ",
                sum(sum(ply.mass) for ply in plies), " kg (packed charge ",
                sum(polymer.mass), " kg)")
        ply_angles = let raw = strip(get(ENV, "TP_CLIP_PLY_ORIENTATIONS_DEG", "0,90,45,-45"))
            [deg2rad(parse(Float64, s)) for s in split(raw, ',')]
        end
        for (ply_i, source) in enumerate(source_indices)
            θ = ply_angles[mod1(ply_i, length(ply_angles))]
            cθ, sθ = cos(θ), sin(θ)
            @inbounds for p in source
                fiber_direction[1, p] = cθ
                fiber_direction[2, p] = sθ
                fiber_direction[3, p] = 0.0
            end
        end
        println("  - separate-ply fibre angles (deg) = ",
                join(round.(rad2deg.(ply_angles); digits=1), ", "),
                " repeating over ", length(source_indices), " plies")
        plies, source_indices
    else
        [polymer], [collect(1:nparticles(polymer))]
    end

final_thickness_target = compression_thickness_ratio * charge_thickness_discrete

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

const organosheet_mechanics = env_int("TP_CLIP_ORGANOSHEET_MECHANICS", 0) != 0
const organosheet_architecture =
    lowercase(strip(get(ENV, "TP_CLIP_ORGANOSHEET_ARCHITECTURE", "woven")))
organosheet_architecture in ("woven", "ud", "qi", "quasi_isotropic") ||
    error("TP_CLIP_ORGANOSHEET_ARCHITECTURE must be 'woven', 'ud', or 'qi'")
const organosheet_ply_model = OrganosheetPlyModel(
    fiber_stiffness=env_float("TP_CLIP_ORGANOSHEET_FIBER_STIFFNESS_PA", mean(E1_particle)),
    shear_stiffness=env_float("TP_CLIP_ORGANOSHEET_SHEAR_STIFFNESS_PA",
                              mean(E2_particle) /
                              (2 * (1 + ((@isdefined(matrix_nu)) ? matrix_nu : 0.3)))),
    locking_angle=deg2rad(env_float("TP_CLIP_ORGANOSHEET_LOCKING_ANGLE_DEG", 45.0)),
    locking_stiffness=env_float("TP_CLIP_ORGANOSHEET_LOCKING_STIFFNESS_PA",
                                mean(E1_particle)),
    locking_exponent=max(1, env_int("TP_CLIP_ORGANOSHEET_LOCKING_EXPONENT", 3)))
const organosheet_active_in_melt = env_int("TP_CLIP_ORGANOSHEET_ACTIVE_IN_MELT", 1) != 0
organosheet_mechanics &&
    println("  - organosheet constitutive mechanics: ON (", organosheet_architecture,
            "; matrix plasticity remains matrix-only; molten fiber stress = ",
            organosheet_active_in_melt ? "ACTIVE" : "OFF", ")")
# #region agent log
if organosheet_mechanics
    println("  - organosheet stiffness [Pa]: fiber=", organosheet_ply_model.fiber_stiffness,
            " shear=", organosheet_ply_model.shear_stiffness,
            " lock=", organosheet_ply_model.locking_stiffness,
            " | UL=", get(ENV, "TP_CLIP_UPDATED_LAGRANGE", "?"),
            " DT0=", get(ENV, "TP_CLIP_DT0", "?"),
            " mu=", get(ENV, "TP_CLIP_MU_FRICTION", "?"))
    try
        open("/home/sadaat/SPH_Code/TrixiParticles.jl_Saadat/.cursor/debug-b1cbaf.log", "a") do _dbg_io
            _ts = round(Int, time() * 1000)
            println(_dbg_io,
                "{\"sessionId\":\"b1cbaf\",\"runId\":\"post-fix\",\"hypothesisId\":\"H1\",\"location\":\"stamping_cfk_clip_pseudo2d_implicit.jl:organosheet\",\"message\":\"organosheet stiffness at startup\",\"timestamp\":",
                _ts,
                ",\"data\":{\"architecture\":\"", organosheet_architecture,
                "\",\"fiber_Pa\":", organosheet_ply_model.fiber_stiffness,
                ",\"shear_Pa\":", organosheet_ply_model.shear_stiffness,
                ",\"lock_Pa\":", organosheet_ply_model.locking_stiffness,
                ",\"UL\":\"", get(ENV, "TP_CLIP_UPDATED_LAGRANGE", "?"),
                "\",\"DT0\":\"", get(ENV, "TP_CLIP_DT0", "?"),
                "\",\"mu\":\"", get(ENV, "TP_CLIP_MU_FRICTION", "?"),
                "\",\"session\":\"", get(ENV, "SESSION", "?"), "\"}}")
        end
    catch
    end
end
# #endregion

# Homogenized stack: QI (or user) fibre angles by through-thickness ply ID.
# Separate-ply mode already assigned these from the source maps above.
if organosheet_mechanics && !separate_charge_plies
    ply_ids = assign_organosheet_ply_ids(polymer.coordinates, n_plies)
    ply_angles = let raw = strip(get(ENV, "TP_CLIP_PLY_ORIENTATIONS_DEG", "0,90,45,-45"))
        [deg2rad(parse(Float64, s)) for s in split(raw, ',')]
    end
    @inbounds for particle in eachindex(ply_ids)
        θ = ply_angles[mod1(ply_ids[particle], length(ply_angles))]
        fiber_direction[1, particle] = cos(θ)
        fiber_direction[2, particle] = sin(θ)
        fiber_direction[3, particle] = 0.0
    end
    println("  - homogenized ply fibre angles (deg) = ",
            join(round.(rad2deg.(ply_angles); digits=1), ", "),
            " on ", n_plies, " material ply IDs (one TLSPH charge system)")
end

@inline function clip_organosheet_weft_direction(warp)
    organosheet_architecture == "ud" && return nothing
    weft = StaticArrays.SVector(-warp[2], warp[1], 0.0)
    if norm(weft) <= sqrt(eps(Float64))
        return StaticArrays.SVector(0.0, 1.0, 0.0)
    end
    return weft / norm(weft)
end

@inline function clip_organosheet_reinforcement_stress(F, warp, model::OrganosheetPlyModel,
                                                       arch::AbstractString)
    if arch == "ud"
        return organosheet_fiber_pk1(F, warp, model) * transpose(F)
    elseif arch == "woven"
        weft = clip_organosheet_weft_direction(warp)
        return organosheet_kirchhoff_stress(F, warp, weft, model)
    elseif arch in ("qi", "quasi_isotropic")
        # Homogenized [0, 90, 45, -45]s stack represented as two balanced orthogonal pairs:
        # Pair 1: (0°, 90°)
        weft1 = StaticArrays.SVector(-warp[2], warp[1], 0.0)
        weft1_u = norm(weft1) > sqrt(eps(Float64)) ? weft1 / norm(weft1) : StaticArrays.SVector(0.0, 1.0, 0.0)
        tau_pair1 = organosheet_kirchhoff_stress(F, warp, weft1_u, model)

        # Pair 2: (+45°, -45°) rotated by 45° in-plane from warp
        inv_sqrt2 = 0.7071067811865476
        warp2 = StaticArrays.SVector(inv_sqrt2 * (warp[1] - warp[2]),
                                     inv_sqrt2 * (warp[1] + warp[2]),
                                     0.0)
        warp2_u = norm(warp2) > sqrt(eps(Float64)) ? warp2 / norm(warp2) : StaticArrays.SVector(inv_sqrt2, inv_sqrt2, 0.0)
        weft2 = StaticArrays.SVector(-warp2_u[2], warp2_u[1], 0.0)
        weft2_u = norm(weft2) > sqrt(eps(Float64)) ? weft2 / norm(weft2) : StaticArrays.SVector(-inv_sqrt2, inv_sqrt2, 0.0)
        tau_pair2 = organosheet_kirchhoff_stress(F, warp2_u, weft2_u, model)

        # 50% volume fraction weighting per balanced pair
        return 0.5 * (tau_pair1 + tau_pair2)
    else
        return organosheet_kirchhoff_stress(F, warp, nothing, model)
    end
end

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
# Mechanical mass scaling must not slow boundary cooling. Internal SPH conduction
# retains the existing scaled-density/k calibration, while ambient convection and
# radiation use the physical composite density.
@inline physical_thermal_density(cyl_sys, particle) =
    cyl_sys.material_density[particle] / mass_scaling
const contact_htc_scale = env_float("TP_CLIP_CONTACT_HTC_SCALE", 1.0)
if contact_htc_scale != 1.0
    h_contact_particle .*= contact_htc_scale
    println(">>> Contact HTC scale = ", contact_htc_scale,
            "  (mean h=", round(sum(h_contact_particle) / length(h_contact_particle); sigdigits=3),
            " W/m²K)")
end

orientation_scalar_particle = ones(length(phase_vf_particle))

viscosity_min_clip = env_float("TP_CLIP_VISCOSITY_MIN_CLIP", 1.0e2)
viscosity_max_clip = env_float("TP_CLIP_VISCOSITY_MAX_CLIP", 5.0e7)
# Cross–WLF update smoothing: 1.0 = instant (legacy); lower values lag η toward the
# target law and reduce TRBDF2/implicit-Euler dt collapse from sharp η jumps at contact.
viscosity_update_alpha = clamp(env_float("TP_CLIP_VISCOSITY_UPDATE_ALPHA", 1.0), 1.0e-4, 1.0)
# Cross–WLF η refresh cadence. RHS interval 0 = skip during implicit Newton (update on
# accepted steps only). N≥1 = recompute η every N RHS evaluations.
const viscosity_rhs_update_interval = env_int("TP_CLIP_VISCOSITY_UPDATE_INTERVAL", 0)
const viscosity_step_update_interval = max(1, env_int("TP_CLIP_VISCOSITY_STEP_UPDATE_INTERVAL", 1))
const viscosity_rhs_call_count = Ref(0)
const viscosity_step_count = Ref(0)
viscosity_update_alpha < 1.0 - 1.0e-12 &&
    println("  - Cross–WLF viscosity smoothing: alpha=", viscosity_update_alpha,
            " (log-blend per η update)")
(viscosity_rhs_update_interval != 1 || viscosity_step_update_interval != 1) &&
    println("  - viscosity update cadence: RHS every ",
            viscosity_rhs_update_interval == 0 ? "never (accepted steps only)" :
            string(viscosity_rhs_update_interval),
            ", accepted step every ", viscosity_step_update_interval,
            viscosity_rhs_update_interval == 0 ?
                " (Cross–WLF incl. shear rate via vel_grad on accepted steps)" : "")
temp_min_clip = 200.0
temp_max_clip = 2000.0

println(use_stl_geometry ?
        "✓ CFK raw charge initialized (STL + packing):" :
        "✓ CFK raw charge initialized (pseudo-2D slab):")
println("  - size [mm] = ", round.(StaticArrays.SVector(charge_length_discrete, charge_width,
                                              charge_thickness_discrete) .* 1e3,
                                      digits=2))
println("  - particles = ", nparticles(polymer))
println("  - physical charge mass = ",
        round(sum(polymer.mass) / mass_scaling * 1e3; digits=3),
        " g (SPH mass / TP_CLIP_MASS_SCALING=", mass_scaling, ")")
haskey(ENV, "TP_CLIP_UNIFORM_FIBER_VF") &&
    println("  - uniform fibre Vf = ", round(fiber_vf_in_tow, digits=4),
            " (TP_CLIP_UNIFORM_FIBER_VF; tow/gap bands disabled in material)")
uniform_fiber_orientation !== nothing &&
    println("  - uniform fibre orientation = [",
            uniform_fiber_orientation[1], ", ",
            uniform_fiber_orientation[2], ", ",
            uniform_fiber_orientation[3], "] (TP_CLIP_UNIFORM_FIBER_ORIENTATION)")
uniform_charge_temp_k !== nothing &&
    println("  - uniform charge temperature = ", fmt_temp(uniform_charge_temp_k),
            " (TP_CLIP_UNIFORM_CHARGE_TEMP_C; applied after checkpoint restore)")
clip_tensile_stress !== nothing &&
    println("  - tensile artificial stress: Monaghan psi=",
            clip_tensile_stress.psi, " (TP_CLIP_TENSILE_PSI)")
haskey(ENV, "TP_CLIP_GAP_VF") && !haskey(ENV, "TP_CLIP_UNIFORM_FIBER_VF") &&
    println("  - gap / resin-rich Vf = ", round(fiber_vf_resin_rich, digits=4),
            " (TP_CLIP_GAP_VF)")
println("  - architecture = ", n_plies, " plies (alternating), tow width=", tow_width * 1e3,
        " mm, tow gap=", tow_gap * 1e3, " mm")
println("  - mean fibre volume fraction Vf = ", round(vf_mean, digits=4))
println("  - phase counts (fibre-rich/matrix-rich) = ", length(idx_fiber_rich), " / ",
        length(idx_matrix_rich))
println("  - homogenized rho = ", round(density_cylinder, digits=2),
        " kg/m^3, E = ", round(E_composite_eff / 1e9, digits=3), " GPa")
println("  - effective conductivity scale = ", thermal_conductivity_scale,
    "x (pseudo-2D accelerated cooling)")
println("  - phase-specific tmelt matrix/glass = ", fmt_temp(matrix_tmelt), " / ",
        fmt_temp(glass_tmelt))
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
if !use_stl_geometry
    initial_gap       = 2 * smoothing_length
    initial_gap_upper = max(initial_gap_upper_target, 2 * smoothing_length)
end

if sim_phase == "retraction"
    # Checkpoint already loaded before packing (hold_checkpoint_early); reuse it.
    hold_checkpoint_early[] === nothing &&
        (hold_checkpoint_early[] = load_hold_checkpoint(checkpoint_path))
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

# Reference frame / tooling: procedural trapezoidal cavity, or STL-packed user geometry.
use_pseudo2d_projection = geometry_mode == :pseudo2d
apply_y_plane_constraint = if haskey(ENV, "TP_CLIP_APPLY_Y_PLANE_CONSTRAINT")
    env_int("TP_CLIP_APPLY_Y_PLANE_CONSTRAINT", 1) != 0
else
    geometry_mode in (:pseudo2d, :thin3d)
end

let c_el = sqrt(matrix_E / matrix_density)
    s = env_float("TP_CLIP_Y_CONSTRAINT_SCALE", 1.0)
    global y_constraint_damp   = sqrt(s) * 20.0 * c_el / particle_spacing
    global y_constraint_spring = (0.5 * y_constraint_damp)^2
end
const y_constraint_explicit_scale = if !retraction_use_explicit
    1.0
elseif haskey(ENV, "TP_CLIP_Y_CONSTRAINT_EXPLICIT_SCALE")
    env_float("TP_CLIP_Y_CONSTRAINT_EXPLICIT_SCALE", 1.0)
else
    omega_n = 0.5 * y_constraint_damp
    min(1.0, env_float("TP_CLIP_Y_CONSTRAINT_EXPLICIT_SAFETY", 0.9) * 2.0 /
        (retraction_explicit_dt_springback * omega_n))
end

@inline function thin3d_y_constraint_coeffs()
    if y_constraint_explicit_scale >= 1.0 - 1.0e-12
        return y_constraint_damp, y_constraint_spring
    end
    s = y_constraint_explicit_scale
    return s * y_constraint_damp, s * s * y_constraint_spring
end

let c_el = sqrt(matrix_E / matrix_density)
    s = env_float("TP_CLIP_RESTING_SHELF_Z_SCALE", 1.0)
    global shelf_z_constraint_damp   = sqrt(s) * 20.0 * c_el / particle_spacing
    global shelf_z_constraint_spring = (0.5 * shelf_z_constraint_damp)^2
end
const shelf_z_constraint_explicit_scale = if !compression_use_explicit
    1.0
elseif haskey(ENV, "TP_CLIP_SHELF_Z_CONSTRAINT_EXPLICIT_SCALE")
    env_float("TP_CLIP_SHELF_Z_CONSTRAINT_EXPLICIT_SCALE", 1.0)
else
    omega_n = 0.5 * shelf_z_constraint_damp
    min(1.0, env_float("TP_CLIP_SHELF_Z_CONSTRAINT_EXPLICIT_SAFETY", 0.9) * 2.0 /
        (compression_explicit_dt * omega_n))
end
const apply_resting_shelf_z_constraint = if haskey(ENV, "TP_CLIP_RESTING_SHELF_Z_CONSTRAINT")
    env_int("TP_CLIP_RESTING_SHELF_Z_CONSTRAINT", 0) != 0
else
    !clip_blankholder_enabled() && apply_y_plane_constraint
end

# Constraint-free thin3d: full 3D WCSPH pressure/viscosity resists out-of-plane bulging via
# neighbor bulk pressure (no y/z spring penalties that stiffen the implicit Jacobian).
const wcsph_full3d_pressure = if haskey(ENV, "TP_CLIP_WCSPH_FULL3D")
    env_int("TP_CLIP_WCSPH_FULL3D", 0) != 0
elseif geometry_mode == :thin3d && !apply_y_plane_constraint
    true
else
    false
end
const wcsph_bulk_scale = env_float("TP_CLIP_WCSPH_BULK_SCALE", 1.0)
const wcsph_bulk_viscosity_pa_s = if haskey(ENV, "TP_CLIP_WCSPH_BULK_VISCOSITY_PA_S")
    max(0.0, env_float("TP_CLIP_WCSPH_BULK_VISCOSITY_PA_S", 0.0))
elseif wcsph_full3d_pressure
    1.0e6   # Pa·s volumetric damping when y/z kinematic constraints are off
else
    0.0
end
const enable_wcsph_liquid_viscosity = if haskey(ENV, "TP_CLIP_WCSPH_LIQUID_VISCOSITY")
    env_int("TP_CLIP_WCSPH_LIQUID_VISCOSITY", 0) != 0
else
    wcsph_full3d_pressure
end
const lag_wcsph_liquid_viscosity = true

@inline function resting_shelf_z_constraint_coeffs()
    if shelf_z_constraint_explicit_scale >= 1.0 - 1.0e-12
        return shelf_z_constraint_damp, shelf_z_constraint_spring
    end
    s = shelf_z_constraint_explicit_scale
    return s * shelf_z_constraint_damp, s * s * shelf_z_constraint_spring
end

@info "Y-layer mode" geometry_mode charge_layers_y use_pseudo2d_projection apply_y_plane_constraint charge_width_mm=(charge_width*1e3) particle_spacing_mm=(particle_spacing*1e3)
wcsph_full3d_pressure &&
    println("  - WCSPH full-3D pressure: ON (bulk confinement via neighbor pressure, not y/z springs)")
enable_wcsph_liquid_viscosity &&
    println("  - WCSPH liquid viscosity: ON (lagged current-config shear dissipation)")
wcsph_bulk_viscosity_pa_s > 0.0 &&
    println("  - WCSPH bulk viscosity: ζ=", wcsph_bulk_viscosity_pa_s, " Pa·s")
wcsph_bulk_scale < 1.0 - 1.0e-12 &&
    println("  - WCSPH bulk scale: ", wcsph_bulk_scale, " (softened melt bulk modulus)")
geometry_mode == :thin3d &&
    @info "Thin-3D mode active: for similar runtime to 1-layer, set TP_CLIP_PS_MM=$(round(0.8*sqrt(Float64(charge_layers_y)), digits=2))"
geometry_mode == :full3d &&
    @info "Full-3D mode: resolved y-width $(round(charge_width*1e3; digits=2)) mm ($(charge_layers_y) layers); no y hard-lock"
const retraction_lock_y = sim_phase == "retraction" && geometry_mode == :thin3d &&
    env_int("TP_CLIP_RETRACTION_LOCK_Y", 1) != 0

@inline function hard_lock_y_kinematics()
    return use_pseudo2d_projection || retraction_lock_y
end

# Default OFF: full-3D y velocity/position caps are unphysical; enable only for diagnostics.
const retraction_full3d_y_cap = sim_phase == "retraction" && geometry_mode == :full3d &&
    env_int("TP_CLIP_RETRACTION_FULL3D_Y_CAP", 0) != 0

@inline function retraction_full3d_y_cap_active()
    return retraction_full3d_y_cap
end

retraction_lock_y && println("Retraction: thin-3D y kinematics hard-locked (v_y=du_y=dv_y=0; layers fixed)")
sim_phase == "retraction" && geometry_mode == :thin3d && !retraction_lock_y &&
    println("Retraction: thin-3D y spring-damper active (lock_y off)",
            retraction_use_explicit ?
                "; explicit y-scale=$(round(y_constraint_explicit_scale; sigdigits=3))" :
                "")

if use_stl_geometry
    geo = stl_molding_geometry[]
    z_base_top = geo.z_base_top
    z_base_bot = geo.z_base_bot
    z_blocks_top = z_base_top + geo.cavity_depth
    z_punch_bot = geo.z_punch_bot
    z_punch_top = geo.z_punch_top
    tool_length = geo.tool_length
    cavity_depth = geo.cavity_depth
    initial_gap_upper = geo.gap_punch
    initial_stroke = z_punch_bot - (z_base_top + final_thickness_target)
else
# Reference frame: cavity FLOOR at z = 0 (this is where the charge sits).
z_base_top   = 0.0
z_blocks_top = z_base_top + cavity_depth
z_base_bot   = z_base_top - base_layers * particle_spacing

nx_tool = ceil(Int, tool_length / particle_spacing)
ny_tool = charge_layers_y
tool_origin_y = -0.5 * ny_tool * particle_spacing
tool_origin_x = -0.5 * tool_length

# Redefined geometry: set bottom width directly, derive draft angle.
cavity_top_half_w = 0.5 * cavity_internal_width
cavity_bot_half_w = 0.5 * (charge_length + 0.003)
wall_inset = cavity_top_half_w - cavity_bot_half_w
draft_angle = atan(wall_inset / cavity_depth)
@info "Derived draft angle for wider cavity" angle_deg=rad2deg(draft_angle)

@inline function cavity_half_width_at(z)
    frac = (z - z_base_top) / (z_blocks_top - z_base_top)
    frac = clamp(frac, 0.0, 1.0)
    return cavity_bot_half_w + frac * (cavity_top_half_w - cavity_bot_half_w)
end

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
end  # !use_stl_geometry

floor_z_max = maximum(floor_particles.coordinates[3,:])
mold_z_min  = minimum(mold_particles.coordinates[3,:])
if use_stl_geometry
    println("✓ Compression-mold tooling built (STL / user L-die):")
    println("  - tool extent x [mm] = ", round(tool_length * 1e3; digits=2))
    println("  - charge size [mm] = ", round.(StaticArrays.SVector(charge_length, charge_width,
                                                  charge_thickness) .* 1e3; digits=2))
    println("  - gap (mold bot ↔ charge top) = ", round(mold_z_min - cyl_z_max; digits=6),
            " m, initial stroke to seat = ", round(initial_stroke * 1e3; digits=3), " mm")
    println("  - h_charge = ", smoothing_length * 1e3, " mm, h_tool = ",
            tool_smoothing_length * 1e3, " mm")
    println("  - Tool particles (floor/mold) = ", nparticles(floor_particles), " / ",
            nparticles(mold_particles))
else
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
end

# Numerical safety rails for pseudo-2D debug runs.
# With all fundamental issues fixed (dt sync, 2D kernel correction, and plane-strain),
# these artificial limiters are no longer needed and just ruin Newton convergence.
enable_safety_clamps = env_int("TP_CLIP_SAFETY_CLAMPS", 0) != 0
enable_velocity_safety_clamps = env_int("TP_CLIP_VELOCITY_SAFETY_CLAMPS", 0) != 0
# F/Fp repair runs in the accepted-step callback only. Doing it inside kick_implicit!
# corrupts the finite-difference Jacobian (each column perturbation can flip the
# J-threshold check and write a different F into shared buffers).
enable_emergency_repair_clamp = env_int("TP_CLIP_EMERGENCY_REPAIR_CLAMP", 0) != 0
x_safety_bound = 0.5 * tool_length + safety_margin
z_safety_min = z_base_bot - safety_margin
z_safety_max = z_punch_top + safety_margin
y_safety_min = -0.5 * charge_width - safety_margin
y_safety_max = 0.5 * charge_width + safety_margin
max_cylinder_speed = 2.0      # m/s
retraction_full3d_y_cap &&
    println("Retraction: full-3D y caps ON (|v_y|≤", max_cylinder_speed,
            " m/s, |dv_y|≤retraction accel cap; y∈[",
            round(y_safety_min * 1e3; digits=2), ", ",
            round(y_safety_max * 1e3; digits=2), "] mm)")
sim_phase == "retraction" && geometry_mode == :full3d && !retraction_full3d_y_cap &&
    println("Retraction: full-3D y caps OFF (TP_CLIP_RETRACTION_FULL3D_Y_CAP=0)")
max_cylinder_accel = 1.0e5    # m/s^2
# Stable capped retraction default: global kick |a| clamp (hardClamp_noY). Set to 0 for nocap.
const retraction_accel_clamp = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_ACCEL_CLAMP", 1) != 0
const retraction_max_accel = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_MAX_ACCEL", 500.0) : max_cylinder_accel
# Deep-overlap hard |a|/|v| while tools penetrate (ON by default with accel clamp path).
const retraction_explicit_kick_soft_limit = sim_phase == "retraction" && retraction_use_explicit &&
    env_int("TP_CLIP_RETRACTION_EXPLICIT_KICK_SOFT_LIMIT", 1) != 0
# While charge–tool overlap is deeper than this (·ps), apply tight HARD clamps on
# accel and velocity (needed to avoid σ-ramp detonation under deep floor overlap).
const retraction_deep_contact_kick_gap_ps = sim_phase == "retraction" &&
    retraction_explicit_kick_soft_limit ?
    env_float("TP_CLIP_RETRACTION_DEEP_CONTACT_KICK_GAP_PS", -0.25) : -Inf
const retraction_deep_contact_accel_cap = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_DEEP_CONTACT_ACCEL_CAP", 500.0) : Inf
const retraction_deep_contact_speed_cap = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_DEEP_CONTACT_SPEED_CAP", 0.01) : Inf
retraction_deep_contact_kick_active = Ref(false)
# Targeted limiter for liquid/mushy mechanics (WCSPH branch). This is applied
# even when global safety clamps are disabled, since the dt-collapse trigger is
# liquid-force acceleration spikes in the latent regime.
# Overwritten after melt-aware contact caps are configured (Layer 4 alignment).
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
# NOTE: global baseline left at 0 for compress/hold (artificial damping can suppress
# viscous gradients). Retraction demold enables Rayleigh + contact dashpot via ENV.
velocity_damping_base = env_float("TP_CLIP_VELOCITY_DAMPING_BASE", 0.0)      # 1/s
velocity_damping_contact = env_float("TP_CLIP_VELOCITY_DAMPING_CONTACT", 0.0) # 1/s
# Retraction demold: mass-proportional Rayleigh (a -= γ v) + near-tool boost.
# Defaults ON for retraction so elastic kick can dissipate without hard |a|/|v| clamps.
retraction_rayleigh_damp = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_RAYLEIGH_DAMP", 200.0) : 0.0          # 1/s
retraction_rayleigh_damp_contact = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_RAYLEIGH_DAMP_CONTACT", 2.0e3) : 0.0  # 1/s
# Soft viscoplastic overstress ceiling on charge |a| while deep tool overlap remains.
# Continuous tanh law (not a hard clamp): a ← a_y·tanh(|a|/a_y)·â. 0 = off.
retraction_viscoplastic_accel_yield = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_VISCOPLASTIC_ACCEL_YIELD", 5.0e3) : 0.0  # m/s²
retraction_viscoplastic_gap_ps = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_VISCOPLASTIC_GAP_PS", -0.25) : -Inf
# Cap INTERNAL (pre-contact) charge |a| only; Reimann/deep contact stays uncapped.
# Soft magnitude law: a ← a_cap·tanh(|a|/a_cap)·â. 0 = off.
retraction_internal_accel_cap = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_INTERNAL_ACCEL_CAP", 0.0) : 0.0  # m/s²
molten_wall_drag_factor = env_float("TP_CLIP_MOLTEN_WALL_DRAG_FACTOR", 0.0005)
molten_wall_drag_max_rate = env_float("TP_CLIP_MOLTEN_WALL_DRAG_MAX_RATE", 1.0e5) # 1/s

# Target stroke: drive the punch from its start position (z_punch_bot) down until
# the charge is compressed to `final_thickness_target` (= compression_thickness_ratio × initial).
target_stroke = z_punch_bot - (z_base_top + final_thickness_target)
const _clip_cad_data_dir = get(ENV, "TP_CLIP_CAD_DIR",
                               joinpath(@__DIR__, "..", "preprocessing", "data_user_l_die"))
const _clip_user_l_die_pack = use_stl_geometry && clip_pack_use_user_l_die_tools(_clip_cad_data_dir)
if !haskey(ENV, "TP_CLIP_TARGET_STROKE_MM") &&
   env_int("TP_CLIP_SEAT_PUNCH_TO_MOLD", _clip_user_l_die_pack ? 1 : 0) != 0 &&
   _clip_user_l_die_pack
    target_stroke = clip_user_l_die_mating_stroke(mold_particles)
    println(">>> User L-die: seat punch to mold (mating stroke) = ",
            round(target_stroke * 1e3; digits=3), " mm")
elseif haskey(ENV, "TP_CLIP_TARGET_STROKE_MM")
    _stroke_raw = lowercase(strip(get(ENV, "TP_CLIP_TARGET_STROKE_MM", "")))
    if _stroke_raw in ("auto", "seat", "geometry")
        # Seat punch to compressed thickness on cavity floor:
        #   stroke = z_punch_bot − (z_base_top + final_thickness_target)
        # For STL packs, z_base_top is the packed cavity-floor z (see clip_stl_geometry_setup).
        target_stroke = max(0.0, z_punch_bot - (z_base_top + final_thickness_target))
        println(">>> Target stroke from geometry (TP_CLIP_TARGET_STROKE_MM=",
                _stroke_raw, ") = ", round(target_stroke * 1e3; digits=3),
                " mm  [punch_bot=", round(z_punch_bot * 1e3; digits=3),
                " floor=", round(z_base_top * 1e3; digits=3),
                " final_t=", round(final_thickness_target * 1e3; digits=3), " mm]")
    else
        target_stroke = max(0.0, env_float("TP_CLIP_TARGET_STROKE_MM", 0.0) * 1e-3)
        println(">>> Override: TP_CLIP_TARGET_STROKE_MM = ",
                round(target_stroke * 1e3; digits=3), " mm")
    end
end

tp_clip_max_press_force_n = env_float("TP_CLIP_MAX_PRESS_FORCE_N", 0.0)
const press_force_locked = Ref(false)
const press_force_locked_z = Ref(0.0)
const punch_contact_force_z = Ref(0.0)

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
# Optional: stop hold when mean charge temperature (°C) cools to this value.
# When set, this replaces the solidification-fraction completion criterion.
hold_until_temp_C = env_float("TP_CLIP_HOLD_UNTIL_TEMP_C", NaN)
# ODE tspan upper-bound when hold is unlimited (hold/hold-tail terminate earlier).
solidification_hold_plan_s = env_float("TP_CLIP_SOLIDIFICATION_HOLD_PLAN_S", 1200.0)
solidification_hold_budget = isfinite(solidification_hold_max) ?
    solidification_hold_max : solidification_hold_plan_s
# Per-particle Nakamura sf to count as crystallized (0.99 = fully cooled/crystallized).
solidification_fraction_threshold = env_float("TP_CLIP_SOLIDIFICATION_THRESHOLD", 0.99)
# Fraction of charge particles that must meet the threshold before mold opens (1.0 = all).
solidification_fraction_target = env_float("TP_CLIP_SOLIDIFICATION_TARGET", 1.0)
const vtu_save_dt = env_float("TP_CLIP_VTU_SAVE_DT", 2.5)
# Compression VTU stays dense; hold uses vtu_save_dt (default 2.5 s).
const compression_vtu_save_scale = env_float("TP_CLIP_COMPRESSION_VTU_SAVE_DT_SCALE", 2.0)
const compression_vtu_save_dt = env_float("TP_CLIP_COMPRESSION_VTU_SAVE_DT", 0.002)
# Retraction VTU denser than hold by default (≥20× vs prior 0.625 s → 0.03125 s).
const retraction_vtu_save_scale = env_float("TP_CLIP_RETRACTION_VTU_SAVE_DT_SCALE", 80.0)
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
# Demold-only: lift punch ABOVE the pre-compression open position so the ejected
# workpiece can sit between mold and punch with clearance from both.
# Non-eject runs keep 0 (no kinematics change). Override with
# TP_CLIP_RETRACTION_PUNCH_EXTRA_LIFT_MM (mm above open).
const retraction_punch_extra_lift = begin
    if !retraction_eject_enable
        0.0
    elseif haskey(ENV, "TP_CLIP_RETRACTION_PUNCH_EXTRA_LIFT_MM") &&
           !isempty(strip(get(ENV, "TP_CLIP_RETRACTION_PUNCH_EXTRA_LIFT_MM", "")))
        max(0.0, env_float("TP_CLIP_RETRACTION_PUNCH_EXTRA_LIFT_MM", 0.0) * 1e-3)
    else
        clear = max(retraction_eject_clear_ps * particle_spacing, 0.0)
        max(0.0, cavity_depth + charge_thickness_discrete + 2.0 * clear)
    end
end
retraction_duration_max = (abs(z_shift_down_end) + retraction_punch_extra_lift) /
                          abs(mold_velocity_state) +
                          0.5 * env_float("TP_CLIP_RETRACTION_RAMP_S",
                                          env_float("TP_CLIP_RETRACTION_KINEMATICS_RELEASE_S", 0.05))
t_total = t_compress + solidification_hold_budget + post_solidification_dwell +
      hold_tlsph_tail_s + retraction_duration_max +
      (retraction_eject_enable ? retraction_eject_s : 0.0) +
      retraction_springback_dwell_s
println("  - compression target = ", round(100 * compression_thickness_ratio; digits=0),
        "% of initial charge thickness (", round(charge_thickness_discrete * 1e3; digits=2),
        " mm -> ", round(final_thickness_target * 1e3; digits=2), " mm)")
println("  - press speed = ", abs(mold_velocity_state), " m/s, ramp = ", t_ramp_mold,
        " s, target stroke = ", round(target_stroke * 1e3; digits=2),
        ", t_down_end = ", round(t_down_end; digits=6),
        " s, t_motion = ", round(t_compress; digits=6),
        " s, solidification hold ",
        isfinite(hold_until_temp_C) ?
            ("until mean charge T ≤ " * string(round(hold_until_temp_C; digits=1)) *
             " °C (no time cap; plan " *
             string(round(solidification_hold_plan_s; digits=6)) * " s)") :
        isfinite(solidification_hold_max) ?
            ("<= " * string(round(solidification_hold_max; digits=6)) * " s") :
            "until 100% particles solidify (no time cap; plan " *
            string(round(solidification_hold_plan_s; digits=6)) * " s)",
        isfinite(hold_until_temp_C) ? "" :
            (" (target " * string(round(100 * solidification_fraction_target; digits=0)) *
             "% particles sf>=" * string(solidification_fraction_threshold) * ")"),
    ", post-solidification dwell = ", round(post_solidification_dwell; digits=6),
    " s, TLSPH equilibration tail = ", round(hold_tlsph_tail_s; digits=6),
    " s, retract <= ", round(retraction_duration_max; digits=6),
    (retraction_punch_extra_lift > 0.0 ?
        (", punch extra lift = " * string(round(retraction_punch_extra_lift * 1e3; digits=2)) * " mm") :
        ""),
    ", eject/demold = ",
    retraction_eject_enable ?
        (string(round(retraction_eject_s; digits=4)) * " s") : "off",
    ", springback dwell <= ", round(retraction_springback_dwell_s; digits=6),
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
        return mold_lift_z_velocity_at(t)
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
retraction_lift_start_time = Ref(Inf)
mold_retraction_complete_time = Ref(Inf)
solidification_reached_time = Ref(Inf)
retraction_state_prepared = Ref(false)
# Solidification hold: freeze charge u/v and skip mechanics RHS; thermal + Nakamura continue.
const compression_explicit_hold_dt_announced = Ref(false)
const charge_hold_phase_dtmax = env_float("TP_CLIP_HOLD_DTMAX", 5.0e-3)
# Kinematics + kin-scaled punch lift ramp (default 0.05 s; starts after elastic σ is on).
const retraction_kinematics_release_s = env_float("TP_CLIP_RETRACTION_KINEMATICS_RELEASE_S", 0.05)
# Mold lift velocity ramp: default tracks kinematics release (override via TP_CLIP_RETRACTION_RAMP_S).
const retraction_ramp_s = env_float("TP_CLIP_RETRACTION_RAMP_S", retraction_kinematics_release_s)
# Contact/hold: instant-on unless TP_CLIP_RETRACTION_MECH_RAMP_S or HOLD_RELEASE_S is set.
const retraction_mech_ramp_instant = !haskey(ENV, "TP_CLIP_RETRACTION_MECH_RAMP_S") &&
    !haskey(ENV, "TP_CLIP_RETRACTION_HOLD_RELEASE_S")
const retraction_mech_ramp_s = (sim_phase == "retraction" || sim_phase == "full") ?
    env_float("TP_CLIP_RETRACTION_MECH_RAMP_S", 0.05) : 0.1
const retraction_hold_release_s = haskey(ENV, "TP_CLIP_RETRACTION_HOLD_RELEASE_S") ?
    env_float("TP_CLIP_RETRACTION_HOLD_RELEASE_S", retraction_mech_ramp_s) :
    retraction_mech_ramp_s
# Elastic σ only: short default ramp during retraction (contact stays instant-on).
const retraction_elastic_ramp_instant = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_ELASTIC_RAMP_INSTANT", 1) != 0
const retraction_elastic_ramp_s = if sim_phase == "retraction"
    env_float("TP_CLIP_RETRACTION_ELASTIC_RAMP_S", 0.03)
elseif haskey(ENV, "TP_CLIP_RETRACTION_ELASTIC_RAMP_S")
    env_float("TP_CLIP_RETRACTION_ELASTIC_RAMP_S", retraction_mech_ramp_s)
else
    retraction_mech_ramp_s
end
# Kin/lift hold at 0 until elastic σ ramp finishes (skipped when elastic is instant-on or pre-lift creep).
const retraction_kinematics_ramp_delay_s =
    (!retraction_elastic_ramp_instant && retraction_pre_lift_creep_s <= 0.0) ?
    retraction_elastic_ramp_s : 0.0
const retraction_contact_gap_off_ps = env_float("TP_CLIP_RETRACTION_CONTACT_GAP_OFF_PS", 2.5)
const retraction_contact_gap_on_ps = env_float("TP_CLIP_RETRACTION_CONTACT_GAP_ON_PS", 0.75)
# Smooth mold-contact release when span grows past hold reference (not a step to zero).
const retraction_span_contact_band = env_float("TP_CLIP_RETRACTION_SPAN_CONTACT_BAND", 0.05)
# Retraction inherits compression/hold contact stiffness unless TP_CLIP_RETRACTION_CONTACT_E_SCALE is set.
const retraction_contact_e_explicit = sim_phase == "retraction" &&
    haskey(ENV, "TP_CLIP_RETRACTION_CONTACT_E_SCALE")
const retraction_contact_e_scale_value = retraction_contact_e_explicit ?
    env_float("TP_CLIP_RETRACTION_CONTACT_E_SCALE", 1.0) : 1.0
const retraction_contact_e_ramp = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_CONTACT_E_RAMP", 0) != 0
const retraction_contact_e_start = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_CONTACT_E_START", 1.0) : 1.0
const retraction_contact_e_ramp_s = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_CONTACT_E_RAMP_S", 0.05) : 0.0
# Keep floor/mold Reimann contact during springback (do not fade with span growth).
const retraction_contact_on_springback = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_CONTACT_ON_SPRINGBACK", 1) != 0
if sim_phase == "retraction"
    if retraction_contact_e_ramp
        retraction_contact_e_scale[] = retraction_contact_e_start
    elseif retraction_contact_e_explicit
        retraction_contact_e_scale[] = retraction_contact_e_scale_value
    end
end
const retraction_contact_e_desc =
    retraction_contact_e_ramp ?
        ("contact E ramp " * string(retraction_contact_e_start) * "→" *
         string(retraction_contact_e_scale_value) * " over " *
         string(retraction_contact_e_ramp_s) * " s") :
    retraction_contact_e_explicit ?
        ("contact E scale=" * string(retraction_contact_e_scale_value)) :
        "contact E inherits compression/hold scale"
sim_phase == "retraction" && println(
    "Retraction stability fixes: implicit default; contact ",
    retraction_mech_ramp_instant ? "instant-on at checkpoint; " :
        "ramp " * string(retraction_hold_release_s) * " s (legacy); ",
    "elastic σ ",
    retraction_elastic_ramp_instant ? "instant-on; " :
        "ramp " * string(retraction_elastic_ramp_s) * " s; ",
    "kin ramp ", retraction_kinematics_release_s, " s",
    retraction_kinematics_ramp_delay_s > 0.0 ?
        " (after elastic " * string(retraction_elastic_ramp_s) * " s); " : "; ",
    "mold lift kin-scaled (ramp ", retraction_ramp_s, " s); ",
    retraction_pre_lift_creep_s > 0.0 ?
        "pre-lift creep " * string(retraction_pre_lift_creep_s) *
        " s (continuous post-lift σ/contact/kin); " :
        "kin+lift ramp from checkpoint; ",
    retraction_contact_on_springback ?
        "floor+mold contact stay on during springback; " :
        "floor contact off on span growth; smooth mold span release (band=" *
        string(retraction_span_contact_band) * "); ",
    "springback dwell=", retraction_springback_dwell_s, " s; ",
    retraction_contact_e_desc,
    retraction_accel_clamp ? "; accel clamp=" * string(retraction_max_accel) * " m/s²" :
        (retraction_explicit_kick_soft_limit ?
            "; hard |a|≤" * string(retraction_deep_contact_accel_cap) *
            " m/s² |v|≤" * string(retraction_deep_contact_speed_cap) *
            " m/s on deep overlap; ramp |a|≤" * string(retraction_max_accel) * " m/s²" :
            ""),
    hold_end_J_floor > 0.0 ? "; hold-end J floor=" * string(hold_end_J_floor) : "",
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
    max(1, env_int("TP_CLIP_RETRACTION_INFO_INTERVAL", 1)) :
    max(1, env_int("TP_CLIP_INFO_INTERVAL", 1))
# Per-step blow-up diagnostics (max |dv|, gaps, spans, J) during retraction-only runs.
const retraction_diag_enabled = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_DIAG", 1) != 0
const retraction_diag_interval = max(1, env_int("TP_CLIP_RETRACTION_DIAG_INTERVAL", 500))
const retraction_diag_log_path = get(ENV, "TP_CLIP_RETRACTION_DIAG_LOG",
                                     joinpath(clip_out_dir, "retraction_diag.csv"))
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
retraction_pre_lift_creep_announced = Ref(false)
retraction_lift_started_announced = Ref(false)
retraction_creep_audit_next_t = Ref(Inf)
retraction_pre_lift_settle_announced = Ref(false)
retraction_pre_lift_settle_done = Ref(false)
retraction_lift_planned_time = Ref(Inf)  # nominal t0+creep; settle may postpone lift_start
retraction_creep_floor_settle_announced = Ref(false)

@inline function retraction_pre_lift_creep_active(t)
    retraction_pre_lift_creep_s <= 0.0 && return false
    !retraction_started[] && return false
    return t < retraction_lift_start_time[]
end

@inline function retraction_pre_lift_settle_active(t)
    retraction_pre_lift_settle_s <= 0.0 && return false
    !retraction_started[] && return false
    t0 = retraction_start_time[] + retraction_pre_lift_creep_s
    return t >= t0 - 1.0e-14 && t < t0 + retraction_pre_lift_settle_s - 1.0e-14
end

function arm_retraction_phase!(t)
    retraction_started[] = true
    retraction_start_time[] = t
    # Reserve closed-die floor-settle budget up front so punch cannot lift one step early.
    retraction_lift_planned_time[] = t + retraction_pre_lift_creep_s +
                                     retraction_pre_lift_settle_s
    sigma_budget = retraction_closed_die_contact_settle ?
        retraction_contact_settle_sigma_ramp_s : 0.0
    retraction_lift_start_time[] = retraction_cool_before_lift ? Inf :
        (retraction_lift_planned_time[] +
         retraction_creep_floor_settle_max_s + sigma_budget)
    retraction_closed_die_cool_active[] = false
    retraction_closed_die_cool_done[] = false
    retraction_closed_die_cool_start_t[] = Inf
    reset_thermal_Fth_state!()
    retraction_creep_floor_settle_announced[] = false
    retraction_pre_lift_settle_announced[] = false
    retraction_pre_lift_settle_done[] = false
    retraction_contact_settle_done[] = false
    retraction_sigma_ramp_start_t[] = Inf
    retraction_contact_settle_announced[] = false
    retraction_peel_done[] = false
    retraction_peel_announced[] = false
    retraction_peel_hold_for_eject_announced[] = false
    retraction_peel_release_announced[] = false
    retraction_peel_release_time[] = Inf
    retraction_floor_open_z_shift[] = 0.0
    retraction_floor_open_target_dz[] = 0.0
    retraction_floor_open_start_t[] = Inf
    retraction_floor_open_active[] = false
    retraction_floor_open_done[] = false
    retraction_floor_open_announced[] = false
    retraction_floor_projected[] = false
    retraction_eject_active[] = false
    retraction_eject_complete[] = false
    retraction_eject_announced[] = false
    retraction_eject_start_t[] = Inf
    retraction_eject_target_dz[] = 0.0
    retraction_eject_z_applied[] = 0.0
    retraction_indie_cool_announced[] = false
    retraction_indie_cool_last_print_t[] = -Inf
    retraction_closed_die_cool_active[] = false
    retraction_closed_die_cool_done[] = false
    retraction_closed_die_cool_start_t[] = Inf
    retraction_post_eject_bang_done[] = false
    retraction_post_eject_bang_t[] = Inf
    retraction_tool_temp_after_punch_lift_done[] = false
    retraction_closed30_ckpt_saved[] = false
    retraction_closed30_settle_start_t[] = Inf
    reset_thermal_Fth_state!()
    retraction_post_cool_springback_active[] = false
    retraction_post_cool_springback_start_t[] = Inf
    retraction_post_cool_springback_announced[] = false
    if retraction_pre_lift_creep_s > 0.0 && retraction_creep_audit_dt > 0.0
        retraction_creep_audit_next_t[] = t + retraction_creep_audit_dt
    else
        retraction_creep_audit_next_t[] = Inf
    end
    return nothing
end

function arm_retraction_from_preeject!(t)
    retraction_lift_planned_time[] = t - retraction_duration_max
    retraction_lift_start_time[] = retraction_lift_planned_time[]
    mold_retraction_complete[] = true
    mold_retraction_complete_time[] = t
    retraction_peel_done[] = true
    retraction_peel_announced[] = true
    retraction_peel_release_announced[] = true
    retraction_peel_release_time[] = t
    retraction_qs_cte_scale[] = max(retraction_qs_cte_scale[], 1.0)
    retraction_lift_started_announced[] = true
    retraction_pre_lift_creep_announced[] = true
    retraction_tool_temp_after_punch_lift_done[] = true
    retraction_closed_die_cool_done[] = true
    retraction_closed_die_cool_active[] = false
    println(">>> Resume pre-eject: skip creep/lift/in-die cool; punch stays up; ",
            "freeze mech during eject stroke, then live mechanics")
    flush(stdout)
    return nothing
end

@inline function retraction_pre_lift_ramp_elapsed(t)
    return max(t - retraction_start_time[], 0.0)
end

@inline function retraction_pre_lift_ramp_scale(t, duration_frac)
    env_int("TP_CLIP_RETRACTION_CREEP_RAMP_FREE", 0) != 0 && return 1.0
    duration_frac <= 0.0 && return 1.0
    !retraction_pre_lift_creep_active(t) && retraction_started[] && return 1.0
    dur = max(retraction_pre_lift_creep_s * duration_frac, eps(Float64))
    return smoothstep01(retraction_pre_lift_ramp_elapsed(t) / dur)
end

# With pre-lift creep, σ/contact/kin reach 1 before punch lift — do not re-ramp from 0 at lift.
@inline function retraction_continuous_post_lift_ramps()
    return retraction_pre_lift_creep_s > 0.0
end

# Optional post-lift kin hold (pre-lift path only); kin motion otherwise follows kin_scale ramp.
# Kinematic peel keeps charge gated from lift start until gaps clear (or peel timeout).
@inline function retraction_charge_kinematics_gated(t)
    !retraction_started[] && return false
    # Rigid eject: freeze deformational kinematics while prescribed +z demold runs.
    retraction_eject_active[] && return true
    retraction_pre_lift_creep_active(t) && return false
    if retraction_kinematic_peel
        # Stepped closed-die QS (lift-first OFF): charge must move in the mold
        # before punch lift; peel gating applies only after lift starts.
        if retraction_qs_cool && !retraction_lift_first_after_frozen_cool &&
           retraction_closed_die_cool_active[]
            return false
        end
        return !retraction_peel_done[]
    end
    retraction_post_lift_kin_freeze_s > 0.0 || return false
    return (t - retraction_lift_start_time[]) < retraction_post_lift_kin_freeze_s
end

@inline function hold_tlsph_tail_start_t()
    return solidification_reached_time[] + post_solidification_dwell
end

@inline function hold_tlsph_tail_active(t)
    sim_phase == "retraction" && return false
    hold_tlsph_tail_s <= 0.0 && return false
    solidification_complete[] || return false
    hold_stress_rate_converged[] && return false
    elapsed = t - hold_tlsph_tail_start_t()
    return elapsed >= 0.0 && elapsed < hold_tlsph_tail_s
end

@inline function hold_tlsph_checkpoint_ready(t)
    sim_phase == "retraction" && return false
    solidification_complete[] || return false
    dwell_elapsed = t - solidification_reached_time[]
    dwell_elapsed < post_solidification_dwell && return false
    hold_stress_rate_converged[] && return true
    return dwell_elapsed >= post_solidification_dwell + hold_tlsph_tail_s
end

@inline function hold_tlsph_tail_kinematics_scale(t)
    sim_phase == "retraction" && return 1.0
    hold_tlsph_tail_active(t) || return 0.0
    hold_tlsph_tail_ramp_s <= 0.0 && return 1.0
    elapsed = t - hold_tlsph_tail_start_t()
    return smoothstep01(elapsed / hold_tlsph_tail_ramp_s)
end

# Mold closed, charge v/u frozen — thermal-only hold OR TLSPH equilibration tail / pre-lift creep.
# Pre-lift default FREEZE=1: closed-die contact settle with ACTIVE kin detonates (deep soft
# overlap + free motion). Fe_iso relaxes under frozen kin; residual floor overlap after peel
# must clear via contact mechanics (particle projection is scrapped).
@inline function hold_kinematics_frozen(t)
    if retraction_pre_lift_creep_active(t)
        if retraction_pre_lift_settle_active(t)
            return false
        end
        # Closed-die QS cool needs live charge motion while punch+floor stay down.
        if retraction_cool_before_lift && retraction_closed_die_cool_active[]
            return false
        end
        # Kinematic peel: charge stays glued through closed-die creep (mold lifts later).
        # Option B (free kin in creep) REJECTED: abort ~37 ms in with floor/mold soft-overlap
        # (kick≈1.3e6, J_min 0.22→0.07) while Strategy E bulk≈0 — tool contact, not bulk.
        retraction_kinematic_peel && return true
        # Contact-led closed-die settle: kin must be ACTIVE while σ is held at 0.
        retraction_closed_die_contact_settle && return false
        # Cool-then-lift: freeze-T creep stays frozen (Fe_iso, moldClear6).
        # Closed-die cool: kin ON only after T ≤ QS_KIN_AFTER_C (r15 extruded at 229 °C).
        if retraction_cool_before_lift && !retraction_closed_die_cool_done[]
            retraction_closed_die_cool_active[] && return false
            return true
        end
        env_int("TP_CLIP_RETRACTION_CREEP_KINEMATICS_FREEZE", 1) != 0 && return true
        if env_int("TP_CLIP_RETRACTION_CREEP_KIN_AFTER_SIGMA", 1) != 0
            return retraction_elastic_scale(t) < 1.0 - 1.0e-8
        end
        return false
    end
    retraction_charge_kinematics_gated(t) && return true
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

# Unilateral thermal shrink while packing is frozen and T is actually falling.
# Hold solidification cool AND closed-die cool. Freeze-T creep is excluded (dT=0).
@inline function frozen_thermal_contraction_active(t)
    hold_kinematics_frozen(t) || return false
    sim_phase == "retraction" && retraction_thermal_frozen_now() && return false
    return true
end

# 0 during frozen CTE-off cool; ramped or 1 after kin ON; 1 after lift.
# SKIP_CTE=0: still off while creep kin is frozen (do not store τ_th), then
# follow T in the closed die so the 190 K mismatch is never restored later.
@inline function retraction_thermal_cte_scale()
    if !retraction_qs_skip_cte_until_kin
        retraction_closed_die_cool_done[] && return 1.0
        retraction_closed_die_cool_active[] || return 0.0
        return retraction_qs_cte_ramp ? retraction_qs_cte_scale[] : 1.0
    end
    if retraction_lift_first_after_frozen_cool
        retraction_peel_done[] || return 0.0
        return retraction_qs_cte_scale[]
    end
    retraction_closed_die_cool_done[] && return 1.0
    retraction_closed_die_cool_active[] || return 0.0
    return retraction_qs_cte_scale[]
end

# Closed-die hold (compression end → retraction): plastic commit substeps + optional skip_heat.
@inline function hold_plastic_relax_active(t)
    t < t_compress && return false
    retraction_pre_lift_creep_active(t) && return true
    retraction_closed_die_cool_active[] && retraction_closed_die_cool_plastic && return true
    sim_phase == "retraction" && return false
    return !retraction_started[]
end

@inline function charge_hold_thermal_only(t)
    retraction_pre_lift_creep_active(t) && return false
    # Free-cool path: after demold eject, drop mechanics and integrate ambient/conduction only.
    # Brief post-cool springback re-enables mechanics (returns false here).
    if retraction_free_cool_thermal_only_now()
        return true
    end
    hold_kinematics_frozen(t) || return false
    return !hold_tlsph_tail_active(t)
end

# Hold only: zero charge v/du and skip elastic RHS (thermal + crystallization path).
@inline charge_hold_mechanics_frozen(t) = charge_hold_thermal_only(t)

@inline function smoothstep01(x)
    y = clamp(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)
end

# After peel ungate: shared 0→1 scale for σ/contact soft re-engagement.
# Returns 1 while peel is still freezing the charge (forces discarded by freeze).
@inline function retraction_peel_post_mech_scale(t)
    retraction_kinematic_peel || return 1.0
    retraction_peel_sigma_ramp_s <= 0.0 && return 1.0
    !retraction_peel_done[] && return 1.0
    t0 = retraction_peel_release_time[]
    !isfinite(t0) && return 0.0
    return smoothstep01((t - t0) / max(retraction_peel_sigma_ramp_s, eps(Float64)))
end

@inline function hold_release_factor(t)
    if !retraction_started[]
        return 1.0
    end
    if retraction_pre_lift_creep_active(t)
        # Contact-led settle: contact fully on from t0 (σ ramps separately).
        retraction_closed_die_contact_settle && return 1.0
        return retraction_pre_lift_ramp_scale(t, retraction_pre_lift_elastic_frac)
    end
    if retraction_continuous_post_lift_ramps()
        return retraction_peel_post_mech_scale(t)
    end
    retraction_mech_ramp_instant && return 1.0
    elapsed = t - retraction_lift_start_time[]
    return smoothstep01(elapsed / retraction_hold_release_s)
end

@inline function retraction_elastic_scale(t)
    if !retraction_started[]
        return 1.0
    end
    if retraction_closed_die_contact_settle
        # Phase A: σ=0 while contact settles. Phase B: σ ramp after settle done.
        !retraction_contact_settle_done[] && return 0.0
        t0σ = retraction_sigma_ramp_start_t[]
        !isfinite(t0σ) && return 0.0
        dur = max(retraction_contact_settle_sigma_ramp_s, eps(Float64))
        return smoothstep01((t - t0σ) / dur)
    end
    if retraction_pre_lift_creep_active(t)
        return retraction_pre_lift_ramp_scale(t, retraction_pre_lift_elastic_frac)
    end
    if retraction_continuous_post_lift_ramps()
        return retraction_peel_post_mech_scale(t)
    end
    retraction_elastic_ramp_instant && return 1.0
    elapsed = t - retraction_lift_start_time[]
    return smoothstep01(elapsed / retraction_elastic_ramp_s)
end

@inline function retraction_kinematics_elapsed(t)
    elapsed = t - retraction_lift_start_time[]
    return max(0.0, elapsed - retraction_kinematics_ramp_delay_s)
end

@inline function retraction_kinematics_scale(t)
    if !retraction_started[]
        return 1.0
    end
    if retraction_pre_lift_creep_active(t)
        if retraction_closed_die_cool_active[]
            retraction_closed_die_kin_ramp_s <= 0.0 && return 1.0
            elapsed = t - retraction_closed_die_cool_start_t[]
            return smoothstep01(elapsed / retraction_closed_die_kin_ramp_s)
        end
        retraction_closed_die_contact_settle && return 1.0
        return retraction_pre_lift_ramp_scale(t, 1.0)
    end
    # Charge-only gate: do NOT use this scale for mold lift (see retraction_mold_lift_scale).
    retraction_charge_kinematics_gated(t) && return 0.0
    retraction_continuous_post_lift_ramps() && return 1.0
    kin_elapsed = retraction_kinematics_elapsed(t)
    kin_elapsed <= 0.0 && return 0.0
    return smoothstep01(kin_elapsed / retraction_kinematics_release_s)
end

# Mold / punch lift scale — independent of charge post-lift kin freeze.
# Previously mold used retraction_kinematics_scale, which returned 0 while charge was
# gated, so the punch never opened the contact gap before charge release → blowup.
@inline function retraction_mold_lift_scale(t)
    if !retraction_started[] || retraction_pre_lift_creep_active(t)
        return 0.0
    end
    retraction_continuous_post_lift_ramps() && return 1.0
    kin_elapsed = retraction_kinematics_elapsed(t)
    kin_elapsed <= 0.0 && return 0.0
    return smoothstep01(kin_elapsed / retraction_kinematics_release_s)
end

@inline function smoothstep01_deriv(x)
    y = clamp(x, 0.0, 1.0)
    return 6.0 * y * (1.0 - y)
end

# Unscaled punch lift trajectory (as if mold lift scale were 1 from lift t0).
@inline function mold_lift_z_full_shift(dt_lift)
    T_ramp = retraction_ramp_s
    if dt_lift <= 0.0
        return 0.0
    elseif dt_lift < T_ramp
        return 0.5 * abs(mold_velocity_state) / T_ramp * dt_lift^2
    else
        dt_capped = min(dt_lift, retraction_duration_max)
        return abs(mold_velocity_state) * (dt_capped - 0.5 * T_ramp)
    end
end

@inline function mold_lift_z_full_velocity(dt_lift)
    T_ramp = retraction_ramp_s
    if dt_lift <= 0.0
        return 0.0
    elseif dt_lift < T_ramp
        return abs(mold_velocity_state) * dt_lift / T_ramp
    else
        return abs(mold_velocity_state)
    end
end

# Punch lift: uses mold scale (not charge gate). Charge may stay frozen while mold moves.
@inline function mold_lift_z_shift_at(t)
    if mold_retraction_complete[]
        return z_shift_motion_end + mold_lift_z_full_shift(retraction_duration_max)
    end
    if !retraction_started[] || retraction_pre_lift_creep_active(t)
        return z_shift_motion_end
    end
    kin_elapsed = retraction_kinematics_elapsed(t)
    kin = retraction_mold_lift_scale(t)
    return z_shift_motion_end + kin * mold_lift_z_full_shift(kin_elapsed)
end

@inline function mold_lift_z_velocity_at(t)
    if !retraction_started[] || retraction_pre_lift_creep_active(t)
        return 0.0
    end
    kin_elapsed = retraction_kinematics_elapsed(t)
    kin_elapsed <= 0.0 && return 0.0
    kin = retraction_mold_lift_scale(t)
    # With continuous post-lift ramps, mold scale is identically 1 — velocity follows
    # the geometric ramp only (no smoothstep kin_dot spike at charge ungating).
    if retraction_continuous_post_lift_ramps()
        return kin * mold_lift_z_full_velocity(min(kin_elapsed, retraction_duration_max))
    end
    T_kin = retraction_kinematics_release_s
    kin_dot = if kin_elapsed >= T_kin
        0.0
    else
        smoothstep01_deriv(kin_elapsed / T_kin) / T_kin
    end
    z_delta = mold_lift_z_full_shift(kin_elapsed)
    v_full = mold_lift_z_full_velocity(min(kin_elapsed, retraction_duration_max))
    return kin_dot * z_delta + kin * v_full
end

@inline function clip_mold_z_shift_unified(t)
    z_shift = if t <= t_compress
        mold_z_shift_at_time(t)
    elseif retraction_started[]
        mold_lift_z_shift_at(t)
    else
        z_shift_motion_end
    end
    # Upper bound 0 = pre-compression open; demold may lift further (+extra).
    return clamp(z_shift, z_shift_motion_end, retraction_punch_extra_lift)
end

function clip_apply_mold_kinematics_cpu!(mold_sys, t)
    mold_sys.clamped_particles_moving[] = true
    z_shift = clip_mold_z_shift_unified(t)
    ε = sqrt(eps(Float64))
    z_m = clip_mold_z_shift_unified(t - ε)
    z_p = clip_mold_z_shift_unified(t + ε)
    vz = (z_p - z_m) / (2ε)
    az = (z_p - 2z_shift + z_m) / (ε^2)
    initial_coords = TrixiParticles.initial_coordinates(mold_sys)
    cache = mold_sys.cache
    has_motion_cache = hasproperty(cache, :velocity)
    @inbounds for particle in TrixiParticles.eachparticle(mold_sys)
        for i in 1:2
            mold_sys.current_coordinates[i, particle] = initial_coords[i, particle]
            if has_motion_cache
                cache.velocity[i, particle] = 0.0
                cache.acceleration[i, particle] = 0.0
            end
        end
        mold_sys.current_coordinates[3, particle] = initial_coords[3, particle] + z_shift
        if has_motion_cache
            cache.velocity[3, particle] = vz
            cache.acceleration[3, particle] = az
        end
    end
    return nothing
end

function clip_sync_tool_kinematics_gpu!(semi_local, t)
    clip_use_gpu || return nothing
    clip_apply_mold_kinematics_cpu!(clip_mold_system(semi_local), t)
    return nothing
end

function clip_set_stress_cache!(cyl_sys)
    TrixiParticles.STRESS_TENSOR_CACHE[] = (objectid(cyl_sys), v_stress_buf)
    return nothing
end

# Optional: ramp tool–charge penalty stiffness over a dedicated short window (not kin ramp).
# Post-peel: soft absolute contact E while σ ramps (L-die wall soft-contact survives z-open).
@inline function update_retraction_contact_e_scale!(t)
    if retraction_kinematic_peel && retraction_peel_sigma_ramp_s > 0.0 &&
       retraction_started[] && retraction_peel_done[]
        s = retraction_peel_post_mech_scale(t)
        e_end = compression_contact_e_scale[]
        retraction_contact_e_scale[] = retraction_peel_contact_e_start +
            (e_end - retraction_peel_contact_e_start) * s
        return
    end
    if !retraction_contact_e_ramp || !retraction_started[]
        return
    end
    if retraction_continuous_post_lift_ramps() && !retraction_pre_lift_creep_active(t)
        retraction_contact_e_scale[] = retraction_contact_e_scale_value
        return
    end
    elapsed = t - retraction_lift_start_time[]
    ramp = smoothstep01(elapsed / retraction_contact_e_ramp_s)
    retraction_contact_e_scale[] = retraction_contact_e_start +
        (retraction_contact_e_scale_value - retraction_contact_e_start) * ramp
end

# Mechanical dv blend: contact ramp × elastic σ ramp (kinematics uses separate du scale).
# Pre-lift creep: σ and contact share one smoothstep — do NOT multiply (that was s² and
# left restoring forces ≪ kin while Fe_iso≈30, which inverted J around 0.10 s).
# Post-peel: same single-ramp rule (hold×elas would be s² again).
@inline function retraction_dv_mech_scale(t)
    if !retraction_started[]
        return 1.0
    end
    if retraction_pre_lift_creep_active(t)
        # Contact settle: keep full dv so Reimann contact is not scaled by σ=0.
        # Stress is zeroed via tensor elas_ramp instead (see skip_tensor_elas_ramp).
        retraction_closed_die_contact_settle && return 1.0
        return retraction_elastic_scale(t)
    end
    if retraction_kinematic_peel && retraction_peel_sigma_ramp_s > 0.0
        return retraction_peel_post_mech_scale(t)
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

function apply_imex_wall_drag_and_friction!(cyl_sys, floor_sys, mold_sys, semi_local,
                                            v_wrap_state, t::Real, dt::Real)
    dt <= 0 && return nothing
    charge_hold_thermal_only(t) && return nothing

    contact_damp_dist = 0.8 * smoothing_length
    bh_sys = blankholder_neighbor_system[]
    resting_buf = charge_resting_on_shelf_buf[]

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        charge_particle_is_flyaway(particle) && continue
        x_i = cyl_sys.current_coordinates[1, particle]
        z_i = cyl_sys.current_coordinates[3, particle]
        gap_floor_drag = tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower)
        gap_mold_drag = tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper)
        near_floor = gap_floor_drag <= contact_damp_dist
        near_mold = gap_mold_drag <= contact_damp_dist
        near_blankholder = false
        gap_bh_drag = Inf
        if bh_sys !== nothing && resting_buf !== nothing && particle <= length(resting_buf) &&
           resting_buf[particle]
            gap_bh_drag = tool_surface_gap(bh_sys, x_i, z_i, particle_spacing; side=:upper)
            near_blankholder = gap_bh_drag <= contact_damp_dist
        end

        # Molten wall no-slip drag (off during solid retraction).
        lf_wall = liquid_fraction_particle_buf[particle]
        if !retraction_solid_mechanics_only() && lf_wall > 1.0e-4 &&
           (near_floor || near_mold || near_blankholder)
            ρ_i = cyl_sys.material_density[particle]
            η_i = vis_particle_buf[particle]
            drag_rate = molten_wall_drag_factor * lf_wall * η_i /
                        (ρ_i * smoothing_length * smoothing_length)
            drag_rate = min(drag_rate, molten_wall_drag_max_rate)

            if near_floor
                act = 1.0 - clamp(gap_floor_drag / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)
                act = act * act * (3.0 - 2.0 * act)
                v_wrap_state[1, particle] -= dt * act * drag_rate * v_wrap_state[1, particle]
                v_wrap_state[2, particle] -= dt * act * drag_rate * v_wrap_state[2, particle]
                v_wrap_state[3, particle] -= dt * act * drag_rate * v_wrap_state[3, particle]
            end

            if near_mold
                act = 1.0 - clamp(gap_mold_drag / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)
                act = act * act * (3.0 - 2.0 * act)
                v_wall_z = mold_z_velocity_at_time(t)
                v_wrap_state[1, particle] -= dt * act * drag_rate * v_wrap_state[1, particle]
                v_wrap_state[2, particle] -= dt * act * drag_rate * v_wrap_state[2, particle]
                v_wrap_state[3, particle] -= dt * act * drag_rate *
                                             (v_wrap_state[3, particle] - v_wall_z)
            end

            if near_blankholder
                act = 1.0 - clamp(gap_bh_drag / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)
                act = act * act * (3.0 - 2.0 * act)
                v_wrap_state[1, particle] -= dt * act * drag_rate * v_wrap_state[1, particle]
                v_wrap_state[2, particle] -= dt * act * drag_rate * v_wrap_state[2, particle]
                v_wrap_state[3, particle] -= dt * act * drag_rate * v_wrap_state[3, particle]
            end
        end

        # Coulomb friction at tool–part contact (solid-fraction weighted).
        if near_floor || near_mold || near_blankholder
            sf = solid_fraction_particle_buf[particle]
            if sf > 1.0e-4
                mu_eff  = mu_friction * sf
                p_n     = max(0.0, -v_stress_buf[3, 3, particle])
                rho_i   = cyl_sys.material_density[particle]
                a_fric  = mu_eff * p_n / (rho_i * particle_spacing)
                vx = v_wrap_state[1, particle]
                vy = v_wrap_state[2, particle]
                v_tan = hypot(vx, vy)
                if v_tan > 1.0e-12
                    change = min(v_tan, dt * a_fric)
                    scale = change / v_tan
                    v_wrap_state[1, particle] -= scale * vx
                    v_wrap_state[2, particle] -= scale * vy
                end
            end
        end
    end
    return nothing
end

@inline function charge_mechanics_fully_released(hold_release)
    return !charge_partial_mechanics_ramp(hold_release)
end

@inline function charge_kinematics_fully_released(t)
    return !retraction_charge_kinematics_gated(t) && !retraction_kinematics_ramping(t)
end

function charge_span_ratio(cyl_sys)
    if retraction_reference_span_x[] <= 0.0 || retraction_reference_span_z[] <= 0.0
        return 1.0
    end
    xmin = Inf
    xmax = -Inf
    zmin = Inf
    zmax = -Inf
    n_used = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        charge_particle_is_flyaway(particle) && continue
        x = cyl_sys.current_coordinates[1, particle]
        z = cyl_sys.current_coordinates[3, particle]
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        zmin = min(zmin, z)
        zmax = max(zmax, z)
        n_used += 1
    end
    n_used == 0 && return 1.0
    span_x = xmax - xmin
    span_z = zmax - zmin
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
    # Rigid eject stroke: keep floor contact unless explicitly disabled.
    # Punch/mold stay off (see mold_reimann_activation_factor).
    if retraction_eject_enable && retraction_eject_active[]
        return retraction_keep_floor_during_eject ? 1.0 : 0.0
    end
    # After eject: keep the floor as a resting table; punch/mold stay off.
    if retraction_eject_enable && retraction_eject_complete[]
        return retraction_keep_floor_after_eject ? 1.0 : 0.0
    end
    # Post-peel diagnostic: TLSPH self only (no tool Reimann).
    if retraction_peel_tools_off && retraction_peel_done[]
        return 0.0
    end
    retraction_pre_lift_creep_active(t) && return 1.0
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
    if retraction_eject_enable && (retraction_eject_active[] || retraction_eject_complete[])
        return 0.0
    end
    if retraction_peel_tools_off && retraction_peel_done[]
        return 0.0
    end
    retraction_pre_lift_creep_active(t) && return 1.0
    hold_release_factor(t) < 1.0 - 1.0e-8 && return 0.0
    min_gap = Inf
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        charge_particle_is_flyaway(particle) && continue
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
        charge_particle_is_flyaway(particle) && continue
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

@inline function charge_particle_is_flyaway(particle::Int)
    buf = charge_flyaway_buf[]
    buf === nothing && return false
    return 1 <= particle <= length(buf) && buf[particle]
end

function update_charge_flyaway_mask!(cyl_sys)
    n = nparticles(cyl_sys)
    mask = charge_flyaway_buf[]
    if mask === nothing || length(mask) != n
        mask = fill(false, n)
        charge_flyaway_buf[] = mask
    else
        fill!(mask, false)
    end
    if !ignore_charge_flyaways
        charge_flyaway_count[] = 0
        return 0
    end

    coords = cyl_sys.current_coordinates
    r_blank = 0.5 * max(charge_length, charge_width)
    r_cut = r_blank + flyaway_r_pad
    r_cut2 = r_cut * r_cut
    z_lo = z_base_top - flyaway_below_floor
    z_hi = z_base_top + max(cavity_depth, 0.0) + flyaway_above_cavity
    r_neigh2 = (2.5 * particle_spacing)^2
    n_neigh = zeros(Int, n)
    @inbounds for i in 1:n
        xi = coords[1, i]
        yi = coords[2, i]
        zi = coords[3, i]
        for j in (i + 1):n
            dx = xi - coords[1, j]
            dy = yi - coords[2, j]
            dz = zi - coords[3, j]
            if dx * dx + dy * dy + dz * dz <= r_neigh2
                n_neigh[i] += 1
                n_neigh[j] += 1
            end
        end
    end

    sx = 0.0
    sy = 0.0
    sz = 0.0
    n_bulk = 0
    @inbounds for i in 1:n
        n_neigh[i] < 6 && continue
        sx += coords[1, i]
        sy += coords[2, i]
        sz += coords[3, i]
        n_bulk += 1
    end
    if n_bulk == 0
        @inbounds for i in 1:n
            sx += coords[1, i]
            sy += coords[2, i]
            sz += coords[3, i]
        end
        n_bulk = n
    end
    inv_n = 1.0 / max(n_bulk, 1)
    cx = sx * inv_n
    cy = sy * inv_n
    cz = sz * inv_n
    isolate2 = flyaway_isolate_dist * flyaway_isolate_dist

    n_fly = 0
    @inbounds for i in 1:n
        x = coords[1, i]
        y = coords[2, i]
        z = coords[3, i]
        through_floor = z < z_lo
        above_tools = z > z_hi
        outside_r = (x * x + y * y) > r_cut2
        dx = x - cx
        dy = y - cy
        dz = z - cz
        isolated = n_neigh[i] < 3 && (dx * dx + dy * dy + dz * dz) > isolate2
        if through_floor || outside_r || above_tools || isolated
            mask[i] = true
            n_fly += 1
        end
    end
    charge_flyaway_count[] = n_fly
    return n_fly
end

function apply_charge_flyaway_freeze!(dv_cyl, v_cyl, cyl_sys)
    buf = charge_flyaway_buf[]
    buf === nothing && return nothing
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        particle <= length(buf) && buf[particle] || continue
        v_cyl[1, particle] = 0.0
        v_cyl[2, particle] = 0.0
        v_cyl[3, particle] = 0.0
        dv_cyl[1, particle] = 0.0
        dv_cyl[2, particle] = 0.0
        dv_cyl[3, particle] = 0.0
    end
    return nothing
end

function charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    min_mold_gap = Inf
    min_floor_gap = Inf
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        charge_particle_is_flyaway(particle) && continue
        x_i = cyl_sys.current_coordinates[1, particle]
        z_i = cyl_sys.current_coordinates[3, particle]
        min_mold_gap = min(min_mold_gap,
                           tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper))
        min_floor_gap = min(min_floor_gap,
                            tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower))
    end
    return min_mold_gap, min_floor_gap
end

# Closed-die: after nominal creep, advance punch lift as soon as floor overlap is within
# tolerance (or at settle deadline). Charge kinematics stay on — physical demold.
function maybe_finish_closed_die_contact_settle!(t, cyl_sys, floor_sys, mold_sys)
    retraction_cool_before_lift && !retraction_closed_die_cool_done[] && return false
    retraction_closed_die_contact_settle || return false
    retraction_started[] || return false
    retraction_contact_settle_done[] && return false

    settle_tol = retraction_creep_floor_settle_gap_ps * particle_spacing
    min_mold_gap, min_floor_gap = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    # Allow at least pre-lift creep, plus optional floor-settle budget, for contact clearing.
    settle_deadline = retraction_start_time[] + retraction_pre_lift_creep_s +
        max(retraction_creep_floor_settle_max_s, 0.0)
    gap_ok = min_floor_gap >= settle_tol
    timed_out = t >= settle_deadline - 1.0e-14

    if !retraction_contact_settle_announced[]
        println(">>> Closed-die CONTACT settle: kin ACTIVE, σ=0, contact ON; ",
                "clear floor gap≥", round(settle_tol * 1e3; digits=4),
                " mm (deadline t=", round(settle_deadline; digits=6), " s)")
        flush(stdout)
        retraction_contact_settle_announced[] = true
    end

    if !(gap_ok || timed_out)
        return true  # still settling
    end

    retraction_contact_settle_done[] = true
    retraction_sigma_ramp_start_t[] = t
    # After σ ramp, punch may lift (keep any remaining floor-settle budget).
    retraction_lift_start_time[] = t + retraction_contact_settle_sigma_ramp_s +
        max(retraction_creep_floor_settle_max_s, 0.0)
    retraction_lift_planned_time[] = t + retraction_contact_settle_sigma_ramp_s
    if gap_ok
        println(">>> Contact settle OK at t=", round(t; digits=6),
                " s: min_floor_gap=", round(min_floor_gap * 1e3; digits=4),
                " mm min_mold_gap=", round(min_mold_gap * 1e3; digits=4),
                " mm; starting σ ramp (", round(retraction_contact_settle_sigma_ramp_s; digits=4),
                " s) before punch lift")
    else
        println(">>> Contact settle timeout at t=", round(t; digits=6),
                " s: min_floor_gap=", round(min_floor_gap * 1e3; digits=4),
                " mm min_mold_gap=", round(min_mold_gap * 1e3; digits=4),
                " mm — starting σ ramp anyway")
    end
    flush(stdout)
    return false
end

function maybe_postpone_lift_for_floor_settle!(t, cyl_sys, floor_sys, mold_sys)
    retraction_cool_before_lift && !retraction_closed_die_cool_done[] && return false
    retraction_creep_floor_settle_max_s <= 0.0 && return false
    retraction_started[] || return false
    retraction_lift_started_announced[] && return false
    retraction_elastic_scale(t) < 1.0 - 1.0e-8 && return false
    planned = retraction_lift_planned_time[]
    isfinite(planned) || return false
    t + 1.0e-14 < planned && return false

    settle_tol = retraction_creep_floor_settle_gap_ps * particle_spacing
    _, min_floor_gap = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    lift_cap = planned + retraction_creep_floor_settle_max_s
    gap_ok = min_floor_gap >= settle_tol
    timed_out = t >= lift_cap - 1.0e-14

    if gap_ok || timed_out
        # Pull lift start to now so punch opens with charge kinematics already on.
        if retraction_lift_start_time[] > t + 1.0e-14
            retraction_lift_start_time[] = min(t, lift_cap)
            if gap_ok
                println(">>> Floor settle OK at t=", round(t; digits=6),
                        " s: min_floor_gap=", round(min_floor_gap * 1e3; digits=4),
                        " mm (tol ", round(settle_tol * 1e3; digits=4),
                        " mm); punch lift with charge kinematics ON")
            else
                println(">>> Floor settle timeout at t=", round(t; digits=6),
                        " s: min_floor_gap=", round(min_floor_gap * 1e3; digits=4),
                        " mm (tol ", round(settle_tol * 1e3; digits=4),
                        " mm) — starting lift anyway")
            end
            flush(stdout)
            retraction_creep_floor_settle_announced[] = true
        end
        return false
    end

    if !retraction_creep_floor_settle_announced[]
        println(">>> Floor settle: holding punch (min_floor_gap=",
                round(min_floor_gap * 1e3; digits=4), " mm < tol ",
                round(settle_tol * 1e3; digits=4),
                " mm); charge kin ACTIVE, mold fixed until gap clears or t=",
                round(lift_cap; digits=6), " s")
        flush(stdout)
        retraction_creep_floor_settle_announced[] = true
    end
    # Keep lift at deadline until gap clears (arm already set lift_start = lift_cap).
    retraction_lift_start_time[] = lift_cap
    return true
end

# Floor particle projection is SCRAPPED (unphysical teleport of charge out of tools).
# Kept as a no-op so old env flags cannot re-enable it.
function project_charge_floor_overlap_after_mold_clear!(integrator, cyl_sys, floor_sys, mold_sys)
    return false
end

# Kinematic peel: freeze charge while punch lifts; optionally open floor/die to
# clear residual soft-contact penetration (tool motion); then ungate springback.
function maybe_update_kinematic_peel!(integrator, cyl_sys, floor_sys, mold_sys)
    retraction_kinematic_peel || return false
    retraction_started[] || return false
    retraction_peel_done[] && return false
    retraction_pre_lift_creep_active(integrator.t) && return false

    if !retraction_peel_announced[]
        floor_note = if retraction_peel_floor_open
            "then floor/die opens over " * string(round(retraction_peel_floor_open_s; digits=4)) * " s"
        elseif retraction_peel_tools_off
            "post-peel σ ramp " * string(round(retraction_peel_sigma_ramp_s; digits=4)) *
                " s with TOOLS OFF (TLSPH self only)"
        else
            "post-peel σ ramp " * string(round(retraction_peel_sigma_ramp_s; digits=4)) * " s"
        end
        println(">>> Kinematic peel: charge FROZEN while mold lifts; ",
                "mold gap ≥ ",
                round(retraction_peel_gap_ps * particle_spacing * 1e3; digits=4),
                " mm; ", floor_note,
                " (NO particle projection; timeout ",
                round(retraction_peel_max_s; digits=4), " s after lift)")
        flush(stdout)
        retraction_peel_announced[] = true
    end

    peel_tol = retraction_peel_gap_ps * particle_spacing
    min_mold_gap, min_floor_gap_vol = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    min_floor_gap = retraction_peel_floor_open ?
        charge_min_floor_top_gap(cyl_sys, floor_sys) : min_floor_gap_vol
    jmin, jmax, jp = _agent_debug_jminmax(cyl_sys)
    mold_ok = min_mold_gap >= peel_tol
    timed_out = (integrator.t - retraction_lift_start_time[]) >=
        retraction_peel_max_s - 1.0e-14

    if !mold_ok && !timed_out
        return true
    end

    # F_th + cavity eject: punch-up is not separation. Ungating here left the
    # charge nested on the floor (fth_aniso30: floor_gap≈−1.5 mm, span_z 11→20 mm).
    # Stay frozen until the eject stroke has lifted the part out of the die.
    if clip_fth_skip_post_peel_cte && retraction_eject_enable &&
       !retraction_eject_complete[]
        if !retraction_peel_hold_for_eject_announced[]
            retraction_peel_hold_for_eject_announced[] = true
            println(">>> Kinematic peel: punch clearance ",
                    round(min_mold_gap * 1e3; digits=3),
                    " mm, floor_gap=", round(min_floor_gap * 1e3; digits=4),
                    " mm — charge stays FROZEN until eject (no in-cavity springback)")
            flush(stdout)
        end
        return true
    end

    # Phase 2: open floor/die downward while charge stays frozen.
    if retraction_peel_floor_open && !retraction_floor_open_done[]
        if !retraction_floor_open_active[]
            if min_floor_gap >= peel_tol
                retraction_floor_open_done[] = true
            else
                extra = retraction_peel_floor_open_extra_ps * particle_spacing
                retraction_floor_open_target_dz[] = (peel_tol - min_floor_gap) + extra
                retraction_floor_open_start_t[] = integrator.t
                retraction_floor_open_active[] = true
                zmin0, zmax0 = floor_tool_z_extents(floor_sys)
                if !retraction_floor_open_announced[]
                    println(">>> Floor/die open at t=", round(integrator.t; digits=6),
                            " s: floor_top_gap=", round(min_floor_gap * 1e3; digits=4),
                            " mm (vol_gap=", round(min_floor_gap_vol * 1e3; digits=4),
                            " mm) → shift floor down ",
                            round(retraction_floor_open_target_dz[] * 1e3; digits=4),
                            " mm over ", round(retraction_peel_floor_open_s; digits=4),
                            " s; z_floor=[", round(zmin0 * 1e3; digits=3), ",",
                            round(zmax0 * 1e3; digits=3), "] mm (charge still FROZEN)")
                    flush(stdout)
                    retraction_floor_open_announced[] = true
                    # #region agent log
                    agent_debug_log!("H6", "maybe_update_kinematic_peel!",
                                     "floor open start",
                                     Dict("t" => integrator.t,
                                          "min_mold_gap_mm" => min_mold_gap * 1e3,
                                          "min_floor_top_gap_mm" => min_floor_gap * 1e3,
                                          "min_floor_vol_gap_mm" => min_floor_gap_vol * 1e3,
                                          "J_min" => jmin, "J_max" => jmax,
                                          "target_dz_mm" => retraction_floor_open_target_dz[] * 1e3,
                                          "z_floor_max_mm" => zmax0 * 1e3,
                                          "z_floor_min_mm" => zmin0 * 1e3,
                                          "floor_z_shift_mm" => 0.0);
                                     runId="floor_open")
                    # #endregion
                end
            end
        end

        if retraction_floor_open_active[] && !retraction_floor_open_done[]
            t0 = retraction_floor_open_start_t[]
            dur = max(retraction_peel_floor_open_s, eps(Float64))
            progress = smoothstep01((integrator.t - t0) / dur)
            retraction_floor_open_z_shift[] = -retraction_floor_open_target_dz[] * progress
            apply_floor_open_z_shift!(floor_sys)
            min_floor_gap2 = charge_min_floor_top_gap(cyl_sys, floor_sys)
            _, min_floor_vol2 = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
            zmin2, zmax2 = floor_tool_z_extents(floor_sys)
            # #region agent log
            agent_debug_log!("H6", "maybe_update_kinematic_peel!",
                             "floor open progress",
                             Dict("t" => integrator.t,
                                  "progress" => progress,
                                  "floor_z_shift_mm" => retraction_floor_open_z_shift[] * 1e3,
                                  "min_floor_top_gap_mm" => min_floor_gap2 * 1e3,
                                  "min_floor_vol_gap_mm" => min_floor_vol2 * 1e3,
                                  "z_floor_max_mm" => zmax2 * 1e3,
                                  "z_floor_min_mm" => zmin2 * 1e3,
                                  "J_min" => jmin,
                                  "target_dz_mm" => retraction_floor_open_target_dz[] * 1e3);
                             runId="floor_open")
            # #endregion
            if progress < 1.0 - 1.0e-8 && min_floor_gap2 < peel_tol
                return true
            end
            retraction_floor_open_z_shift[] = -retraction_floor_open_target_dz[]
            apply_floor_open_z_shift!(floor_sys)
            retraction_floor_open_done[] = true
            min_floor_gap3 = charge_min_floor_top_gap(cyl_sys, floor_sys)
            _, min_floor_vol3 = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
            zmin3, zmax3 = floor_tool_z_extents(floor_sys)
            jmin3, jmax3, _ = _agent_debug_jminmax(cyl_sys)
            println(">>> Floor/die open complete at t=", round(integrator.t; digits=6),
                    " s: floor_z_shift=",
                    round(retraction_floor_open_z_shift[] * 1e3; digits=4),
                    " mm floor_top_gap=", round(min_floor_gap3 * 1e3; digits=4),
                    " mm vol_gap=", round(min_floor_vol3 * 1e3; digits=4),
                    " mm z_floor_max=", round(zmax3 * 1e3; digits=4),
                    " mm J_min=", round(jmin3; digits=4),
                    " (charge still frozen until springback ungate)")
            flush(stdout)
            # #region agent log
            agent_debug_log!("H6", "maybe_update_kinematic_peel!",
                             "floor open complete",
                             Dict("t" => integrator.t,
                                  "floor_z_shift_mm" => retraction_floor_open_z_shift[] * 1e3,
                                  "min_floor_top_gap_mm" => min_floor_gap3 * 1e3,
                                  "min_floor_vol_gap_mm" => min_floor_vol3 * 1e3,
                                  "z_floor_max_mm" => zmax3 * 1e3,
                                  "z_floor_min_mm" => zmin3 * 1e3,
                                  "min_mold_gap_mm" => min_mold_gap * 1e3,
                                  "J_min" => jmin3, "J_max" => jmax3,
                                  "J_unchanged" => abs(jmin3 - jmin) < 1.0e-6);
                             runId="floor_open")
            # #endregion
        end

        !retraction_floor_open_done[] && return true
    end

    # Keep floor shift applied after open (PrescribedMotion + explicit write).
    retraction_peel_floor_open && apply_floor_open_z_shift!(floor_sys)

    retraction_peel_done[] = true
    retraction_peel_release_time[] = integrator.t
    if !retraction_peel_release_announced[]
        min_mold_gap, min_floor_vol = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
        min_floor_gap = retraction_peel_floor_open ?
            charge_min_floor_top_gap(cyl_sys, floor_sys) : min_floor_vol
        jmin, jmax, jp = _agent_debug_jminmax(cyl_sys)
        if mold_ok
            println(">>> Kinematic peel complete at t=", round(integrator.t; digits=6),
                    " s: min_mold_gap=", round(min_mold_gap * 1e3; digits=4),
                    " mm floor_top_gap=", round(min_floor_gap * 1e3; digits=4),
                    " mm vol_gap=", round(min_floor_vol * 1e3; digits=4),
                    " mm J_min=", round(jmin; digits=4),
                    " floor_shift=", round(retraction_floor_open_z_shift[] * 1e3; digits=4),
                    retraction_eject_complete[] ?
                        " mm — enabling springback on free part" :
                        " mm — enabling springback")
        else
            println(">>> Kinematic peel timeout at t=", round(integrator.t; digits=6),
                    " s: min_mold_gap=", round(min_mold_gap * 1e3; digits=4),
                    " mm floor_top_gap=", round(min_floor_gap * 1e3; digits=4),
                    " mm J_min=", round(jmin; digits=4),
                    " — enabling springback anyway")
        end
        flush(stdout)
        retraction_peel_release_announced[] = true
        # #region agent log
        agent_debug_log!("H7", "maybe_update_kinematic_peel!",
                         mold_ok ? "peel complete — springback ungate" :
                                   "peel timeout — springback ungate",
                         Dict("t" => integrator.t, "mold_ok" => mold_ok,
                              "min_mold_gap_mm" => min_mold_gap * 1e3,
                              "min_floor_top_gap_mm" => min_floor_gap * 1e3,
                              "min_floor_vol_gap_mm" => min_floor_vol * 1e3,
                              "J_min" => jmin, "J_max" => jmax, "J_min_pid" => jp,
                              "timed_out" => timed_out,
                              "floor_z_shift_mm" => retraction_floor_open_z_shift[] * 1e3,
                              "floor_open" => retraction_peel_floor_open,
                              "peel_sigma_ramp_s" => retraction_peel_sigma_ramp_s);
                         runId="floor_open")
        # #endregion
    end
    return false
end

# Restore the 150 °C thermal stress only after punch clearance. This is a
# prescribed continuation parameter, not an equilibrium gate: each stage adds
# the stress associated with QS_COOL_STEP_C until the full ΔT from T_stress_free
# is active. Charge kinematics and the post-peel σ/contact ramp remain active.
function maybe_update_post_peel_cte_ramp!(integrator, cyl_sys)
    retraction_lift_first_after_frozen_cool || return false
    retraction_peel_done[] || return false
    if clip_fth_skip_post_peel_cte
        retraction_qs_cte_scale[] = 0.0
        return false
    end
    retraction_qs_cte_scale[] >= 1.0 - 1.0e-12 && return false

    if !retraction_qs_cte_ramp
        retraction_qs_cte_scale[] = 1.0
        println(">>> Post-peel CTE enabled in one step at t=",
                round(integrator.t; digits=6), " s")
        flush(stdout)
        return true
    end

    if !isfinite(retraction_post_peel_cte_next_t[])
        retraction_post_peel_cte_next_t[] = integrator.t
    end
    integrator.t + 1.0e-12 >= retraction_post_peel_cte_next_t[] || return false

    Tmean_C = mean(cyl_sys.temp) - 273.15
    T_sf_C = T_stress_free - 273.15
    dT_full = max(abs(Tmean_C - T_sf_C), retraction_qs_cool_step_C)
    d_scale = min(1.0 - retraction_qs_cte_scale[],
                  retraction_qs_cool_step_C / dT_full)
    scale_from = retraction_qs_cte_scale[]
    retraction_qs_cte_scale[] = scale_from + d_scale
    retraction_qs_stage_index[] += 1
    retraction_post_peel_cte_next_t[] =
        integrator.t + retraction_post_peel_cte_interval_s

    println(">>> Post-peel CTE stage ", retraction_qs_stage_index[],
            ": cte_scale ", round(scale_from; sigdigits=4), " → ",
            round(retraction_qs_cte_scale[]; sigdigits=4),
            " (≈", retraction_qs_cool_step_C, " K of ",
            round(dT_full; digits=1), " K); Tmean=",
            round(Tmean_C; digits=2), " °C; punch clear; kin ON")
    flush(stdout)
    return true
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
    nd = TrixiParticles.ndims(cyl_sys)
    check_y = retraction_full3d_y_cap_active() && nd >= 2
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        x = cyl_sys.current_coordinates[1, particle]
        z = cyl_sys.current_coordinates[3, particle]
        vx = v_cyl[1, particle]
        vz = v_cyl[3, particle]
        bad = !isfinite(x) || !isfinite(z) || !isfinite(vx) || !isfinite(vz)
        if check_y
            y = cyl_sys.current_coordinates[2, particle]
            vy = v_cyl[2, particle]
            bad = bad || !isfinite(y) || !isfinite(vy)
        end
        bad && (n_bad += 1)
    end
    return n_bad
end

function build_retraction_diag_snapshot(integrator, cyl_sys, floor_sys, mold_sys, v_cyl)
    xs = @view cyl_sys.current_coordinates[1, :]
    zs = @view cyl_sys.current_coordinates[3, :]
    span_x = maximum(xs) - minimum(xs)
    span_z = maximum(zs) - minimum(zs)
    J_min, J_max, J_min_particle = charge_jacobian_extremes(cyl_sys)
    in_springback = retraction_post_cool_springback_active[] ||
                    (mold_retraction_complete[] && (!retraction_eject_enable || retraction_eject_complete[]))
    min_mold_gap, min_floor_gap = in_springback ? (NaN, NaN) : charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    max_speed = 0.0
    max_abs_vz = 0.0
    sum_speed2 = 0.0
    n_speed = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        vx = v_cyl[1, particle]
        vy = v_cyl[2, particle]
        vz = v_cyl[3, particle]
        speed2 = vx * vx + vy * vy + vz * vz
        max_speed = max(max_speed, sqrt(speed2))
        max_abs_vz = max(max_abs_vz, abs(vz))
        sum_speed2 += speed2
        n_speed += 1
    end
    rms_speed = n_speed > 0 ? sqrt(sum_speed2 / n_speed) : 0.0
    t_now = integrator.t
    return (
        step = retraction_diag_step_count[],
        t = t_now,
        dt = integrator.dt,
        kick_max_dv = retraction_diag_kick_max_dv[],
        max_speed = max_speed,
        rms_speed = rms_speed,
        max_abs_vz = max_abs_vz,
        span_x_mm = span_x * 1e3,
        span_z_mm = span_z * 1e3,
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
            round(snap.span_z_mm; digits=4), " mm span_ratio=",
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
            snap.span_z_mm, ",", snap.span_ratio, ",", snap.J_min, ",",
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
    n_fly = update_charge_flyaway_mask!(cyl_sys)
    xmin = Inf
    xmax = -Inf
    zmin = Inf
    zmax = -Inf
    n_used = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        charge_particle_is_flyaway(particle) && continue
        x = cyl_sys.current_coordinates[1, particle]
        z = cyl_sys.current_coordinates[3, particle]
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        zmin = min(zmin, z)
        zmax = max(zmax, z)
        n_used += 1
    end
    if n_used == 0
        xs = @view cyl_sys.current_coordinates[1, :]
        zs = @view cyl_sys.current_coordinates[3, :]
        retraction_reference_span_x[] = maximum(xs) - minimum(xs)
        retraction_reference_span_z[] = maximum(zs) - minimum(zs)
    else
        retraction_reference_span_x[] = xmax - xmin
        retraction_reference_span_z[] = zmax - zmin
    end
    if ignore_charge_flyaways
        println(">>> Ignoring ", n_fly, " / ", nparticles(cyl_sys),
                " charge flyaways (frozen; skipped in contact, peel gaps, span, J-floor)")
        flush(stdout)
    end
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

function repair_hold_end_fp_committed!(cyl_sys, Fp_buf)
    I3 = StaticArrays.SMatrix{3, 3, Float64}(I)
    n_repaired = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        Fp_view = @view Fp_buf[:, :, particle]
        any(!isfinite, Fp_view) || continue
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        det_F = det(F)
        if !isfinite(det_F) || abs(det_F) < 1e-10
            for jj in 1:3, ii in 1:3
                Fp_buf[ii, jj, particle] = I3[ii, jj]
            end
        else
            F_iso = StaticArrays.SMatrix{3, 3, Float64}(F) / cbrt(det_F)
            for jj in 1:3, ii in 1:3
                Fp_buf[ii, jj, particle] = F_iso[ii, jj]
            end
        end
        n_repaired += 1
    end
    n_repaired > 0 &&
        println(">>> Hold-end Fp_committed repair: ", n_repaired, " particles (nonfinite -> F_iso)")
    return n_repaired
end

function clip_invert_correction_matrix(L)
    L_inv = Matrix(1.0I, 3, 3)
    Lx, Ly, Lz = L[1, 1], L[2, 2], L[3, 3]

    is_thin_sheet_xy = (geometry_mode != :pseudo2d) &&
                       (abs(Lz) < 0.2 * min(abs(Lx), abs(Ly)) ||
                        (haskey(ENV, "TP_CLIP_PS_ANISOTROPIC") && env_int("TP_CLIP_PS_ANISOTROPIC", 0) != 0))

    if is_thin_sheet_xy
        # 3D thin sheet in x-y plane (thickness in z):
        # Invert the well-conditioned 2x2 in-plane (x, y) block.
        # Decouple z from in-plane to prevent boundary truncation cross-coupling (cond(L) ~ 1.0).
        Lxy, Lyx = L[1, 2], L[2, 1]
        detL_xy = Lx * Ly - Lxy * Lyx
        detL_xy_reg = sign(detL_xy) * max(abs(detL_xy), 1e-6)
        if abs(Lx) + abs(Ly) + abs(Lxy) + abs(Lyx) > 1e-9
            L_inv[1, 1] = Ly / detL_xy_reg
            L_inv[1, 2] = -Lxy / detL_xy_reg
            L_inv[2, 1] = -Lyx / detL_xy_reg
            L_inv[2, 2] = Lx / detL_xy_reg
        end
        L_inv[3, 3] = abs(Lz) > 1e-9 ? 1.0 / max(Lz, 1e-6) : 1.0
        L_inv[1, 3] = 0.0
        L_inv[3, 1] = 0.0
        L_inv[2, 3] = 0.0
        L_inv[3, 2] = 0.0
        return L_inv
    elseif geometry_mode == :pseudo2d || (abs(Ly) < 0.2 * min(abs(Lx), abs(Lz)))
        # Pseudo-2D or thin-3D in x-z plane (out-of-plane is y):
        # Invert the active 2x2 (x, z) block.
        Lxz, Lzx = L[1, 3], L[3, 1]
        detL_xz = Lx * Lz - Lxz * Lzx
        detL_xz_reg = sign(detL_xz) * max(abs(detL_xz), 1e-6)
        if abs(Lx) + abs(Lz) + abs(Lxz) + abs(Lzx) > 1e-9
            L_inv[1, 1] = Lz / detL_xz_reg
            L_inv[1, 3] = -Lxz / detL_xz_reg
            L_inv[3, 1] = -Lzx / detL_xz_reg
            L_inv[3, 3] = Lx / detL_xz_reg
        end
        L_inv[2, 2] = abs(Ly) > 1e-9 ? 1.0 / max(Ly, 1e-6) : 1.0
        L_inv[1, 2] = 0.0
        L_inv[2, 1] = 0.0
        L_inv[2, 3] = 0.0
        L_inv[3, 2] = 0.0
        return L_inv
    else
        # Full 3D resolved / isotropic:
        detL = det(L)
        return abs(detL) < 1.0e-9 ? Matrix(1.0I, 3, 3) : Matrix(inv(L))
    end
end

function floor_hold_end_deformation_j!(cyl_sys, J_p_buf, J_floor; update_jp::Bool=true)
    J_floor <= 0.0 && return 0
    I3 = StaticArrays.SMatrix{3, 3, Float64}(I)
    n_floored = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        charge_particle_is_flyaway(particle) && continue
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        det_F = det(F)
        isfinite(det_F) && det_F >= J_floor && continue
        # Inverted or collapsed neighborhoods: iso-rescale of a negative det keeps
        # det(F_new)<0. Reset to I (or scaled I) so J_min stays positive.
        if !isfinite(det_F) || det_F <= 0.0 || abs(det_F) < 1e-10
            s = cbrt(J_floor)
            for jj in 1:3, ii in 1:3
                cyl_sys.deformation_grad[ii, jj, particle] = (ii == jj ? s : 0.0)
            end
            (update_jp && use_volumetric_plasticity) && (J_p_buf[particle] = J_floor)
        else
            F_s = StaticArrays.SMatrix{3, 3, Float64}(F)
            F_iso = F_s / cbrt(det_F)
            F_new = F_iso * cbrt(J_floor)
            for jj in 1:3, ii in 1:3
                cyl_sys.deformation_grad[ii, jj, particle] = F_new[ii, jj]
            end
            (update_jp && use_volumetric_plasticity) && (J_p_buf[particle] = J_floor)
        end
        n_floored += 1
    end
    return n_floored
end

const compress_J_floor_log_count = Ref(0)
function apply_compress_j_floor_accepted!(cyl_sys, J_p_buf, t)
    compress_J_floor <= 0.0 && return 0
    # Compression stroke only (punch still closing). Hold/retract use hold-end / retract path.
    t >= t_compress && return 0
    n = floor_hold_end_deformation_j!(cyl_sys, J_p_buf, compress_J_floor; update_jp=false)
    if n > 0
        compress_J_floor_log_count[] += 1
        if compress_J_floor_log_count[] <= 5 || compress_J_floor_log_count[] % 50 == 0
            println(">>> Compress J floor: det(F)→≥", compress_J_floor,
                    " on ", n, " particles at t=", round(t; digits=6),
                    " s (event #", compress_J_floor_log_count[], ")")
            flush(stdout)
        end
    end
    return n
end

const retraction_J_floor_log_count = Ref(0)
const retraction_last_J_floor_n = Ref(0)
function apply_retraction_j_floor_accepted!(cyl_sys, J_p_buf, t)
    if retraction_J_floor <= 0.0 || sim_phase != "retraction"
        retraction_last_J_floor_n[] = 0
        return 0
    end
    n = floor_hold_end_deformation_j!(cyl_sys, J_p_buf, retraction_J_floor)
    retraction_last_J_floor_n[] = n
    if n > 0
        retraction_J_floor_log_count[] += 1
        if retraction_J_floor_log_count[] <= 8 || retraction_J_floor_log_count[] % 40 == 0
            println(">>> Retraction J floor: det(F)→≥", retraction_J_floor,
                    " on ", n, " particles at t=", round(t; digits=6),
                    " s (event #", retraction_J_floor_log_count[], ")")
            flush(stdout)
        end
    end
    return n
end

function prepare_charge_state_for_retraction!(cyl_sys, semi_local, v_ode, u_ode,
                                              Fp_buf, J_ref_buf, J_p_buf, vel_grad_buf,
                                              dv_marker_buf; align_spencer::Bool=hold_end_spencer_align,
                                              reset_fp::Bool=hold_end_fp_reset)
    TrixiParticles.update_nhs!(semi_local, u_ode)
    update_charge_flyaway_mask!(cyl_sys)

    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    u_cyl = TrixiParticles.wrap_u(u_ode, cyl_sys, semi_local)
    freeze_charge_mechanical_state!(v_cyl, cyl_sys)

    # Hold seating improvement: reduce excessive floor overlap on pocket corners.
    floor_sys = length(semi_local.systems) >= floor_system_index ? clip_floor_system(semi_local) : nothing
    if floor_sys !== nothing && env_int("TP_CLIP_IMPROVE_HOLD_SEATING", 1) != 0
        max_overlap_mm = env_float("TP_CLIP_HOLD_SEATING_MAX_OVERLAP_MM", 0.25)
        tol_gap = -max_overlap_mm * 1e-3
        n_lifted = 0
        max_dz = 0.0
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            x_i = u_cyl[1, particle]
            z_i = u_cyl[3, particle]
            gap = tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower)
            if gap < tol_gap
                dz = tol_gap - gap
                u_cyl[3, particle] += dz
                cyl_sys.current_coordinates[3, particle] += dz
                n_lifted += 1
                max_dz = max(max_dz, dz)
            end
        end
        if n_lifted > 0
            TrixiParticles.update_nhs!(semi_local, u_ode)
            println(">>> Improved hold seating on ", n_lifted, " pocket corner particles: max lift dz=",
                    round(max_dz * 1e3; digits=3), " mm (floor overlap clamped to <= ",
                    max_overlap_mm, " mm)")
            flush(stdout)
        end
    end

    # Keep the hold-end F from compression/hold. Recalculating F from SPH neighborhoods on
    # frozen charge geometry can yield det(F) ≪ 1 and GPa-scale bulk stress at retraction.
    apply_charge_plane_strain_F_fix!(cyl_sys)

    if hold_end_J_floor > 0.0
        n_j = floor_hold_end_deformation_j!(cyl_sys, J_p_buf, hold_end_J_floor)
        n_j > 0 &&
            println(">>> Hold-end J floor: raised det(F) to ", hold_end_J_floor,
                    " on ", n_j, " crushed particles")
    end
    hold_end_fp_repair && repair_hold_end_fp_committed!(cyl_sys, Fp_buf)

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
                    # Re-reference solid TLSPH initial_coordinates to current formed clip coordinates
                    for d in 1:3
                        cyl_sys.initial_coordinates[d, particle] = u_cyl[d, particle]
                    end
                    for jj in 1:3, ii in 1:3
                        cyl_sys.deformation_grad[ii, jj, particle] = I3[ii, jj]
                        Fp_buf[ii, jj, particle] = I3[ii, jj]
                    end
                    J_ref_buf[particle] = 1.0
                    J_p_buf[particle] = 1.0
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

    if reset_fp
        TrixiParticles.set_zero!(cyl_sys.correction_matrix)
        initial_coords = TrixiParticles.initial_coordinates(cyl_sys)
        TrixiParticles.foreach_point_neighbor(cyl_sys, cyl_sys, initial_coords, initial_coords,
                                              semi_local;
                                              parallelization_backend=PolyesterBackend()) do particle, neighbor, pos_diff, distance
            grad_kernel = TrixiParticles.smoothing_kernel_grad(cyl_sys, pos_diff, distance, particle)
            iszero(grad_kernel) && return
            volume = cyl_sys.mass[neighbor] / cyl_sys.material_density[neighbor]
            result = volume * grad_kernel * pos_diff'
            for j in 1:3, i in 1:3
                cyl_sys.correction_matrix[i, j, particle] -= result[i, j]
            end
        end
        for particle in TrixiParticles.eachparticle(cyl_sys)
            L = TrixiParticles.extract_smatrix(cyl_sys.correction_matrix, cyl_sys, particle)
            L_inv = clip_invert_correction_matrix(L)
            for j in 1:3, i in 1:3
                cyl_sys.correction_matrix[i, j, particle] = L_inv[i, j]
            end
        end
    end

    align_spencer && align_spencer_fibre_reference!(cyl_sys, fiber_direction)

    enforce_charge_plastic_audit!(cyl_sys, Fp_buf, J_p_buf; label="hold_end_prep")

    return nothing
end

function snapshot_charge_dv!(marker, dv_ode, cyl_sys, semi_local)
    if separate_charge_plies
        clip_foreach_charge_system(semi_local) do system, source
            dv = TrixiParticles.wrap_v(dv_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                g = source[particle]
                marker[1, g] = dv[1, particle]
                marker[2, g] = dv[2, particle]
                marker[3, g] = dv[3, particle]
            end
        end
        return nothing
    end
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
    if separate_charge_plies
        clip_foreach_charge_system(semi_local) do system, source
            dv = TrixiParticles.wrap_v(dv_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                g = source[particle]
                dv[1, particle] = marker[1, g] + scale * (dv[1, particle] - marker[1, g])
                dv[2, particle] = marker[2, g] + scale * (dv[2, particle] - marker[2, g])
                dv[3, particle] = marker[3, g] + scale * (dv[3, particle] - marker[3, g])
            end
        end
        return nothing
    end
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

const _vtu_output_prefix_default = sim_phase == "retraction" ?
                          "molding_cfk_" * geometry_name_tag * "_elastic_" * ps_run_tag * "_ret_solution" :
                          "molding_cfk_" * geometry_name_tag * "_elastic_" * ps_run_tag * "_solution"
const vtu_output_prefix = get(ENV, "TP_CLIP_VTU_PREFIX", _vtu_output_prefix_default)
sim_phase == "retraction" &&
    println(">>> VTU prefix (retraction): ", vtu_output_prefix)
sim_phase == "retraction" &&
    println(">>> VTU save dt (retraction): ", retraction_vtu_save_dt,
            " s (compress/hold default ", vtu_save_dt, " s)")

function vtu_fiber_vf_quantity(sys, data, t)
    return TrixiParticles.nparticles(sys) == length(phase_vf_particle) ?
           phase_vf_particle : fill(NaN, TrixiParticles.nparticles(sys))
end

# A11/A22/A33 from flow-orientation tensor (Song uses A11 for validation)
function vtu_orientation_component(sys, data, t, i, j)
    n = TrixiParticles.nparticles(sys)
    A = orientation_tensor_state[]
    size(A, 3) == n || return fill(NaN, n)
    out = Vector{Float64}(undef, n)
    @inbounds for p in 1:n
        out[p] = A[i, j, p]
    end
    return out
end
vtu_a11_quantity(sys, data, t) = vtu_orientation_component(sys, data, t, 1, 1)
vtu_a22_quantity(sys, data, t) = vtu_orientation_component(sys, data, t, 2, 2)
vtu_a33_quantity(sys, data, t) = vtu_orientation_component(sys, data, t, 3, 3)

function save_vtu_snapshot!(integrator; iter_override=nothing,
                            save_dt=nothing, time_origin=nothing)
    iter = if iter_override === nothing
        TrixiParticles.take_vtu_save_iter!()
    else
        iter_override
    end
    dvdu_ode = try
        get_du(integrator)
    catch
        zero(integrator.u)
    end
    TrixiParticles.trixi2vtk(dvdu_ode, clip_integrator_vu_partition(integrator), integrator.p,
                             integrator.t;
                             iter=iter,
                             output_directory=clip_out_dir,
                             prefix=vtu_output_prefix,
                             max_coordinates=Inf,
                             fiber_vf=vtu_fiber_vf_quantity,
                             a11=vtu_a11_quantity,
                             a22=vtu_a22_quantity,
                             a33=vtu_a33_quantity)
    return nothing
end

function compute_retraction_eject_target_dz(cyl_sys, floor_sys, punch_sys)
    zs = @view cyl_sys.current_coordinates[3, :]
    z_charge_min = minimum(zs)
    z_charge_max = maximum(zs)
    z_charge_mean = 0.5 * (z_charge_min + z_charge_max)
    span = max(z_charge_max - z_charge_min, 0.0)
    # Die deck / land top (for norim STL this is the forming-face max).
    z_deck = maximum(@view floor_sys.current_coordinates[3, :])
    z_punch_min = minimum(@view punch_sys.current_coordinates[3, :])
    clear = max(retraction_eject_clear_ps * particle_spacing, 0.0)
    # Lift until the lowest charge particle is above the deck + clearance.
    need_clear = max(0.0, (z_deck - z_charge_min) + clear)
    # Also clear at least the nominal cavity/pocket depth when known.
    need_cavity = max(0.0, cavity_depth + clear)
    need_min = max(0.0, retraction_eject_min_dz_mm * 1e-3)
    # Optional placement into the free mold↔punch gap.
    # gap_frac=0 → mold-clear only (default). gap_frac=0.5 → old mid-gap aim.
    gap_lo = z_deck + clear
    gap_hi = z_punch_min - clear
    need_gap = 0.0
    if retraction_eject_gap_frac > 0.0 && gap_hi > gap_lo + span
        target_center = gap_lo + retraction_eject_gap_frac * (gap_hi - gap_lo)
        need_gap = max(0.0, target_center - z_charge_mean)
        # Keep charge fully inside [gap_lo, gap_hi].
        need_gap = max(need_gap, gap_lo - z_charge_min)
        need_gap = min(need_gap, max(0.0, gap_hi - span - z_charge_min))
    end
    need = max(need_clear, need_cavity, need_min, need_gap)
    if retraction_eject_max_dz_mm > 0.0
        need = min(need, retraction_eject_max_dz_mm * 1e-3)
    end
    # Hard cap: do not drive the charge into / above the punch.
    room = max(0.0, (z_punch_min - clear) - z_charge_max)
    return min(need, room)
end

function apply_charge_eject_z!(integrator, z_abs)
    prev = retraction_eject_z_applied[]
    dz = z_abs - prev
    abs(dz) <= 0.0 && return nothing
    v_ode, u_ode = clip_integrator_vu(integrator)
    cyl_sys = integrator.p.systems[1]
    u_cyl = TrixiParticles.wrap_u(u_ode, cyl_sys, integrator.p)
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, integrator.p)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        u_cyl[3, particle] += dz
        cyl_sys.current_coordinates[3, particle] += dz
        v_cyl[1, particle] = 0.0
        if NDIMS_CYL >= 2
            v_cyl[2, particle] = 0.0
        end
        v_cyl[3, particle] = 0.0
    end
    retraction_eject_z_applied[] = z_abs
    # Force NHS refresh on next RHS — charge left the cavity.
    nhs_updated_at_t[] = -Inf
    return nothing
end

function maybe_bang_room_temp_after_eject!(integrator)
    retraction_bang_t_after_eject || return nothing
    retraction_eject_complete[] || return nothing
    retraction_post_eject_bang_done[] && return nothing
    cyl_sys = integrator.p.systems[1]
    v_ode, _ = clip_integrator_vu(integrator)
    Tmean_C = mean(cyl_sys.temp) - 273.15
    target_C = retraction_cool_until_temp_C
    delta_C = target_C - Tmean_C
    apply_charge_temperature_delta!(cyl_sys, integrator.p, v_ode, delta_C)
    thermal_Fth_u_modified[] = true
    retraction_post_eject_bang_done[] = true
    retraction_post_eject_bang_t[] = integrator.t
    println(">>> Post-eject bang T: Tmean ", round(Tmean_C; digits=2),
            " → ", round(target_C; digits=2),
            " °C (room-temp target); T frozen; kin ON for ",
            retraction_post_eject_dwell_s, " s springback")
    flush(stdout)
    save_vtu_snapshot!(integrator)
    return nothing
end

function start_retraction_eject!(integrator)
    if !retraction_eject_announced[]
        save_preeject_checkpoint!(integrator)
    end
    cyl_sys = integrator.p.systems[1]
    floor_sys = clip_floor_system(integrator.p)
    punch_sys = clip_mold_system(integrator.p)
    retraction_eject_active[] = true
    retraction_eject_complete[] = false
    retraction_eject_start_t[] = integrator.t
    retraction_eject_z_applied[] = 0.0
    retraction_eject_target_dz[] = compute_retraction_eject_target_dz(cyl_sys, floor_sys, punch_sys)
    if !retraction_eject_announced[]
        z_deck = maximum(@view floor_sys.current_coordinates[3, :])
        z_punch_min = minimum(@view punch_sys.current_coordinates[3, :])
        println(">>> Starting workpiece eject / demold at t=",
                round(integrator.t; digits=6),
                " s: target_dz=",
                round(retraction_eject_target_dz[] * 1e3; digits=3),
                " mm over ", round(retraction_eject_s; digits=4),
                " s (punch/mold OFF; floor ",
                retraction_keep_floor_during_eject ? "ON during stroke" : "OFF",
                retraction_freeze_mech_during_eject ? "; mech/F FROZEN during stroke" : "",
                "; rigid +z into gap mold↔punch; deck=",
                round(z_deck * 1e3; digits=2),
                " mm, punch_min=",
                round(z_punch_min * 1e3; digits=2), " mm)")
        flush(stdout)
        retraction_eject_announced[] = true
    end
    save_vtu_snapshot!(integrator)
    return nothing
end

function maybe_update_retraction_eject!(integrator)
    retraction_eject_enable || return false
    retraction_eject_active[] || return false
    retraction_eject_complete[] && return false
    dur = max(retraction_eject_s, eps(Float64))
    progress = clamp((integrator.t - retraction_eject_start_t[]) / dur, 0.0, 1.0)
    # Smoothstep eject stroke.
    s = progress * progress * (3.0 - 2.0 * progress)
    apply_charge_eject_z!(integrator, s * retraction_eject_target_dz[])
    if progress < 1.0 - 1.0e-12
        return true
    end
    apply_charge_eject_z!(integrator, retraction_eject_target_dz[])
    retraction_eject_active[] = false
    retraction_eject_complete[] = true
    save_vtu_snapshot!(integrator)
    println(">>> Workpiece eject complete at t=", round(integrator.t; digits=6),
            " s; lifted ",
            round(retraction_eject_z_applied[] * 1e3; digits=3),
            " mm out of mold; punch/mold OFF",
            retraction_keep_floor_after_eject ? "; floor table ON" : "; floor OFF")
    maybe_bang_room_temp_after_eject!(integrator)
    if retraction_use_explicit
        # Demold finished → allow larger free-cool / springback dt from here on.
        enforce_explicit_retraction_dt!(integrator)
        println(">>> Free cool after eject: dt→", retraction_explicit_dt_springback,
                " s (demold stayed at ", retraction_explicit_dt, " s)")
        if retraction_unfreeze_t_after_eject
            println(">>> Free cool after eject: T UNFROZEN (demold used FREEZE_T=",
                    retraction_explicit_freeze_t, " unchanged; ambient ON",
                    retraction_skip_sph_conduction_after_eject ?
                        "; SPH conduction OFF" : "; SPH conduction ON",
                    retraction_thermal_shrinkage_only_after_eject ?
                        "; plastic OFF; CTE shrinkage ON" :
                        ("; plastic heat=" * (retraction_skip_plastic_heat_after_eject ? "OFF" : "ON")),
                    ")")
        end
    end
    if retraction_freeze_mech_after_eject
        println(">>> Free cool: charge mechanics FROZEN after eject (thermal-only + ambient",
                retraction_free_cool_exposure_floor > 0.0 ?
                    "; exposure floor=" * string(retraction_free_cool_exposure_floor) : "",
                "; post-cool springback=", retraction_post_cool_springback_s, " s)")
    elseif retraction_thermal_shrinkage_only_after_eject
        println(">>> Free cool: mechanics ON after eject (thermal shrinkage only; dt=",
                retraction_explicit_dt_springback, " s)")
    end
    if retraction_keep_floor_after_eject || retraction_gravity_after_eject != 0.0 ||
       retraction_rayleigh_after_eject > 0.0
        println(">>> Free cool support: floor=",
                retraction_keep_floor_after_eject ? "ON" : "OFF",
                "  g=", retraction_gravity_after_eject,
                " m/s²  extra Rayleigh γ=", retraction_rayleigh_after_eject, " 1/s")
    end
    flush(stdout)
    # Arm free springback dwell clock from eject end.
    mold_retraction_complete_time[] = integrator.t
    cyl_sys = integrator.p.systems[1]
    v_cyl_ref, _ = clip_integrator_vu(integrator)
    v_cyl = TrixiParticles.wrap_v(v_cyl_ref, cyl_sys, integrator.p)
    springback_equil_streak[] = 0
    springback_equil_prev_snap[] = build_retraction_diag_snapshot(integrator, cyl_sys,
        clip_floor_system(integrator.p), clip_mold_system(integrator.p), v_cyl)
    return false
end

function apply_tool_temp_after_punch_lift!(integrator; when="after punch lift")
    retraction_tool_temp_after_punch_lift_active || return nothing
    retraction_tool_temp_after_punch_lift_done[] && return nothing
    target_K = celsius_to_kelvin(retraction_tool_temp_after_punch_lift_C)
    tool_names = String[]
    for (idx, sys) in enumerate(integrator.p.systems[2:end])
        sys.temp .= target_K
        push!(tool_names, "systems[" * string(idx + 1) * "]")
    end
    retraction_tool_temp_after_punch_lift_done[] = true
    println(">>> Tool temperature ", when, ": ",
            join(tool_names, ", "), " → ",
            round(retraction_tool_temp_after_punch_lift_C; digits=2),
            " °C (closed-die / in-die cool sink)")
    flush(stdout)
    save_vtu_snapshot!(integrator)
    return nothing
end

function complete_mold_retraction!(integrator)
    mold_retraction_complete[] = true
    mold_retraction_complete_time[] = integrator.t
    save_vtu_snapshot!(integrator)
    if retraction_eject_enable
        if retraction_eject_after_cool
            apply_tool_temp_after_punch_lift!(integrator)
            println(">>> Punch stroke complete at t=", round(integrator.t; digits=6),
                    " s; punch UP — workpiece stays in die (floor ON, plastic OFF)")
            println(">>> In-die cool: T UNFROZEN, convection ON, plastic OFF; eject when mean T ≤ ",
                    retraction_cool_until_temp_C, " °C (dwell cap ",
                    retraction_springback_dwell_s, " s)")
            flush(stdout)
            return nothing
        end
        println(">>> Punch stroke complete at t=", round(integrator.t; digits=6),
                " s; beginning demold eject of workpiece from cavity")
        flush(stdout)
        return start_retraction_eject!(integrator)
    end
    apply_tool_temp_after_punch_lift!(integrator)
    if retraction_springback_dwell_s <= 0.0 && !retraction_cool_until_active
        return complete_retraction_simulation!(integrator; reason="mold stroke complete (dwell=0)")
    end
    cyl_sys = integrator.p.systems[1]
    v_cyl_ref, _ = clip_integrator_vu(integrator)
    v_cyl = TrixiParticles.wrap_v(v_cyl_ref, cyl_sys, integrator.p)
    springback_equil_streak[] = 0
    springback_equil_prev_snap[] = build_retraction_diag_snapshot(integrator, cyl_sys,
        clip_floor_system(integrator.p), clip_mold_system(integrator.p), v_cyl)
    if retraction_cool_until_active
        println(">>> Punch stroke complete at t=", round(integrator.t; digits=6),
                " s; punch UP — no eject; workpiece stays in die (floor ON, plastic OFF)")
        println(">>> In-die cool (no eject): T UNFROZEN, convection ON; stop when mean T ≤ ",
                retraction_cool_until_temp_C, " °C (dwell cap ",
                retraction_springback_dwell_s, " s)")
    else
        println(">>> Mold stroke complete at t=", round(integrator.t; digits=6),
                " s; springback equilibration dwell (max ",
                round(retraction_springback_dwell_s; digits=4), " s)")
    end
    flush(stdout)
    return nothing
end

function complete_retraction_simulation!(integrator; reason="springback equilibrated")
    retraction_complete[] = true
    apply_retraction_j_floor_accepted!(integrator.p.systems[1],
                                       J_p_particle_buf, integrator.t)
    save_vtu_snapshot!(integrator)
    save_cool_end_checkpoint!(integrator)
    println(">>> Retraction finished (", reason, ") at t=", round(integrator.t; digits=6), " s")
    cyl = integrator.p.systems[1]
    xs = @view cyl.current_coordinates[1, :]
    ys = @view cyl.current_coordinates[2, :]
    zs = @view cyl.current_coordinates[3, :]
    span_x_mm = (maximum(xs) - minimum(xs)) * 1e3
    span_y_mm = (maximum(ys) - minimum(ys)) * 1e3
    span_z_mm = (maximum(zs) - minimum(zs)) * 1e3
    ref_x = retraction_reference_span_x[] * 1e3
    ref_z = retraction_reference_span_z[] * 1e3
    println(">>> Free-part warpage vs hold span: span_x=", round(span_x_mm; digits=3),
            " mm (Δ", round(span_x_mm - ref_x; digits=3), " mm)  span_y=",
            round(span_y_mm; digits=3), " mm  span_z=", round(span_z_mm; digits=3),
            " mm (Δ", round(span_z_mm - ref_z; digits=3),
            " mm)  Song Table 4 target 1.61 mm")
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

# Transversely isotropic F_th = λ⊥ I + (λ∥ − λ⊥) n⊗n, n = F n_ref / |F n_ref|
# (Fig. 4 principal fibre). Positions: same F_th about (cx, cy, z_floor).
@inline function clip_aniso_thermal_stretch(cyl_sys, particle, F, dT)
    α_par = alpha1_particle[particle]
    α_perp = alpha2_particle[particle]
    lam_par = clamp(exp(α_par * dT), 0.90, 1.02)
    lam_perp = clamp(exp(α_perp * dT), 0.90, 1.02)
    n_ref = StaticArrays.SVector{3, Float64}(fiber_direction[1, particle],
                                             fiber_direction[2, particle],
                                             fiber_direction[3, particle])
    F_s = StaticArrays.SMatrix{3, 3, Float64}(F)
    Fn = F_s * n_ref
    nrm2 = Fn[1] * Fn[1] + Fn[2] * Fn[2] + Fn[3] * Fn[3]
    n = if nrm2 > 1.0e-24
        Fn / sqrt(nrm2)
    else
        nrm_ref = sqrt(n_ref[1] * n_ref[1] + n_ref[2] * n_ref[2] + n_ref[3] * n_ref[3])
        nrm_ref > 1.0e-14 ? n_ref / nrm_ref :
            StaticArrays.SVector{3, Float64}(1.0, 0.0, 0.0)
    end
    dlam = lam_par - lam_perp
    Fth = lam_perp * one(StaticArrays.SMatrix{3, 3, Float64}) + dlam * (n * n')
    return Fth, lam_par, lam_perp
end

# Die-aware F_th (not r9 centroid homothety): shrink onto the floor in z and
# about the charge xy centroid, and scale F so det(F) tracks J_th. Kin stays frozen.
function apply_frozen_thermal_contraction!(cyl_sys, u_ode, semi_local, t;
                                          v_ode=nothing, force::Bool=false)
    clip_fth_die_aware || return false
    if !force
        frozen_thermal_contraction_active(t) || return false
    else
        hold_kinematics_frozen(t) || return false
    end
    n = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        n += 1
    end
    n == 0 && return false
    if !thermal_Fth_initialized[]
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            thermal_Fth_T_prev[particle] = cyl_sys.temp[particle]
        end
        thermal_Fth_initialized[] = true
        return false
    end

    u_cyl = TrixiParticles.wrap_u(u_ode, cyl_sys, semi_local)
    nd = TrixiParticles.ndims(cyl_sys)
    v_cyl = v_ode === nothing ? nothing : TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    floor_sys = length(semi_local.systems) >= floor_system_index ? clip_floor_system(semi_local) : nothing
    z_floor = if floor_sys !== nothing
        maximum(@view floor_sys.current_coordinates[3, :])
    else
        zmin = Inf
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            zmin = min(zmin, u_cyl[3, particle])
        end
        zmin
    end

    cx = 0.0
    cy = 0.0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        cx += u_cyl[1, particle]
        nd >= 2 && (cy += u_cyl[2, particle])
    end
    invn = 1.0 / n
    cx *= invn
    cy *= invn

    sum_lam_par = 0.0
    sum_lam_perp = 0.0
    sum_dT = 0.0
    n_move = 0
    xmin0 = Inf
    xmax0 = -Inf
    ymin0 = Inf
    ymax0 = -Inf
    zmin0 = Inf
    zmax0 = -Inf
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        xmin0 = min(xmin0, u_cyl[1, particle])
        xmax0 = max(xmax0, u_cyl[1, particle])
        if nd >= 2
            ymin0 = min(ymin0, u_cyl[2, particle])
            ymax0 = max(ymax0, u_cyl[2, particle])
        end
        zmin0 = min(zmin0, u_cyl[3, particle])
        zmax0 = max(zmax0, u_cyl[3, particle])
        T_now = cyl_sys.temp[particle]
        dT = T_now - thermal_Fth_T_prev[particle]
        thermal_Fth_T_prev[particle] = T_now
        dT >= -1.0e-8 && continue
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        Fth, lam_par, lam_perp = clip_aniso_thermal_stretch(cyl_sys, particle, F, dT)
        F_new = Fth * StaticArrays.SMatrix{3, 3, Float64}(F)
        for jj in 1:3, ii in 1:3
            cyl_sys.deformation_grad[ii, jj, particle] = F_new[ii, jj]
        end
        J_scale = lam_par * lam_perp * lam_perp
        J_ref_particle_buf[particle] = max(J_ref_particle_buf[particle] * J_scale, 1e-6)
        use_volumetric_plasticity &&
            (J_p_particle_buf[particle] = max(J_p_particle_buf[particle] * J_scale, 1e-6))
        dx = u_cyl[1, particle] - cx
        dy = nd >= 2 ? (u_cyl[2, particle] - cy) : 0.0
        dz = u_cyl[3, particle] - z_floor
        x_new = cx + Fth[1, 1] * dx + Fth[1, 2] * dy + Fth[1, 3] * dz
        y_new = cy + Fth[2, 1] * dx + Fth[2, 2] * dy + Fth[2, 3] * dz
        z_new = z_floor + Fth[3, 1] * dx + Fth[3, 2] * dy + Fth[3, 3] * dz
        u_cyl[1, particle] = x_new
        cyl_sys.current_coordinates[1, particle] = x_new
        if nd >= 2
            u_cyl[2, particle] = y_new
            cyl_sys.current_coordinates[2, particle] = y_new
        end
        u_cyl[3, particle] = z_new
        cyl_sys.current_coordinates[3, particle] = z_new
        if v_cyl !== nothing
            v_cyl[1, particle] = 0.0
            nd >= 2 && (v_cyl[2, particle] = 0.0)
            v_cyl[3, particle] = 0.0
        end
        sum_lam_par += lam_par
        sum_lam_perp += lam_perp
        sum_dT += dT
        n_move += 1
    end
    n_move == 0 && return false

    thermal_Fth_skip_Jp[] = true
    TrixiParticles.update_nhs!(semi_local, u_ode)
    xmin1 = Inf
    xmax1 = -Inf
    ymin1 = Inf
    ymax1 = -Inf
    zmin1 = Inf
    zmax1 = -Inf
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        xmin1 = min(xmin1, u_cyl[1, particle])
        xmax1 = max(xmax1, u_cyl[1, particle])
        if nd >= 2
            ymin1 = min(ymin1, u_cyl[2, particle])
            ymax1 = max(ymax1, u_cyl[2, particle])
        end
        zmin1 = min(zmin1, u_cyl[3, particle])
        zmax1 = max(zmax1, u_cyl[3, particle])
    end
    println(">>> Die-aware F_th (aniso Fig.4): n=", n_move, " mean λ∥=",
            round(sum_lam_par / n_move; sigdigits=6), " mean λ⊥=",
            round(sum_lam_perp / n_move; sigdigits=6), " mean dT=",
            round(sum_dT / n_move; digits=2), " K; span_x ",
            round((xmax0 - xmin0) * 1e3; digits=3), " → ",
            round((xmax1 - xmin1) * 1e3; digits=3), " mm; span_y ",
            nd >= 2 ? round((ymax0 - ymin0) * 1e3; digits=3) : 0.0, " → ",
            nd >= 2 ? round((ymax1 - ymin1) * 1e3; digits=3) : 0.0,
            " mm; span_z ",
            round((zmax0 - zmin0) * 1e3; digits=3), " → ",
            round((zmax1 - zmin1) * 1e3; digits=3), " mm; z_floor=",
            round(z_floor * 1e3; digits=3), " mm; kin frozen")
    flush(stdout)
    return true
end

@inline function clip_thermal_freeze_restore!(dv_cyl, cyl_sys, t)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    if clip_jac_thermal_frozen && clip_in_jacobian_fill[] && clip_thermal_jac_cache_valid[]
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[NDIMS_CYL + 1, particle] = clip_thermal_dv_cache[particle]
        end
        return true
    end
    if clip_thermal_same_t_cache && !clip_in_jacobian_fill[] && t == clip_thermal_same_t_updated_at[]
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[NDIMS_CYL + 1, particle] = clip_thermal_dv_cache[particle]
        end
        return true
    end
    return false
end

function clip_thermal_freeze_capture!(dv_cyl, cyl_sys, t)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        clip_thermal_dv_cache[particle] = dv_cyl[NDIMS_CYL + 1, particle]
    end
    if clip_in_jacobian_fill[] && clip_jac_thermal_frozen
        clip_thermal_jac_cache_valid[] = true
    end
    if clip_thermal_same_t_cache && !clip_in_jacobian_fill[]
        clip_thermal_same_t_updated_at[] = t
    end
    return nothing
end

@inline function clip_contact_freeze_restore_active(t)
    if clip_jac_contact_frozen && clip_in_jacobian_fill[] && clip_contact_jac_cache_valid[]
        return true
    end
    if clip_contact_same_t_cache && !clip_in_jacobian_fill[] &&
       t == clip_contact_same_t_updated_at[]
        return true
    end
    return false
end

function clip_apply_cached_contact_increment!(dv_ode, cyl_sys, semi_local)
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        dv_cyl[1, particle] += clip_contact_dv_inc_cache[1, particle]
        dv_cyl[2, particle] += clip_contact_dv_inc_cache[2, particle]
        dv_cyl[3, particle] += clip_contact_dv_inc_cache[3, particle]
    end
    return nothing
end

function clip_capture_contact_increment!(dv_ode, pre_snap, cyl_sys, semi_local, t)
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        clip_contact_dv_inc_cache[1, particle] = dv_cyl[1, particle] - pre_snap[1, particle]
        clip_contact_dv_inc_cache[2, particle] = dv_cyl[2, particle] - pre_snap[2, particle]
        clip_contact_dv_inc_cache[3, particle] = dv_cyl[3, particle] - pre_snap[3, particle]
    end
    if clip_in_jacobian_fill[] && clip_jac_contact_frozen
        clip_contact_jac_cache_valid[] = true
    end
    if clip_contact_same_t_cache && !clip_in_jacobian_fill[]
        clip_contact_same_t_updated_at[] = t
    end
    return nothing
end

# #region agent log
const _agent_debug_log_path =
    "/home/sadaat/SPH_Code/TrixiParticles.jl_Saadat/.cursor/debug-a799af.log"
const _agent_debug_last_t = Ref(-Inf)
function _agent_debug_json_escape(s::AbstractString)
    return replace(replace(s, '\\' => "\\\\"), '"' => "\\\"")
end
function _agent_debug_max_abs3(buf, cyl_sys)
    m = 0.0
    pid = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        ax = buf[1, particle]; ay = buf[2, particle]; az = buf[3, particle]
        mag = sqrt(ax * ax + ay * ay + az * az)
        if mag > m
            m = mag
            pid = particle
        end
    end
    return m, pid
end
function _agent_debug_jminmax(cyl_sys)
    jmin = Inf
    jmax = -Inf
    pmin = 0
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
        if J < jmin
            jmin = J
            pmin = particle
        end
        jmax = max(jmax, J)
    end
    return jmin, jmax, pmin
end
# Compare legacy ln(det(Fe)/J_ref) vs Strategy E ln(J_e_vol) on the charge.
function _agent_debug_bulk_logs(cyl_sys, Fp_buf)
    max_legacy = 0.0
    max_stratE = 0.0
    max_dev = 0.0
    max_i4 = 0.0
    pid_legacy = 0
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        F = TrixiParticles.deformation_gradient(cyl_sys, particle)
        det_F = det(F)
        (!isfinite(det_F) || abs(det_F) < 1e-10) && continue
        Fp_local = StaticArrays.SMatrix{3, 3}(@view Fp_buf[:, :, particle])
        d_fp = det(Fp_local)
        if isfinite(d_fp) && d_fp > 1e-4
            Fp_local = Fp_local / cbrt(d_fp)
        else
            Fp_local = F / cbrt(det_F)
        end
        Fe = F * inv(Fp_local)
        J_e = max(det(Fe), 1e-6)
        J_ref_p = max(J_ref_particle_buf[particle], 1e-6)
        legacy = abs(log(max(J_e / J_ref_p, 1e-6)))
        stratE = abs(matrix_bulk_log_J(F, particle, Fp_local, 1.0, J_ref_p))
        be = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - (1 / 3) * tr(be_bar) * I
        dmag = sqrt(sum(dev_be .* dev_be))
        n_ref_p = StaticArrays.SVector{3}(fiber_direction[1, particle],
                                          fiber_direction[2, particle],
                                          fiber_direction[3, particle])
        Fn_p = F * n_ref_p
        i4dev = abs(dot(Fn_p, Fn_p) - 1.0)
        if legacy > max_legacy
            max_legacy = legacy
            pid_legacy = particle
        end
        max_stratE = max(max_stratE, stratE)
        max_dev = max(max_dev, dmag)
        max_i4 = max(max_i4, i4dev)
    end
    return max_legacy, max_stratE, max_dev, max_i4, pid_legacy
end
function agent_debug_log!(hypothesisId::AbstractString, location::AbstractString,
                          message::AbstractString, data::AbstractDict;
                          runId::AbstractString="peel-decomp")
    try
        open(_agent_debug_log_path, "a") do io
            print(io, "{\"sessionId\":\"a799af\",\"runId\":\"", runId,
                  "\",\"hypothesisId\":\"", _agent_debug_json_escape(hypothesisId),
                  "\",\"location\":\"", _agent_debug_json_escape(location),
                  "\",\"message\":\"", _agent_debug_json_escape(message),
                  "\",\"timestamp\":", round(Int, time() * 1000), ",\"data\":{")
            first = true
            for (k, v) in data
                first || print(io, ",")
                first = false
                print(io, "\"", k, "\":")
                if v isa AbstractString
                    print(io, "\"", _agent_debug_json_escape(v), "\"")
                elseif v isa Bool
                    print(io, v ? "true" : "false")
                elseif v isa Integer
                    print(io, v)
                elseif v isa Real
                    if isnan(v)
                        print(io, "\"NaN\"")
                    elseif isinf(v)
                        print(io, v > 0 ? "\"Inf\"" : "\"-Inf\"")
                    else
                        print(io, v)
                    end
                else
                    print(io, "\"", _agent_debug_json_escape(string(v)), "\"")
                end
            end
            println(io, "}}")
        end
    catch
    end
    return nothing
end
function agent_debug_peel_rhs!(dv_ode, cyl_sys, semi_local, floor_sys, mold_sys, t;
                               stage::AbstractString)
    retraction_kinematic_peel || return nothing
    retraction_started[] || return nothing
    in_creep = retraction_pre_lift_creep_active(t)
    near_release = !in_creep &&
        (t - retraction_lift_start_time[]) >= (retraction_peel_max_s - 0.01)
    post_release = retraction_peel_done[]
    in_sigma_ramp = post_release && retraction_peel_sigma_ramp_s > 0.0 &&
        (t - retraction_peel_release_time[]) <= (retraction_peel_sigma_ramp_s + 0.02)
    sample_creep = in_creep && (t - _agent_debug_last_t[] >= 0.05)
    sample_peel = !in_creep && !post_release &&
                  (t - _agent_debug_last_t[] >= 0.25) &&
                  (t - retraction_lift_start_time[] >= 0.0)
    sample_ramp = in_sigma_ramp && (t - _agent_debug_last_t[] >= 0.01)
    (near_release || post_release || sample_peel || sample_ramp || sample_creep) || return nothing
    # Throttle: during sigma ramp log every ~10 ms; otherwise keep prior cadence.
    if post_release && !in_sigma_ramp && (t - _agent_debug_last_t[] < 0.05)
        return nothing
    end
    _agent_debug_last_t[] = t
    internal_max, internal_pid = _agent_debug_max_abs3(clip_contact_dv_pre_snap, cyl_sys)
    contact_max, contact_pid = _agent_debug_max_abs3(clip_contact_dv_inc_cache, cyl_sys)
    total_max = max_charge_kick_dv(dv_ode, cyl_sys, semi_local)
    jmin, jmax, jp = _agent_debug_jminmax(cyl_sys)
    max_leg, max_e, max_dev, max_i4, pid_leg = _agent_debug_bulk_logs(cyl_sys, Fp_committed[])
    gmin, gfloor = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    frozen = hold_kinematics_frozen(t)
    peel_scale = retraction_peel_post_mech_scale(t)
    hyp = if in_creep
        "H-creepFree"
    elseif max_leg > 0.05 && max_e < 0.05
        "H-JeVolBug"
    elseif contact_max > internal_max * 2
        "A"
    elseif internal_max > contact_max * 2
        "B"
    else
        "A+B"
    end
    msg = if in_creep
        "closed-die creep RHS (free kin)"
    elseif in_sigma_ramp
        "post-peel sigma-ramp RHS"
    else
        "peel RHS decomposition before freeze-zero"
    end
    agent_debug_log!(hyp, "kick_implicit_visible!:$(stage)", msg,
                     Dict(
                         "t" => t,
                         "stage" => stage,
                         "frozen" => frozen,
                         "in_creep" => in_creep,
                         "peel_done" => retraction_peel_done[],
                         "peel_scale" => peel_scale,
                         "peel_sigma_ramp_s" => retraction_peel_sigma_ramp_s,
                         "kin" => retraction_kinematics_scale(t),
                         "elas" => retraction_elastic_scale(t),
                         "mech_dv" => retraction_dv_mech_scale(t),
                         "floor_act" => floor_reimann_activation_factor(t, cyl_sys),
                         "mold_act" => mold_reimann_activation_factor(cyl_sys, mold_sys, t),
                         "internal_max_dv" => internal_max,
                         "internal_pid" => internal_pid,
                         "contact_max_dv" => contact_max,
                         "contact_pid" => contact_pid,
                         "total_max_dv" => total_max,
                         "J_min" => jmin,
                         "J_max" => jmax,
                         "J_min_pid" => jp,
                         "max_legacy_bulk_log" => max_leg,
                         "max_stratE_bulk_log" => max_e,
                         "max_dev_be" => max_dev,
                         "max_abs_I4_m1" => max_i4,
                         "legacy_bulk_pid" => pid_leg,
                         "min_mold_gap_mm" => gmin * 1e3,
                         "min_floor_gap_mm" => gfloor * 1e3,
                         "min_floor_top_gap_mm" => charge_min_floor_top_gap(cyl_sys, floor_sys) * 1e3,
                         "floor_z_shift_mm" => retraction_floor_open_z_shift[] * 1e3,
                         "z_floor_max_mm" => floor_tool_z_extents(floor_sys)[2] * 1e3,
                         "contact_e" => (isdefined(Main, :active_contact_e_scale) ?
                             Float64(active_contact_e_scale()) :
                             Float64(retraction_contact_e_scale[])),
                     ); runId="jeVol")
    return nothing
end
# #endregion

@inline function compute_ambient_cooling_rate(T_i, exposure, rho_i, cp_eff, dx, t)
    T_i <= temp_ambient && return 0.0
    flux_conv = h_ambient * (T_i - temp_ambient)
    flux_rad = stefan_boltzmann * emissivity_charge * (T_i^4 - temp_ambient^4)
    raw_dT_dt = exposure * (flux_conv + flux_rad) / (rho_i * cp_eff * dx)
    # Unconditional exponential decay safeguard against explicit Euler overshoot:
    dt_step = retraction_use_explicit ? retraction_explicit_target_dt(t) : 0.005
    decay_rate = raw_dT_dt / max(T_i - temp_ambient, 1e-6)
    decay_factor = (1.0 - exp(-decay_rate * dt_step)) / max(dt_step, 1e-6)
    return (T_i - temp_ambient) * decay_factor
end

function apply_charge_hold_thermal_kick!(dv_ode, v_ode, semi_local, t)
    cyl_sys = semi_local.systems[1]
    floor_sys = clip_floor_system(semi_local)
    mold_sys = clip_mold_system(semi_local)
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)

    freeze_charge_mechanical_state!(v_cyl, cyl_sys)

    T_floor = ambient_heat_loss_active_now() ? max(temp_ambient, temp_min_clip) : temp_min_clip
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        t_trial = v_cyl[NDIMS_CYL + 1, particle]
        if !isfinite(t_trial)
            t_trial = cyl_sys.temp[particle]
        end
        t_clamped = clamp(t_trial, T_floor, temp_max_clip)
        cyl_sys.temp[particle] = t_clamped
        v_cyl[NDIMS_CYL + 1, particle] = t_clamped
    end

    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    if clip_thermal_freeze_restore!(dv_cyl, cyl_sys, t)
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[1, particle] = 0.0
            dv_cyl[2, particle] = 0.0
            dv_cyl[3, particle] = 0.0
        end
        return dv_ode
    end

    contact_heat_flux = if retraction_post_eject_phase()
        fill!(contact_heat_flux_buf, 0.0)
    else
        compute_contact_heat_flux!(contact_heat_flux_buf, cyl_sys,
                                   floor_sys, mold_sys, particle_spacing, t)
    end
    if !retraction_skip_sph_conduction_now()
        TrixiParticles.thermal_rhs_sph3d!(cyl_sys, dv_cyl, v_cyl, 0.0,
                                          particle_spacing, bound_coordinate_thermal, semi_local)
    end
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
            rho_i = physical_thermal_density(cyl_sys, particle)
            dv_cyl[NDIMS_CYL + 1, particle] -= contact_heat_flux[particle] /
                                              (rho_i * cp_eff * particle_spacing)
        end
        if ambient_heat_loss_active_now()
            exposure = free_cool_ambient_exposure(particle)
            if exposure > 0.0
                rho_i = physical_thermal_density(cyl_sys, particle)
                dT_dt_loss = compute_ambient_cooling_rate(T_i, exposure, rho_i, cp_eff, particle_spacing, t)
                dv_cyl[NDIMS_CYL + 1, particle] -= dT_dt_loss
            end
        end
        dv_cyl[1, particle] = 0.0
        dv_cyl[2, particle] = 0.0
        dv_cyl[3, particle] = 0.0
    end
    clip_thermal_freeze_capture!(dv_cyl, cyl_sys, t)

    return dv_ode
end

@inline function free_cool_ambient_exposure(particle)
    exp0 = charge_boundary_exposure[particle]
    if retraction_free_cool_thermal_only_now() && retraction_free_cool_exposure_floor > 0.0
        return max(exp0, retraction_free_cool_exposure_floor)
    end
    return exp0
end

function apply_charge_hold_rhs!(dv_ode, v_ode, u_ode, semi_local, t)
    apply_charge_hold_thermal_kick!(dv_ode, v_ode, semi_local, t)
    # Post-eject free cool: skip frozen stress every step (thermal-only + large dt).
    # Thermal residual is applied when post-cool springback re-enables mechanics.
    if retraction_free_cool_thermal_only_now()
        return dv_ode
    end
    # Frozen-geometry stress evaluation (no contact / momentum): keeps σ and Fp commit
    # consistent through solidification without reintroducing stiff tool interaction.
    enable_elastic_stress || return dv_ode
    cyl_sys = semi_local.systems[1]
    if clip_in_jacobian_fill[] && clip_jac_nhs_refresh
        TrixiParticles.update_nhs!(semi_local, u_ode)
    elseif t != nhs_updated_at_t[] || (clip_ode_flat[] && CURRENT_JACOBIAN_STRATEGY[] == "matrix_free")
        TrixiParticles.update_nhs!(semi_local, u_ode)
        nhs_updated_at_t[] = t
    end
    TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] =
        clip_deformation_only_selector(semi_local)
    apply_all_charge_plane_strain_F_fixes!(semi_local)
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
    deform = cyl_sys.deformation_grad
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        deform[2, 1, particle] = 0.0
        deform[2, 2, particle] = 1.0
        deform[2, 3, particle] = 0.0
        deform[1, 2, particle] = 0.0
        deform[3, 2, particle] = 0.0
    end
    return nothing
end

function apply_all_charge_plane_strain_F_fixes!(semi_local)
    clip_foreach_charge_system(semi_local) do system, _
        apply_charge_plane_strain_F_fix!(system)
    end
    return nothing
end

# Ḟ = L·F on accepted steps during retraction (position-based F skipped in NL iterates).
function advance_F_from_vel_grad!(cyl_sys, vel_grad_cache, dt)
    I_mat = StaticArrays.SMatrix{TrixiParticles.ndims(cyl_sys),
                                 TrixiParticles.ndims(cyl_sys), Float64}(I)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        L_corr = TrixiParticles.correction_matrix(cyl_sys, particle)
        L = TrixiParticles.extract_smatrix(vel_grad_cache, Val(TrixiParticles.ndims(cyl_sys)),
                                           particle) * L_corr'
        F_old = TrixiParticles.deformation_gradient(cyl_sys, particle)
        F_new = (I_mat + dt * L) * F_old
        for jj in 1:TrixiParticles.ndims(cyl_sys), ii in 1:TrixiParticles.ndims(cyl_sys)
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

    _cache_stress = TrixiParticles.stress_tensor_cache_for(cyl_sys)
    _use_cache = _cache_stress !== nothing
    _tensile_cache = TrixiParticles.tensile_stress_cache_tlsph(cyl_sys.tensile_stress, cyl_sys,
                                                               _use_cache, _cache_stress)

    system_coords = TrixiParticles.initial_coordinates(cyl_sys)
    nhs = TrixiParticles.get_neighborhood_search(cyl_sys, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)

    PointNeighbors.foreach_point_neighbor(
        system_coords, system_coords, nhs;
        parallelization_backend=clip_parallelization_backend,
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

function clip_update_punch_contact_force_z!(dv_ode, mold_sys, semi_local)
    # Sum contact force on the punch from ∂v/∂t * m (action–reaction after interact!).
    # Must run whenever press tonnage limiting is enabled — including the fully-active
    # system_interaction! fast path (previously skipped → force stuck at 0).
    tp_clip_max_press_force_n > 0.0 || return nothing
    dv_mold = TrixiParticles.wrap_v(dv_ode, mold_sys, semi_local)
    Fz = 0.0
    @inbounds for i in 1:size(dv_mold, 2)
        Fz += dv_mold[3, i] * mold_sys.mass[i]
    end
    punch_contact_force_z[] = Fz
    return nothing
end

function system_interaction_ramped!(dv_ode, v_ode, u_ode, semi_local,
                                    floor_activation, mold_activation)
    if floor_activation >= 1.0 - 1.0e-8 && mold_activation >= 1.0 - 1.0e-8
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi_local)
        mold_sys = clip_mold_system(semi_local)
        clip_update_punch_contact_force_z!(dv_ode, mold_sys, semi_local)
        return dv_ode
    end

    apply_charge_interact_skip_penalty_only!(dv_ode, v_ode, semi_local)

    floor_sys = clip_floor_system(semi_local)
    mold_sys = clip_mold_system(semi_local)
    bh_sys = blankholder_neighbor_system[]

    if floor_activation > 1.0e-10
        clip_foreach_charge_system(semi_local) do cyl_sys, _
            for (system, neighbor) in ((cyl_sys, floor_sys), (floor_sys, cyl_sys))
                TrixiParticles.interact!(dv_ode, v_ode, u_ode, system, neighbor, semi_local)
            end
        end
    end

    if mold_activation > 1.0e-10
        clip_foreach_charge_system(semi_local) do cyl_sys, _
            for (system, neighbor) in ((cyl_sys, mold_sys), (mold_sys, cyl_sys))
                TrixiParticles.interact!(dv_ode, v_ode, u_ode, system, neighbor, semi_local)
            end
        end
        clip_update_punch_contact_force_z!(dv_ode, mold_sys, semi_local)
    end

    if bh_sys !== nothing
        clip_foreach_charge_system(semi_local) do cyl_sys, _
            for (system, neighbor) in ((cyl_sys, bh_sys), (bh_sys, cyl_sys))
                TrixiParticles.interact!(dv_ode, v_ode, u_ode, system, neighbor, semi_local)
            end
        end
    end

    return dv_ode
end

mold_motion = if clip_use_gpu
    # Dummy motion: kinematics applied on CPU in clip_apply_mold_kinematics_cpu!.
    # Keeps velocity/acceleration buffers in the system cache for GPU layout.
    PrescribedMotion((x, t) -> x, t -> false)
else
    PrescribedMotion(
        (x, t) -> begin
            if press_force_locked[] && !retraction_started[]
                z_shift = press_force_locked_z[]
            else
                z_shift = if t <= t_compress
                    mold_z_shift_at_time(t)
                elseif retraction_started[]
                    mold_lift_z_shift_at(t)
                else
                    z_shift_motion_end
                end
                z_shift = clamp(z_shift, z_shift_motion_end, retraction_punch_extra_lift)
            end
            x + StaticArrays.SVector(0.0, 0.0, z_shift)
        end,
        t -> true)
end

# Floor/die open after peel: prescribed −z shift while charge remains frozen.
floor_motion = if retraction_peel_floor_open
    PrescribedMotion(
        (x, t) -> x + StaticArrays.SVector(0.0, 0.0, retraction_floor_open_z_shift[]),
        # Always "moving" once feature is on so NHS refreshes and shift is reapplied each step.
        t -> true)
else
    nothing
end

bound_coordinate_thermal = (3, minimum(floor_particles.coordinates[3, :]))

println("✓ Thermomechanical regime configured:")
println("  - Viscous regime: activated when T > matrix_temp_liq = ",
        fmt_temp(matrix_temp_liq))
println("  - Composite feedstock: carbon fibres in thermoplastic matrix (CF–PP organosheet)")

# ==========================================================================================
# STEP 5: Build the TLSPH systems
# ==========================================================================================
(@isdefined(matrix_nu)) || (matrix_nu = 0.3)
material_polymer = (density=density_cylinder, E=E_composite_eff, nu=matrix_nu, beta=0.000,
                    temp=thermal_softening_reference_temp,
                    temp_ref=thermal_softening_reference_temp,
                    cp=mean(cp_particle), k=mean(k_particle) * mass_scaling,
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

# Molten compression: Reimann k_n = (E_scale * E)/ps.
# Default E_scale=0.1 (L-die nocap_v4); override with TP_CLIP_COMPRESSION_CONTACT_E_SCALE.
const compression_contact_e_scale_default = 0.1
compression_contact_e_scale[] = env_float("TP_CLIP_COMPRESSION_CONTACT_E_SCALE",
                                        compression_contact_e_scale_default)
contact_penalty_alpha[] = env_float("TP_CLIP_CONTACT_PENALTY_ALPHA", 0.25)
contact_overlap_sat_frac[] = env_float("TP_CLIP_CONTACT_OVERLAP_SAT_FRAC", 0.5)
_legacy_contact_cap = env_float("TP_CLIP_CONTACT_TOTAL_ACCEL_CAP", 1.0e10)
contact_total_accel_cap[] = _legacy_contact_cap
contact_accel_cap_solid[] = env_float("TP_CLIP_CONTACT_ACCEL_CAP_SOLID", 5.0e4)
contact_accel_cap_molten[] = env_float("TP_CLIP_CONTACT_ACCEL_CAP_MOLENT", 1.0e10)
melt_contact_softness[] = env_float("TP_CLIP_MELT_CONTACT_SOFTNESS", 0.15)
# Deep contact |a| caps: OFF by default for compress/hold/retraction (nocap baseline).
# Opt into contact-only deep caps via TP_CLIP_CONTACT_DEEP_ACCEL_CAP > 0.
if sim_phase == "retraction"
    contact_deep_overlap_frac[] = env_float("TP_CLIP_CONTACT_DEEP_OVERLAP_FRAC", 0.25)
    contact_deep_accel_cap[] = env_float("TP_CLIP_CONTACT_DEEP_ACCEL_CAP", 0.0)
    contact_dashpot_zeta[] = env_float("TP_CLIP_CONTACT_DASHPOT_ZETA", 0.05)
    contact_dashpot_zeta_deep[] = env_float("TP_CLIP_CONTACT_DASHPOT_ZETA_DEEP", 0.0)
else
    contact_deep_overlap_frac[] = env_float("TP_CLIP_CONTACT_DEEP_OVERLAP_FRAC", 0.5)
    contact_deep_accel_cap[] = env_float("TP_CLIP_CONTACT_DEEP_ACCEL_CAP", 0.0)
    contact_dashpot_zeta[] = env_float("TP_CLIP_CONTACT_DASHPOT_ZETA", 0.05)
    contact_dashpot_zeta_deep[] = env_float("TP_CLIP_CONTACT_DASHPOT_ZETA_DEEP", 0.0)
end
println("  - compression Reimann contact E scale = ",
        round(compression_contact_e_scale[], digits=4),
        " (driver default = ",
        round(compression_contact_e_scale_default, digits=4), ")")
println("  - molten wall drag factor = ", molten_wall_drag_factor)
println("  - contact penalty: α=", contact_penalty_alpha[],
        " ps deadband, overlap sat=", contact_overlap_sat_frac[], " ps")
println("  - contact dashpot ζ=", contact_dashpot_zeta[],
        contact_dashpot_zeta_deep[] > 0.0 ?
            (" (deep ζ=" * string(contact_dashpot_zeta_deep[]) * ")") : "")
println("  - melt-aware contact: k_n scale (molten)=", melt_contact_softness[],
        ", |a|_cap solid=", contact_accel_cap_solid[],
        " molten=", contact_accel_cap_molten[], " m/s²")
if contact_deep_accel_cap[] > 0.0
    println("  - deep-overlap contact |a|_cap=", contact_deep_accel_cap[],
            " m/s² when δ>", contact_deep_overlap_frac[], "·ps",
            sim_phase == "retraction" ? " (retraction contact-only)" : " (0=off)")
else
    println("  - deep-overlap contact |a|_cap=off",
            sim_phase == "retraction" ? "" : " (compress/hold default)")
end
sim_phase == "retraction" && retraction_internal_accel_cap > 0.0 &&
    println("  - internal (pre-contact) |a|_cap=", retraction_internal_accel_cap,
            " m/s² (soft tanh; contact uncapped)")
sim_phase == "retraction" && retraction_explicit_kick_soft_limit &&
    println("  - deep-contact HARD clamps: |a|≤", retraction_deep_contact_accel_cap,
            " m/s², |v|≤", retraction_deep_contact_speed_cap,
            " m/s when gap<", retraction_deep_contact_kick_gap_ps, "·ps")
if sim_phase == "retraction" &&
   (retraction_rayleigh_damp > 0.0 || retraction_rayleigh_damp_contact > 0.0 ||
    retraction_viscoplastic_accel_yield > 0.0)
    println("  - demold visco/Rayleigh: γ=", retraction_rayleigh_damp,
            " 1/s, γ_contact=", retraction_rayleigh_damp_contact,
            " 1/s, viscoplastic a_y=", retraction_viscoplastic_accel_yield,
            " m/s² (gap<", retraction_viscoplastic_gap_ps, "·ps)")
end
sim_phase == "retraction" && retraction_eject_enable &&
    println("  - eject after cool=", retraction_eject_after_cool,
            "  during eject: floor=",
            retraction_keep_floor_during_eject ? "ON" : "OFF",
            "  after eject: floor=",
            retraction_keep_floor_after_eject ? "ON" : "OFF",
            "  g=", retraction_gravity_after_eject,
            " m/s²  extra Rayleigh γ=", retraction_rayleigh_after_eject, " 1/s")
_clip_max_liquid = env_float("TP_CLIP_MAX_LIQUID_ACCEL", 2.5e5)
max_liquid_accel = max(_clip_max_liquid, contact_accel_cap_molten[])
println("  - max liquid mechanics |a| = ", max_liquid_accel,
        " m/s² (≥ molten contact cap)")
sim_phase == "retraction" && !retraction_contact_e_explicit && !retraction_contact_e_ramp &&
    println("  - retraction active contact E scale = ",
            round(active_contact_e_scale(), digits=4),
            " (inherits compression/hold; α=", contact_penalty_alpha[],
            ", overlap sat=", contact_overlap_sat_frac[], " ps)")

import PointNeighbors: DictionaryCellList

const tool_smoothing_h = use_stl_geometry ? tool_smoothing_length : smoothing_length
bh_particles, blankholder_system, charge_resting_on_shelf =
    clip_setup_blankholder(polymer, particle_spacing, floor_density, tool_smoothing_h,
                           smoothing_kernel, E_boundary, material_polymer)
blankholder_neighbor_system[] = blankholder_system
charge_resting_on_shelf_buf[] = charge_resting_on_shelf
n_bh_particles = blankholder_system === nothing ? 0 : nparticles(bh_particles)
apply_resting_shelf_z_constraint &&
    println("  - resting-shelf z spring-damper: ON (",
            count(charge_resting_on_shelf), " / ", nparticles(polymer),
            " charge particles; replaces blankholder hold-down on flange)")
const _clip_n_tool_particles = nparticles(floor_particles) + nparticles(mold_particles) +
    n_bh_particles
_clip_n_tool_particles > 2000 &&
    println("  - NHS max_points_per_cell = ",
            clip_nhs_max_points_per_cell(_clip_n_tool_particles),
            " (tool particles = ", _clip_n_tool_particles, ")")
nhs_template = clip_charge_nhs_template(;
    max_neighbors=use_stl_geometry ? max(300, env_int("TP_CLIP_NHS_MAX_NEIGHBORS", 400)) : 200,
    n_tool=_clip_n_tool_particles)

floor_system = TotalLagrangianSPHSystem(floor_particles,
                                        smoothing_kernel,
                                        tool_smoothing_h,
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
                                        clamped_particles_motion=floor_motion,
                                        acceleration=(0.0, 0.0, 0.0),
                                        self_interaction_nhs=nothing)

mold_system = TotalLagrangianSPHSystem(mold_particles,
                                       smoothing_kernel,
                                       tool_smoothing_h,
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
                                       self_interaction_nhs=nothing)

function build_charge_tlsph_system(initial_condition; self_interaction_nhs=nhs_template)
    return TotalLagrangianSPHSystem(initial_condition,
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
                                    penalty_force=let a = env_float("TP_CLIP_PENALTY_ALPHA", 0.05)
                                        a > 0.0 ? PenaltyForceGanzenmueller(alpha=a) : nothing
                                    end,
                                    viscosity=ArtificialViscosityMonaghan(alpha=0.5,
                                                                            beta=4.0),
                                    tensile_stress=clip_tensile_stress,
                                    self_interaction_nhs=self_interaction_nhs)
end

charge_systems = if separate_charge_plies
    map(charge_initial_conditions) do ply
        ply_nhs = clip_charge_nhs_template(;
            max_neighbors=use_stl_geometry ?
                          max(300, env_int("TP_CLIP_NHS_MAX_NEIGHBORS", 400)) : 200,
            n_tool=_clip_n_tool_particles)
        build_charge_tlsph_system(ply; self_interaction_nhs=ply_nhs)
    end
else
    [build_charge_tlsph_system(polymer)]
end
cylinder_system = first(charge_systems) # Legacy alias; never denotes all plies.
n_charge_systems = length(charge_systems)
println("  - Updated Lagrangian re-referencing: ", clip_updated_lagrange ? "ON (interval=$(clip_updated_lagrange_interval) s)" : "OFF")
charge_system_indices = 1:n_charge_systems
floor_system_index = n_charge_systems + 1
mold_system_index = n_charge_systems + 2
blankholder_system_index = blankholder_system === nothing ? nothing : n_charge_systems + 3

@inline function clip_charge_systems(semi_local)
    return ntuple(i -> semi_local.systems[charge_system_indices[i]],
                  length(charge_system_indices))
end
@inline clip_floor_system(semi_local) = semi_local.systems[floor_system_index]
@inline clip_mold_system(semi_local) = semi_local.systems[mold_system_index]
@inline clip_is_charge_system_index(index::Integer) = index in charge_system_indices

function clip_register_charge_source_maps!(semi_local)
    systems = collect(clip_charge_systems(semi_local))
    source_map = clip_build_charge_source_map(systems, charge_ply_source_indices,
                                              n_cylinder_particles)
    contact_charge_source_indices[] = source_map
    return source_map
end

function clip_foreach_charge_system(f, semi_local)
    source_map = contact_charge_source_indices[]
    source_map === nothing &&
        error("charge source maps have not been registered")
    for system in clip_charge_systems(semi_local)
        indices = get(source_map, objectid(system), nothing)
        indices === nothing &&
            error("encountered an unregistered charge system")
        f(system, indices)
    end
    return nothing
end

@inline function clip_deformation_only_selector(semi_local)
    separate_charge_plies ||
        return objectid(first(clip_charge_systems(semi_local)))
    return Set(objectid(system) for system in clip_charge_systems(semi_local))
end

@inline function clip_charge_packed_index(source, local_particle::Integer)
    return source[local_particle]
end

function clip_pack_charge_deformation_grad(semi_local)
    packed = zeros(Float64, 3, 3, n_cylinder_particles)
    clip_foreach_charge_system(semi_local) do system, source
        clip_charge_particle_scatter!(packed, source, system.deformation_grad)
    end
    return packed
end

function clip_pack_charge_temp(semi_local)
    packed = zeros(Float64, n_cylinder_particles)
    clip_foreach_charge_system(semi_local) do system, source
        clip_charge_particle_scatter!(packed, source, system.temp)
    end
    return packed
end

function clip_unpack_charge_deformation_grad!(semi_local, packed)
    clip_foreach_charge_system(semi_local) do system, source
        system.deformation_grad .= clip_charge_particle_view(packed, source)
    end
    return nothing
end

function clip_unpack_charge_temp!(semi_local, packed)
    clip_foreach_charge_system(semi_local) do system, source
        system.temp .= clip_charge_particle_view(packed, source)
    end
    return nothing
end

function clip_preheat_charge_systems!(semi_local, charge_temp, tool_temp;
                                      punch_temp=tool_temp, die_temp=tool_temp,
                                      blankholder_temp=tool_temp)
    # floor = lower die (female); mold = upper punch (male). Optional split temps
    # match Haanappel ECCM14 / thesis Ch.2 (preheated steel male + cold female).
    clip_foreach_charge_system(semi_local) do system, _
        system.temp .= charge_temp
    end
    clip_floor_system(semi_local).temp .= die_temp
    clip_mold_system(semi_local).temp .= punch_temp
    bh = blankholder_system
    bh !== nothing && (bh.temp .= blankholder_temp)
    return nothing
end

# ==========================================================================================
# STEP 6: Semidiscretization and solve
# ==========================================================================================
mold_nhs_coords = blankholder_system === nothing ?
    mold_particles.coordinates :
    hcat(mold_particles.coordinates, bh_particles.coordinates)
semi = if !separate_charge_plies && blankholder_system === nothing
    Semidiscretization(cylinder_system, floor_system, mold_system;
        neighborhood_search=clip_build_neighborhood_search(
            smoothing_length;
            polymer_coords=polymer.coordinates,
            floor_coords=floor_particles.coordinates,
            mold_coords=mold_nhs_coords),
        parallelization_backend=clip_parallelization_backend)
elseif !separate_charge_plies
    Semidiscretization(cylinder_system, floor_system, mold_system, blankholder_system;
        neighborhood_search=clip_build_neighborhood_search(
            smoothing_length;
            polymer_coords=polymer.coordinates,
            floor_coords=floor_particles.coordinates,
            mold_coords=mold_nhs_coords),
        parallelization_backend=clip_parallelization_backend)
else
    ordered_systems = blankholder_system === nothing ?
                      (charge_systems..., floor_system, mold_system) :
                      (charge_systems..., floor_system, mold_system, blankholder_system)
    Semidiscretization(ordered_systems...;
        neighborhood_search=clip_build_neighborhood_search(
            smoothing_length;
            polymer_coords=polymer.coordinates,
            floor_coords=floor_particles.coordinates,
            mold_coords=mold_nhs_coords),
        parallelization_backend=clip_parallelization_backend)
end

charge_source_map = clip_register_charge_source_maps!(semi)
if blankholder_system !== nothing
    # NHS lookup is by === on semi.systems. Re-bind after Semidiscretization
    # so gripper contact does not throw "system is not in the semidiscretization".
    bh_idx = findfirst(s -> s === blankholder_system, semi.systems)
    blankholder_neighbor_system[] = bh_idx === nothing ? semi.systems[end] :
        semi.systems[bh_idx]
end
contact_ply_interface_rest_separation[] =
    env_float("TP_CLIP_PLY_INTERFACE_REST_SEPARATION_MM",
              particle_spacing_z * 1e3) * 1e-3
# Ellipsoidal tool–charge rest (default):
#   rest_xy = Δxy so the punch cannot slip between charge points
#   rest_z  = sheet + ½ tool Δz on thin packs (n_z ≤ 8), else ½(Δz_c+Δz_t) (bulk ≈ sphere)
# Ply–ply rest stays spherical Δz.
# Overrides: TP_CLIP_TOOL_CHARGE_REST_XY_MM / _Z_MM.
# Legacy TP_CLIP_TOOL_CHARGE_REST_SEPARATION_MM without XY/Z is xy only, not isotropic.
_tool_ps_z_contact = if haskey(ENV, "TP_CLIP_TOOL_PS_Z_MM")
    env_float("TP_CLIP_TOOL_PS_Z_MM", tool_particle_spacing * 1e3) * 1e-3
else
    Float64(tool_particle_spacing)
end
_n_z_layers = max(1, round(Int, charge_thickness_discrete /
                               max(particle_spacing_z, eps(Float64))))
_rest_xy_default_mm = particle_spacing * 1e3
_rest_z_default_mm = if _n_z_layers <= 8
    (charge_thickness_discrete + 0.5 * _tool_ps_z_contact) * 1e3
else
    0.5 * (particle_spacing_z + _tool_ps_z_contact) * 1e3
end
_legacy_tool_rest_mm = haskey(ENV, "TP_CLIP_TOOL_CHARGE_REST_SEPARATION_MM") &&
                       !haskey(ENV, "TP_CLIP_TOOL_CHARGE_REST_XY_MM") ?
    env_float("TP_CLIP_TOOL_CHARGE_REST_SEPARATION_MM", _rest_xy_default_mm) :
    _rest_xy_default_mm
contact_tool_charge_rest_xy[] =
    env_float("TP_CLIP_TOOL_CHARGE_REST_XY_MM", _legacy_tool_rest_mm) * 1e-3
contact_tool_charge_rest_z[] =
    env_float("TP_CLIP_TOOL_CHARGE_REST_Z_MM", _rest_z_default_mm) * 1e-3
contact_tool_charge_rest_separation[] = contact_tool_charge_rest_xy[]
_legacy_grip_rest_mm = haskey(ENV, "TP_CLIP_GRIPPER_REST_SEPARATION_MM") &&
                       !haskey(ENV, "TP_CLIP_GRIPPER_REST_XY_MM") ?
    env_float("TP_CLIP_GRIPPER_REST_SEPARATION_MM",
              contact_tool_charge_rest_xy[] * 1e3) :
    contact_tool_charge_rest_xy[] * 1e3
contact_gripper_rest_xy[] =
    env_float("TP_CLIP_GRIPPER_REST_XY_MM", _legacy_grip_rest_mm) * 1e-3
contact_gripper_rest_z[] =
    env_float("TP_CLIP_GRIPPER_REST_Z_MM",
              contact_tool_charge_rest_z[] * 1e3) * 1e-3
contact_gripper_rest_separation[] = contact_gripper_rest_xy[]
contact_search_radius_h_mult[] = env_float("TP_CLIP_CONTACT_RADIUS", 0.0)
_rest_kind = _n_z_layers <= 8 ? "organosheet (n_z≤8)" : "bulk (n_z>8)"
println(">>> Tool–charge contact rest ellipsoid: xy = ",
        round(contact_tool_charge_rest_xy[] * 1e3; digits=3),
        " mm, z = ",
        round(contact_tool_charge_rest_z[] * 1e3; digits=3),
        " mm (", _rest_kind, ", n_z=", _n_z_layers, "; punch/die)")
println(">>> Gripper rest ellipsoid: xy = ",
        round(contact_gripper_rest_xy[] * 1e3; digits=3),
        " mm, z = ",
        round(contact_gripper_rest_z[] * 1e3; digits=3),
        " mm")
if contact_search_radius_h_mult[] > 0.0
    println(">>> Contact search radius = ", contact_search_radius_h_mult[],
            " h  (Jacobian radius = ", clip_jac_radius_mult, " h)")
end
flush(stdout)
if separate_charge_plies &&
   contact_ply_interface_rest_separation[] > 1.5 * particle_spacing
    error("TP_CLIP_PLY_INTERFACE_REST_SEPARATION_MM=" *
          "$(contact_ply_interface_rest_separation[] * 1e3) exceeds the " *
          "cross-system Reimann search radius $(1.5 * particle_spacing * 1e3) mm; " *
          "increase the physics particle spacing/search radius or use a compatible " *
          "through-thickness packing")
end

if separate_charge_plies
    println(">>> Separate-ply hold: ", n_charge_systems, " charge TLSPH systems; ",
            "floor index=", floor_system_index, " punch index=", mold_system_index,
            " (retraction/full still unsupported)")
    flush(stdout)
end

function initialize_charge_boundary_exposure!(exposure_vec, semi_local)
    cyl_sys = semi_local.systems[1]
    n_part = nparticles(cyl_sys)
    resize!(exposure_vec, n_part)
    fill!(exposure_vec, 0.0)
    
    if !use_ambient_heat_loss
        return nothing
    end
    
    initial_coords = TrixiParticles.initial_coordinates(cyl_sys)
    rho = cyl_sys.material_density[1]
    
    W_sum = zeros(n_part)
    TrixiParticles.foreach_point_neighbor(cyl_sys, cyl_sys, initial_coords, initial_coords, semi_local) do particle, neighbor, r_nb, _
        vol_nb = cyl_sys.mass[neighbor] / rho
        r_scalar = sqrt(TrixiParticles.dot(r_nb, r_nb))
        W_sum[particle] += vol_nb * TrixiParticles.smoothing_kernel(cyl_sys, r_scalar, particle)
    end
    
    W_max = maximum(W_sum)
    if W_max > 0.0
        for i in 1:n_part
            exposure_vec[i] = max(0.0, 1.0 - W_sum[i] / W_max)
        end
    end
    
    println(">>> Boundary exposure factor initialized for charge:")
    println("    - max exposure = ", round(maximum(exposure_vec); digits=4))
    println("    - mean exposure = ", round(mean(exposure_vec); digits=4))
    println("    - count of exposed particles (exposure > 0.01) = ", count(x -> x > 0.01, exposure_vec))
    flush(stdout)
    return nothing
end

initialize_charge_boundary_exposure!(charge_boundary_exposure, semi)

# Preheat temperatures are specified in °C and stored as K for thermal laws.
# TP_CLIP_TOOL_TEMP remains the common default. Optional split:
#   TP_CLIP_PUNCH_TEMP → upper punch / male (mold_system)
#   TP_CLIP_DIE_TEMP   → lower die / female (floor_system)
#   TP_CLIP_BLANKHOLDER_TEMP → grippers (default = punch temp)
preheating_target_temp = celsius_to_kelvin(env_float("TP_CLIP_CHARGE_TEMP", 453.0))
_tool_temp_C = env_float("TP_CLIP_TOOL_TEMP", 320.0)
tool_preheat_temp = celsius_to_kelvin(_tool_temp_C)
_punch_temp_C = haskey(ENV, "TP_CLIP_PUNCH_TEMP") ?
    env_float("TP_CLIP_PUNCH_TEMP", _tool_temp_C) : _tool_temp_C
_die_temp_C = haskey(ENV, "TP_CLIP_DIE_TEMP") ?
    env_float("TP_CLIP_DIE_TEMP", _tool_temp_C) : _tool_temp_C
_bh_temp_C = haskey(ENV, "TP_CLIP_BLANKHOLDER_TEMP") ?
    env_float("TP_CLIP_BLANKHOLDER_TEMP", _punch_temp_C) : _punch_temp_C
punch_preheat_temp = celsius_to_kelvin(_punch_temp_C)
die_preheat_temp = celsius_to_kelvin(_die_temp_C)
blankholder_preheat_temp = celsius_to_kelvin(_bh_temp_C)

println("\n=== PREHEATING PHASE ===")
clip_preheat_charge_systems!(semi, preheating_target_temp, tool_preheat_temp;
                             punch_temp=punch_preheat_temp,
                             die_temp=die_preheat_temp,
                             blankholder_temp=blankholder_preheat_temp)
println("✓ Cylinder preheated: T = ", fmt_temp(preheating_target_temp))
println("✓ Floor (die/female) preheated:  T = ", fmt_temp(die_preheat_temp))
println("✓ Mold  (punch/male) preheated:  T = ", fmt_temp(punch_preheat_temp))
if _punch_temp_C != _die_temp_C
    println("  - split tool temps ON (TP_CLIP_PUNCH_TEMP=", _punch_temp_C,
            " °C, TP_CLIP_DIE_TEMP=", _die_temp_C, " °C)")
end
blankholder_system !== nothing &&
    println("✓ Blankholder preheated: T = ", fmt_temp(blankholder_preheat_temp),
            " (kinematically fixed)")
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
    # Include optional pre-lift settle + floor-settle + contact-settle σ ramp past nominal creep.
    t_ret1 = t_ret0 + retraction_pre_lift_creep_s + retraction_pre_lift_settle_s +
        retraction_creep_floor_settle_max_s +
        retraction_contact_settle_sigma_ramp_s +
        max(retraction_duration_max,
            retraction_post_lift_kin_freeze_s,
            retraction_kinematic_peel ? retraction_peel_max_s : 0.0) +
        (retraction_kinematic_peel ? retraction_peel_sigma_ramp_s : 0.0) +
        (retraction_peel_floor_open ? retraction_peel_floor_open_s : 0.0) +
        (retraction_eject_enable ? retraction_eject_s : 0.0) +
        retraction_springback_dwell_s
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
    hold_probe_s = env_float("TP_CLIP_HOLD_PROBE_S", 0.0)
    if hold_probe_s > 0.0
        tspan = (0.0, min(t_total, hold_probe_s))
        println(">>> Hold PROBE: tspan capped to ", hold_probe_s,
                " s for solver validation/benchmark")
    end
    println(">>> Hold-only run: will save checkpoint and stop before retraction")
    flush(stdout)
else
    tspan = (0.0, t_total)
end

if sim_phase in ("full", "hold") && env_int("TP_CLIP_RESTORE_CHECKPOINT", 0) != 0
    isfile(checkpoint_path) ||
        error("TP_CLIP_RESTORE_CHECKPOINT=1 but checkpoint not found: $(abspath(checkpoint_path))")
    hold_checkpoint_restore = load_hold_checkpoint(checkpoint_path)
    meta_now = checkpoint_meta(; particle_spacing=particle_spacing,
                               n_charge=n_cylinder_particles,
                               n_floor=nparticles(floor_particles),
                               n_mold=nparticles(mold_particles),
                               t_compress=t_compress,
                               z_shift_down_end=z_shift_down_end,
                               mold_velocity_state=mold_velocity_state,
                               compression_thickness_ratio=compression_thickness_ratio)
    validate_checkpoint_meta!(hold_checkpoint_restore, meta_now; motion_from_checkpoint=(env_int("TP_CLIP_PROFILE_FROM_CHECKPOINT", 1) != 0))
    t_ck = hold_checkpoint_restore["t"]
    tspan = (t_ck, t_total)
    # Skip leftover punch travel and enter closed-die hold at the restored time.
    # Compute freeze_z on the original downstroke timeline before shortening t_down_end.
    if env_int("TP_CLIP_ENTER_HOLD_NOW", 0) != 0
        freeze_z = mold_z_shift_at_time(t_ck)
        t_compress = t_ck
        t_down_end = t_ck
        z_shift_down_end = freeze_z
        z_shift_motion_end = freeze_z
        retraction_duration_max = (abs(z_shift_down_end) + retraction_punch_extra_lift) /
                                  abs(mold_velocity_state) +
                                  0.5 * env_float("TP_CLIP_RETRACTION_RAMP_S",
                                                  env_float("TP_CLIP_RETRACTION_KINEMATICS_RELEASE_S", 0.05))
        t_total = t_compress + solidification_hold_budget + post_solidification_dwell +
              hold_tlsph_tail_s + retraction_duration_max +
              (retraction_eject_enable ? retraction_eject_s : 0.0) +
              retraction_springback_dwell_s
        tspan = (t_ck, t_total)
        println(">>> ENTER HOLD NOW: punch frozen at t=", round(t_ck; digits=6),
                " s, z_shift=", round(freeze_z * 1e3; digits=3),
                " mm (remaining downstroke skipped; kinematics frozen)")
        flush(stdout)
    end
    println(">>> RESTORE from checkpoint: t0=", round(t_ck; digits=6),
            " s, tspan=", tspan)
    flush(stdout)
end

# VTU save schedule (compress/hold/full). Retraction-only uses fixed dt instead.
const compress_hold_vtu_save_times = if sim_phase == "retraction"
    Float64[]
else
    times = filter(t -> t <= last(tspan) + 1.0e-12,
                   build_compress_hold_vtu_save_times(t_total))
    if last(tspan) < t_total - 1.0e-12
        push!(times, last(tspan))
        sort!(unique!(times))
    end
    times
end
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

# ==========================================================================================
# KERNEL CORRECTION MATRIX
# Coplanar pseudo-2D particles require inversion of only the active x-z block. Resolved
# full-3D particles must retain all matrix components and use the full 3x3 inverse.
# Must run on CPU before semidiscretize (GPU transfer happens in semidiscretize).
# ==========================================================================================
println(">>> Applying geometry-adapted kernel correction matrix...")
TrixiParticles.foreach_system(semi) do system
    TrixiParticles.set_zero!(system.correction_matrix)
    initial_coords = TrixiParticles.initial_coordinates(system)

    TrixiParticles.foreach_point_neighbor(system, system, initial_coords, initial_coords,
                                          semi;
                                          parallelization_backend=PolyesterBackend()) do particle, neighbor, pos_diff, distance
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
        L_inv = clip_invert_correction_matrix(L)
        for j in 1:3, i in 1:3
            system.correction_matrix[i, j, particle] = L_inv[i, j]
        end
    end
end
println(">>> Geometry-adapted kernel correction applied.")
# ==========================================================================================

ode_base = clip_gpu_allowscalar(() -> semidiscretize(semi, tspan))
semi = ode_base.p
if clip_use_gpu
    clip_gpu_allowscalar() do
        clip_apply_mold_kinematics_cpu!(clip_mold_system(semi), tspan[1])
    end
end

v_stress_buf          = clip_particle_buffer(Float64, 3, 3, n_cylinder_particles)
v_stress_elastic_buf  = clip_particle_buffer(Float64, 3, 3, n_cylinder_particles)
v_stress_viscous_buf  = clip_particle_buffer(Float64, 3, 3, n_cylinder_particles)
vel_grad_buf          = clip_particle_buffer(Float64, 3, 3, n_cylinder_particles)
wcsph_visc_lagged_dv_buf = clip_particle_buffer(Float64, 3, n_cylinder_particles)
charge_dv_mech_marker_buf = clip_particle_buffer(Float64, 3, n_cylinder_particles)
wcsph_visc_scratch_dv_ode = clip_ode_vector(length(ode_base.u0.x[1]))
wcsph_liquid_density_ref_buf = clip_particle_buffer(Float64, n_cylinder_particles)
wcsph_liquid_density_curr_buf = clip_particle_buffer(Float64, n_cylinder_particles)
wcsph_pressure_buf = clip_particle_buffer(Float64, n_cylinder_particles)
wcsph_rho0_buf = clip_particle_buffer(Float64, n_cylinder_particles)
wcsph_latent_weight_buf = clip_particle_buffer(Float64, n_cylinder_particles)
wcsph_L_curr_buf = wcsph_full3d_pressure ?
    clip_particle_buffer(Float64, 3, 3, n_cylinder_particles) :
    clip_particle_buffer(Float64, 2, 2, n_cylinder_particles)
wcsph_tau_vis_buf = wcsph_full3d_pressure ?
    clip_particle_buffer(Float64, 3, 3, n_cylinder_particles) :
    clip_particle_buffer(Float64, 2, 2, n_cylinder_particles)
# Use systems from `semi` (TLSPH copies inside Semidiscretization differ from local constructors).
const wcsph_nhs_search_radius = TrixiParticles.compact_support(semi.systems[1], semi.systems[1])
const wcsph_current_nhs = clip_use_gpu ? nothing :
    clip_build_wcsph_neighborhood_search(wcsph_nhs_search_radius, semi)
const wcsph_liquid_density_ref_initialized = Ref(false)

nhs_updated_at_t = Ref(-Inf)
contact_diag_interval = 5000

enable_contact_diag = false
contact_heat_flux_buf = clip_particle_buffer(Float64, n_cylinder_particles)
clip_thermal_dv_cache = clip_particle_buffer(Float64, n_cylinder_particles)
clip_contact_dv_inc_cache = clip_particle_buffer(Float64, 3, n_cylinder_particles)
clip_contact_dv_pre_snap = clip_particle_buffer(Float64, 3, n_cylinder_particles)
ys_particle_buf = clip_particle_buffer(Float64, n_cylinder_particles)
hard_particle_buf = clip_particle_buffer(Float64, n_cylinder_particles)
vis_particle_buf = clip_particle_buffer(Float64, n_cylinder_particles)
thermal_softening_particle_buf = clip_particle_buffer(Float64, n_cylinder_particles)
liquid_fraction_particle_buf = clip_particle_buffer(Float64, n_cylinder_particles)
solid_fraction_particle_buf  = clip_particle_buffer(Float64, n_cylinder_particles)
# J_ref: volumetric reference Jacobian set to det(F) while a particle is liquid.
# At re-solidification J_ref = J_at_solidification, so K·log(J_e/J_ref) = K·log(1) = 0,
# giving zero volumetric elastic stress at the moment of re-solidification and
# eliminating the sudden 1–2 GPa pressure spike that collapses TRBDF2 Δt.
J_ref_particle_buf = clip_use_gpu ? CUDA.ones(Float64, n_cylinder_particles) :
                                    ones(n_cylinder_particles)
# Volumetric plastic Jacobian J_p (Strategy E). Stress-free bulk: J_e_vol = det(F)/J_p = 1.
J_p_particle_buf = clip_use_gpu ? CUDA.ones(Float64, n_cylinder_particles) :
                                  ones(n_cylinder_particles)
thermal_Fth_T_prev = clip_particle_buffer(Float64, n_cylinder_particles)
# Nakamura crystallinity state: α_c ∈ [0, 1].  0 = fully liquid, 1 = fully crystallized.
# Initialised to 0 because the charge starts fully molten at ~430 K > T_melt.
crystallinity_particle_buf = clip_particle_buffer(Float64, n_cylinder_particles)
# The charge is preheated above matrix_tmelt, so initialize phase buffers as molten.
# The Nakamura seed is applied only when cooling enters the crystallization range.
fill!(liquid_fraction_particle_buf, 1.0)
fill!(solid_fraction_particle_buf, 0.0)
fill!(thermal_softening_particle_buf, 0.0)
contact_liquid_fraction_buf[] = liquid_fraction_particle_buf
clip_host_assign!(vis_particle_buf, viscosity_base_particle)
orientation_tensor_state = Ref(zeros(3, 3, n_cylinder_particles))
solidification_hold_announced = Ref(false)
# When restoring a solidified hold checkpoint with TP_CLIP_HOLD_UNTIL_TEMP_C, keep the
# die closed and continue thermal cool until mean charge T reaches the target.
hold_cool_continue_announced = Ref(false)
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
const clip_integrator_dt = Ref(dt_cap)

@inline function clip_sync_trial_dt_state!()
    trial_dt_state[] = clip_integrator_dt[]
    return nothing
end

function clip_reduced_trbdf2_refresh_fsalfirst!(integrator)
    clip_reduced_trbdf2_active() || return nothing
    fs = integrator.fsalfirst
    clip_reduced_trbdf2_fsalfirst[] = fs
    abs(integrator.t - clip_reduced_prev_t[]) <= 1.0e-14 * max(1.0, abs(integrator.t)) ||
        return nothing
    integrator.f(fs, integrator.u, integrator.p, integrator.t)
    return nothing
end

function clip_sync_trial_dt_from_integrator!(integrator)
    clip_integrator_dt[] = integrator.dt
    trial_dt_state[] = integrator.dt
    if clip_reduced_trbdf2_active()
        clip_reduced_trbdf2_refresh_fsalfirst!(integrator)
    end
    return nothing
end

rhs_time_pos_ns = Ref(0)
rhs_time_nhs_ns = Ref(0)
rhs_time_quant_ns = Ref(0)
rhs_time_implicit_ns = Ref(0)
rhs_time_pressure_ns = Ref(0)
rhs_time_boundary_ns = Ref(0)
rhs_time_final_ns = Ref(0)
rhs_time_stress_interact_ns = Ref(0)
rhs_time_thermal_ns = Ref(0)

# Cache tool pos/quant updates within repeated RHS evaluations at the same time `t`.
# During implicit Newton/Jacobian fills we call RHS many times with identical `t`,
# while tool kinematics depend only on `t` (charge depends on trial `u`).
const clip_tool_pos_quant_cache = env_int("TP_CLIP_TOOL_POS_QUANT_CACHE", 0) != 0
const clip_tool_pos_updated_at_t = Ref(-Inf)
const clip_tool_quant_updated_at_t = Ref(-Inf)

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
const formulation_profile_time_dt = env_float("TP_CLIP_FORMULATION_PROFILE_TIME_DT", 0.0)
const formulation_profile_path = get(ENV, "TP_CLIP_FORMULATION_PROFILE_LOG",
                                     joinpath(clip_out_dir, "formulation_profile.log"))
const formulation_profile_step_count = Ref(0)
const rhs_eval_count = Ref(0)
const rhs_heartbeat_interval = max(0, env_int("TP_CLIP_RHS_HEARTBEAT", 0))
const formulation_profile_last_t = Ref(0.0)
const formulation_profile_next_t = Ref(formulation_profile_time_dt > 0.0 ? formulation_profile_time_dt : Inf)
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
        "Note: implicit Jacobian + linear solves not included.",
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
        "Note: implicit Jacobian + linear solves are NOT in RHS timers (dominant during micro-dt).",
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
const stall_diag_enabled = env_int("TP_CLIP_STALL_DIAG", 0) != 0
const stiff_diag_enabled = env_int("TP_CLIP_STIFF_DIAG", stall_diag_enabled ? 1 : 0) != 0
const force_stage_diag_enabled = env_int("TP_CLIP_FORCE_STAGE_DIAG", stall_diag_enabled ? 1 : 0) != 0
const solver_diag_enabled = env_int("TP_CLIP_SOLVER_DIAG", stall_diag_enabled ? 1 : 0) != 0
const stiff_diag_start_t = env_float("TP_CLIP_STIFF_DIAG_START_T", 0.18)
const solver_diag_dt_threshold = env_float("TP_CLIP_SOLVER_DIAG_DT", 1.0e-7)
const diag_stiff_last_t = Ref(-1.0)   # sim time of last diagnostic print
const diag_force_stage_last_t = Ref(-1.0)
const diag_force_stage_start_t = env_float("TP_CLIP_FORCE_STAGE_START_T", 0.18)
const diag_force_stage_dt = env_float("TP_CLIP_FORCE_STAGE_DT", 1.0e-4)

if stall_diag_enabled
    println(">>> Stall diagnostics ON: stiff=", stiff_diag_enabled,
            " (t≥", stiff_diag_start_t, " s), force-stage=", force_stage_diag_enabled,
            ", solver-dt<", solver_diag_dt_threshold, " s")
    formulation_profile_enabled || @warn "TP_CLIP_STALL_DIAG=1 without TP_CLIP_FORMULATION_PROFILE=1: " *
        "enable formulation profile to see RHS timing breakdown."
    flush(stdout)
end

# function contact_condition(cyl_sys, floor_sys, mold_sys, i, contact_dist)
#     z_i         = cyl_sys.current_coordinates[3, i]
#     z_floor_top = maximum(floor_sys.current_coordinates[3, :])
#     z_mold_bot  = minimum(mold_sys.current_coordinates[3, :])
#     return (z_i - z_floor_top) <= contact_dist || (z_mold_bot - z_i) <= contact_dist
# end

@inline soft_limit(x, lim) = lim * tanh(x / max(lim, eps(Float64)))
@inline hard_limit(x, lim) = clamp(x, -lim, lim)

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

function charge_y_repair_value(cyl_sys, particle::Integer)
    y0 = TrixiParticles.initial_coordinates(cyl_sys)
    return y0[2, particle]
end

function clamp_charge_y_coordinate(y, cyl_sys, particle::Integer)
    if !isfinite(y)
        return charge_y_repair_value(cyl_sys, particle)
    end
    return clamp(y, y_safety_min, y_safety_max)
end

function apply_retraction_full3d_y_accel_cap!(dv_cyl, cyl_sys, t)
    retraction_full3d_y_cap_active() || return nothing
    accel_cap = retraction_charge_accel_cap(t)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        dv_cyl[2, particle] = soft_limit(dv_cyl[2, particle], accel_cap)
    end
    return nothing
end

function apply_retraction_full3d_y_velocity_cap!(v_cyl, cyl_sys)
    retraction_full3d_y_cap_active() || return nothing
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        v_cyl[2, particle] = soft_limit(v_cyl[2, particle], max_cylinder_speed)
    end
    return nothing
end

@inline function retraction_explicit_kick_cap(t)
    if retraction_accel_clamp
        return retraction_charge_accel_cap(t)
    elseif retraction_explicit_kick_soft_limit && retraction_started[] &&
           retraction_deep_contact_kick_active[]
        return retraction_deep_contact_accel_cap
    elseif retraction_explicit_kick_soft_limit && retraction_started[] &&
           retraction_dv_mech_ramping(t)
        kin = retraction_kinematics_ramping(t) ?
              max(retraction_kinematics_scale(t), 0.1) : 1.0
        # Hard clamp during σ ramp too (was tanh soft-limit).
        return min(retraction_max_accel * kin, retraction_deep_contact_accel_cap)
    end
    return Inf
end

function update_retraction_deep_contact_kick_flag!(cyl_sys, floor_sys, mold_sys)
    retraction_deep_contact_kick_gap_ps == -Inf && return nothing
    !retraction_started[] && return nothing
    tol = retraction_deep_contact_kick_gap_ps * particle_spacing
    min_mold_gap, min_floor_gap = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
    retraction_deep_contact_kick_active[] =
        (min_floor_gap < tol) || (min_mold_gap < tol)
    return nothing
end

function apply_retraction_explicit_kick_soft_limit!(dv_cyl, cyl_sys, t)
    cap = retraction_explicit_kick_cap(t)
    !isfinite(cap) && return nothing
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        dv_cyl[1, particle] = soft_limit(dv_cyl[1, particle], cap)
        dv_cyl[2, particle] = soft_limit(dv_cyl[2, particle], cap)
        dv_cyl[3, particle] = soft_limit(dv_cyl[3, particle], cap)
    end
    return nothing
end

function apply_retraction_deep_contact_velocity_hard_clamp!(v_cyl, cyl_sys)
    retraction_deep_contact_kick_active[] || return nothing
    !isfinite(retraction_deep_contact_speed_cap) && return nothing
    vmax = retraction_deep_contact_speed_cap
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        v_cyl[1, particle] = soft_limit(v_cyl[1, particle], vmax)
        v_cyl[2, particle] = soft_limit(v_cyl[2, particle], vmax)
        v_cyl[3, particle] = soft_limit(v_cyl[3, particle], vmax)
    end
    return nothing
end

# Cap charge |a| from internal TLSPH/stress only (before Reimann contact is added).
function apply_retraction_internal_accel_cap!(dv_ode, cyl_sys, semi_local)
    a_cap = retraction_internal_accel_cap
    (a_cap > 0.0 && isfinite(a_cap)) || return nothing
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        ax = dv_cyl[1, particle]
        ay = dv_cyl[2, particle]
        az = dv_cyl[3, particle]
        mag = sqrt(ax * ax + ay * ay + az * az)
        if mag > a_cap
            s = (a_cap * tanh(mag / a_cap)) / mag
            dv_cyl[1, particle] = ax * s
            dv_cyl[2, particle] = ay * s
            dv_cyl[3, particle] = az * s
        end
    end
    return nothing
end

# Rayleigh (mass-proportional) + soft viscoplastic |a| law for demold.
# Prefer these over hard |a|/|v| clamps: dissipate / regularize via force laws.
function apply_retraction_visco_rayleigh_damping!(dv_cyl, v_cyl, cyl_sys,
                                                  floor_sys, mold_sys, t)
    in_retract = sim_phase == "retraction" && retraction_started[]
    in_hold_tail = hold_tlsph_tail_active(t)
    (in_retract || in_hold_tail) || return nothing
    γ0 = in_hold_tail ? max(retraction_rayleigh_damp, 200.0) : retraction_rayleigh_damp
    γc = in_hold_tail ? max(retraction_rayleigh_damp_contact, 2.0e3) : retraction_rayleigh_damp_contact
    γ_base = velocity_damping_base
    γ_near = velocity_damping_contact
    a_y = retraction_viscoplastic_accel_yield
    use_rayleigh = (γ0 > 0.0) || (γc > 0.0) || (γ_base > 0.0) || (γ_near > 0.0) ||
        (retraction_rayleigh_after_eject > 0.0)
    use_vp = in_retract && a_y > 0.0 && isfinite(a_y)
    use_rayleigh || use_vp || return nothing

    contact_damp_dist = 0.8 * smoothing_length
    gap_tol = retraction_viscoplastic_gap_ps * particle_spacing
    v_wall_z = mold_z_velocity_at_time(t)

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        x_i = cyl_sys.current_coordinates[1, particle]
        z_i = cyl_sys.current_coordinates[3, particle]
        gap_f = tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower)
        gap_m = tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper)
        near_f = gap_f <= contact_damp_dist
        near_m = gap_m <= contact_damp_dist
        deep = (gap_f < gap_tol) || (gap_m < gap_tol)

        if use_rayleigh
            γ = γ0 + γ_base
            if retraction_post_eject_phase()
                γ += retraction_rayleigh_after_eject
            end
            if near_f || near_m
                act_f = near_f ?
                    (1.0 - clamp(gap_f / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)) : 0.0
                act_m = near_m ?
                    (1.0 - clamp(gap_m / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)) : 0.0
                act = max(act_f, act_m)
                act = act * act * (3.0 - 2.0 * act)
                γ += act * (γc + γ_near)
            end
            if γ > 0.0
                dv_cyl[1, particle] -= γ * v_cyl[1, particle]
                dv_cyl[2, particle] -= γ * v_cyl[2, particle]
                # Relative to punch wall when near mold (moving tool).
                vz_rel = near_m ? (v_cyl[3, particle] - v_wall_z) : v_cyl[3, particle]
                dv_cyl[3, particle] -= γ * vz_rel
            end
        end

        if use_vp && deep
            ax = dv_cyl[1, particle]
            ay = dv_cyl[2, particle]
            az = dv_cyl[3, particle]
            mag = sqrt(ax * ax + ay * ay + az * az)
            if mag > a_y
                # Regularized viscoplastic overstress: a ← a_y·tanh(|a|/a_y)·â
                s = (a_y * tanh(mag / a_y)) / mag
                dv_cyl[1, particle] = ax * s
                dv_cyl[2, particle] = ay * s
                dv_cyl[3, particle] = az * s
            end
        end
    end
    return nothing
end

function apply_retraction_charge_accel_caps!(dv_cyl, cyl_sys, t; cap_y::Bool=false)
    apply_velocity_safety_clamps() || return nothing
    accel_cap = retraction_charge_accel_cap(t)
    cap_y = cap_y || retraction_full3d_y_cap_active()
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        dv_cyl[1, particle] = soft_limit(dv_cyl[1, particle], accel_cap)
        cap_y && (dv_cyl[2, particle] = soft_limit(dv_cyl[2, particle], accel_cap))
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
    force_stage_diag_enabled || return nothing
    max_abs, pid_abs, max_delta, pid_delta =
        force_stage_maxima(prev_dv, dv_ode, cyl_sys, semi_local)
    dv_now = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    prev_dv .= dv_now

    pid = pid_delta > 0 ? pid_delta : pid_abs
    if pid > 0
        println(stderr,
                "  FORCE-STAGE ", label,
                "  |dv|max=", round(max_abs; digits=4),
                " p=", pid_abs,
                "  |Δdv|=", round(max_delta; digits=4),
                " pΔ=", pid_delta,
                "  T=", round(cyl_sys.temp[pid]; digits=2),
                "  lf=", round(liquid_fraction_particle_buf[pid]; digits=3),
                "  sf=", round(solid_fraction_particle_buf[pid]; digits=3),
                "  pos=(",
                round(cyl_sys.current_coordinates[1, pid] * 1e3; digits=2),
                ", ",
                round(cyl_sys.current_coordinates[3, pid] * 1e3; digits=2),
                ") mm")
    else
        println(stderr,
                "  FORCE-STAGE ", label,
                "  |dv|max=", round(max_abs; digits=4),
                "  |Δdv|=", round(max_delta; digits=4))
    end
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

# Supporting-surface floor gap under the charge (not volumetric, not global max-z).
# - Volumetric tool_surface_gap(:lower) is invariant to rigid −z shifts of a thick die
#   when the charge sits inside the tool's z-extent.
# - Global max-z nearby picks L-die vertical wall tops (floor_top_gap≈−49 mm evidence).
# Local support: highest floor particle near x that is not above the charge.
function floor_top_surface_gap(floor_sys, x_target, z_target, particle_spacing)
    x_band = 2.0 * particle_spacing
    z_above = z_target + 0.5 * particle_spacing
    best_z = -Inf
    found = false
    @inbounds for p in 1:nparticles(floor_sys)
        abs(floor_sys.current_coordinates[1, p] - x_target) > x_band && continue
        z_p = floor_sys.current_coordinates[3, p]
        z_p > z_above && continue  # skip vertical walls / shelf above the charge
        if z_p > best_z
            best_z = z_p
            found = true
        end
    end
    !found && return tool_surface_gap(floor_sys, x_target, z_target, particle_spacing;
                                      side=:lower)
    return (z_target - best_z) - particle_spacing
end

function charge_min_floor_top_gap(cyl_sys, floor_sys)
    min_gap = Inf
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        charge_particle_is_flyaway(particle) && continue
        x_i = cyl_sys.current_coordinates[1, particle]
        z_i = cyl_sys.current_coordinates[3, particle]
        min_gap = min(min_gap,
                      floor_top_surface_gap(floor_sys, x_i, z_i, particle_spacing))
    end
    return min_gap
end

@inline function floor_tool_z_extents(floor_sys)
    zmin = Inf
    zmax = -Inf
    @inbounds for p in 1:nparticles(floor_sys)
        z = floor_sys.current_coordinates[3, p]
        zmin = min(zmin, z)
        zmax = max(zmax, z)
    end
    return zmin, zmax
end

function apply_floor_open_z_shift!(floor_sys)
    shift = retraction_floor_open_z_shift[]
    init_f = TrixiParticles.initial_coordinates(floor_sys)
    @inbounds for p in 1:nparticles(floor_sys)
        floor_sys.current_coordinates[1, p] = init_f[1, p]
        floor_sys.current_coordinates[2, p] = init_f[2, p]
        floor_sys.current_coordinates[3, p] = init_f[3, p] + shift
    end
    return shift
end

@inline function tool_surface_gap(system_tool, x_target, z_target, particle_spacing; side)
    z_tool_center = local_tool_surface_center_z(system_tool, x_target, z_target; side=side)
    center_gap = side === :lower ? (z_target - z_tool_center) : (z_tool_center - z_target)
    return center_gap - particle_spacing
end

const thermal_gap_floor_cache = Float64[]
const thermal_gap_mold_cache = Float64[]
const thermal_gap_cache_valid = Ref(false)

@inline function tool_thermal_gap(system_tool, x_target, z_target, particle_spacing)
    best_distance2 = Inf
    x_band = 5.0 * particle_spacing
    z_band = 5.0 * particle_spacing
    @inbounds for p in 1:nparticles(system_tool)
        dx = x_target - system_tool.current_coordinates[1, p]
        abs(dx) > x_band && continue
        dz = z_target - system_tool.current_coordinates[3, p]
        abs(dz) > z_band && continue
        distance2 = dx * dx + dz * dz
        if distance2 < best_distance2
            best_distance2 = distance2
        end
    end
    if isinf(best_distance2)
        @inbounds for p in 1:nparticles(system_tool)
            dx = x_target - system_tool.current_coordinates[1, p]
            dz = z_target - system_tool.current_coordinates[3, p]
            distance2 = dx * dx + dz * dz
            if distance2 < best_distance2
                best_distance2 = distance2
            end
        end
    end
    return sqrt(best_distance2) - particle_spacing
end

function compute_contact_heat_flux!(ext_heat_per_particle, system_cyl, system_floor,
                                    system_mold, particle_spacing, t=-Inf;
                                    source_indices=nothing, reset::Bool=true)
    reset && fill!(ext_heat_per_particle, 0.0)

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

    n_cyl = nparticles(system_cyl)
    # Cache gaps only in closed-die solidification hold: punch has finished its
    # stroke (t ≥ t_down_end) and charge kinematics are frozen. During compression
    # the punch moves, so gaps must be recomputed every call.
    use_thermal_gap_cache = !separate_charge_plies &&
        (sim_phase == "hold" || sim_phase == "full") &&
        hold_kinematics_mode == :frozen &&
        isfinite(t) && t >= t_down_end
    if use_thermal_gap_cache
        if !thermal_gap_cache_valid[] || length(thermal_gap_floor_cache) != n_cyl
            resize!(thermal_gap_floor_cache, n_cyl)
            resize!(thermal_gap_mold_cache, n_cyl)
            @inbounds for i in 1:n_cyl
                x_i = system_cyl.current_coordinates[1, i]
                z_i = system_cyl.current_coordinates[3, i]
                thermal_gap_floor_cache[i] = tool_thermal_gap(system_floor, x_i, z_i, particle_spacing)
                thermal_gap_mold_cache[i] = tool_thermal_gap(system_mold, x_i, z_i, particle_spacing)
            end
            thermal_gap_cache_valid[] = true
            println(">>> Pre-cached tool thermal gaps for ", n_cyl,
                    " workpiece particles at t≥t_down_end=", round(t_down_end; digits=6),
                    " s (closed-die hold; no gap searches until retract).")
            flush(stdout)
        end
    else
        thermal_gap_cache_valid[] = false
    end

    @inbounds for i in 1:n_cyl
        g = source_indices === nothing ? i : source_indices[i]
        x_i = system_cyl.current_coordinates[1, i]
        z_i = system_cyl.current_coordinates[3, i]
        T_i = system_cyl.temp[i]
        h_contact_i = h_contact_particle[g]

        gap_floor = use_thermal_gap_cache ? thermal_gap_floor_cache[i] :
            tool_thermal_gap(system_floor, x_i, z_i, particle_spacing)
        if gap_floor <= contact_threshold
            gap_floor_eff = clamp(gap_floor, 0.0, contact_threshold)
            dT = T_i - T_floor
            flux_floor = h_contact_i * dT * gap_activation(gap_floor_eff) * dT_activation(dT)
            if flux_floor != 0.0
                ext_heat_per_particle[g] += flux_floor
            end
        end

        gap_mold = use_thermal_gap_cache ? thermal_gap_mold_cache[i] :
            tool_thermal_gap(system_mold, x_i, z_i, particle_spacing)
        if gap_mold <= contact_threshold
            gap_mold_eff = clamp(gap_mold, 0.0, contact_threshold)
            dT = T_i - T_mold
            flux_mold = h_contact_i * dT * gap_activation(gap_mold_eff) * dT_activation(dT)
            if flux_mold != 0.0
                ext_heat_per_particle[g] += flux_mold
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
    @inbounds for ij in eachindex(A)
        if !isfinite(A[ij])
            fill!(A, 0.0)
            A[1, 1] = A[2, 2] = A[3, 3] = 1.0 / 3.0
            return A
        end
    end
    A_sym = 0.5 .* (A .+ A')
    eig = try
        eigen(Symmetric(A_sym))
    catch err
        err isa ArgumentError || rethrow()
        fill!(A, 0.0)
        A[1, 1] = A[2, 2] = A[3, 3] = 1.0 / 3.0
        return A
    end
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

    if organosheet_architecture in ("qi", "quasi_isotropic")
        @inbounds for i in 1:n
            A_state[1, 1, i] = 0.5
            A_state[2, 2, i] = 0.5
            A_state[3, 3, i] = 0.0
        end
        return A_state
    end

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

# Song et al. CompA 114 (2018) Fig. 4 — digitized initial FO at charge location P.
# Fig.4 frame: A11 along plotted X (thickness, lowest), A22 along Y (highest), A33 along Z.
const song_fig4_zn = (-1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125,
                      0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
const song_fig4_A11 = (0.18, 0.16, 0.14, 0.15, 0.17, 0.16, 0.15, 0.17,
                       0.18, 0.16, 0.14, 0.15, 0.17, 0.19, 0.17, 0.16, 0.18)
const song_fig4_A22 = (0.50, 0.48, 0.45, 0.43, 0.42, 0.46, 0.48, 0.47,
                       0.45, 0.44, 0.48, 0.52, 0.50, 0.47, 0.48, 0.50, 0.52)
const song_fig4_A33 = (0.32, 0.36, 0.41, 0.42, 0.41, 0.38, 0.37, 0.36,
                       0.37, 0.40, 0.38, 0.33, 0.33, 0.34, 0.35, 0.34, 0.30)

function _lerp_fig4(zn_query, zn_tab, a_tab)
    z = clamp(zn_query, zn_tab[1], zn_tab[end])
    if z <= zn_tab[1]
        return Float64(a_tab[1])
    elseif z >= zn_tab[end]
        return Float64(a_tab[end])
    end
    i = 1
    while i < length(zn_tab) && zn_tab[i + 1] < z
        i += 1
    end
    z0 = Float64(zn_tab[i]); z1 = Float64(zn_tab[i + 1])
    a0 = Float64(a_tab[i]); a1 = Float64(a_tab[i + 1])
    t = (z - z0) / max(z1 - z0, eps(Float64))
    return (1.0 - t) * a0 + t * a1
end

"""Map Song Fig.4 CT FO onto the SPH charge.

Default mold-frame map (`length_x_preferred`):
  charge thickness z ← Fig.4 X (low A11) → A_zz
  charge length x    ← Fig.4 Y (high A22) → A_xx
  charge width y     ← Fig.4 Z (mid A33)  → A_yy

`fiber_direction` is set to the principal eigenvector of A (Spencer n_ref).
"""
function initialize_orientation_tensor_song_fig4!(A_state, fiber_dir, coords;
                                                  map_mode::AbstractString="length_x_preferred")
    n = size(A_state, 3)
    zs = @view coords[3, 1:n]
    zmin = minimum(zs); zmax = maximum(zs)
    zspan = max(zmax - zmin, eps(Float64))
    mode = lowercase(strip(map_mode))
    mean_axx = 0.0; mean_ayy = 0.0; mean_azz = 0.0

    @inbounds for i in 1:n
        zn = 2.0 * (coords[3, i] - zmin) / zspan - 1.0
        a11 = _lerp_fig4(zn, song_fig4_zn, song_fig4_A11)
        a22 = _lerp_fig4(zn, song_fig4_zn, song_fig4_A22)
        a33 = _lerp_fig4(zn, song_fig4_zn, song_fig4_A33)
        s = a11 + a22 + a33
        a11 /= s; a22 /= s; a33 /= s

        # Diagonal A in mold frame (off-diagonals ≈ 0 in Fig.4 mean sense).
        if mode in ("length_x_preferred", "default", "song", "x")
            a_xx, a_yy, a_zz = a22, a33, a11
        elseif mode in ("fig4_xyz", "ct", "raw")
            a_xx, a_yy, a_zz = a11, a22, a33
        else
            error("TP_CLIP_FO_FIG4_MAP must be 'length_x_preferred' or 'fig4_xyz' (got $map_mode)")
        end

        fill!(view(A_state, :, :, i), 0.0)
        A_state[1, 1, i] = a_xx
        A_state[2, 2, i] = a_yy
        A_state[3, 3, i] = a_zz
        project_orientation_tensor!(view(A_state, :, :, i))

        A = Symmetric(A_state[:, :, i])
        evals, evecs = eigen(A)
        p = evecs[:, argmax(evals)]
        nrm = norm(p)
        if nrm < eps(Float64)
            p = [1.0, 0.0, 0.0]
        else
            p = p ./ nrm
            # Prefer +x hemisphere for a stable Spencer reference.
            p[1] < 0.0 && (p = -p)
        end
        fiber_dir[1, i] = p[1]
        fiber_dir[2, i] = p[2]
        fiber_dir[3, i] = p[3]
        mean_axx += A_state[1, 1, i]
        mean_ayy += A_state[2, 2, i]
        mean_azz += A_state[3, 3, i]
    end
    invn = 1.0 / n
    println("  - fiber orientation: Song Fig.4 mapped init (TP_CLIP_FO_INIT=song_fig4, map=",
            mode, "); mean Axx/Ayy/Azz = ",
            round(mean_axx * invn; digits=3), " / ",
            round(mean_ayy * invn; digits=3), " / ",
            round(mean_azz * invn; digits=3))
    return A_state
end

fo_init_mode = lowercase(strip(get(ENV, "TP_CLIP_FO_INIT", "")))
if fo_init_mode in ("song_fig4", "fig4", "song-fig4")
    orientation_tensor_state[] = zeros(3, 3, n_cylinder_particles)
    initialize_orientation_tensor_song_fig4!(orientation_tensor_state[], fiber_direction,
                                             polymer.coordinates;
                                             map_mode=get(ENV, "TP_CLIP_FO_FIG4_MAP",
                                                          "length_x_preferred"))
else
    orientation_tensor_state[] = initialize_orientation_tensor_state(fiber_direction)
end

function update_flow_orientation_kinetics!(A_state, orientation_scalar, vel_grad, dt,
                                           fiber_direction;
                                           xi_ft, ci_ft, kappa_rsc,
                                           ci_ard_parallel, ci_ard_perp)
    I3 = Matrix{Float64}(I, 3, 3)
    dt_eff = max(dt, 1.0e-12)

    @inbounds for i in 1:size(A_state, 3)
        A = view(A_state, :, :, i)
        L = view(vel_grad, :, :, i)

        bad_L = false
        for ij in eachindex(L)
            if !isfinite(L[ij])
                bad_L = true
                break
            end
        end
        bad_L && continue

        D = 0.5 .* (L .+ L')
        W = 0.5 .* (L .- L')
        gamma_dot = sqrt(2.0 * sum(D .* D))
        if !isfinite(gamma_dot) || gamma_dot > 1.0e8
            continue
        end

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
        parallelization_backend=clip_parallelization_backend
    ) do particle, neighbor, pos_diff_initial, initial_distance
        initial_distance^2 < eps(TrixiParticles.initial_smoothing_length(system)^2) && return

        volume = @inbounds system.mass[neighbor] / system.material_density[neighbor]
        NDIMS = TrixiParticles.ndims(system)
        vel_diff = TrixiParticles.extract_svector(v, Val(NDIMS), particle) - TrixiParticles.extract_svector(v, Val(NDIMS), neighbor)
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
    n_particles = size(vel_grad_buf, 3)
    v_vis = v_vis_buf !== nothing ? v_vis_buf :
            zeros(eltype(system.young_modulus), 3, 3, n_particles)

    Threads.@threads for particle in 1:n_particles
        if solid_fraction_buf !== nothing && solid_fraction_buf[particle] >= skip_solid_fraction
            @inbounds v_vis[:, :, particle] .= 0.0
            continue
        end

        F = TrixiParticles.deformation_gradient(system, particle)
        L_corr = @inbounds TrixiParticles.correction_matrix(system, particle)
        J = max(det(F), 1e-6)

        _L = StaticArrays.SMatrix{3, 3}(@view vel_grad_buf[:, :, particle]) * L_corr'
        d = 0.5 * (_L + _L')
        dev_d = d - 1 / 3 * tr(d) * I

        tau = 2.0 * vis[particle] * dev_d
        tau_norm = sqrt(sum(tau .^ 2))
        if tau_norm > matrix_yield_stress
            tau = tau * (matrix_yield_stress / tau_norm)
        end

        if include_bulk_term
            lf = solid_fraction_buf !== nothing ? (1.0 - solid_fraction_buf[particle]) : 1.0
            p_vol = (1.0 - lf) * matrix_K_melt * log(J)
            tau += p_vol * I
        end

        temp_p = system.temp[particle]
        visc_blend = clamp((matrix_tmelt + 2.5 - temp_p) / 5.0, 0.0, 1.0)
        if visc_blend < 1e-6
            @inbounds v_vis[:, :, particle] .= 0.0
            continue
        end

        FinvT = inv(F)'
        v_vis[:, :, particle] .= (tau * FinvT) * L_corr * visc_blend
    end

    return v_vis
end

@inline function particle_J_total(F_reg)
    return max(det(F_reg), 1e-6)
end

@inline function elastic_volumetric_J(F_reg, particle, Fp_local;
                                      global_particle=particle)
    if use_volumetric_plasticity
        J_p = max(J_p_particle_buf[global_particle], 1e-6)
        return max(particle_J_total(F_reg) / J_p, 1e-6)
    end
    Fp_inv = inv(Fp_local)
    return max(det(F_reg * Fp_inv), 1e-6)
end

@inline function matrix_bulk_log_J(F_reg, particle, Fp_local, matrix_blend, J_ref_legacy;
                                   global_particle=particle)
    if use_volumetric_plasticity
        matrix_blend < 1.0 - 1e-6 && return 0.0
        J_e_vol = elastic_volumetric_J(F_reg, particle, Fp_local; global_particle)
        return log(max(J_e_vol / elastic_bulk_J_ref, 1e-6))
    end
    if matrix_blend < 1.0 - 1e-6
        J_e = elastic_volumetric_J(F_reg, particle, Fp_local; global_particle)
        return log(max(J_e / J_e, 1e-6))
    end
    J_e = elastic_volumetric_J(F_reg, particle, Fp_local; global_particle)
    return log(max(J_e / max(J_ref_legacy, 1e-6), 1e-6))
end

function commit_volumetric_plastic_relax!(F_reg, particle; active, commit_blend, matrix_blend,
                                          global_particle=particle)
    use_volumetric_plasticity || return
    active || return
    commit_blend < 1e-6 && return
    J_tot = particle_J_total(F_reg)
    if matrix_blend < 1.0 - 1e-6
        J_p_particle_buf[global_particle] = J_tot
        return
    end
    J_p = max(J_p_particle_buf[global_particle], 1e-6)
    rate = clamp(volumetric_plastic_relax_rate, 0.0, 1.0)
    J_p_new = J_p + rate * (J_tot - J_p)
    J_p_particle_buf[global_particle] = clamp(J_p_new, 1e-6, max(J_tot, 1e-6))
    return nothing
end



function elastic_stress3d_trial_skip_liquid!(system, ys, hard, vis, dt, _alpha, _Fp, semi;
                                             solid_fraction_buf,
                                             v_elas_buf=nothing,
                                             skip_solid_fraction=0.0,
                                             source_indices=nothing)
    (; deformation_grad, young_modulus, poisson_ratio, temp, tmelt, hardening) = system

    n_particles = size(deformation_grad, 3)
    v_elas = v_elas_buf !== nothing ? v_elas_buf : zeros(eltype(young_modulus), 3, 3, n_particles)

    mu = young_modulus / (2 + 2 * poisson_ratio)
    K  = young_modulus / (3 - 6 * poisson_ratio)
    H_theta = 1.0 / (tmelt - system.temp_ref[1])
    cte_s = retraction_thermal_cte_scale()

    Threads.@threads for particle in 1:n_particles
        global_particle = source_indices === nothing ? particle : source_indices[particle]
        if solid_fraction_buf[particle] <= skip_solid_fraction
            if organosheet_mechanics && organosheet_active_in_melt
                F = TrixiParticles.deformation_gradient(system, particle)
                L_corr = @inbounds TrixiParticles.correction_matrix(system, particle)
                det_F = det(F)
                F_reg = if !isfinite(det_F) || abs(det_F) < 1e-10
                    F + 1e-4 * one(StaticArrays.SMatrix{3, 3, eltype(F)})
                else
                    F
                end
                n_ref_p = StaticArrays.SVector{3}(fiber_direction[1, global_particle],
                                                  fiber_direction[2, global_particle],
                                                  fiber_direction[3, global_particle])
                reinforcement_tau_p = clip_organosheet_reinforcement_stress(
                    F_reg, n_ref_p, organosheet_ply_model, organosheet_architecture)
                if all(isfinite, reinforcement_tau_p)
                    FinvT = inv(F_reg)'
                    v_local = (reinforcement_tau_p * FinvT) * L_corr
                    if all(isfinite, v_local)
                        @inbounds v_elas[:, :, particle] .= v_local
                        continue
                    end
                end
            end
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
        if isfinite(d_fp) && d_fp > 1e-4
            Fp_local = Fp_local / cbrt(d_fp)
        else
            det_F_val = det(F_reg)
            if isfinite(det_F_val) && det_F_val > 1e-10
                Fp_local = F_reg / cbrt(det_F_val)
            else
                Fp_local = one(StaticArrays.SMatrix{3, 3, eltype(F_reg)})
            end
        end

        n_ref_p = StaticArrays.SVector{3}(fiber_direction[1, global_particle],
                                          fiber_direction[2, global_particle],
                                          fiber_direction[3, global_particle])

        # Kinematic thermal contraction (F = Fe * Fp * Fth  =>  F_mech = F * inv(Fth))
        F_mech = if (clip_kinematic_thermal_shrinkage && sim_phase == "retraction" &&
                     retraction_post_cool_springback_active[])
            delta_T_th = temp[particle] - retraction_post_cool_thermal_Tref_K
            ramp_th = retraction_post_cool_thermal_scale[]
            dT_eff = ramp_th * delta_T_th
            α_par = alpha1_particle[global_particle]
            α_perp = alpha2_particle[global_particle]
            lam_par = clamp(exp(α_par * dT_eff), 0.90, 1.02)
            lam_perp = clamp(exp(α_perp * dT_eff), 0.90, 1.02)
            inv_lam_perp = 1.0 / lam_perp
            d_inv_lam = (1.0 / lam_par) - inv_lam_perp
            inv_Fth = inv_lam_perp * one(StaticArrays.SMatrix{3, 3, Float64}) + d_inv_lam * (n_ref_p * n_ref_p')
            F_reg * inv_Fth
        else
            F_reg
        end

        Fp_inv = inv(Fp_local)
        Fe = F_mech * Fp_inv
        J_e = max(det(Fe), 1e-6)
        be = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - 1 / 3 * tr(be_bar) * I
        # Per-particle Lamé constants: Halpin–Tsai transverse E2 as isotropic base.
        mu = E2_particle[global_particle] / (2 + 2 * poisson_ratio)
        K  = E2_particle[global_particle] / (3 - 6 * poisson_ratio)

        Fn_p = F_mech * n_ref_p
        I4_p = dot(Fn_p, Fn_p)
        I4_tensile = max(I4_p - 1.0, 0.0)
        reinforcement_tau_p = if organosheet_mechanics
            clip_organosheet_reinforcement_stress(F_reg, n_ref_p, organosheet_ply_model,
                                                  organosheet_architecture)
        else
            (2 * k_fiber_particle[global_particle] * I4_tensile) .*
                (Fn_p * Fn_p')
        end

        # Smooth blend of matrix solid fraction over a 20 K window centred on matrix_tmelt.
        # blend = 1 → fully solid  (T ≤ matrix_tmelt - matrix_blend_half_K)
        # blend = 0 → fully molten (T ≥ matrix_tmelt + matrix_blend_half_K)
        # matrix_blend_half_K is a module-level const (= 10 K) shared with the
        # accepted-step callback so the two blend windows are always consistent.
        matrix_blend = clamp((matrix_tmelt + matrix_blend_half_K - temp[particle]) /
                             (2 * matrix_blend_half_K), 0.0, 1.0)

        # Compute full solid-state tau (elastic + return mapping) then scale by blend.
        # Compress/hold: keep legacy ln(det(Fe)/J_ref) so molding physics is unchanged.
        # Retraction only: Strategy E ln(J_e_vol)=ln(det(F)/J_p) via matrix_bulk_log_J
        # (matches commit/audit; avoids fake GPa bulk from isochoric Fp).
        # NOTE: do NOT write J_ref_particle_buf here (Jacobian corruption).
        J_ref_p = max(J_ref_particle_buf[global_particle], 1e-6)
        if matrix_blend < 1.0 - 1e-6
            J_ref_p = J_e
        end
        bulk_log = if use_volumetric_plasticity && sim_phase == "retraction"
            matrix_bulk_log_J(F_reg, particle, Fp_local, matrix_blend, J_ref_p;
                              global_particle)
        else
            log(max(J_e / J_ref_p, 1e-6))
        end
        # Matrix Kirchhoff only for yield / plastic flow. Spencer is hyperelastic
        # fiber reinforcement (depends on total F, not Fe) — keep it in the elastic
        # stress below, but do not let it drive Fp updates (same philosophy as heating).
        tau_matrix = K * bulk_log * I + mu * dev_be
        tau_solid = tau_matrix + reinforcement_tau_p

        dev_tau = tau_matrix - 1 / 3 * tr(tau_matrix) * I
        yf = sqrt(1.5) * sqrt(sum(dev_tau .^ 2)) - (ys[particle] + hard[particle])

        if matrix_blend > 1e-6 && yf > 1e-6 && !retraction_thermal_shrinkage_only_now()
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
                Fe = F_mech * Fp_inv
                J_e = max(det(Fe), 1e-6)
                be = Fe * Fe'
                be_bar = be / (J_e^(2 / 3))
                dev_be = be_bar - 1 / 3 * tr(be_bar) * I
                bulk_log = if use_volumetric_plasticity && sim_phase == "retraction"
                    matrix_bulk_log_J(F_reg, particle, Fp_local, matrix_blend, J_ref_p;
                                      global_particle)
                else
                    log(max(J_e / J_ref_p, 1e-6))
                end
                tau_matrix = K * bulk_log * I + mu * dev_be
                tau_solid = tau_matrix + reinforcement_tau_p
            end
        end

        # Thermal stress for springback:
        # Transversely isotropic deviatoric thermal stress drives angle spring-in:
        # τ_th,dev = -2·μ·(α₁ - α₂)·ΔT · (n⊗n - 1/3·I)
        # where n is the current deformed fiber direction.
        # This is strictly traceless (zero artificial hydrostatic pressure) and
        # directly reproduces the classical Radford & Rennick (1999) / Song et al. (2018)
        # corner spring-in bending moment without causing unphysical volumetric cavitation/expansion.
        T_ref_th = sim_phase == "retraction" ?
            retraction_post_cool_thermal_Tref_K : T_stress_free
        delta_T_th = temp[particle] - T_ref_th
        ramp_th = retraction_post_cool_thermal_scale[]
        if ramp_th < 1.0
            delta_T_th *= ramp_th
        end

        Fn_p = F_reg * n_ref_p
        nrm2 = dot(Fn_p, Fn_p)
        n_curr = nrm2 > 1.0e-24 ? Fn_p / sqrt(nrm2) : n_ref_p
        M_dev = (n_curr * n_curr') - (1.0 / 3.0) * one(StaticArrays.SMatrix{3, 3, Float64})
        alpha_diff = alpha1_particle[global_particle] - alpha2_particle[global_particle]
        # Optional deviatoric thermal stress (governed by TP_CLIP_DEVIATORIC_THERMAL_CTE, default 0 in retraction to prevent unphysical compressive buckling and flange waviness)
        dev_cte_scale = env_float("TP_CLIP_DEVIATORIC_THERMAL_CTE", sim_phase == "retraction" ? 0.0 : 1.0)
        tau_th_dev = dev_cte_scale > 0.0 ?
            (dev_cte_scale * (-2.0 * mu * alpha_diff * delta_T_th * M_dev * (cte_s * solid_fraction_buf[particle]))) :
            zero(StaticArrays.SMatrix{3, 3, Float64})

        # Optional isotropic volumetric thermal pressure (governed by TP_CLIP_VOLUMETRIC_THERMAL_CTE, default 0 in retraction to prevent boundary blowup)
        vol_cte_scale = env_float("TP_CLIP_VOLUMETRIC_THERMAL_CTE", sim_phase == "retraction" ? 0.0 : 1.0)
        tau_th_vol = if vol_cte_scale > 0.0
            (vol_cte_scale * cte_s * (-K * 3.0 * alpha_eff_particle[global_particle] * delta_T_th *
             solid_fraction_buf[particle])) * one(StaticArrays.SMatrix{3, 3, Float64})
        else
            zero(StaticArrays.SMatrix{3, 3, Float64})
        end

        tau_matrix += (tau_th_dev + tau_th_vol)
        tau_solid += (tau_th_dev + tau_th_vol)

        # Blend: above melt the matrix contribution fades to zero continuously,
        # while continuous fiber reinforcement remains active when organosheet_active_in_melt is on.
        tau = if organosheet_mechanics && organosheet_active_in_melt
            matrix_blend * tau_matrix + reinforcement_tau_p
        else
            matrix_blend * tau_solid
        end

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
                                       vel_grad_buf=nothing, update_viscosity::Bool=true,
                                       source_indices=nothing)
    gas_constant = 8.31446261815324
    t_prop_start = time_ns()
    @inbounds for i in eachindex(temp)
        global_i = source_indices === nothing ? i : source_indices[i]
        temp_i = temp[i]
        if !isfinite(temp_i)
            temp_i = matrix_temp_liq
        end
        temp_i = clamp(temp_i, temp_min_clip, temp_max_clip)

        liquid_fraction = use_nakamura_kinetics ?
                  liquid_fraction_particle_buf[i] :
                  liquid_fraction_particle(temp_i, global_i)
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

        orient_scale = 1.0 + phase_vf_particle[global_i] * orientation_coupling_gain *
                             (orientation_scalar_particle[global_i] - 1.0)

        yield_base = organosheet_mechanics ?
                     matrix_yield_stress * (1.0 - phase_vf_particle[global_i]) :
                     yield_base_particle[global_i]
        hardening_base = organosheet_mechanics ?
                         matrix_hardening * (1.0 - phase_vf_particle[global_i]) :
                         hardening_base_particle[global_i]
        ys[i] = max(1.0e3,
                (yield_base + hardening_base * alpha[i]) *
                thermal_softening)
        ys[i] *= orient_scale
        hard[i] = hardening_base * thermal_softening * orient_scale
    end
    rhs_time_property_update_ns[] += time_ns() - t_prop_start

    update_viscosity || return nothing

    t_visc_start = time_ns()
    @inbounds for i in eachindex(temp)
        global_i = source_indices === nothing ? i : source_indices[i]
        temp_i = temp[i]
        if !isfinite(temp_i)
            temp_i = matrix_temp_liq
        end
        temp_i = clamp(temp_i, temp_min_clip, temp_max_clip)
        liquid_fraction = liquid_fraction_particle_buf[i]
        melt_softening = clamp((temp_i - 0.8 * matrix_tmelt) /
                       (0.4 * matrix_tmelt + eps(Float64)), 0.0, 1.0)
        solid_factor = 1.0 - melt_softening
        orient_scale = 1.0 + phase_vf_particle[global_i] * orientation_coupling_gain *
                             (orientation_scalar_particle[global_i] - 1.0)
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
            L = StaticArrays.SMatrix{3, 3}(@view vel_grad_buf[:, :, i])
            D = 0.5 * (L + L')
            shear_rate = sqrt(max(2.0 * sum(abs2, D), 0.0))
            # Previous Cross denominator retained here for comparison:
            # cross_denominator = 1.0 + (viscosity_cross_time_constant * shear_rate)^
            #                       (1.0 - viscosity_cross_power_law_index)
            cross_argument = eta_zero_shear * shear_rate / cross_wlf_critical_stress
            cross_denominator = 1.0 + cross_argument^
                                  (1.0 - cross_wlf_power_law_index)
            concentration_base = max(1.0 - phase_vf_particle[global_i] /
                                     fiber_max_packing_fraction,
                                     eps(Float64))
            concentration_factor = concentration_base^
                                   (-fiber_intrinsic_viscosity * fiber_max_packing_fraction)
            vis[i] = eta_zero_shear / cross_denominator * concentration_factor
        end
        # vis[i] = viscosity_base_particle[i] * (0.2 + 0.8 * solid_factor) * orient_scale
        vis_target = clamp(vis[i], viscosity_min_clip, viscosity_max_clip)
        if viscosity_update_alpha >= 1.0 - 1.0e-12
            vis[i] = vis_target
        else
            η_old = max(vis[i], viscosity_min_clip)
            η_new = vis_target
            log_blend = (1.0 - viscosity_update_alpha) * log(η_old) +
                        viscosity_update_alpha * log(η_new)
            vis[i] = clamp(exp(log_blend), viscosity_min_clip, viscosity_max_clip)
        end
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

@inline function clip_viscosity_update_on_rhs()
    interval = viscosity_rhs_update_interval
    interval <= 0 && return false
    viscosity_rhs_call_count[] += 1
    return viscosity_rhs_call_count[] % interval == 1
end

function clip_viscosity_update_on_step()
    interval = viscosity_step_update_interval
    viscosity_step_count[] += 1
    return viscosity_step_count[] % interval == 1
end

function update_implicit_stress_cache_separate!(semi_local, v_ode, t)
    fill!(v_stress_buf, 0.0)
    fill!(v_stress_viscous_buf, 0.0)
    fill!(v_stress_elastic_buf, 0.0)
    update_viscosity = clip_viscosity_update_on_rhs()
    caches = Dict{UInt64, TrixiParticles.StressTensorArray}()

    clip_foreach_charge_system(semi_local) do system, source
        ys = clip_charge_particle_view(ys_particle_buf, source)
        hard = clip_charge_particle_view(hard_particle_buf, source)
        vis = clip_charge_particle_view(vis_particle_buf, source)
        thermal_soft = clip_charge_particle_view(thermal_softening_particle_buf, source)
        liquid = clip_charge_particle_view(liquid_fraction_particle_buf, source)
        solid = clip_charge_particle_view(solid_fraction_particle_buf, source)
        alpha = clip_charge_particle_view(alpha_committed[], source)
        velocity_gradient = clip_charge_particle_view(vel_grad_buf, source)
        stress = clip_charge_particle_view(v_stress_buf, source)
        stress_viscous = clip_charge_particle_view(v_stress_viscous_buf, source)
        stress_elastic = clip_charge_particle_view(v_stress_elastic_buf, source)
        Fp = clip_charge_particle_view(Fp_committed[], source)

        update_dual_phase_properties!(ys, hard, vis, system.temp, alpha,
                                      thermal_soft, liquid, solid;
                                      vel_grad_buf=velocity_gradient,
                                      update_viscosity, source_indices=source)
        stress_visc = if retraction_solid_mechanics_only()
            fill!(stress_viscous, 0.0)
        else
            viscous_stress_from_cached_grad!(
                system, vis; v_vis_buf=stress_viscous,
                vel_grad_buf=velocity_gradient,
                include_bulk_term=enable_elastic_stress,
                solid_fraction_buf=solid,
                skip_solid_fraction=viscous_skip_solid_fraction)
        end

        if !enable_elastic_stress
            stress .= stress_visc
            caches[objectid(system)] = Array(stress)
            return
        end

        stress_elas = elastic_stress3d_trial_skip_liquid!(
            system, ys, hard, vis, trial_dt_state[], alpha, Fp, semi_local;
            solid_fraction_buf=solid,
            skip_solid_fraction=elastic_skip_solid_fraction,
            v_elas_buf=stress_elastic,
            source_indices=source)

        @inbounds for particle in TrixiParticles.each_integrated_particle(system)
            elas_ok = all(isfinite, @view stress_elas[:, :, particle])
            mech_w = (organosheet_mechanics && organosheet_active_in_melt) ? 1.0 : solid_mechanics_weight(solid[particle])
            for j in 1:3, i in 1:3
                stress[i, j, particle] =
                    elas_ok ? mech_w * stress_elas[i, j, particle] : 0.0
            end
        end
        caches[objectid(system)] = Array(stress)
    end
    refresh_retraction_solid_phase_buffers!()
    TrixiParticles.STRESS_TENSOR_CACHE[] = caches
    return nothing
end

function update_implicit_stress_cache!(semi_local, v_ode, t)
    if separate_charge_plies
        return update_implicit_stress_cache_separate!(semi_local, v_ode, t)
    end
    cyl_sys   = semi_local.systems[1]
    floor_sys = clip_floor_system(semi_local)
    mold_sys  = clip_mold_system(semi_local)
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)

    # ===== Properties + constitutive stress cache =====
    update_dual_phase_properties!(ys_particle_buf, hard_particle_buf, vis_particle_buf,
                                  cyl_sys.temp, alpha_committed[],
                                  thermal_softening_particle_buf,
                                  liquid_fraction_particle_buf,
                                  solid_fraction_particle_buf;
                                  vel_grad_buf=vel_grad_buf,
                                  update_viscosity=clip_viscosity_update_on_rhs())
    refresh_retraction_solid_phase_buffers!()

    # Fp / alpha / J_ref updates for liquid and solidifying particles happen only in
    # commit_plastic_history_and_heat! (accepted-step callback). No history-variable
    # writes here — the RHS must be a pure function of ODE state + committed buffers.

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
        clip_set_stress_cache!(cyl_sys)
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
                if organosheet_mechanics && organosheet_active_in_melt
                    for j in 1:3, i in 1:3
                        v_stress_buf[i, j, particle] = stress_elas[i, j, particle]
                    end
                else
                    # Mushy/latent/liquid: mechanics is liquid-only (WCSPH branch).
                    # Keep TLSPH constitutive stress fully off to avoid mixed-state
                    # history coupling (Fp/J_ref/plasticity) that collapses TRBDF2 dt.
                    for j in 1:3, i in 1:3
                        v_stress_buf[i, j, particle] = 0.0
                    end
                end
            elseif mech_w >= 1.0 - 1.0e-10
                # Fully solid: use solid constitutive stress.
                for j in 1:3, i in 1:3
                    v_stress_buf[i, j, particle] = stress_elas[i, j, particle]
                end
            else
                # Smooth handover in the semi-solid window to avoid stiffness
                # spikes when particles first re-enter solid mechanics.
                if organosheet_mechanics && organosheet_active_in_melt
                    for j in 1:3, i in 1:3
                        v_stress_buf[i, j, particle] = stress_elas[i, j, particle]
                    end
                else
                    for j in 1:3, i in 1:3
                        v_stress_buf[i, j, particle] = mech_w * stress_elas[i, j, particle]
                    end
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
    # Pre-lift: elastic fade is applied once via retraction_dv_mech_scale (avoid s×s),
    # except contact-settle mode where mech_dv=1 and σ is applied on the stress tensor.
    skip_tensor_elas_ramp = retraction_pre_lift_creep_active(t) &&
        !retraction_closed_die_contact_settle
    if retraction_started[] && !retraction_solid_mechanics_only() &&
       !skip_tensor_elas_ramp && elas_ramp < 1.0 - 1.0e-8
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            for j in 1:3, i in 1:3
                tau_v = v_stress_viscous_buf[i, j, particle]
                tau_e = v_stress_buf[i, j, particle] - tau_v
                v_stress_buf[i, j, particle] = tau_v + elas_ramp * tau_e
            end
        end
    elseif retraction_started[] && retraction_solid_mechanics_only() &&
           !skip_tensor_elas_ramp && elas_ramp < 1.0 - 1.0e-8
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

    clip_set_stress_cache!(cyl_sys)
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
    clip_set_stress_cache!(cyl_sys)
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

    if hard_lock_y_kinematics()
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
        clip_sync_tool_kinematics_gpu!(semi_local, t)
        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        if t != nhs_updated_at_t[]
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end

        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] =
            clip_deformation_only_selector(semi_local)

        TrixiParticles.foreach_system(semi_local) do system
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi_local, t)
        end

        apply_all_charge_plane_strain_F_fixes!(semi_local)

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

    floor_sys = clip_floor_system(semi_local)
    mold_sys  = clip_mold_system(semi_local)
    contact_heat_flux = if retraction_post_eject_phase()
        fill!(contact_heat_flux_buf, 0.0)
    else
        compute_contact_heat_flux!(contact_heat_flux_buf, cyl_sys,
                                   floor_sys, mold_sys, particle_spacing, t)
    end
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    dx = particle_spacing
    rho_cyl = cyl_sys.material_density[1]
    rho_thermal = physical_thermal_density(cyl_sys, 1)
    cp_cyl = cyl_sys.cp
    TrixiParticles.thermal_rhs_sph3d!(cyl_sys, dv_cyl, v_cyl, 0.0,
                                      particle_spacing, bound_coordinate_thermal, semi_local)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        if contact_heat_flux[particle] > 0.0
            dv_cyl[NDIMS_CYL + 1, particle] -= contact_heat_flux[particle] /
                                                (rho_thermal * cp_cyl * dx)
        end
        if ambient_heat_loss_active_now()
            exposure = free_cool_ambient_exposure(particle)
            if exposure > 0.0
                T_i = cyl_sys.temp[particle]
                dT_dt_loss = compute_ambient_cooling_rate(T_i, exposure, rho_thermal, cp_cyl, dx, t)
                dv_cyl[NDIMS_CYL + 1, particle] -= dT_dt_loss
            end
        end
    end

    if hard_lock_y_kinematics()
        TrixiParticles.foreach_system(semi_local) do system
            dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                dv_sys[2, particle] = 0.0
            end
        end
    end

    apply_retraction_charge_accel_caps!(dv_cyl, cyl_sys, t)
    update_retraction_deep_contact_kick_flag!(cyl_sys, floor_sys, mold_sys)
    if retraction_post_eject_phase() && retraction_gravity_after_eject != 0.0
        g = retraction_gravity_after_eject
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[3, particle] -= g
        end
    end
    apply_retraction_visco_rayleigh_damping!(dv_cyl, v_cyl, cyl_sys, floor_sys, mold_sys, t)
    apply_retraction_explicit_kick_soft_limit!(dv_cyl, cyl_sys, t)
    apply_retraction_deep_contact_velocity_hard_clamp!(v_cyl, cyl_sys)
    apply_retraction_full3d_y_accel_cap!(dv_cyl, cyl_sys, t)
    apply_charge_flyaway_freeze!(dv_cyl, v_cyl, cyl_sys)

    if retraction_diag_enabled && retraction_started[]
        retraction_diag_kick_max_dv[] = max_charge_kick_dv(dv_ode, cyl_sys, semi_local)
    end

    return dv_ode
end

function commit_plastic_history_and_heat!(system, ys, hard, vis, dt, _alpha, _Fp;
                                          skip_heat::Bool=false, hold_creep::Bool=false,
                                          source_indices=nothing)
    (; deformation_grad, young_modulus, poisson_ratio, temp, tmelt, hardening,
       material_density, cp, temp_ref) = system

    mu = young_modulus / (2 + 2 * poisson_ratio)
    K  = young_modulus / (3 - 6 * poisson_ratio)
    H_theta = 1.0 / (tmelt - temp_ref[1])

    @inbounds for particle in TrixiParticles.eachparticle(system)
        global_particle = source_indices === nothing ? particle : source_indices[particle]
        # Mirror the early-skip used in elastic_stress3d_trial_skip_liquid!:
        # liquid particles (sf ≤ 0) contribute no elastic stress to the physics
        # RHS, so they must also produce no plastic dissipation here.
        # CRUCIALLY: while a particle is liquid, Fp must track F so that when it
        # re-solidifies Fe = F*Fp_inv ≈ I (stress-free). Without this, Fp stays
        # at its last solid value while F keeps compressing → J_e→0.675 → K*log(J_e)
        # produces GPa-scale elastic stress the instant the particle re-solidifies,
        # collapsing Δt to femtoseconds.
        sf_particle = solid_fraction_particle_buf[global_particle]
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
                    (J_p_particle_buf[global_particle] = max(det_F_liq, 1e-6))
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
        if isfinite(d_fp) && d_fp > 1e-4
            Fp_sm = Fp_sm / cbrt(d_fp)
        else
            det_F_val = det(F_reg)
            if isfinite(det_F_val) && det_F_val > 1e-10
                Fp_sm = F_reg / cbrt(det_F_val)
            else
                Fp_sm = one(StaticArrays.SMatrix{3, 3, eltype(F_reg)})
            end
        end

        Fp_inv = inv(Fp_sm)
        Fe = F_reg * Fp_inv
        J_e = max(det(Fe), 1e-6)
        J_e_vol = elastic_volumetric_J(F_reg, particle, Fp_sm; global_particle)
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
                    (J_p_particle_buf[global_particle] = max(det_Freg, 1e-6))
            end
            continue
        end
        be = Fe * Fe'
        be_bar = be / (J_e^(2 / 3))
        dev_be = be_bar - 1 / 3 * tr(be_bar) * I
        # Per-particle Lamé constants: Halpin–Tsai transverse E2 as isotropic base.
        # (Matches elastic_stress3d_trial_skip_liquid!; the sf protection comes
        # from the early-skip above, not from scaling.)
        mu = E2_particle[global_particle] / (2 + 2 * poisson_ratio)
        K  = E2_particle[global_particle] / (3 - 6 * poisson_ratio)
        # While semi-solid (commit_blend_vol < 1), WCSPH carries volumetric load:
        # track J_ref = J_e so elastic bulk pressure stays zero until fully solid.
        # Uses matrix_blend_half_K (module-level const, same as the RHS blend window)
        # so the two blend windows are consistent.
        commit_blend_vol = clamp((matrix_tmelt + matrix_blend_half_K - temp[particle]) /
                                 (2 * matrix_blend_half_K), 0.0, 1.0)
        matrix_blend = clamp((matrix_tmelt + matrix_blend_half_K - temp[particle]) /
                               (2 * matrix_blend_half_K), 0.0, 1.0)
        J_ref_p = max(J_ref_particle_buf[global_particle], 1e-6)
        if use_volumetric_plasticity
            commit_blend_vol < 1.0 - 1e-6 &&
                (J_p_particle_buf[global_particle] = particle_J_total(F_reg))
        elseif commit_blend_vol < 1.0 - 1e-6
            J_ref_p = J_e
            J_ref_particle_buf[global_particle] = J_e
        end
        bulk_log = matrix_bulk_log_J(F_reg, particle, Fp_sm, matrix_blend, J_ref_p;
                                     global_particle)
        # Matrix Kirchhoff only for yield / plastic flow / heating. Spencer stays
        # hyperelastic (on total F) and must not rewrite Fp (matches elastic RHS split).
        tau_matrix = K * bulk_log * I + mu * dev_be

        # Spencer (1984) transversely isotropic fiber reinforcement.
        n_ref_p = StaticArrays.SVector{3}(fiber_direction[1, global_particle],
                                          fiber_direction[2, global_particle],
                                          fiber_direction[3, global_particle])
        Fn_p = F_reg * n_ref_p
        I4_p = dot(Fn_p, Fn_p)
        I4_tensile = max(I4_p - 1.0, 0.0)
        reinforcement_tau_p = organosheet_mechanics ?
            clip_organosheet_reinforcement_stress(F_reg, n_ref_p, organosheet_ply_model,
                                                  organosheet_architecture) :
            (2 * k_fiber_particle[global_particle] * I4_tensile) .*
                (Fn_p * Fn_p')

        # Guard: skip return mapping and heating if stress is non-finite
        # (e.g. Spencer term overflow from near-singular F at corners).
        if !all(isfinite, tau_matrix) || !all(isfinite, reinforcement_tau_p)
            continue
        end

        dev_tau = tau_matrix - 1 / 3 * tr(tau_matrix) * I
        # Matrix J2 surface: during hold_creep use matrix yield only (not RoM with
        # fiber UTS). Matches Fix B (matrix Kirchhoff drives flow).
        ys_flow = if hold_creep
            matrix_yield_stress
        else
            ys[particle] + hard[particle]
        end
        yf = sqrt(1.5) * sqrt(sum(dev_tau .^ 2)) - ys_flow

        # Gate plasticity on matrix temperature with a 5 K smooth blend (same window as
        # elastic_stress3d_trial_skip_liquid!) to keep ∂RHS/∂T continuous across the
        # melt transition.  A hard cutoff at matrix_tmelt creates a C⁰ kink that the
        # TRBDF2 local-error estimator sees as a large f″ jump, collapsing Δt.
        #   blend = 1 → fully solid  (T ≤ matrix_tmelt - 2.5 K)
        #   blend = 0 → fully molten (T ≥ matrix_tmelt + 2.5 K) → yf set negative
        commit_blend = clamp((matrix_tmelt + 2.5 - temp[particle]) / 5.0, 0.0, 1.0)
        # Closed-die creep + freeze-T: checkpoint particles can sit near/above matrix_tmelt
        # while sf already marks them solid. T-only gating then sets yf=-1 forever and
        # Fe_iso never relaxes. Allow flow from solid fraction during hold_creep.
        if hold_creep && sf_particle >= solid_mechanics_solid_fraction_start
            commit_blend = max(commit_blend, solid_mechanics_weight(sf_particle))
        end
        if commit_blend < 1e-6
            yf = -1.0  # fully molten: no plastic work injection
        end

        # hold_creep: always track J_p→det(F), except while F_th is putting thermal
        # volume into F (hold cool / closed-die cool). J2 must not eat 3αΔT.
        vol_relax = !thermal_Fth_skip_Jp[] && (hold_creep || yf > 1e-6)
        commit_volumetric_plastic_relax!(F_reg, particle;
            active=vol_relax, commit_blend=commit_blend, matrix_blend=matrix_blend,
            global_particle)

        if yf > 1e-6
            H_alpha_theta = hardening * (1 - H_theta * (temp[particle] - temp_ref[1]))
            # hold_creep: prefer rate-independent return (clears Fe_iso on frozen F);
            # viscous branch with high η left residual stress unchanged for 0.15+ s.
            if hold_creep || temp[particle] < 0.5 * tmelt
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
                det_Fp = det(Fp_sm)
                if isfinite(det_Fp) && det_Fp > 1e-4
                    Fp_sm = Fp_sm / cbrt(det_Fp)
                else
                    det_F_val = det(F_reg)
                    if isfinite(det_F_val) && det_F_val > 1e-10
                        Fp_sm = F_reg / cbrt(det_F_val)
                    else
                        Fp_sm = one(StaticArrays.SMatrix{3,3,eltype(F_reg)})
                    end
                end

                Fp_inv = inv(Fp_sm)
                Fe = F_reg * Fp_inv
                J_e = max(det(Fe), 1e-6)
                be = Fe * Fe'
                be_bar = be / (J_e^(2 / 3))
                dev_be = be_bar - 1 / 3 * tr(be_bar) * I
                J_ref_p2 = max(J_ref_particle_buf[global_particle], 1e-6)
                bulk_log = matrix_bulk_log_J(F_reg, particle, Fp_sm, matrix_blend,
                                             J_ref_p2; global_particle)
                tau_matrix = K * bulk_log * I + mu * dev_be
                commit_volumetric_plastic_relax!(F_reg, particle;
                    active=!thermal_Fth_skip_Jp[], commit_blend=commit_blend,
                    matrix_blend=matrix_blend, global_particle)
                _alpha[particle] += delta_gamma

                dFp_total = Fp_sm - Fp_old

                depsilon = 0.5 * (dFp_total * Fp_old_inv + (dFp_total * Fp_old_inv)')
                # Matrix-only plastic heating (Spencer is hyperelastic / non-dissipative).
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
# Hot-melt bulk modulus — Wang 2010 Tait K = B/C @ 180°C preheat (validation 07 WCSPH set).
const matrix_K_liq = 721_327_590.0
const wcsph_hot_soften_span = 35.0    # K above melt window to relax liquid bulk stiffness smoothly
const wcsph_pressure_cap = 1.0e8        # Pa: 100 MPa hard upper bound (compression moulding)
const wcsph_visc_pair_accel_cap = 5.0e4 # m/s², pairwise cap before global liquid accel limiter
# Polymer-melt cohesion: fraction of bulk modulus applied as tensile (negative) pressure when
# particles separate (J > 1).  0.0 = no tension (original water-like behaviour),
# 0.1–0.5 = mild cohesion suitable for molten polymer that resists free-surface breakup.
const wcsph_tension_fraction = env_float("TP_CLIP_WCSPH_TENSION_FRACTION", 0.0)

@inline function wcsph_pressure_force_dims(cyl_sys)
    return wcsph_full3d_pressure ? (1:TrixiParticles.ndims(cyl_sys)) : (1, 3)
end

@inline function wcsph_pressure_smooth(J_raw, K)
    if J_raw >= 1.0
        # Linear cohesive tension with a larger cap (0.5) to provide physically significant cohesion
        wcsph_tension_fraction <= 0.0 && return 0.0
        Δ = J_raw - 1.0
        Δ_capped = min(Δ, 0.5)
        return -wcsph_tension_fraction * K * Δ_capped
    end
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
function clip_sanitize_coords_for_wcsph!(coords, cyl_sys)
    nd = TrixiParticles.ndims(cyl_sys)
    init = TrixiParticles.initial_coordinates(cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        for d in 1:nd
            x = coords[d, particle]
            if !isfinite(x)
                coords[d, particle] = init[d, particle]
            end
        end
        coords[1, particle] = clamp(coords[1, particle], -x_safety_bound, x_safety_bound)
        # Full-3D: clamp y too — runaway trial coords overflow PointNeighbors Int32 cell hashes.
        if nd >= 2
            if apply_y_plane_constraint || use_pseudo2d_projection
                coords[2, particle] = 0.0
            else
                coords[2, particle] = clamp(coords[2, particle], y_safety_min, y_safety_max)
            end
        end
        if nd >= 3
            coords[3, particle] = clamp(coords[3, particle], z_safety_min, z_safety_max)
        end
    end
    return nothing
end

function update_wcsph_current_nhs!(nhs, coords; cyl_sys=nothing, sanitize::Bool=true)
    if sanitize && cyl_sys !== nothing
        clip_sanitize_coords_for_wcsph!(coords, cyl_sys)
    end
    try
        PointNeighbors.initialize!(nhs, coords, coords)
    catch e
        # Last-resort: NaN/Inf or extreme FD trial coords can still overflow Int32 cell indices.
        e isa InexactError || rethrow(e)
        if cyl_sys !== nothing
            init = TrixiParticles.initial_coordinates(cyl_sys)
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                for d in 1:TrixiParticles.ndims(cyl_sys)
                    coords[d, particle] = init[d, particle]
                end
            end
            clip_sanitize_coords_for_wcsph!(coords, cyl_sys)
            PointNeighbors.initialize!(nhs, coords, coords)
            @warn "WCSPH NHS reset to initial coords after InexactError (runaway trial state)" maxlog=5
        else
            rethrow(e)
        end
    end
    return nhs
end

function compute_wcsph_current_density!(ρ_out, cyl_sys, coords, nhs)
    fill!(ρ_out, 0.0)
    h = smoothing_length
    h2_eps = eps(h * h)

    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        ρ_out[particle] += cyl_sys.mass[particle] *
            TrixiParticles.smoothing_kernel(cyl_sys, 0.0, particle)
    end

    PointNeighbors.foreach_point_neighbor(
        coords, coords, nhs;
        parallelization_backend=clip_parallelization_backend,
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, pos_diff, distance
        distance^2 < h2_eps && return
        w = TrixiParticles.smoothing_kernel(cyl_sys, distance, particle)
        @inbounds ρ_out[particle] += cyl_sys.mass[neighbor] * w
    end

    return ρ_out
end

function apply_wcsph_liquid_pressure!(dv_ode, cyl_sys, semi_local)
    n_p = nparticles(cyl_sys)
    nhs = clip_wcsph_nhs(semi_local)
    dv = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    h = smoothing_length
    h2_eps = eps(h * h)
    ρ0_buf = wcsph_rho0_buf
    latent_w_buf = wcsph_latent_weight_buf

    if !wcsph_liquid_density_ref_initialized[]
        ref_coords = TrixiParticles.initial_coordinates(cyl_sys)
        update_wcsph_current_nhs!(nhs, ref_coords; cyl_sys=cyl_sys, sanitize=false)
        compute_wcsph_current_density!(wcsph_liquid_density_ref_buf, cyl_sys,
                                       ref_coords, nhs)
        wcsph_liquid_density_ref_initialized[] = true
    end

    # Pass 1: current-space SPH density sum (grid NHS at current coordinates).
    current_coords = cyl_sys.current_coordinates
    update_wcsph_current_nhs!(nhs, current_coords; cyl_sys=cyl_sys)
    ρ_curr = wcsph_liquid_density_curr_buf
    compute_wcsph_current_density!(ρ_curr, cyl_sys, current_coords, nhs)

    # Pass 2: WC pressure from J = ρ_ref/ρ_curr (liquid-fraction weighted).
    p_wc = wcsph_pressure_buf
    ρ_ref_buf = wcsph_liquid_density_ref_buf
    mat_rho = cyl_sys.material_density
    temp = cyl_sys.temp
    fill!(p_wc, 0.0)
    @inbounds for i in 1:n_p
        lf = liquid_fraction_particle_buf[i]
        lf < 1e-12 && continue
        ρ_ref_i = max(ρ_ref_buf[i], 1e-12 * mat_rho[i])
        ρ0_buf[i] = ρ_ref_i
        latent_w_buf[i] = wcsph_latent_window_weight(temp[i])
        ρc = max(ρ_curr[i], 1e-12 * ρ_ref_i)
        J = ρ_ref_i / ρc
        K_liq = wcsph_bulk_scale * wcsph_liquid_bulk_modulus(temp[i])
        p_raw = lf * wcsph_pressure_smooth(J, K_liq)
        p_wc[i] = min(p_raw, wcsph_pressure_cap * lf)
    end

    # Pass 3: WCSPH symmetric pressure force at current positions (grid NHS).
    latent_force_min = wcsph_latent_force_min
    PointNeighbors.foreach_point_neighbor(
        current_coords, current_coords, nhs;
        parallelization_backend=clip_parallelization_backend,
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, pos_diff, distance
        distance^2 < h2_eps && return
        pi = @inbounds p_wc[particle]
        pj = @inbounds p_wc[neighbor]
        (pi == 0.0 && pj == 0.0) && return

        grad_W = TrixiParticles.smoothing_kernel_grad(cyl_sys, pos_diff, distance, particle)
        ρ0i = @inbounds ρ0_buf[particle]
        ρ0j = @inbounds ρ0_buf[neighbor]
        mj = @inbounds cyl_sys.mass[neighbor]

        coeff = pi / (ρ0i * ρ0i) + pj / (ρ0j * ρ0j)
        w_i = @inbounds latent_w_buf[particle]
        w_j = @inbounds latent_w_buf[neighbor]
        latent_scale = 1.0 - (1.0 - latent_force_min) * (w_i > w_j ? w_i : w_j)
        coeff *= latent_scale

        pressure_dims = wcsph_pressure_force_dims(cyl_sys)
        @inbounds for d in pressure_dims
            dv[d, particle] -= mj * coeff * grad_W[d]
        end
    end

    scale = wcsph_ramp_scale[]
    if scale < 1.0 - 1.0e-12
        pressure_dims = wcsph_pressure_force_dims(cyl_sys)
        @inbounds for i in 1:n_p
            for d in pressure_dims
                dv[d, i] *= scale
            end
        end
    end

    return ρ_curr
end

function apply_wcsph_bulk_viscosity!(dv_ode, cyl_sys, semi_local)
    ζ0 = wcsph_bulk_viscosity_pa_s
    ζ0 <= 0.0 && return nothing
    dv = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    nd = TrixiParticles.ndims(cyl_sys)
    scale = wcsph_ramp_scale[]
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        solid_fraction_particle_buf[particle] >= solid_mechanics_solid_fraction_min && continue
        lf = liquid_fraction_particle_buf[particle]
        lf < 1e-12 && continue
        L11 = vel_grad_buf[1, 1, particle]
        L22 = vel_grad_buf[2, 2, particle]
        L33 = vel_grad_buf[3, 3, particle]
        div_v = L11 + L22 + L33
        ζ = ζ0 * lf * scale
        for d in 1:nd
            dv[d, particle] -= ζ * div_v
        end
    end
    return nothing
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
    nhs = clip_wcsph_nhs(semi_local)
    dv = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    v = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    h = smoothing_length
    h2_eps = eps(h * h)
    current_coords = cyl_sys.current_coordinates
    update_wcsph_current_nhs!(nhs, current_coords; cyl_sys=cyl_sys)
    nd = TrixiParticles.ndims(cyl_sys)
    force_dims = wcsph_full3d_pressure ? (1:nd) : (1, 3)

    lf_buf = liquid_fraction_particle_buf
    rho_inv_buf = wcsph_rho0_buf
    @inbounds for i in 1:n_p
        rho_inv_buf[i] = 1.0 / max(ρ_curr[i], 1e-12)
    end

    L_curr = wcsph_L_curr_buf
    fill!(L_curr, 0.0)

    PointNeighbors.foreach_point_neighbor(
        current_coords, current_coords, nhs;
        parallelization_backend=clip_parallelization_backend,
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, pos_diff, distance
        distance^2 < h2_eps && return
        @inbounds lf_i = lf_buf[particle]
        lf_i < 1e-12 && return

        grad_W = TrixiParticles.smoothing_kernel_grad(cyl_sys, pos_diff, distance, particle)
        fac = @inbounds cyl_sys.mass[neighbor] * rho_inv_buf[particle]

        if wcsph_full3d_pressure
            @inbounds for α in 1:nd, β in 1:nd
                dα = v[α, neighbor] - v[α, particle]
                L_curr[α, β, particle] += fac * dα * grad_W[β]
            end
        else
            dvx = @inbounds v[1, neighbor] - v[1, particle]
            dvz = @inbounds v[3, neighbor] - v[3, particle]
            @inbounds begin
                L_curr[1, 1, particle] += fac * dvx * grad_W[1]
                L_curr[1, 2, particle] += fac * dvx * grad_W[3]
                L_curr[2, 1, particle] += fac * dvz * grad_W[1]
                L_curr[2, 2, particle] += fac * dvz * grad_W[3]
            end
        end
    end

    tau_vis = wcsph_tau_vis_buf
    fill!(tau_vis, 0.0)
    @inbounds for i in 1:n_p
        lf_i = liquid_fraction_particle_buf[i]
        lf_i < 1e-12 && continue
        η = vis_particle_buf[i] * lf_i

        if wcsph_full3d_pressure
            L = StaticArrays.SMatrix{3, 3}(@view L_curr[:, :, i])
            D = 0.5 * (L + L')
            div_v = tr(D)
            dev = D - div_v / 3.0 * StaticArrays.SMatrix{3, 3}(I)
            tau = 2.0 * η * dev
            τ_norm = sqrt(sum(abs2, tau))
            if τ_norm > matrix_yield_stress
                tau = tau * (matrix_yield_stress / τ_norm)
            end
            @inbounds for α in 1:nd, β in 1:nd
                tau_vis[α, β, i] = tau[α, β]
            end
        else
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
    end

    visc_accel_cap = wcsph_visc_pair_accel_cap
    PointNeighbors.foreach_point_neighbor(
        current_coords, current_coords, nhs;
        parallelization_backend=clip_parallelization_backend,
        points=TrixiParticles.each_integrated_particle(cyl_sys)
    ) do particle, neighbor, pos_diff, distance
        distance^2 < h2_eps && return
        @inbounds begin
            lf_i = lf_buf[particle]
            lf_j = lf_buf[neighbor]
        end
        (lf_i < 1e-12 && lf_j < 1e-12) && return

        grad_W = TrixiParticles.smoothing_kernel_grad(cyl_sys, pos_diff, distance, particle)
        @inbounds begin
            ρi2_inv = rho_inv_buf[particle] * rho_inv_buf[particle]
            ρj2_inv = rho_inv_buf[neighbor] * rho_inv_buf[neighbor]
            mj = cyl_sys.mass[neighbor]
        end

        ai_x = 0.0
        ai_y = 0.0
        ai_z = 0.0
        @inbounds for α in force_dims
            acc = 0.0
            for β in force_dims
                acc += (tau_vis[α, β, particle] * ρi2_inv +
                        tau_vis[α, β, neighbor] * ρj2_inv) * grad_W[β]
            end
            if α == 1
                ai_x = mj * acc
            elseif α == 2
                ai_y = mj * acc
            else
                ai_z = mj * acc
            end
        end
        ai_norm = sqrt(ai_x * ai_x + ai_y * ai_y + ai_z * ai_z)
        if ai_norm > visc_accel_cap
            s = visc_accel_cap / ai_norm
            ai_x *= s
            ai_y *= s
            ai_z *= s
        end
        @inbounds for α in force_dims
            if α == 1
                dv[1, particle] += ai_x
            elseif α == 2
                dv[2, particle] += ai_y
            else
                dv[3, particle] += ai_z
            end
        end
    end

    scale = wcsph_ramp_scale[]
    if scale < 1.0 - 1.0e-12
        @inbounds for i in 1:n_p
            for d in force_dims
                dv[d, i] *= scale
            end
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
    clip_sync_trial_dt_state!()
    TrixiParticles.set_zero!(dv_ode)
    rhs_eval_count[] += 1
    if rhs_heartbeat_interval > 0 && rhs_eval_count[] % rhs_heartbeat_interval == 0
        println(">>> RHS eval #", rhs_eval_count[], "  t=", round(t; digits=9),
                "  trial_dt=", trial_dt_state[], " s")
        flush(stdout)
    end
    hold_tlsph_equilibration_now[] = hold_tlsph_tail_active(t) ||
        retraction_pre_lift_creep_active(t)

    if retraction_cfrp_style_active()
        return kick_implicit_cfrp_retraction!(dv_ode, v_ode, u_ode, semi_local, t)
    end

    cyl_sys = semi_local.systems[1]
    v_cyl = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
    NDIMS_CYL = TrixiParticles.ndims(cyl_sys)

    # Pseudo-2D only: hard-zero v_y before force evaluation.
    # Thin-3D uses a spring-damper in compute_implicit_rhs! instead so that
    # AutoFiniteDiff sees ∂(dv_y)/∂v_y = -y_constraint_damp ≠ 0  (non-singular Jacobian).
    if hard_lock_y_kinematics()
        TrixiParticles.foreach_system(semi_local) do system
            v_sys = TrixiParticles.wrap_v(v_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                v_sys[2, particle] = 0.0
            end
        end
    end

    if separate_charge_plies
        clip_foreach_charge_system(semi_local) do system, _
            v_charge = TrixiParticles.wrap_v(v_ode, system, semi_local)
            ndims_charge = TrixiParticles.ndims(system)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                if apply_velocity_safety_clamps()
                    v_charge[1, particle] = soft_limit(v_charge[1, particle],
                                                       max_cylinder_speed)
                    v_charge[3, particle] = soft_limit(v_charge[3, particle],
                                                       max_cylinder_speed)
                end
                T_floor = ambient_heat_loss_active_now() ? max(temp_ambient, temp_min_clip) : temp_min_clip
                t_trial = v_charge[ndims_charge + 1, particle]
                if !isfinite(t_trial)
                    t_trial = system.temp[particle]
                end
                t_clamped = clamp(t_trial, T_floor, temp_max_clip)
                system.temp[particle] = t_clamped
                v_charge[ndims_charge + 1, particle] = t_clamped
            end
        end
    else
        T_floor = ambient_heat_loss_active_now() ? max(temp_ambient, temp_min_clip) : temp_min_clip
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            if apply_velocity_safety_clamps()
                v_cyl[1, particle] = soft_limit(v_cyl[1, particle], max_cylinder_speed)
                v_cyl[3, particle] = soft_limit(v_cyl[3, particle], max_cylinder_speed)
            end
            t_trial = v_cyl[NDIMS_CYL + 1, particle]
            if !isfinite(t_trial)
                t_trial = cyl_sys.temp[particle]
            end
            t_clamped = clamp(t_trial, T_floor, temp_max_clip)
            cyl_sys.temp[particle] = t_clamped
            v_cyl[NDIMS_CYL + 1, particle] = t_clamped
        end
    end
    apply_retraction_full3d_y_velocity_cap!(v_cyl, cyl_sys)

    if charge_hold_thermal_only(t)
        return apply_charge_hold_rhs!(dv_ode, v_ode, u_ode, semi_local, t)
    end

    floor_sys = clip_floor_system(semi_local)
    mold_sys  = clip_mold_system(semi_local)

    force_stage_diag = force_stage_diag_enabled &&
                       t > diag_force_stage_start_t &&
                       t > diag_force_stage_last_t[] + diag_force_stage_dt
    force_stage_prev = force_stage_diag ?
                       copy(TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)) :
                       nothing
    if force_stage_diag
        diag_force_stage_last_t[] = t
        println(stderr, "\n>>> FORCE-STAGE-DIAG t=", round(t; digits=9),
                " trial_dt=", trial_dt_state[],
                " n_particles=", nparticles(cyl_sys))
    end

    try
        t_pos_start = time_ns()
        clip_sync_tool_kinematics_gpu!(semi_local, t)
        # Charge positions depend on trial `u` and must update every RHS.
        # Tool positions/kinematics depend only on `t` and can be cached across repeated RHS
        # evaluations at the same `t` (Newton iterations, Jacobian fills).
        do_tool_pos = !clip_tool_pos_quant_cache || (t != clip_tool_pos_updated_at_t[])
        for (sys_idx, system) in enumerate(semi_local.systems)
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            if clip_is_charge_system_index(sys_idx) || do_tool_pos
                TrixiParticles.update_positions!(system, v, u, v_ode, u_ode, semi_local, t)
            end
        end
        do_tool_pos && (clip_tool_pos_updated_at_t[] = t)
        # Ensure peel floor/die open shift is present even if PrescribedMotion AD path misfires.
        if retraction_peel_floor_open &&
           (retraction_floor_open_active[] || retraction_floor_open_z_shift[] != 0.0)
            apply_floor_open_z_shift!(clip_floor_system(semi_local))
        end

        # Safety clamp for charge coordinates during nonlinear iterations.
        if apply_coordinate_safety_clamps() || retraction_full3d_y_cap_active()
            coords_cyl = cyl_sys.current_coordinates
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                if apply_coordinate_safety_clamps()
                    coords_cyl[1, particle] = clamp(coords_cyl[1, particle],
                                                    -x_safety_bound, x_safety_bound)
                    coords_cyl[3, particle] = clamp(coords_cyl[3, particle],
                                                      z_safety_min, z_safety_max)
                    if use_pseudo2d_projection
                        coords_cyl[2, particle] = 0.0
                    end
                end
                if retraction_full3d_y_cap_active()
                    coords_cyl[2, particle] = clamp_charge_y_coordinate(
                        coords_cyl[2, particle], cyl_sys, particle)
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
            if retraction_pre_lift_creep_s > 0.0
                creep_free = env_int("TP_CLIP_RETRACTION_CREEP_RAMP_FREE", 0) != 0
                println(">>> Pre-lift physics at t=", round(t; digits=6),
                        retraction_closed_die_contact_settle ?
                            " s (CONTACT settle: kin ACTIVE, σ=0, contact ON; punch lift later)" :
                        creep_free ?
                            " s (σ/contact/kin INSTANT; mold fixed; hold_creep ON; punch lift later)" :
                        retraction_cool_before_lift ?
                            (" s (σ/contact over " *
                             string(round(retraction_pre_lift_creep_s * retraction_pre_lift_elastic_frac; digits=4)) *
                             " s; kin FROZEN until punch lift; mold fixed; hold_creep ON)") :
                            (" s (σ/contact over " *
                             string(round(retraction_pre_lift_creep_s * retraction_pre_lift_elastic_frac; digits=4)) *
                             " s; kin over " * string(round(retraction_pre_lift_creep_s; digits=4)) *
                             " s; mold fixed; hold_creep ON)"))
            else
                println(">>> Retraction at t=", round(t; digits=6),
                        " s (contact ",
                        retraction_mech_ramp_instant ? "instant-on" :
                            "ramp " * string(retraction_hold_release_s) * " s",
                        "; elastic σ ",
                        retraction_elastic_ramp_instant ? "instant-on" :
                            "ramp " * string(retraction_elastic_ramp_s) * " s",
                        "; kinematics ramp ", retraction_kinematics_release_s,
                        " s; mold lift kin-scaled; solid TLSPH only)")
            end
            flush(stdout)
        end
        wcsph_ramp_scale[] = wcsph_mechanics_active() ?
                             charge_wcsph_ramp_scale_value(cyl_sys) : 0.0

        t_nhs_start = time_ns()
        # Colored/sparse Jacobian: optional per-perturbation NHS refresh (TP_CLIP_JAC_NHS_REFRESH=1).
        # Default freezes NHS at the trial state, matching dense AutoFiniteDiff.
        if clip_in_jacobian_fill[] && clip_jac_nhs_refresh
            TrixiParticles.update_nhs!(semi_local, u_ode)
        elseif t != nhs_updated_at_t[] || (clip_ode_flat[] && CURRENT_JACOBIAN_STRATEGY[] == "matrix_free")
            TrixiParticles.update_nhs!(semi_local, u_ode)
            nhs_updated_at_t[] = t
        end
        rhs_time_nhs_ns[] += time_ns() - t_nhs_start

        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] =
            clip_deformation_only_selector(semi_local)

        t_quant_start = time_ns()
        do_tool_quant = !clip_tool_pos_quant_cache || (t != clip_tool_quant_updated_at_t[])
        for (sys_idx, system) in enumerate(semi_local.systems)
            v = TrixiParticles.wrap_v(v_ode, system, semi_local)
            u = TrixiParticles.wrap_u(u_ode, system, semi_local)
            if clip_is_charge_system_index(sys_idx) || do_tool_quant
                TrixiParticles.update_quantities!(system, v, u, v_ode, u_ode, semi_local, t)
            end
        end
        do_tool_quant && (clip_tool_quant_updated_at_t[] = t)
        rhs_time_quant_ns[] += time_ns() - t_quant_start

        apply_all_charge_plane_strain_F_fixes!(semi_local)

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
            deform_cyl = cyl_sys.deformation_grad
            coords_cyl = cyl_sys.current_coordinates
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                F11 = deform_cyl[1, 1, particle]
                F12 = deform_cyl[1, 2, particle]
                F13 = deform_cyl[1, 3, particle]
                F21 = deform_cyl[2, 1, particle]
                F22 = deform_cyl[2, 2, particle]
                F23 = deform_cyl[2, 3, particle]
                F31 = deform_cyl[3, 1, particle]
                F32 = deform_cyl[3, 2, particle]
                F33 = deform_cyl[3, 3, particle]
                J = F11 * (F22 * F33 - F23 * F32) -
                    F12 * (F21 * F33 - F23 * F31) +
                    F13 * (F21 * F32 - F22 * F31)
                if J < worst_J
                    worst_J = J
                    worst_J_particle = particle
                end
                if !isfinite(J) || J <= J_min || J >= J_max
                    deform_cyl[1, 1, particle] = 1.0
                    deform_cyl[1, 2, particle] = 0.0
                    deform_cyl[1, 3, particle] = 0.0
                    deform_cyl[2, 1, particle] = 0.0
                    deform_cyl[2, 2, particle] = 1.0
                    deform_cyl[2, 3, particle] = 0.0
                    deform_cyl[3, 1, particle] = 0.0
                    deform_cyl[3, 2, particle] = 0.0
                    deform_cyl[3, 3, particle] = 1.0
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
                    elseif retraction_full3d_y_cap_active()
                        u_cyl[2, particle] = clamp_charge_y_coordinate(
                            u_cyl[2, particle], cyl_sys, particle)
                    end
                    u_cyl[3, particle] = clamp(u_cyl[3, particle], z_safety_min,
                                               z_safety_max)

                    coords_cyl[1, particle] = clamp(coords_cyl[1, particle],
                                                    -x_safety_bound,
                                                    x_safety_bound)
                    if use_pseudo2d_projection
                        coords_cyl[2, particle] = 0.0
                    elseif retraction_full3d_y_cap_active()
                        coords_cyl[2, particle] = u_cyl[2, particle]
                    end
                    coords_cyl[3, particle] = clamp(coords_cyl[3, particle],
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
        # Cap internal/stress |a| before contact; deep contact remains uncapped.
        apply_retraction_internal_accel_cap!(dv_ode, cyl_sys, semi_local)
        snapshot_charge_dv!(clip_contact_dv_pre_snap, dv_ode, cyl_sys, semi_local)
        if clip_contact_freeze_restore_active(t)
            clip_apply_cached_contact_increment!(dv_ode, cyl_sys, semi_local)
        else
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
                if wcsph_bulk_viscosity_pa_s > 0.0
                    apply_wcsph_bulk_viscosity!(dv_ode, cyl_sys, semi_local)
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
            clip_capture_contact_increment!(dv_ode, clip_contact_dv_pre_snap, cyl_sys, semi_local, t)
        end

        blend_charge_dv_from_marker!(dv_ode, cyl_sys, semi_local,
                                     charge_dv_mech_marker_buf, mech_dv_scale)

        # ── STIFFNESS DIAGNOSTIC (TP_CLIP_STIFF_DIAG / TP_CLIP_STALL_DIAG) ─────────
        if stiff_diag_enabled && t >= stiff_diag_start_t && t > diag_stiff_last_t[] + 1e-4
            diag_stiff_last_t[] = t
            n_diag = nparticles(cyl_sys)

            max_sig_e = 0.0; pid_e = 0
            max_sig_v = 0.0; pid_v = 0
            max_dv_x  = 0.0; pid_dv = 0
            max_p_wc  = 0.0; pid_p = 0
            min_J_wc  = Inf
            max_J_wc  = 0.0

            dv_diag = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
            NDIMS_D = TrixiParticles.ndims(cyl_sys)

            @inbounds for p in 1:n_diag
                sig_e = 0.0; sig_v = 0.0
                for jj in 1:3, ii in 1:3
                    de = v_stress_buf[ii, jj, p] - v_stress_viscous_buf[ii, jj, p]
                    sig_e += de * de
                    sig_v += v_stress_viscous_buf[ii, jj, p]^2
                end
                sig_e = sqrt(sig_e); sig_v = sqrt(sig_v)
                if sig_e > max_sig_e; max_sig_e = sig_e; pid_e = p; end
                if sig_v > max_sig_v; max_sig_v = sig_v; pid_v = p; end

                dvp = sqrt(dv_diag[1, p]^2 + dv_diag[NDIMS_D, p]^2)
                if dvp > max_dv_x; max_dv_x = dvp; pid_dv = p; end

                if wcsph_pressure_buf[p] > max_p_wc
                    max_p_wc = wcsph_pressure_buf[p]
                    pid_p = p
                end
                if wcsph_liquid_density_ref_initialized[]
                    ρ_ref_i = max(wcsph_liquid_density_ref_buf[p],
                                  1e-12 * cyl_sys.material_density[p])
                    ρc = max(wcsph_liquid_density_curr_buf[p], 1e-12 * ρ_ref_i)
                    J_wc = ρ_ref_i / ρc
                    min_J_wc = min(min_J_wc, J_wc)
                    max_J_wc = max(max_J_wc, J_wc)
                end
            end

            println(stderr, "\n>>> STIFF-DIAG  t=", round(t; digits=9),
                    "  n_particles=", n_diag,
                    "  WCSPH J∈[", isfinite(min_J_wc) ? round(min_J_wc; digits=4) : "n/a",
                    ", ", isfinite(max_J_wc) ? round(max_J_wc; digits=4) : "n/a", "]",
                    "  p_wc_max=", round(max_p_wc / 1e6; digits=1), " MPa")

            if pid_e > 0
                p = pid_e
                F11 = cyl_sys.deformation_grad[1, 1, p]; F13 = cyl_sys.deformation_grad[1, 3, p]
                F31 = cyl_sys.deformation_grad[3, 1, p]; F33 = cyl_sys.deformation_grad[3, 3, p]
                J_p = F11 * F33 - F13 * F31
                Fp_p = Fp_committed[]
                Fpe11 = Fp_p[1, 1, p]; Fpe13 = Fp_p[1, 3, p]
                Fpe31 = Fp_p[3, 1, p]; Fpe33 = Fp_p[3, 3, p]
                det_Fp = Fpe11 * Fpe33 - Fpe13 * Fpe31
                n_ref = StaticArrays.SVector(fiber_direction[1, p], fiber_direction[2, p],
                                           fiber_direction[3, p])
                F3 = StaticArrays.SMatrix{3, 3}(
                    cyl_sys.deformation_grad[1, 1, p], cyl_sys.deformation_grad[2, 1, p],
                    cyl_sys.deformation_grad[3, 1, p], cyl_sys.deformation_grad[1, 2, p],
                    cyl_sys.deformation_grad[2, 2, p], cyl_sys.deformation_grad[3, 2, p],
                    cyl_sys.deformation_grad[1, 3, p], cyl_sys.deformation_grad[2, 3, p],
                    cyl_sys.deformation_grad[3, 3, p])
                Fn_p = F3 * n_ref
                I4_p = dot(Fn_p, Fn_p)
                println(stderr, "  MAX ELASTIC  p=", p,
                        "  |σ_e|=", round(max_sig_e / 1e6; digits=1), " MPa",
                        "  T=", round(cyl_sys.temp[p]; digits=1), " K",
                        "  lf=", round(liquid_fraction_particle_buf[p]; digits=3),
                        "  sf=", round(solid_fraction_particle_buf[p]; digits=3),
                        "  J=", round(J_p; digits=4),
                        "  det(Fp)=", round(det_Fp; digits=4),
                        "  I4=", round(I4_p; digits=4),
                        "  k_fiber=", round(k_fiber_particle[p] / 1e9; digits=2), " GPa")
            end

            if pid_v > 0
                p = pid_v
                println(stderr, "  MAX VISCOUS  p=", p,
                        "  |σ_v|=", round(max_sig_v / 1e6; digits=1), " MPa",
                        "  T=", round(cyl_sys.temp[p]; digits=1), " K",
                        "  lf=", round(liquid_fraction_particle_buf[p]; digits=3),
                        "  vis=", round(vis_particle_buf[p]; digits=1), " Pa·s")
            end

            if pid_p > 0
                p = pid_p
                println(stderr, "  MAX WCSPH p  p=", p,
                        "  p=", round(wcsph_pressure_buf[p] / 1e6; digits=1), " MPa",
                        "  lf=", round(liquid_fraction_particle_buf[p]; digits=3),
                        "  T=", round(cyl_sys.temp[p]; digits=1), " K")
            end

            if pid_dv > 0
                p = pid_dv
                println(stderr, "  MAX |dv|     p=", p,
                        "  |dv|=", round(max_dv_x; digits=2), " m/s²",
                        "  T=", round(cyl_sys.temp[p]; digits=1), " K",
                        "  lf=", round(liquid_fraction_particle_buf[p]; digits=3),
                        "  pos=(", round(cyl_sys.current_coordinates[1, p] * 1e3; digits=2),
                        ", ", round(cyl_sys.current_coordinates[3, p] * 1e3; digits=2), ") mm")
            end

            sig_e_all = Vector{Tuple{Float64, Int}}(undef, n_diag)
            @inbounds for pp in 1:n_diag
                se = 0.0
                for jj in 1:3, ii in 1:3
                    de = v_stress_buf[ii, jj, pp] - v_stress_viscous_buf[ii, jj, pp]
                    se += de * de
                end
                sig_e_all[pp] = (sqrt(se), pp)
            end
            sort!(sig_e_all; by=x -> x[1], rev=true)
            println(stderr, "  TOP-5 elastic stress particles (MPa / T / lf):")
            for (se, pp) in sig_e_all[1:min(5, n_diag)]
                println(stderr, "    p=", pp,
                        "  |σ_e|=", round(se / 1e6; digits=1),
                        "  T=", round(cyl_sys.temp[pp]; digits=1),
                        "  lf=", round(liquid_fraction_particle_buf[pp]; digits=3))
            end
            flush(stderr)
        end
        # ── END STIFFNESS DIAGNOSTIC ────────────────────────────────────────────────

        rhs_time_stress_interact_ns[] += time_ns() - t_stress_interact_start
    finally
        TrixiParticles.TLSPH_DEFORMATION_GRAD_ONLY_SYSTEM[] = nothing
        TrixiParticles.STRESS_TENSOR_CACHE[] = nothing
    end

    contact_damp_dist = 0.8 * smoothing_length
    dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    floor_sys = clip_floor_system(semi_local)
    mold_sys  = clip_mold_system(semi_local)
    if retraction_thermal_frozen_now()
        clip_foreach_charge_system(semi_local) do system, _
            dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
            nd = TrixiParticles.ndims(system)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                dv_sys[nd + 1, particle] = 0.0
            end
        end
    else
        t_thermal_start = time_ns()
        if clip_thermal_freeze_restore!(dv_cyl, cyl_sys, t)
            if force_stage_diag
                print_force_stage_delta!("thermal_and_contact_heat", force_stage_prev, dv_ode,
                                         cyl_sys, semi_local, t)
            end
            rhs_time_thermal_ns[] += time_ns() - t_thermal_start
        else
            # Tool–charge HTC is off after eject: mechanically removed tools must not
            # reheat the freely cooling workpiece.
            t_contact_heat_start = time_ns()
            if !retraction_skip_sph_conduction_now() && !retraction_post_eject_phase()
                if separate_charge_plies
                    fill!(contact_heat_flux_buf, 0.0)
                    clip_foreach_charge_system(semi_local) do system, source
                        compute_contact_heat_flux!(contact_heat_flux_buf, system,
                                                   floor_sys, mold_sys, particle_spacing, t;
                                                   source_indices=source, reset=false)
                    end
                    contact_heat_flux = contact_heat_flux_buf
                else
                    contact_heat_flux = compute_contact_heat_flux!(contact_heat_flux_buf, cyl_sys,
                                                                   floor_sys, mold_sys, particle_spacing, t)
                end
            else
                fill!(contact_heat_flux_buf, 0.0)
                contact_heat_flux = contact_heat_flux_buf
            end
            rhs_time_contact_heat_ns[] += time_ns() - t_contact_heat_start

            t_thermal_sph_start = time_ns()
            if !retraction_skip_sph_conduction_now()
                clip_foreach_charge_system(semi_local) do system, _
                    dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
                    v_sys = TrixiParticles.wrap_v(v_ode, system, semi_local)
                    TrixiParticles.thermal_rhs_sph3d!(system, dv_sys, v_sys, 0.0,
                                                      particle_spacing, bound_coordinate_thermal, semi_local)
                end
            end
            rhs_time_thermal_sph_ns[] += time_ns() - t_thermal_sph_start
            if use_nakamura_kinetics && !retraction_solid_mechanics_only()
                t_nak_rhs_start = time_ns()
            end
            clip_foreach_charge_system(semi_local) do system, source
                dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
                nd = TrixiParticles.ndims(system)
                @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                    g = source[particle]
                    T_i = system.temp[particle]
                    cp_eff = effective_heat_capacity_particle(T_i, g)
                    dv_sys[nd + 1, particle] *= system.cp / cp_eff
                    if use_nakamura_kinetics && !retraction_solid_mechanics_only()
                        α_i = crystallinity_particle_buf[g]
                        dα_dt = nakamura_dalpha_dt(clamp(T_i, temp_min_clip, temp_max_clip), α_i)
                        L_i = latent_heat_particle[g]
                        cp_i = cp_particle[g]
                        dv_sys[nd + 1, particle] += L_i * dα_dt / cp_i
                    end
                    if contact_heat_flux[g] != 0.0
                        rho_i = physical_thermal_density(system, particle)
                        dv_sys[nd + 1, particle] -= contact_heat_flux[g] /
                                                    (rho_i * cp_eff * particle_spacing)
                    end
                    if ambient_heat_loss_active_now()
                        exposure = free_cool_ambient_exposure(g)
                        if exposure > 0.0
                            rho_i = physical_thermal_density(system, particle)
                            dT_dt_loss = compute_ambient_cooling_rate(T_i, exposure, rho_i, cp_eff, particle_spacing, t)
                            dv_sys[nd + 1, particle] -= dT_dt_loss
                        end
                    end
                end
            end
            if use_nakamura_kinetics && !retraction_solid_mechanics_only()
                rhs_time_nakamura_rhs_ns[] += time_ns() - t_nak_rhs_start
            end
            clip_thermal_freeze_capture!(dv_cyl, cyl_sys, t)
            if force_stage_diag
                print_force_stage_delta!("thermal_and_contact_heat", force_stage_prev, dv_ode,
                                         cyl_sys, semi_local, t)
            end
            rhs_time_thermal_ns[] += time_ns() - t_thermal_start
        end
        dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
    end

    if tool_wall_coupling_active(t) && !imex_soft_terms
        bh_sys = blankholder_neighbor_system[]
        resting_buf = charge_resting_on_shelf_buf[]
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            charge_particle_is_flyaway(particle) && continue
            x_i = cyl_sys.current_coordinates[1, particle]
            z_i = cyl_sys.current_coordinates[3, particle]
            gap_floor_drag = tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower)
            gap_mold_drag = tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper)
            near_floor = gap_floor_drag <= contact_damp_dist
            near_mold = gap_mold_drag <= contact_damp_dist
            near_blankholder = false
            gap_bh_drag = Inf
            if bh_sys !== nothing && resting_buf !== nothing && particle <= length(resting_buf) &&
               resting_buf[particle]
                gap_bh_drag = tool_surface_gap(bh_sys, x_i, z_i, particle_spacing; side=:upper)
                near_blankholder = gap_bh_drag <= contact_damp_dist
            end

            # Molten wall no-slip drag (off during solid retraction).
            lf_wall = liquid_fraction_particle_buf[particle]
            if !retraction_solid_mechanics_only() && lf_wall > 1.0e-4 &&
               (near_floor || near_mold || near_blankholder)
                ρ_i = cyl_sys.material_density[particle]
                η_i = vis_particle_buf[particle]
                drag_rate = molten_wall_drag_factor * lf_wall * η_i /
                            (ρ_i * smoothing_length * smoothing_length)
                drag_rate = min(drag_rate, molten_wall_drag_max_rate)

                if near_floor
                    act = 1.0 - clamp(gap_floor_drag / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)
                    act = act * act * (3.0 - 2.0 * act)
                    dv_cyl[1, particle] -= act * drag_rate * v_cyl[1, particle]
                    dv_cyl[2, particle] -= act * drag_rate * v_cyl[2, particle]
                    dv_cyl[3, particle] -= act * drag_rate * v_cyl[3, particle]
                end

                if near_mold
                    act = 1.0 - clamp(gap_mold_drag / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)
                    act = act * act * (3.0 - 2.0 * act)
                    v_wall_z = mold_z_velocity_at_time(t)
                    dv_cyl[1, particle] -= act * drag_rate * v_cyl[1, particle]
                    dv_cyl[2, particle] -= act * drag_rate * v_cyl[2, particle]
                    dv_cyl[3, particle] -= act * drag_rate * (v_cyl[3, particle] - v_wall_z)
                end

                if near_blankholder
                    act = 1.0 - clamp(gap_bh_drag / max(contact_damp_dist, eps(Float64)), 0.0, 1.0)
                    act = act * act * (3.0 - 2.0 * act)
                    dv_cyl[1, particle] -= act * drag_rate * v_cyl[1, particle]
                    dv_cyl[2, particle] -= act * drag_rate * v_cyl[2, particle]
                    dv_cyl[3, particle] -= act * drag_rate * v_cyl[3, particle]
                end
            end

            # Coulomb friction at tool–part contact (solid-fraction weighted).
            if near_floor || near_mold || near_blankholder
                sf = solid_fraction_particle_buf[particle]
                if sf > 1.0e-4
                    mu_eff   = mu_friction * sf
                    p_n      = max(0.0, -v_stress_buf[3, 3, particle])
                    rho_i    = cyl_sys.material_density[particle]
                    a_fric   = mu_eff * p_n / (rho_i * particle_spacing)
                    vx       = v_cyl[1, particle]
                    vy       = v_cyl[2, particle]
                    v_tan    = hypot(vx, vy)
                    if v_tan > 1.0e-12
                        dv_cyl[1, particle] -= a_fric * (vx / v_tan)
                        dv_cyl[2, particle] -= a_fric * (vy / v_tan)
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
    update_retraction_deep_contact_kick_flag!(cyl_sys, floor_sys, mold_sys)
    if retraction_post_eject_phase() && retraction_gravity_after_eject != 0.0
        g = retraction_gravity_after_eject
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[3, particle] -= g
        end
    end
    apply_retraction_visco_rayleigh_damping!(dv_cyl, v_cyl, cyl_sys, floor_sys, mold_sys, t)
    apply_retraction_explicit_kick_soft_limit!(dv_cyl, cyl_sys, t)
    apply_retraction_deep_contact_velocity_hard_clamp!(v_cyl, cyl_sys)
    apply_retraction_full3d_y_accel_cap!(dv_cyl, cyl_sys, t)
    apply_charge_flyaway_freeze!(dv_cyl, v_cyl, cyl_sys)
    @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
        # Liquid/mushy-only acceleration guard: keep WCSPH force spikes from
        # forcing TRBDF2 into nanosecond dt around the melt transition.
        if solid_fraction_particle_buf[particle] < solid_mechanics_solid_fraction_min
            dv_cyl[1, particle] = soft_limit(dv_cyl[1, particle], max_liquid_accel)
            if wcsph_full3d_pressure || !apply_y_plane_constraint
                dv_cyl[2, particle] = soft_limit(dv_cyl[2, particle], max_liquid_accel)
            end
            dv_cyl[3, particle] = soft_limit(dv_cyl[3, particle], max_liquid_accel)
        end
    end
    if force_stage_diag
        print_force_stage_delta!("friction_and_accel_caps", force_stage_prev, dv_ode,
                                 cyl_sys, semi_local, t)
    end
    if hard_lock_y_kinematics()
        # Hard zero v_y/du_y/dv_y: pseudo-2D collapses to y=0; thin-3D retraction keeps layer planes.
        TrixiParticles.foreach_system(semi_local) do system
            dv_sys = TrixiParticles.wrap_v(dv_ode, system, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                dv_sys[2, particle] = 0.0
            end
        end
    elseif apply_y_plane_constraint && geometry_mode == :thin3d
        # Thin-3D: stiff spring-damper keeps each particle at its initial y-plane.
        # dv_y += -γ·v_y - k·dy  where dy = current_y - initial_y (displacement, not absolute coord).
        # Gives TRBDF2 a non-zero Jacobian column → Newton converges cleanly.
        # TRBDF2's L-stability damps y-motion to zero within 1–2 steps.
        y_damp, y_spring = thin3d_y_constraint_coeffs()
        TrixiParticles.foreach_system(semi_local) do system
            v_sys  = TrixiParticles.wrap_v(v_ode,   system, semi_local)
            u_sys  = TrixiParticles.wrap_u(u_ode,   system, semi_local)
            dv_sys = TrixiParticles.wrap_v(dv_ode,  system, semi_local)
            y0 = TrixiParticles.initial_coordinates(system)  # [dim, particle], reference positions
            @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                dy = u_sys[2, particle] - y0[2, particle]  # displacement from initial y-plane
                dv_sys[2, particle] += -y_damp * v_sys[2, particle] - y_spring * dy
            end
        end
        # Spring runs after xz caps and is unbounded — clamp y for explicit stability.
        apply_retraction_charge_accel_caps!(dv_cyl, cyl_sys, t; cap_y=true)
    end
    if apply_resting_shelf_z_constraint
        resting_buf = charge_resting_on_shelf_buf[]
        if resting_buf !== nothing
            z_damp, z_spring = resting_shelf_z_constraint_coeffs()
            v_cyl_z = TrixiParticles.wrap_v(v_ode, cyl_sys, semi_local)
            u_cyl_z = TrixiParticles.wrap_u(u_ode, cyl_sys, semi_local)
            z0 = TrixiParticles.initial_coordinates(cyl_sys)
            r_shelf = if haskey(ENV, "TP_CLIP_SHELF_R_MIN_MM")
                env_float("TP_CLIP_SHELF_R_MIN_MM", 21.125) * 1.0e-3
            elseif haskey(ENV, "TP_CLIP_CAD_DIR") && occursin("hemisphere", ENV["TP_CLIP_CAD_DIR"])
                21.125e-3
            else
                0.0
            end
            is_circular = r_shelf > 0.0
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                particle > length(resting_buf) && continue
                resting_buf[particle] || continue
                if is_circular
                    r_curr = hypot(u_cyl_z[1, particle], u_cyl_z[2, particle])
                    r_curr < r_shelf && continue
                end
                dz = u_cyl_z[3, particle] - z0[3, particle]
                dv_cyl[3, particle] += -z_damp * v_cyl_z[3, particle] - z_spring * dz
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
            if hard_lock_y_kinematics() || (apply_y_plane_constraint && geometry_mode == :thin3d)
                dv_cyl[2, particle] *= kin_scale
            end
            dv_cyl[3, particle] *= kin_scale
        end
    end

    # #region agent log
    agent_debug_peel_rhs!(dv_ode, cyl_sys, semi_local, floor_sys, mold_sys, t;
                          stage="pre_freeze_zero")
    # #endregion

    if (hold_tlsph_tail_active(t) && hold_kinematics_mode == :frozen) ||
       hold_kinematics_frozen(t)
        dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            dv_cyl[1, particle] = 0.0
            dv_cyl[2, particle] = 0.0
            dv_cyl[3, particle] = 0.0
        end
    elseif sim_phase != "retraction" && hold_tlsph_tail_active(t)
        tail_k = hold_tlsph_tail_kinematics_scale(t)
        if tail_k < 1.0 - 1.0e-8
            dv_cyl = TrixiParticles.wrap_v(dv_ode, cyl_sys, semi_local)
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                dv_cyl[1, particle] *= tail_k
                dv_cyl[2, particle] *= tail_k
                dv_cyl[3, particle] *= tail_k
            end
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

    kin_scale = if sim_phase != "retraction" && hold_tlsph_tail_active(t)
        hold_tlsph_tail_kinematics_scale(t)
    elseif hold_kinematics_frozen(t)
        0.0
    elseif retraction_started[]
        retraction_kinematics_scale(t)
    else
        1.0
    end
    if kin_scale < 1.0 - 1.0e-8
        if separate_charge_plies
            clip_foreach_charge_system(semi_local) do system, _
                du_charge = TrixiParticles.wrap_u(du_ode, system, semi_local)
                ndims_charge = TrixiParticles.ndims(system)
                @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                    for d in 1:ndims_charge
                        du_charge[d, particle] *= kin_scale
                    end
                end
            end
        else
            cyl_sys = semi_local.systems[1]
            du_cyl = TrixiParticles.wrap_u(du_ode, cyl_sys, semi_local)
            NDIMS_CYL = TrixiParticles.ndims(cyl_sys)
            @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                for d in 1:NDIMS_CYL
                    du_cyl[d, particle] *= kin_scale
                end
            end
        end
    end

    # Hard-zero du_y when y motion is locked (pseudo-2D or thin-3D retraction lock).
    if hard_lock_y_kinematics()
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
    if uniform_fiber_orientation !== nothing
        apply_uniform_fiber_orientation!(fiber_direction, uniform_fiber_orientation)
        println(">>> Overrode checkpoint fibre directions with uniform orientation ",
                "[", uniform_fiber_orientation[1], ", ", uniform_fiber_orientation[2], ", ",
                uniform_fiber_orientation[3], "] (Spencer re-align at retraction prep)")
        flush(stdout)
    end
    if retraction_tlsph_only
        refresh_retraction_solid_phase_buffers!()
        wcsph_ramp_scale[] = 0.0
    end
end

v0_ode, u0_ode = clip_ode_initial_state(ode_base, hold_checkpoint_restore)
if hold_checkpoint_restore !== nothing && uniform_charge_temp_k !== nothing
    cyl_sys = semi.systems[1]
    apply_uniform_charge_temperature!(cyl_sys, semi, v0_ode, uniform_charge_temp_k)
    println(">>> Overrode checkpoint charge temperature with uniform T=",
            round(uniform_charge_temp_k; digits=2), " K (checkpoint file unchanged)")
    flush(stdout)
end

# Retraction-only: arm ramps before the first Newton step.  If retraction_started flips
# inside accepted_step_callback, dv_mech_scale jumps 1 → 0 and TRBDF2 collapses dt to ~1e-8
# (sim time stays at t0 → InfoCallback shows 0% for hundreds of steps).
if sim_phase == "retraction" && hold_checkpoint_restore !== nothing
    cyl_sys = semi.systems[1]
    t_ret = hold_checkpoint_restore["t"]
    arm_retraction_phase!(t_ret)
    if retraction_from_preeject[]
        arm_retraction_from_preeject!(t_ret)
    end
    capture_retraction_reference_span!(cyl_sys)
    if retraction_prepare_state && !retraction_from_preeject[]
        prepare_charge_state_for_retraction!(cyl_sys, semi, v0_ode, u0_ode,
                                             Fp_committed[], J_ref_particle_buf, J_p_particle_buf,
                                             vel_grad_buf, charge_dv_mech_marker_buf)
    else
        enforce_charge_plastic_audit!(cyl_sys, Fp_committed[], J_p_particle_buf;
                                      label="retraction_load")
    end
    retraction_state_prepared[] = true
    if retraction_pre_lift_creep_s > 0.0 && !retraction_from_preeject[]
        retraction_pre_lift_creep_announced[] = true
        creep_kin_freeze = env_int("TP_CLIP_RETRACTION_CREEP_KINEMATICS_FREEZE", 0) != 0
        creep_kin_after_sigma = env_int("TP_CLIP_RETRACTION_CREEP_KIN_AFTER_SIGMA", 1) != 0
        kin_desc = if retraction_kinematic_peel
            "FROZEN for kinematic peel (lift mold first; springback after gaps ≥0)"
        elseif retraction_closed_die_contact_settle
            "ACTIVE (contact settle; σ held at 0 until floor gap OK)"
        elseif creep_kin_freeze
            "FROZEN (non-physical for demold; floor overlap will not clear)"
        elseif creep_kin_after_sigma
            "gated until σ≈1 then ACTIVE (closed-die contact settle before lift)"
        else
            "ACTIVE from t0 (unsafe while σ≪1)"
        end
        println(">>> Pre-lift closed-die creep: ", round(retraction_pre_lift_creep_s; digits=4),
                " s (",
                retraction_closed_die_contact_settle ?
                    "CONTACT settle: kin ACTIVE + σ=0 until gap OK, then σ ramp " *
                    string(round(retraction_contact_settle_sigma_ramp_s; digits=4)) * " s" :
                env_int("TP_CLIP_RETRACTION_CREEP_RAMP_FREE", 0) != 0 ?
                    "σ/contact/kin INSTANT from checkpoint" :
                    "σ/contact ramp 0→1 once, not s²",
                "; hold_creep; mold fixed through floor settle;",
                " nominal lift t≥", round(retraction_lift_start_time[]; digits=6),
                " s (settle ≤", round(retraction_creep_floor_settle_max_s; digits=4), " s);",
                " charge kinematics ", kin_desc,
                retraction_kinematic_peel ?
                    "; punch lifts while charge frozen" :
                    "; punch lift with charge kin ON",
                retraction_pre_lift_settle_s > 0.0 ?
                    "; extra pre-lift settle " *
                    string(round(retraction_pre_lift_settle_s; digits=4)) *
                    " s (kin ON, T fixed)" : "",
                !retraction_kinematic_peel && retraction_post_lift_kin_freeze_s > 0.0 ?
                    " [WARNING: post-lift freeze " *
                    string(retraction_post_lift_kin_freeze_s) * " s overrides]" : "",
                ")")
    end
    if retraction_cfrp_style
        println(">>> Retraction armed (CFRP-style) at checkpoint t=", round(t_ret; digits=6),
                " s — full system_interaction!, no σ/contact/kin ramps")
    else
        println(">>> Retraction armed at checkpoint t=", round(t_ret; digits=6),
                retraction_pre_lift_creep_s > 0.0 ?
                    " s (pre-lift creep then kin-scaled lift; mechanics continuous at lift)" :
                    " s (contact/elastic/kinematics ramps active from t0; no step-1 discontinuity)")
    end
    hold_end_spencer_align &&
        println(">>> Hold-end Spencer alignment: ON (I4=1 at checkpoint F)")
    retraction_springback_dwell_s > 0.0 &&
        println(">>> Post-lift springback dwell: up to ", retraction_springback_dwell_s,
                " s (equil max_speed<", retraction_springback_equil_max_speed,
                " m/s, span_z rate<", retraction_springback_equil_span_rate_mm_s, " mm/s)")
    if retraction_cool_until_active
        freeze_t_policy = if retraction_cool_before_lift
            "demold frozen (FREEZE_T=$(retraction_explicit_freeze_t)); ON during closed-die cool, then after punch lift"
        elseif retraction_unfreeze_t_after_punch_lift
            "demold frozen (FREEZE_T=$(retraction_explicit_freeze_t)); ON after punch lift"
        elseif retraction_unfreeze_t_after_eject
            "demold frozen (FREEZE_T=$(retraction_explicit_freeze_t)); ON after eject only"
        elseif retraction_explicit_freeze_t
            "always (T will NOT cool)"
        else
            "off"
        end
        cool_where = if retraction_cool_before_lift && retraction_eject_after_cool
            "closed die to $(retraction_cool_before_lift_C) °C, then punch lift + air cool, then eject"
        elseif retraction_eject_after_cool
            "in die after punch lift, then eject"
        else
            "after eject"
        end
        println(">>> Cool until mean charge T ≤ ",
                retraction_cool_until_temp_C, " °C (", cool_where,
                "; ambient heat loss=", use_ambient_heat_loss,
                ", h=", h_ambient, " W/m²K, T_amb=",
                round(temp_ambient - 273.15; digits=1), " °C",
                ", freeze T=", freeze_t_policy,
                retraction_cool_before_lift ?
                    (retraction_closed_die_cool_plastic ?
                        ", plastic ON in closed die (hold_creep); OFF after punch lift" :
                        ", plastic OFF in closed die (elastic+CTE only); OFF after punch lift") :
                    ", plastic OFF",
                ", freeze mech after eject=", retraction_freeze_mech_after_eject,
                retraction_tool_temp_after_punch_lift_active ?
                    ", tool T after punch lift=" *
                    string(retraction_tool_temp_after_punch_lift_C) * " °C" : "",
                ", demold dt=", retraction_explicit_dt,
                " s, free-cool dt=", retraction_explicit_dt_springback, " s)")
        if retraction_explicit_freeze_t &&
           !retraction_unfreeze_t_after_eject && !retraction_unfreeze_t_after_punch_lift
            @warn "TP_CLIP_RETRACTION_COOL_UNTIL_TEMP_C set but FREEZE_T=1 without UNFREEZE — T will not evolve"
        end
        if !use_ambient_heat_loss
            @warn "TP_CLIP_RETRACTION_COOL_UNTIL_TEMP_C set but USE_AMBIENT_HEAT_LOSS=0 — enable ambient convection"
        end
    elseif use_ambient_heat_loss
        println(">>> Ambient heat loss ON: h=", h_ambient, " W/m²K, T_amb=",
                round(temp_ambient - 273.15; digits=1), " °C, ε=", emissivity_charge,
                retraction_unfreeze_t_after_punch_lift ? " (applied after punch lift)" :
                (retraction_unfreeze_t_after_eject ? " (applied after eject only)" : ""))
    end
    flush(stdout)
end

ode = clip_build_ode_problem(v0_ode, u0_ode, tspan, semi)

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
            "span_ratio,J_min,J_max,J_min_particle,min_mold_gap_mm,",
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

const clip_updated_lagrange_last_t = Ref(-1.0)

function clip_recompute_correction_matrix!(system, semi)
    TrixiParticles.set_zero!(system.correction_matrix)
    initial_coords = TrixiParticles.initial_coordinates(system)

    TrixiParticles.foreach_point_neighbor(system, system, initial_coords, initial_coords,
                                          semi;
                                          parallelization_backend=semi.parallelization_backend) do particle, neighbor, pos_diff, distance
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
        L_inv = clip_invert_correction_matrix(L)
        for j in 1:3, i in 1:3
            system.correction_matrix[i, j, particle] = L_inv[i, j]
        end
    end
    return nothing
end

function clip_apply_updated_lagrange_reference_update!(system, source, semi, t)
    n_part = TrixiParticles.nparticles(system)
    # 1. Update fiber orientation in physical space using current deformation gradient
    for p in 1:n_part
        global_p = source === nothing ? p : source[p]
        F_p = TrixiParticles.deformation_gradient(system, p)
        a_old = StaticArrays.SVector{3, Float64}(fiber_direction[1, global_p],
                                                 fiber_direction[2, global_p],
                                                 fiber_direction[3, global_p])
        a_new = F_p * a_old
        n_a = norm(a_new)
        if n_a > 1e-12
            a_unit = a_new / n_a
            fiber_direction[1, global_p] = a_unit[1]
            fiber_direction[2, global_p] = a_unit[2]
            fiber_direction[3, global_p] = a_unit[3]
        end
    end

    # 2. Update reference coordinates to current coordinates
    system.initial_coordinates .= system.current_coordinates

    # 3. Dynamic neighborhood search update on new reference coordinates
    PointNeighbors.update!(system.self_interaction_nhs, system.initial_coordinates, system.initial_coordinates)

    # 4. Recompute geometry-adapted kernel correction matrix
    clip_recompute_correction_matrix!(system, semi)

    # 5. Reset relative deformation gradient to identity
    for p in 1:n_part
        for j in 1:3, i in 1:3
            system.deformation_grad[i, j, p] = (i == j ? 1.0 : 0.0)
        end
    end
    return nothing
end

function clip_maybe_updated_lagrange_step!(integrator)
    clip_updated_lagrange || return nothing
    integrator.t >= t_compress && return nothing

    if clip_updated_lagrange_last_t[] < 0.0
        clip_updated_lagrange_last_t[] = integrator.t
        return nothing
    end

    if integrator.t - clip_updated_lagrange_last_t[] >= clip_updated_lagrange_interval
        println(">>> [UPDATED LAGRANGE] Re-referencing configuration at t = ",
                round(integrator.t; digits=4), " s (interval = ",
                clip_updated_lagrange_interval, " s)")
        flush(stdout)

        if separate_charge_plies
            clip_foreach_charge_system(integrator.p) do system, source
                clip_apply_updated_lagrange_reference_update!(system, source, integrator.p, integrator.t)
            end
        else
            cyl_sys = integrator.p.systems[1]
            clip_apply_updated_lagrange_reference_update!(cyl_sys, nothing, integrator.p, integrator.t)
        end

        clip_reset_jacobian_coloring!()
        clip_updated_lagrange_last_t[] = integrator.t
    end
    return nothing
end

const clip_last_synced_integrator_dt = Ref(-1.0)

# Sync trial_dt_state when the integrator changes Δt (init, rejects, controller) — not every callback poll.
clip_trial_dt_sync_callback = DiscreteCallback(
    function(u, t, integrator)
        integrator.dt != clip_last_synced_integrator_dt[]
    end,
    function(integrator)
        clip_last_synced_integrator_dt[] = integrator.dt
        clip_sync_trial_dt_from_integrator!(integrator)
        u_modified!(integrator, false)
        return nothing
    end;
    save_positions=(false, false))

accepted_step_callback = DiscreteCallback(
    (u, t, integrator) -> integrator.iter > 0 && mod(integrator.iter, 1) == 0,
    function(integrator)
            clip_sync_trial_dt_from_integrator!(integrator)
            enforce_explicit_compression_dt!(integrator)
            enforce_explicit_retraction_dt!(integrator)

            cyl_sys = integrator.p.systems[1]
            floor_sys = clip_floor_system(integrator.p)
            mold_sys = clip_mold_system(integrator.p)
            v_ode_state, u_ode_state = clip_integrator_vu(integrator)
            v_wrap_state = TrixiParticles.wrap_v(v_ode_state, cyl_sys, integrator.p)

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
                if separate_charge_plies
                    clip_foreach_charge_system(integrator.p) do system, _
                        freeze_charge_mechanical_state!(
                            TrixiParticles.wrap_v(v_ode_state, system, integrator.p),
                            system)
                    end
                else
                    freeze_charge_mechanical_state!(v_wrap_state, cyl_sys)
                end
            else
                update_retraction_deep_contact_kick_flag!(cyl_sys, floor_sys, mold_sys)
                apply_retraction_deep_contact_velocity_hard_clamp!(v_wrap_state, cyl_sys)
            end
            thermal_Fth_skip_Jp[] =
                retraction_closed_die_cool_active[] ||
                (retraction_lift_first_after_frozen_cool &&
                 retraction_qs_cte_scale[] < 1.0 - 1.0e-12)

            # Keep J_min ≥ compress_J_floor during punch-down (accepted steps only).
            if separate_charge_plies
                clip_foreach_charge_system(integrator.p) do system, source
                    apply_compress_j_floor_accepted!(
                        system, clip_charge_particle_view(J_p_particle_buf, source),
                        integrator.t)
                end
            else
                apply_compress_j_floor_accepted!(cyl_sys, J_p_particle_buf, integrator.t)
            end
            # Keep J_min ≥ retraction_J_floor through demold (stops creep re-crush).
            apply_retraction_j_floor_accepted!(cyl_sys, J_p_particle_buf, integrator.t)

            clip_maybe_updated_lagrange_step!(integrator)

    # IMEX split: apply wall drag + Coulomb friction explicitly on accepted steps
    # so the implicit Newton/JVP system only sees stiff contact + TLSPH mechanics.
    if imex_soft_terms && tool_wall_coupling_active(integrator.t)
        apply_imex_wall_drag_and_friction!(cyl_sys, floor_sys, mold_sys, integrator.p,
                                           v_wrap_state, integrator.t, integrator.dt)
        update_retraction_deep_contact_kick_flag!(cyl_sys, floor_sys, mold_sys)
        apply_retraction_deep_contact_velocity_hard_clamp!(v_wrap_state, cyl_sys)
    end

            if hold_tlsph_tail_active(integrator.t) && !hold_tlsph_tail_announced[]
                hold_tlsph_tail_announced[] = true
                hold_tail_dtmax = env_float("TP_CLIP_HOLD_TAIL_DTMAX", 1.0e-3)
                integrator.opts.dtmax = min(integrator.opts.dtmax, hold_tail_dtmax)
                integrator.dt = 1.0e-4
                println(">>> TLSPH equilibration tail: WCSPH off, full TLSPH+contact, ",
                        hold_kinematics_mode == :frozen ?
                            ("kinematics frozen for " * string(round(hold_tlsph_tail_s; digits=4)) * " s") :
                            ("kinematics active for " * string(round(hold_tlsph_tail_s; digits=4)) * " s"),
                        "; dtmax=", integrator.opts.dtmax, " s")
                flush(stdout)
            end
            t_vel_grad_start = time_ns()
            if separate_charge_plies
                clip_foreach_charge_system(integrator.p) do system, source
                    v_local = TrixiParticles.wrap_v(v_ode_state, system, integrator.p)
                    refresh_velocity_gradient_cache!(
                        system, v_local, integrator.p,
                        clip_charge_particle_view(vel_grad_buf, source))
                end
            else
                refresh_velocity_gradient_cache!(cyl_sys, v_wrap_state, integrator.p,
                                                 vel_grad_buf)
            end
            rhs_time_callback_vel_grad_ns[] += time_ns() - t_vel_grad_start

            # Orientation stays on in retraction (Jeffery/RSC): fiber alignment can still
            # evolve under demold shear. WCSPH / Nakamura / molten drag remain off.
            if !charge_hold_thermal_only(integrator.t) && !hold_tlsph_tail_active(integrator.t)
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

            u_wrap = TrixiParticles.wrap_u(u_ode_state, cyl_sys, integrator.p)
            if enable_emergency_repair_clamp
                repaired_particles = 0
                @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
                    x = u_wrap[1, particle]
                    y = u_wrap[2, particle]
                    z = u_wrap[3, particle]
                    vx = v_wrap_state[1, particle]
                    vy = v_wrap_state[2, particle]
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
                    if retraction_full3d_y_cap_active()
                        invalid_state = invalid_state ||
                                        !isfinite(y) || !isfinite(vy) ||
                                        y < y_safety_min || y > y_safety_max ||
                                        abs(vy) > max_cylinder_speed
                    end

                    if invalid_state
                        u_wrap[1, particle] = clamp(x, -x_safety_bound, x_safety_bound)
                        if use_pseudo2d_projection
                            u_wrap[2, particle] = 0.0
                        elseif retraction_full3d_y_cap_active()
                            u_wrap[2, particle] = clamp_charge_y_coordinate(y, cyl_sys, particle)
                        end
                        u_wrap[3, particle] = clamp(z, z_safety_min, z_safety_max)

                        cyl_sys.current_coordinates[1, particle] = u_wrap[1, particle]
                        if use_pseudo2d_projection
                            cyl_sys.current_coordinates[2, particle] = 0.0
                        elseif retraction_full3d_y_cap_active()
                            cyl_sys.current_coordinates[2, particle] = u_wrap[2, particle]
                        end
                        cyl_sys.current_coordinates[3, particle] = u_wrap[3, particle]

                        v_wrap_state[1, particle] = 0.0
                        if use_pseudo2d_projection || retraction_full3d_y_cap_active()
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
                    if retraction_full3d_y_cap_active()
                        y_now = u_wrap[2, particle]
                        vy_now = v_wrap_state[2, particle]
                        if !isfinite(y_now) || y_now < y_safety_min || y_now > y_safety_max
                            y_fix = clamp_charge_y_coordinate(y_now, cyl_sys, particle)
                            u_wrap[2, particle] = y_fix
                            cyl_sys.current_coordinates[2, particle] = y_fix
                        end
                        v_wrap_state[2, particle] = soft_limit(vy_now, max_cylinder_speed)
                    end
                end
            end

            if retraction_cfrp_style_active()
                advance_F_from_vel_grad!(cyl_sys, vel_grad_buf, integrator.dt)
                enable_elastic_stress &&
                    commit_plastic_history_cfrp!(cyl_sys, integrator.p, integrator.dt,
                                                 v_ode_state)
            elseif enable_elastic_stress && integrator.t >= t_warmup_end
                # ------------------------------------------------------------------
                # Nakamura crystallization ODE: advance α_c by one accepted step dt.
                # Must run before update_dual_phase_properties! so that liquid/solid
                # fraction buffers are refreshed from the new crystallinity.
                # ------------------------------------------------------------------
                if use_nakamura_kinetics && !retraction_solid_mechanics_only() &&
                   !hold_tlsph_tail_active(integrator.t) &&
                   !retraction_pre_lift_creep_active(integrator.t)
                    t_nak_acc_start = time_ns()
                    if separate_charge_plies
                        clip_foreach_charge_system(integrator.p) do system, source
                            step_nakamura_crystallinity!(
                                clip_charge_particle_view(crystallinity_particle_buf, source),
                                system.temp,
                                clip_charge_particle_view(liquid_fraction_particle_buf, source),
                                clip_charge_particle_view(solid_fraction_particle_buf, source),
                                integrator.dt)
                        end
                    else
                        step_nakamura_crystallinity!(crystallinity_particle_buf,
                                                     cyl_sys.temp,
                                                     liquid_fraction_particle_buf,
                                                     solid_fraction_particle_buf,
                                                     integrator.dt)
                    end
                    rhs_time_nakamura_accepted_ns[] += time_ns() - t_nak_acc_start
                end

                update_viscosity = clip_viscosity_update_on_step()
                if separate_charge_plies
                    clip_foreach_charge_system(integrator.p) do system, source
                        update_dual_phase_properties!(
                            clip_charge_particle_view(ys_particle_buf, source),
                            clip_charge_particle_view(hard_particle_buf, source),
                            clip_charge_particle_view(vis_particle_buf, source),
                            system.temp,
                            clip_charge_particle_view(alpha_committed[], source),
                            clip_charge_particle_view(thermal_softening_particle_buf, source),
                            clip_charge_particle_view(liquid_fraction_particle_buf, source),
                            clip_charge_particle_view(solid_fraction_particle_buf, source);
                            vel_grad_buf=clip_charge_particle_view(vel_grad_buf, source),
                            update_viscosity, source_indices=source)
                    end
                else
                    update_dual_phase_properties!(
                        ys_particle_buf, hard_particle_buf, vis_particle_buf,
                        cyl_sys.temp, alpha_committed[],
                        thermal_softening_particle_buf,
                        liquid_fraction_particle_buf,
                        solid_fraction_particle_buf;
                        vel_grad_buf=vel_grad_buf, update_viscosity)
                end
                refresh_retraction_solid_phase_buffers!()

                hold_active = hold_plastic_relax_active(integrator.t)
                skip_heat_commit = hold_active || retraction_skip_plastic_heat_now()
                n_commit = hold_active ? hold_relax_substeps : 1
                dt_commit = integrator.dt / n_commit
                t_plastic_start = time_ns()
                # CLOSED_DIE_COOL_PLASTIC=0: elastic+CTE only while tools stay closed.
                skip_plastic_closed_cool = retraction_closed_die_cool_active[] &&
                    !retraction_closed_die_cool_plastic
                if !retraction_thermal_shrinkage_only_now() && !skip_plastic_closed_cool
                    for _ in 1:n_commit
                        if separate_charge_plies
                            clip_foreach_charge_system(integrator.p) do system, source
                                commit_plastic_history_and_heat!(
                                    system,
                                    clip_charge_particle_view(ys_particle_buf, source),
                                    clip_charge_particle_view(hard_particle_buf, source),
                                    clip_charge_particle_view(vis_particle_buf, source),
                                    dt_commit,
                                    clip_charge_particle_view(alpha_committed[], source),
                                    clip_charge_particle_view(Fp_committed[], source);
                                    skip_heat=skip_heat_commit,
                                    hold_creep=hold_active,
                                    source_indices=source)
                            end
                        else
                            commit_plastic_history_and_heat!(
                                cyl_sys, ys_particle_buf, hard_particle_buf,
                                vis_particle_buf, dt_commit, alpha_committed[],
                                Fp_committed[]; skip_heat=skip_heat_commit,
                                hold_creep=hold_active)
                        end
                    end
                end
                rhs_time_plastic_commit_ns[] += time_ns() - t_plastic_start

                clip_foreach_charge_system(integrator.p) do system, _
                    v_wrap = TrixiParticles.wrap_v(v_ode_state, system, integrator.p)
                    ndims_charge = TrixiParticles.ndims(system)
                    @inbounds for particle in TrixiParticles.each_integrated_particle(system)
                        v_wrap[ndims_charge + 1, particle] = system.temp[particle]
                    end
                end
            end

            if retraction_cfrp_style_active()
                # F + plastic commit handled in CFRP-style branch above.
            elseif charge_hold_thermal_only(integrator.t)
                # Mechanical F frozen during solidification hold; F_th may still scale F.
            elseif retraction_started[] && !hold_kinematics_frozen(integrator.t)
                kin = retraction_kinematics_scale(integrator.t)
                if kin > 1.0e-10
                    advance_F_from_vel_grad!(cyl_sys, vel_grad_buf, integrator.dt * kin)
                end
            else
                if separate_charge_plies
                    clip_foreach_charge_system(integrator.p) do system, source
                        advance_F_from_vel_grad!(
                            system, clip_charge_particle_view(vel_grad_buf, source),
                            integrator.dt)
                    end
                else
                    advance_F_from_vel_grad!(cyl_sys, vel_grad_buf, integrator.dt)
                end
            end
            thermal_Fth_u_modified[] = apply_frozen_thermal_contraction!(
                cyl_sys, u_ode_state, integrator.p, integrator.t;
                v_ode=v_ode_state)

            if wcsph_mechanics_active() && enable_wcsph_liquid_viscosity &&
               lag_wcsph_liquid_viscosity
                refresh_lagged_wcsph_liquid_viscosity!(wcsph_visc_lagged_dv_buf,
                    wcsph_visc_scratch_dv_ode,
                    v_ode_state,
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
                mean_T_C = mean(cyl_sys.temp) - 273.15
                temp_target_reached = isfinite(hold_until_temp_C) &&
                    mean_T_C <= hold_until_temp_C
                hold_complete = if isfinite(hold_until_temp_C)
                    temp_target_reached || max_hold_reached
                else
                    solidified_fraction >= solidification_fraction_target || max_hold_reached
                end
                if hold_complete
                    if isfinite(hold_until_temp_C) && temp_target_reached
                        println(">>> Hold temperature target reached at t=",
                                round(integrator.t; digits=6), " s (hold ",
                                round(hold_elapsed; digits=3), " s; mean charge T=",
                                round(mean_T_C; digits=1), " °C ≤ ",
                                round(hold_until_temp_C; digits=1), " °C; ",
                                round(100 * solidified_fraction; digits=1),
                                "% with sf>=",
                                round(solidification_fraction_threshold; digits=3), ")")
                    elseif solidified_fraction >= solidification_fraction_target
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
                # Track stress relaxation rate during hold
                curr_stress_mean = 0.0
                n_charge_pts = length(TrixiParticles.each_integrated_particle(cyl_sys))
                @inbounds for p_idx in TrixiParticles.each_integrated_particle(cyl_sys)
                    s11 = v_stress_buf[1, 1, p_idx]
                    s22 = v_stress_buf[2, 2, p_idx]
                    s33 = v_stress_buf[3, 3, p_idx]
                    s12 = v_stress_buf[1, 2, p_idx]
                    s13 = v_stress_buf[1, 3, p_idx]
                    s23 = v_stress_buf[2, 3, p_idx]
                    vm = sqrt(max(0.0, 0.5 * ((s11 - s22)^2 + (s22 - s33)^2 + (s33 - s11)^2 + 6.0 * (s12^2 + s13^2 + s23^2))))
                    curr_stress_mean += vm
                end
                curr_stress_mean /= max(1, n_charge_pts)
                if isfinite(hold_prev_stress_t[]) && (integrator.t - hold_prev_stress_t[]) >= 1.0e-5
                    dt_stress = integrator.t - hold_prev_stress_t[]
                    hold_stress_rate_current[] = abs(curr_stress_mean - hold_prev_stress_mean[]) / dt_stress
                    hold_prev_stress_mean[] = curr_stress_mean
                    hold_prev_stress_t[] = integrator.t
                elseif !isfinite(hold_prev_stress_t[])
                    hold_prev_stress_mean[] = curr_stress_mean
                    hold_prev_stress_t[] = integrator.t
                end

                dwell_elapsed = integrator.t - solidification_reached_time[]
                if isfinite(hold_stress_rate_threshold) &&
                   dwell_elapsed >= (post_solidification_dwell + hold_tlsph_tail_min_s) &&
                   hold_stress_rate_current[] <= hold_stress_rate_threshold &&
                   !hold_stress_rate_converged[]
                    hold_stress_rate_converged[] = true
                    println(">>> Hold stress relaxation converged at t=",
                            round(integrator.t; digits=6), " s (dwell ",
                            round(dwell_elapsed; digits=4), " s): ||dσ/dt|| = ",
                            round(hold_stress_rate_current[] / 1e6; digits=4), " MPa/s ≤ ",
                            round(hold_stress_rate_threshold / 1e6; digits=4), " MPa/s")
                    flush(stdout)
                end

                mean_T_C_hold = mean(cyl_sys.temp) - 273.15
                hold_time_elapsed = integrator.t - t_compress
                max_hold_time_reached = isfinite(solidification_hold_max) &&
                    hold_time_elapsed >= solidification_hold_max
                cool_target_pending = isfinite(hold_until_temp_C) &&
                    mean_T_C_hold > hold_until_temp_C &&
                    !max_hold_time_reached
                if cool_target_pending
                    if !hold_cool_continue_announced[]
                        hold_cool_continue_announced[] = true
                        println(">>> Continuing closed-die cool: mean charge T=",
                                round(mean_T_C_hold; digits=1), " °C → target ≤ ",
                                round(hold_until_temp_C; digits=1),
                                " °C (tools closed; sf already complete)")
                        flush(stdout)
                    end
                elseif hold_tlsph_checkpoint_ready(integrator.t)
                    if isfinite(hold_until_temp_C)
                        println(">>> Hold cool target reached at t=",
                                round(integrator.t; digits=6),
                                " s (mean charge T=", round(mean_T_C_hold; digits=1),
                                " °C ≤ ", round(hold_until_temp_C; digits=1), " °C)")
                        flush(stdout)
                    end
                    if !retraction_state_prepared[]
                        prepare_charge_state_for_retraction!(cyl_sys, integrator.p,
                            v_ode_state, u_ode_state,
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
                        checkpoint_audit_on_save &&
                            enforce_charge_plastic_audit!(cyl_sys, Fp_committed[], J_p_particle_buf;
                                                          label="hold_complete")
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
                    arm_retraction_phase!(integrator.t)
                    capture_retraction_reference_span!(cyl_sys)
                    if !retraction_state_prepared[]
                        prepare_charge_state_for_retraction!(cyl_sys, integrator.p,
                            v_ode_state, u_ode_state,
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
                    if retraction_pre_lift_creep_s > 0.0 && !retraction_pre_lift_creep_announced[]
                        retraction_pre_lift_creep_announced[] = true
                        println(">>> Pre-lift closed-die creep: ",
                                round(retraction_pre_lift_creep_s; digits=4),
                                " s (hold_creep; mold fixed until t=",
                                round(retraction_lift_start_time[]; digits=6),
                                " s; post-lift σ/contact/kin stay at 1)")
                        flush(stdout)
                    end
                    if retraction_pre_lift_creep_s <= 0.0
                        println(">>> Starting mold retraction at t=",
                                round(integrator.t; digits=6),
                                " s (contact ",
                                retraction_mech_ramp_instant ? "instant-on" :
                                    "ramp " * string(retraction_hold_release_s) * " s",
                                "; elastic σ ",
                                retraction_elastic_ramp_instant ? "instant-on" :
                                    "ramp " * string(retraction_elastic_ramp_s) * " s",
                                "; mold lift kin-scaled; velocity ramp ",
                                retraction_ramp_s,
                                " s; kinematics ramp ", retraction_kinematics_release_s,
                                " s)")
                        flush(stdout)
                    end
                end
            end

            if retraction_started[] && !charge_mechanics_release_announced[] &&
               !retraction_dv_mech_ramping(integrator.t)
                charge_mechanics_release_announced[] = true
                if charge_hold_dtmax_active[]
                    # Still in demold (creep/lift/eject): keep demold dt, not free-cool dt.
                    restored_dtmax = retraction_use_explicit ?
                        retraction_explicit_target_dt(integrator.t) :
                        solver_dtmax_before_hold[]
                    integrator.opts.dtmax = restored_dtmax
                    integrator.dt = min(integrator.dt, restored_dtmax)
                    try
                        SciMLBase.set_proposed_dt!(integrator, integrator.dt)
                    catch
                    end
                    charge_hold_dtmax_active[] = false
                end
                println(">>> Retraction contact/σ-dv ramp complete at t=",
                        round(integrator.t; digits=6),
                        " s; dtmax restored to ", integrator.opts.dtmax, " s")
                retraction_use_explicit &&
                    println(">>> Explicit retraction: mech ramp complete at t=",
                            round(integrator.t; digits=6),
                            " s (demold dt=", retraction_explicit_dt,
                            " s until eject; free-cool dt=",
                            retraction_explicit_dt_springback, " s after)")
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

            if retraction_started[] && retraction_pre_lift_creep_active(integrator.t) &&
               integrator.t >= retraction_creep_audit_next_t[] &&
               (integrator.t - retraction_start_time[]) <= retraction_pre_lift_creep_s + 1.0e-6
                creep_elapsed = integrator.t - retraction_start_time[]
                enforce_charge_plastic_audit!(cyl_sys, Fp_committed[], J_p_particle_buf;
                    label="creep_mid_t=$(round(creep_elapsed; digits=4))s")
                retraction_creep_audit_next_t[] = integrator.t + retraction_creep_audit_dt
            end

            if retraction_started[] && retraction_pre_lift_settle_s > 0.0 &&
               !retraction_lift_started_announced[]
                settle_t0 = retraction_start_time[] + retraction_pre_lift_creep_s
                if !retraction_pre_lift_settle_announced[] &&
                   integrator.t >= settle_t0 - 1.0e-14
                    retraction_pre_lift_settle_announced[] = true
                    println(">>> Pre-lift settle start at t=",
                            round(integrator.t; digits=6),
                            " s: punch+floor fixed, kin ON, T fixed, no lift/cooling for ",
                            round(retraction_pre_lift_settle_s; digits=4), " s")
                    flush(stdout)
                end
                if !retraction_pre_lift_settle_done[] &&
                   integrator.t >= settle_t0 + retraction_pre_lift_settle_s - 1.0e-14
                    retraction_pre_lift_settle_done[] = true
                    snap_settle = build_retraction_diag_snapshot(integrator, cyl_sys, floor_sys,
                                                                 mold_sys, v_wrap_state)
                    println(">>> Pre-lift settle done at t=",
                            round(integrator.t; digits=6),
                            " s: span_z=", round(snap_settle.span_z_mm; digits=4),
                            " mm J_min=", round(snap_settle.J_min; sigdigits=4),
                            " min_floor_gap=",
                            round(snap_settle.min_floor_gap_mm; digits=4),
                            " mm rms_speed=", round(snap_settle.rms_speed; sigdigits=4))
                    flush(stdout)
                    save_vtu_snapshot!(integrator)
                end
            end

            if retraction_started[] && !retraction_lift_started_announced[]
                maybe_finish_closed_die_contact_settle!(integrator.t, cyl_sys, floor_sys, mold_sys)
                maybe_postpone_lift_for_floor_settle!(integrator.t, cyl_sys, floor_sys, mold_sys)
            end

            if retraction_started[] && retraction_cool_before_lift &&
               !retraction_lift_started_announced[] && !retraction_closed_die_cool_done[] &&
               !retraction_pre_lift_settle_active(integrator.t)
                creep_elapsed = integrator.t - retraction_start_time[]
                if creep_elapsed >= retraction_pre_lift_creep_s - 1.0e-14
                    Tmean_C = mean(cyl_sys.temp) - 273.15
                    if !retraction_closed_die_cool_active[]
                        if retraction_qs_cool && retraction_lift_first_after_frozen_cool &&
                           isfinite(retraction_qs_kin_after_C) &&
                           retraction_qs_skip_cte_until_kin &&
                           Tmean_C > retraction_qs_kin_after_C + 1.0e-8
                            if clip_fth_die_aware
                                apply_frozen_thermal_contraction!(
                                    cyl_sys, u_ode_state, integrator.p, integrator.t;
                                    v_ode=v_ode_state, force=true)
                            end
                            delta_C = Tmean_C - retraction_qs_kin_after_C
                            apply_charge_temperature_delta!(cyl_sys, integrator.p,
                                                            v_ode_state, -delta_C)
                            thermal_Fth_u_modified[] = apply_frozen_thermal_contraction!(
                                cyl_sys, u_ode_state, integrator.p, integrator.t;
                                v_ode=v_ode_state, force=true)
                            apply_tool_temp_after_punch_lift!(integrator;
                                when="at frozen T drop (punch DOWN)")
                            println(">>> QS frozen T drop: Tmean ",
                                    round(Tmean_C; digits=2), " → ",
                                    round(Tmean_C - delta_C; digits=2),
                                    " °C; kin frozen; τ_th=0; ",
                                    clip_fth_die_aware ?
                                        "aniso die-aware F_th on F+positions (Fig.4 n)" :
                                        "packing unchanged")
                            flush(stdout)
                            save_vtu_snapshot!(integrator)
                            Tmean_C -= delta_C
                        end
                        if retraction_lift_first_after_frozen_cool
                            retraction_closed_die_cool_active[] = false
                            retraction_closed_die_cool_done[] = true
                            retraction_closed_die_cool_start_t[] = integrator.t
                            retraction_qs_cte_scale[] = 0.0
                            retraction_qs_stage_index[] = 0
                            retraction_lift_start_time[] = integrator.t
                            println(">>> Lift-first controller: Tmean=",
                                    round(Tmean_C; digits=2),
                                    " °C, τ_th=0, charge FROZEN; starting punch lift. ",
                                    "Release at mold gap ≥",
                                    round(retraction_peel_gap_ps * particle_spacing * 1e3;
                                          digits=3),
                                    " mm",
                                    clip_fth_skip_post_peel_cte ?
                                        "; overlay CTE stays off (F_th already absorbed ΔT)" :
                                        ("; then restore CTE in " *
                                         string(retraction_qs_cool_step_C) * " K stages"))
                            save_vtu_snapshot!(integrator)
                        else
                            apply_tool_temp_after_punch_lift!(integrator;
                                when="at closed-die cool start (punch DOWN)")
                            retraction_closed_die_cool_active[] = true
                            retraction_closed_die_cool_start_t[] = integrator.t
                            retraction_qs_cte_scale[] =
                                retraction_qs_cte_ramp ? 0.0 : 1.0
                            if retraction_qs_cool
                                retraction_qs_stage_index[] = 0
                                retraction_qs_stage_start_t[] = integrator.t
                                retraction_qs_stage_target_C[] = Tmean_C
                                retraction_qs_equil_streak[] = 0
                                retraction_qs_prev_snap[] =
                                    build_retraction_diag_snapshot(integrator, cyl_sys,
                                        floor_sys, mold_sys, v_wrap_state)
                                retraction_qs_next_sample_t[] =
                                    integrator.t + retraction_qs_sample_s
                                println(">>> Quasi-static closed-die cool: tools FIXED, kin ON at Tmean=",
                                        round(Tmean_C; digits=2), " °C; cte_scale=",
                                        round(retraction_qs_cte_scale[]; sigdigits=3),
                                        retraction_qs_cte_ramp ? " (2 K ramp)" : " (bang)",
                                        "; ΔT=", retraction_qs_cool_step_C,
                                        " K; sample=", retraction_qs_sample_s,
                                        " s; require rms|v|<",
                                        retraction_qs_equil_rms_speed,
                                        " m/s, max|v|<",
                                        retraction_qs_equil_max_speed,
                                        " m/s, |dspan_z/dt|<",
                                        retraction_springback_equil_span_rate_mm_s,
                                        " mm/s, J-floor≤",
                                        retraction_qs_max_j_floor_n,
                                        " before target ",
                                        retraction_cool_before_lift_C, " °C")
                            else
                                println(">>> Closed-die cool (punch+floor ON, convection OFF, kin ON, CTE volumetric, no F_th teleport, ",
                                        retraction_closed_die_cool_plastic ?
                                            "hold_creep plastic ON, heat OFF" :
                                            "plastic OFF (elastic+CTE), heat OFF",
                                        "): t=",
                                        round(integrator.t; digits=6),
                                        " s  Tmean=", round(Tmean_C; digits=1),
                                        " °C → lift when ≤ ",
                                        retraction_cool_before_lift_C, " °C")
                            end
                        end
                        flush(stdout)
                    end
                    if retraction_lift_first_after_frozen_cool
                        # Punch motion + kinematic peel now own the transition.
                    elseif retraction_qs_cool
                        stage_elapsed = integrator.t - retraction_qs_stage_start_t[]
                        snap_qs = nothing
                        if integrator.t + 1.0e-12 >= retraction_qs_next_sample_t[]
                            snap_qs = build_retraction_diag_snapshot(integrator, cyl_sys,
                                                                    floor_sys, mold_sys,
                                                                    v_wrap_state)
                            prev_qs = retraction_qs_prev_snap[]
                            sample_dt = prev_qs === nothing ? 0.0 :
                                snap_qs.t - prev_qs.t
                            span_rate = sample_dt > 0.0 ?
                                abs((snap_qs.span_z_mm - prev_qs.span_z_mm) / sample_dt) :
                                Inf
                            sample_healthy =
                                stage_elapsed >= retraction_qs_equil_min_s &&
                                sample_dt >= 0.5 * retraction_qs_sample_s &&
                                snap_qs.rms_speed <= retraction_qs_equil_rms_speed &&
                                snap_qs.max_speed <= retraction_qs_equil_max_speed &&
                                span_rate <= retraction_springback_equil_span_rate_mm_s &&
                                retraction_last_J_floor_n[] <= retraction_qs_max_j_floor_n &&
                                snap_qs.nonfinite_n == 0
                            if sample_healthy
                                retraction_qs_equil_streak[] += 1
                            else
                                retraction_qs_equil_streak[] = 0
                            end
                            println(">>> QS equilibrium sample: stage=",
                                    retraction_qs_stage_index[], " Tmean=",
                                    round(Tmean_C; digits=2), " °C cte=",
                                    round(retraction_thermal_cte_scale(); sigdigits=3),
                                    " rms|v|=",
                                    round(snap_qs.rms_speed; sigdigits=4),
                                    " max|v|=", round(snap_qs.max_speed; sigdigits=4),
                                    " m/s |dspan_z/dt|=",
                                    round(span_rate; sigdigits=4), " mm/s span_z=",
                                    round(snap_qs.span_z_mm; digits=3),
                                    " mm J_min=", round(snap_qs.J_min; sigdigits=3),
                                    " J-floor=",
                                    retraction_last_J_floor_n[], " streak=",
                                    retraction_qs_equil_streak[], "/",
                                    retraction_qs_equil_patience,
                                    sample_healthy ? " ✓" : "")
                            flush(stdout)
                            if (retraction_qs_abort_span_z_active &&
                                snap_qs.span_z_mm > retraction_qs_abort_span_z_mm) ||
                               (retraction_qs_abort_on_j_min &&
                                snap_qs.J_min < retraction_qs_abort_j_min)
                                println(stderr,
                                        ">>> QS COOL ABORT: span_z=",
                                        round(snap_qs.span_z_mm; digits=3),
                                        " mm J_min=", round(snap_qs.J_min; sigdigits=4),
                                        " (limits span_z≤",
                                        retraction_qs_abort_span_z_mm,
                                        " mm, J_min≥",
                                        retraction_qs_abort_j_min, "). Punch remains closed; no lift.")
                                flush(stderr)
                                save_vtu_snapshot!(integrator)
                                terminate!(integrator)
                            end
                            retraction_qs_prev_snap[] = snap_qs
                            retraction_qs_next_sample_t[] =
                                integrator.t + retraction_qs_sample_s
                        end

                        stage_converged =
                            retraction_qs_equil_streak[] >= retraction_qs_equil_patience
                        if retraction_closed30_ckpt_saved[]
                            settle_elapsed = integrator.t -
                                retraction_closed30_settle_start_t[]
                            if settle_elapsed + 1.0e-12 >= retraction_closed30_settle_s
                                snap_b = build_retraction_diag_snapshot(integrator,
                                    cyl_sys, floor_sys, mold_sys, v_wrap_state)
                                shape_ok = snap_b.J_min >= max(retraction_qs_abort_j_min, 0.0) &&
                                    (!retraction_qs_abort_span_z_active ||
                                     snap_b.span_z_mm <= retraction_qs_abort_span_z_mm) &&
                                    snap_b.nonfinite_n == 0
                                if !shape_ok
                                    println(stderr,
                                            ">>> CLOSED30 SETTLE ABORT: span_z=",
                                            round(snap_b.span_z_mm; digits=3),
                                            " mm J_min=",
                                            round(snap_b.J_min; sigdigits=4),
                                            " — no checkpoint B, punch stays closed.")
                                    flush(stderr)
                                    save_vtu_snapshot!(integrator)
                                    terminate!(integrator)
                                else
                                    save_named_retraction_checkpoint!(integrator,
                                        retraction_closed30_settle_checkpoint_path,
                                        "closed30_settle")
                                    retraction_closed_die_cool_active[] = false
                                    retraction_closed_die_cool_done[] = true
                                    retraction_lift_start_time[] = integrator.t
                                    println(">>> Checkpoint B OK (span_z=",
                                            round(snap_b.span_z_mm; digits=3),
                                            " mm J_min=",
                                            round(snap_b.J_min; sigdigits=4),
                                            "); starting punch lift")
                                    flush(stdout)
                                    save_vtu_snapshot!(integrator)
                                end
                            elseif integrator.t - retraction_indie_cool_last_print_t[] >= 0.5
                                println(">>> Closed-die 30 °C settle: t=",
                                        round(integrator.t; digits=3),
                                        " s  elapsed=",
                                        round(settle_elapsed; digits=3),
                                        "/", retraction_closed30_settle_s,
                                        " s  punch DOWN")
                                flush(stdout)
                                retraction_indie_cool_last_print_t[] = integrator.t
                            end
                        elseif stage_converged
                            cte_now = retraction_thermal_cte_scale()
                            if retraction_qs_cte_ramp && cte_now < 1.0 - 1.0e-12
                                T_sf_C = T_stress_free - 273.15
                                dT_full = max(abs(Tmean_C - T_sf_C),
                                              retraction_qs_cool_step_C)
                                d_scale = min(1.0 - retraction_qs_cte_scale[],
                                              retraction_qs_cool_step_C / dT_full)
                                scale_from = retraction_qs_cte_scale[]
                                retraction_qs_cte_scale[] = scale_from + d_scale
                                retraction_qs_stage_index[] += 1
                                retraction_qs_stage_target_C[] = Tmean_C
                                retraction_qs_stage_start_t[] = integrator.t
                                retraction_qs_equil_streak[] = 0
                                retraction_qs_prev_snap[] =
                                    build_retraction_diag_snapshot(integrator, cyl_sys,
                                        floor_sys, mold_sys, v_wrap_state)
                                retraction_qs_next_sample_t[] =
                                    integrator.t + retraction_qs_sample_s
                                println(">>> QS CTE ramp stage ",
                                        retraction_qs_stage_index[], ": cte_scale ",
                                        round(scale_from; sigdigits=3), " → ",
                                        round(retraction_qs_cte_scale[]; sigdigits=3),
                                        " (≈", retraction_qs_cool_step_C, " K of ",
                                        round(dT_full; digits=1),
                                        " K vs T_stress_free); Tmean=",
                                        round(Tmean_C; digits=2),
                                        " °C; kin ON; tools fixed")
                                flush(stdout)
                                save_vtu_snapshot!(integrator)
                            elseif Tmean_C <= retraction_cool_before_lift_C + 1.0e-8
                                save_named_retraction_checkpoint!(integrator,
                                    retraction_closed30_checkpoint_path,
                                    "closed30")
                                retraction_closed30_ckpt_saved[] = true
                                retraction_closed30_settle_start_t[] = integrator.t
                                println(">>> Checkpoint A (closed-die ",
                                        round(Tmean_C; digits=1),
                                        " °C, punch DOWN) at t=",
                                        round(integrator.t; digits=6), " s")
                                flush(stdout)
                                save_vtu_snapshot!(integrator)
                            else
                                delta_C = min(retraction_qs_cool_step_C,
                                              Tmean_C -
                                              retraction_cool_before_lift_C)
                                apply_charge_temperature_delta!(cyl_sys, integrator.p,
                                                                v_ode_state, -delta_C)
                                thermal_Fth_u_modified[] = true
                                retraction_qs_stage_index[] += 1
                                retraction_qs_stage_target_C[] = Tmean_C - delta_C
                                retraction_qs_stage_start_t[] = integrator.t
                                retraction_qs_equil_streak[] = 0
                                retraction_qs_prev_snap[] =
                                    build_retraction_diag_snapshot(integrator, cyl_sys,
                                        floor_sys, mold_sys, v_wrap_state)
                                retraction_qs_next_sample_t[] =
                                    integrator.t + retraction_qs_sample_s
                                println(">>> QS cool stage ",
                                        retraction_qs_stage_index[], ": Tmean ",
                                        round(Tmean_C; digits=2), " → ",
                                        round(Tmean_C - delta_C; digits=2),
                                        " °C; tools fixed; equilibrating")
                                flush(stdout)
                                save_vtu_snapshot!(integrator)
                            end
                        elseif stage_elapsed >= retraction_qs_equil_max_s
                            snap_abort = snap_qs === nothing ?
                                build_retraction_diag_snapshot(integrator, cyl_sys,
                                    floor_sys, mold_sys, v_wrap_state) : snap_qs
                            println(stderr,
                                    ">>> QS COOL ABORT: stage ",
                                    retraction_qs_stage_index[],
                                    " did not equilibrate within ",
                                    retraction_qs_equil_max_s,
                                    " s at Tmean=", round(Tmean_C; digits=2),
                                    " °C cte_scale=",
                                    round(retraction_thermal_cte_scale(); sigdigits=3),
                                    " (rms_speed=",
                                    round(snap_abort.rms_speed; sigdigits=4),
                                    ", max_speed=",
                                    round(snap_abort.max_speed; sigdigits=4),
                                    " m/s, span_z=", round(snap_abort.span_z_mm; digits=4),
                                    " mm, J_min=", round(snap_abort.J_min; sigdigits=4),
                                    ", J-floor=", retraction_last_J_floor_n[],
                                    "). Punch remains closed; no lift.")
                            flush(stderr)
                            save_vtu_snapshot!(integrator)
                            terminate!(integrator)
                        end
                    else
                        cool_elapsed = integrator.t - retraction_closed_die_cool_start_t[]
                        hit_T = Tmean_C <= retraction_cool_before_lift_C
                        hit_dwell = cool_elapsed >= retraction_springback_dwell_s
                        if hit_T || hit_dwell
                            retraction_closed_die_cool_active[] = false
                            retraction_closed_die_cool_done[] = true
                            retraction_lift_start_time[] = integrator.t
                            println(">>> Closed-die cool done at t=",
                                    round(integrator.t; digits=6),
                                    " s (Tmean=", round(Tmean_C; digits=1), " °C",
                                    hit_T ? " ≤ " *
                                            string(retraction_cool_before_lift_C) * " °C" :
                                            "; dwell cap before T target",
                                    "); starting punch lift")
                            flush(stdout)
                            save_vtu_snapshot!(integrator)
                        elseif integrator.t - retraction_indie_cool_last_print_t[] >= 5.0
                            println(">>> Closed-die cool: t=",
                                    round(integrator.t; digits=3),
                                    " s  Tmean=", round(Tmean_C; digits=1),
                                    " °C  kin ON  (lift at ≤ ",
                                    retraction_cool_before_lift_C, " °C)")
                            flush(stdout)
                            retraction_indie_cool_last_print_t[] = integrator.t
                        end
                    end
                end
            end

            if retraction_started[] && !retraction_lift_started_announced[] &&
               !retraction_pre_lift_creep_active(integrator.t) &&
               retraction_pre_lift_creep_s > 0.0
                retraction_lift_started_announced[] = true
                _, gap_floor_lift = charge_min_tool_gaps(cyl_sys, floor_sys, mold_sys)
                peel_note = retraction_kinematic_peel ?
                    "charge FROZEN (kinematic peel until gaps clear)" :
                    "charge kinematics ON"
                println(">>> Starting mold lift at t=", round(integrator.t; digits=6),
                        " s (mold lift ramp ", retraction_ramp_s,
                        " s; ", peel_note, "; min_floor_gap=",
                        round(gap_floor_lift * 1e3; digits=4),
                        " mm; σ/contact already at 1)")
                flush(stdout)
                # End-of-creep plastic / Fe_iso audit — demold readiness signal.
                enforce_charge_plastic_audit!(cyl_sys, Fp_committed[], J_p_particle_buf;
                                              label="creep_end_pre_lift")
            end

            if retraction_started[] && retraction_lift_started_announced[]
                maybe_update_kinematic_peel!(integrator, cyl_sys, floor_sys, mold_sys)
                maybe_update_post_peel_cte_ramp!(integrator, cyl_sys)
            end

            if retraction_started[] && !mold_retraction_complete[]
                lift_t0 = retraction_pre_lift_creep_s > 0.0 ?
                    retraction_lift_start_time[] : retraction_start_time[]
                retract_elapsed = integrator.t - lift_t0
                if retract_elapsed >= retraction_duration_max
                    complete_mold_retraction!(integrator)
                end
            end

            if mold_retraction_complete[] && retraction_eject_enable &&
               retraction_eject_after_cool &&
               !retraction_eject_active[] && !retraction_eject_complete[] &&
               !retraction_complete[]
                Tmean_C = mean(cyl_sys.temp) - 273.15
                cool_elapsed = integrator.t - mold_retraction_complete_time[]
                hit_T = Tmean_C <= retraction_eject_at_temp_C
                hit_dwell = cool_elapsed >= retraction_springback_dwell_s
                cte_ready = clip_fth_skip_post_peel_cte ||
                    !retraction_lift_first_after_frozen_cool ||
                    retraction_qs_cte_scale[] >= 1.0 - 1.0e-12
                if (hit_T || hit_dwell) && cte_ready
                    println(">>> In-die cool done at t=", round(integrator.t; digits=6),
                            " s (Tmean=", round(Tmean_C; digits=1), " °C",
                            hit_T ? " ≤ " * string(retraction_eject_at_temp_C) * " °C" :
                                    "; dwell cap before T target",
                            "; CTE ready=", cte_ready, "); starting eject")
                    flush(stdout)
                    start_retraction_eject!(integrator)
                elseif integrator.t - retraction_indie_cool_last_print_t[] >= 5.0
                    println(">>> In-die cool: t=", round(integrator.t; digits=3),
                            " s  Tmean=", round(Tmean_C; digits=1),
                            " °C  (eject at ≤ ", retraction_eject_at_temp_C,
                            " °C after CTE ready; cte_scale=",
                            round(retraction_qs_cte_scale[]; sigdigits=4), ")")
                    flush(stdout)
                    retraction_indie_cool_last_print_t[] = integrator.t
                end
            end

            if mold_retraction_complete[] && retraction_eject_enable &&
               !retraction_eject_complete[]
                maybe_update_retraction_eject!(integrator)
            end

            if mold_retraction_complete[] && retraction_eject_enable &&
               retraction_eject_complete[] && !retraction_complete[] &&
               retraction_springback_dwell_s <= 0.0
                complete_retraction_simulation!(integrator;
                    reason="demold eject complete (dwell=0)")
            end

            if mold_retraction_complete[] && !retraction_complete[] &&
               retraction_springback_dwell_s > 0.0 &&
               (!retraction_eject_enable || retraction_eject_complete[])
                # Charge still frozen after mold stroke: max_speed=0 is NOT springback
                # equilibration. Wait until post-lift kin freeze ends, then dwell.
                if retraction_charge_kinematics_gated(integrator.t)
                    springback_equil_streak[] = 0
                    springback_equil_prev_snap[] = nothing
                else
                    peel_t0 = if retraction_kinematic_peel
                        retraction_peel_release_time[] + retraction_peel_sigma_ramp_s
                    else
                        retraction_lift_start_time[] + retraction_post_lift_kin_freeze_s
                    end
                    springback_t0 = max(mold_retraction_complete_time[], peel_t0)
                    dwell_elapsed = integrator.t - springback_t0
                    snap_eq = build_retraction_diag_snapshot(integrator, cyl_sys, floor_sys, mold_sys,
                                                            v_wrap_state)
                    prev_eq = springback_equil_prev_snap[]
                    if springback_equilibration_met(snap_eq, prev_eq)
                        springback_equil_streak[] += 1
                    else
                        springback_equil_streak[] = 0
                    end
                    springback_equil_prev_snap[] = snap_eq
                    # Free-cool path: stop on mean T, not mechanical equilibration alone.
                    # With FREEZE_MECH: cool thermal-only, then brief post-cool springback.
                    if retraction_cool_until_active
                        Tmean_C = mean(cyl_sys.temp) - 273.15
                        if retraction_post_cool_springback_active[]
                            sb_elapsed = integrator.t - retraction_post_cool_springback_start_t[]
                            if retraction_post_cool_thermal_ramp_s > 0.0
                                x_ramp = clamp(sb_elapsed / retraction_post_cool_thermal_ramp_s, 0.0, 1.0)
                                retraction_post_cool_thermal_scale[] = x_ramp * x_ramp * (3.0 - 2.0 * x_ramp)
                                if sb_elapsed < retraction_post_cool_thermal_ramp_s
                                    # Nothing moves yet: do not count this as equilibrium.
                                    springback_equil_streak[] = 0
                                end
                            end
                            sb_dwell_target = retraction_post_cool_springback_s > 0.0 ?
                                retraction_post_cool_springback_s : retraction_springback_dwell_s
                            if springback_equil_streak[] >= retraction_springback_equil_patience ||
                               sb_elapsed >= sb_dwell_target
                                complete_retraction_simulation!(integrator;
                                    reason=(springback_equil_streak[] >= retraction_springback_equil_patience ?
                                            "post-cool springback equilibrated (Tmean=" :
                                            "post-cool springback dwell timeout (Tmean=") *
                                    string(round(Tmean_C; digits=1)) * " °C, span_z=" *
                                    string(round(snap_eq.span_z_mm; digits=4)) * " mm, max_speed=" *
                                    string(round(snap_eq.max_speed; sigdigits=3)) * " m/s)")
                            end
                        elseif Tmean_C <= retraction_cool_until_temp_C + 0.5
                            post_eject_settle = retraction_eject_enable &&
                                retraction_eject_complete[] &&
                                (dwell_elapsed < max(retraction_post_eject_dwell_s, 0.05) &&
                                 springback_equil_streak[] < retraction_springback_equil_patience)
                            if post_eject_settle
                                if integrator.t - retraction_indie_cool_last_print_t[] >= 0.05
                                    println(">>> Post-eject live mechanics: t=",
                                            round(integrator.t; digits=3),
                                            " s  Tmean=", round(Tmean_C; digits=1),
                                            " °C  span_z=", round(snap_eq.span_z_mm; digits=3),
                                            " mm  (dwell ", round(dwell_elapsed; digits=3),
                                            "/", retraction_post_eject_dwell_s, " s)")
                                    flush(stdout)
                                    retraction_indie_cool_last_print_t[] = integrator.t
                                end
                            elseif retraction_bang_t_after_eject &&
                               (!retraction_post_eject_bang_done[] ||
                                integrator.t < retraction_post_eject_bang_t[] +
                                    retraction_post_eject_dwell_s - 1.0e-14)
                                # Wait for post-eject mechanical response at room T.
                            elseif retraction_freeze_mech_after_eject &&
                               retraction_post_cool_springback_s > 0.0
                                retraction_post_cool_springback_active[] = true
                                retraction_post_cool_springback_start_t[] = integrator.t
                                retraction_post_cool_thermal_scale[] =
                                    retraction_post_cool_thermal_ramp_s > 0.0 ? 0.0 : 1.0
                                springback_equil_streak[] = 0
                                enforce_explicit_retraction_dt!(integrator)
                                if !retraction_post_cool_springback_announced[]
                                    println(">>> Cool target reached (Tmean=",
                                            round(Tmean_C; digits=1),
                                            " °C): releasing mechanics for post-cool springback ",
                                            retraction_post_cool_springback_s,
                                            " s (T re-frozen, dt=",
                                            retraction_explicit_dt, " s)")
                                    flush(stdout)
                                    retraction_post_cool_springback_announced[] = true
                                    save_vtu_snapshot!(integrator)
                                end
                            elseif springback_equil_streak[] >= retraction_springback_equil_patience
                                complete_retraction_simulation!(integrator;
                                    reason="free cool target reached and springback equilibrated (Tmean=" *
                                    string(round(Tmean_C; digits=1)) * " °C ≤ " *
                                    string(retraction_cool_until_temp_C) * " °C, max_speed=" *
                                    string(round(snap_eq.max_speed; sigdigits=3)) * " m/s, span_z=" *
                                    string(round(snap_eq.span_z_mm; digits=4)) * " mm)")
                            elseif dwell_elapsed >= retraction_springback_dwell_s
                                complete_retraction_simulation!(integrator;
                                    reason="springback dwell timeout reached at cool target (Tmean=" *
                                    string(round(Tmean_C; digits=1)) * " °C, max_speed=" *
                                    string(round(snap_eq.max_speed; sigdigits=3)) * " m/s, span_z=" *
                                    string(round(snap_eq.span_z_mm; digits=4)) * " mm)")
                            else
                                if integrator.t - retraction_indie_cool_last_print_t[] >= 0.1
                                    println(">>> Post-cool equilibrating: t=",
                                            round(integrator.t; digits=3),
                                            " s  Tmean=", round(Tmean_C; digits=1),
                                            " °C  span_z=", round(snap_eq.span_z_mm; digits=3),
                                            " mm  max_speed=", round(snap_eq.max_speed; sigdigits=3),
                                            " m/s  equil streak=", springback_equil_streak[], "/", retraction_springback_equil_patience)
                                    flush(stdout)
                                    retraction_indie_cool_last_print_t[] = integrator.t
                                end
                            end
                        else
                            print_dt = max(retraction_explicit_target_dt(integrator.t), 1.0)
                            if integrator.t - retraction_indie_cool_last_print_t[] >= print_dt
                                println(">>> Free cool: t=", round(integrator.t; digits=3),
                                        " s  Tmean=", round(Tmean_C; digits=1),
                                        " °C  span_x=", round(snap_eq.span_x_mm; digits=2),
                                        " mm span_z=", round(snap_eq.span_z_mm; digits=2),
                                        " mm J_min=", round(snap_eq.J_min; sigdigits=3),
                                        "  (stop at ≤ ", retraction_cool_until_temp_C,
                                        " °C; dt=", retraction_explicit_target_dt(integrator.t), " s",
                                        retraction_freeze_mech_after_eject ? "; mech frozen" : "",
                                        ")")
                                flush(stdout)
                                retraction_indie_cool_last_print_t[] = integrator.t
                            end
                            if dwell_elapsed >= retraction_springback_dwell_s
                                complete_retraction_simulation!(integrator;
                                    reason="springback dwell time reached before cool target (Tmean=" *
                                    string(round(Tmean_C; digits=1)) * " °C, span_z=" *
                                    string(round(snap_eq.span_z_mm; digits=4)) * " mm)")
                            end
                        end
                    elseif springback_equil_streak[] >= retraction_springback_equil_patience
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
            end

            maybe_save_periodic_checkpoint!(integrator, integrator.p)

            if formulation_profile_enabled
                formulation_profile_step_count[] += 1
                if formulation_profile_step_count[] % formulation_profile_interval == 0
                    print_formulation_timing_report!(integrator;
                        label="every $(formulation_profile_interval) steps")
                end
                if formulation_profile_time_dt > 0.0
                    while integrator.t + 1.0e-14 >= formulation_profile_next_t[]
                        print_formulation_timing_report!(integrator;
                            label="every $(formulation_profile_time_dt) s")
                        formulation_profile_next_t[] += formulation_profile_time_dt
                    end
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

            if solver_diag_enabled && integrator.dt < solver_diag_dt_threshold
                stats = integrator.stats
                dnf       = stats.nf       - diag_prev_nf[]
                dnjacs    = stats.njacs    - diag_prev_njacs[]
                dnsolve   = stats.nsolve   - diag_prev_nsolve[]
                dnnonlin  = stats.nnonliniter - diag_prev_nnonliniter[]
                dnreject  = stats.nreject  - diag_prev_nreject[]
                println(">>> SOLVER-DIAG t=", round(integrator.t; digits=8),
                        " dt=", integrator.dt,
                        " Δnf=", dnf, " Δnjac=", dnjacs,
                        " Δnsolve=", dnsolve,
                        " Δnnonlin=", dnnonlin,
                        " Δnreject=", dnreject)
                flush(stdout)
                if formulation_profile_enabled
                    print_formulation_timing_report!(integrator; label="solver-diag dt-collapse")
                end
                diag_prev_nf[]          = stats.nf
                diag_prev_njacs[]       = stats.njacs
                diag_prev_nsolve[]      = stats.nsolve
                diag_prev_nnonliniter[] = stats.nnonliniter
                diag_prev_nreject[]     = stats.nreject
            end
            clip_commit_reduced_accepted_state!(integrator, v_ode_state, u_ode_state)
            u_modified!(integrator, thermal_Fth_u_modified[])
            thermal_Fth_u_modified[] = false
        end)

const retraction_save_final = sim_phase == "retraction" &&
    env_int("TP_CLIP_RETRACTION_SAVE_FINAL", 1) != 0
vtu_save_callback = if clip_reduced_ode_active()
    # The stock callback assumes integrator.u=[v,u]. The reduced state is [u,T],
    # so route output through the compatibility reconstruction helper.
    let save_index = Ref(1)
        DiscreteCallback(
            (u, t, integrator) ->
                save_index[] <= length(compress_hold_vtu_save_times) &&
                t + 1.0e-12 >= compress_hold_vtu_save_times[save_index[]],
            integrator -> begin
                save_vtu_snapshot!(integrator)
                while save_index[] <= length(compress_hold_vtu_save_times) &&
                      integrator.t + 1.0e-12 >=
                          compress_hold_vtu_save_times[save_index[]]
                    save_index[] += 1
                end
                u_modified!(integrator, false)
            end;
            save_positions=(false, false))
    end
elseif sim_phase == "retraction"
    SolutionSavingCallback(dt=retraction_vtu_save_dt,
                           output_directory=clip_out_dir,
                           prefix=vtu_output_prefix,
                           save_final_solution=retraction_save_final,
                           max_coordinates=Inf,
                           fiber_vf=vtu_fiber_vf_quantity,
                           a11=vtu_a11_quantity,
                           a22=vtu_a22_quantity,
                           a33=vtu_a33_quantity)
else
    SolutionSavingCallback(save_times=compress_hold_vtu_save_times,
                           output_directory=clip_out_dir,
                           prefix=vtu_output_prefix,
                           save_final_solution=true,
                           max_coordinates=Inf,
                           fiber_vf=vtu_fiber_vf_quantity,
                           a11=vtu_a11_quantity,
                           a22=vtu_a22_quantity,
                           a33=vtu_a33_quantity)
end
info_callback = InfoCallback(interval=retraction_info_interval)
const peak_punch_force_z = Ref(0.0)
const force_print_counter = Ref(0)
const force_print_interval = max(1, env_int("TP_CLIP_FORCE_PRINT_INTERVAL", 100))

press_force_limit_callback = DiscreteCallback(
    (u, t, integrator) -> begin
        if t <= t_compress
            if punch_contact_force_z[] > peak_punch_force_z[]
                peak_punch_force_z[] = punch_contact_force_z[]
            end
            force_print_counter[] += 1
            if force_print_counter[] % force_print_interval == 0
                println(">>> [t=", round(t; digits=5), " s] Current punch force: ", round(punch_contact_force_z[]; digits=2), " N (Peak so far: ", round(peak_punch_force_z[]; digits=2), " N)")
                flush(stdout)
            end
        end

        if tp_clip_max_press_force_n > 0.0 && !press_force_locked[] && t <= t_compress
            return punch_contact_force_z[] >= tp_clip_max_press_force_n
        end
        return false
    end,
    (integrator) -> begin
        press_force_locked[] = true
        press_force_locked_z[] = mold_z_shift_at_time(integrator.t)
        println(">>> PRESS TONNAGE LIMIT REACHED! Max force: ", punch_contact_force_z[], " N. Locking stroke at ", round(press_force_locked_z[] * 1e3, digits=3), " mm.")
        flush(stdout)
        return nothing
    end,
    save_positions=(false, false)
)

callbacks = if sim_phase == "retraction"
    CallbackSet(clip_trial_dt_sync_callback, accepted_step_callback, vtu_save_callback,
                info_callback, press_force_limit_callback)
else
    CallbackSet(clip_trial_dt_sync_callback, accepted_step_callback,
                vtu_save_callback,
                info_callback, press_force_limit_callback)
end

default_abstol = 1.0e-3
default_reltol = 1.0e-3
default_dtmax  = 1.0e-3

phase_abstol = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_ABSTOL", retraction_cfrp_style ? 1.0e-5 : default_abstol) :
    default_abstol
phase_reltol = sim_phase == "retraction" ?
    env_float("TP_CLIP_RETRACTION_RELTOL", default_reltol) :
    default_reltol
phase_dtmax = if sim_phase == "retraction"
    env_float("TP_CLIP_RETRACTION_DTMAX", retraction_cfrp_style ? 1.0e-4 : default_dtmax)
elseif sim_phase == "hold"
    charge_hold_phase_dtmax
else
    default_dtmax
end

# Global overrides (useful for debugging stiff starts / hold runs)
retraction_abstol = env_float("TP_CLIP_ABSTOL", phase_abstol)
retraction_reltol = env_float("TP_CLIP_RELTOL", phase_reltol)
retraction_dtmax  = env_float("TP_CLIP_DTMAX",  phase_dtmax)
if sim_phase in ("full", "hold") && charge_hold_dtmax_active[]
    retraction_dtmax = charge_hold_phase_dtmax
end
if sim_phase == "hold" && hold_checkpoint_restore !== nothing && hold_tlsph_tail_active(hold_checkpoint_restore["t"])
    retraction_dtmax = env_float("TP_CLIP_HOLD_TAIL_DTMAX", 1.0e-3)
end


# Offline gap / penetration audit from a hold checkpoint (no ODE solve).
# Opt-in ONLY via TP_CLIP_RUN_GAP_AUDIT_ONLY=1 (TP_CLIP_AUDIT_CHECKPOINT_GAPS is
# force-cleared by the 3d driver — leaked =1 aborted real retractions via exit(0)).
#   TP_CLIP_RUN_GAP_AUDIT_ONLY=1 TP_CLIP_CHECKPOINT=out/...dat julia ... implicit.jl
if env_int("TP_CLIP_RUN_GAP_AUDIT_ONLY", 0) != 0
    let
        ck_aud = load_hold_checkpoint(checkpoint_path)
        restore_hold_checkpoint_state!(ck_aud, semi)
        cyl_sys = semi.systems[1]
        floor_sys = clip_floor_system(semi)
        mold_sys = clip_mold_system(semi)
        local n_floor_pen = 0
        local n_mold_pen = 0
        local min_floor_g = Inf
        local min_mold_g = Inf
        local sum_floor_pen = 0.0
        local sum_mold_pen = 0.0
        local n_deep_floor = 0
        deep_tol = -0.25 * particle_spacing
        @inbounds for particle in TrixiParticles.each_integrated_particle(cyl_sys)
            x_i = cyl_sys.current_coordinates[1, particle]
            z_i = cyl_sys.current_coordinates[3, particle]
            gap_f = tool_surface_gap(floor_sys, x_i, z_i, particle_spacing; side=:lower)
            gap_m = tool_surface_gap(mold_sys, x_i, z_i, particle_spacing; side=:upper)
            min_floor_g = min(min_floor_g, gap_f)
            min_mold_g = min(min_mold_g, gap_m)
            if gap_f < 0.0
                n_floor_pen += 1
                sum_floor_pen += gap_f
            end
            if gap_m < 0.0
                n_mold_pen += 1
                sum_mold_pen += gap_m
            end
            gap_f < deep_tol && (n_deep_floor += 1)
        end
        n_charge = nparticles(cyl_sys)
        println("\n>>> CHECKPOINT GAP AUDIT")
        println("    file=", abspath(checkpoint_path))
        println("    t=", round(ck_aud["t"]; digits=6),
                " s  solidification_complete=", get(ck_aud, "solidification_complete", missing))
        println("    n_charge=", n_charge)
        println("    min_floor_gap=", round(min_floor_g * 1e3; digits=4), " mm",
                "  floor_penetrators=", n_floor_pen, "/", n_charge,
                n_floor_pen > 0 ?
                    "  mean_pen=" * string(round(sum_floor_pen / n_floor_pen * 1e3; digits=4)) * " mm" : "")
        println("    min_mold_gap=", round(min_mold_g * 1e3; digits=4), " mm",
                "  mold_penetrators=", n_mold_pen, "/", n_charge,
                n_mold_pen > 0 ?
                    "  mean_pen=" * string(round(sum_mold_pen / n_mold_pen * 1e3; digits=4)) * " mm" : "")
        println("    deep_floor (gap<-0.25·ps)=", n_deep_floor)
        flush(stdout)
    end
    exit(0)
end

# Optional: benchmark TRBDF2 from a saved hold checkpoint (compression stall diagnosis).
#   TP_CLIP_PROFILE_FROM_CHECKPOINT=1 TP_CLIP_FORMULATION_PROFILE=1 julia ... implicit.jl
if env_int("TP_CLIP_PROFILE_FROM_CHECKPOINT", 0) != 0
    profile_steps = max(1, env_int("TP_CLIP_PROFILE_STEPS", 5))
    ck_prof = load_hold_checkpoint(checkpoint_path)
    restore_hold_checkpoint_state!(ck_prof, semi)
    t_prof0 = ck_prof["t"]
    t_prof1 = t_prof0 + env_float("TP_CLIP_PROFILE_DT_SIM", 1.0e-6)
    v_prof, u_prof = clip_ode_state_from_checkpoint(ck_prof)
    ode_prof = clip_build_ode_problem(v_prof, u_prof, (t_prof0, t_prof1), semi)
    n_ode_dof_prof = length(v_prof) + length(u_prof)
    integrator_mode_prof = clip_integrator_mode()
    jacobian_strategy_prof = clip_jacobian_strategy()
    use_matrix_free_prof = jacobian_strategy_prof == "matrix_free"
    jacobian_mode_prof = clip_jacobian_mode()
    jacobian_autodiff_prof = use_matrix_free_prof ? nothing :
        clip_jacobian_autodiff(jacobian_mode_prof)
    linsolve_default_prof = use_matrix_free_prof ? "gmres" : "dense"
    linsolve_prof = clip_linsolve_solver(n_ode_dof_prof;
                                         default_mode=linsolve_default_prof,
                                         matrix_free=use_matrix_free_prof,
                                         jacobian_strategy=jacobian_strategy_prof)
    alg_prof = clip_build_implicit_algorithm(integrator_mode_prof;
                                           linsolve_solver=linsolve_prof,
                                           use_matrix_free=use_matrix_free_prof,
                                           jacobian_autodiff=jacobian_autodiff_prof,
                                           nlsolve=nlsolve_prof)
    nlsolve_prof = NLNewton(κ=1e-1, max_iter=20, fast_convergence_cutoff=0.9, always_new=false)
    println("\n>>> PROFILE FROM CHECKPOINT t0=", round(t_prof0; digits=6),
            " s, n_dof=", n_ode_dof_prof,
            ", profile_steps<=", profile_steps,
            ", integrator=", integrator_mode_prof,
            ", jacobian_strategy=", jacobian_strategy_prof,
            use_matrix_free_prof ? "" : ", jacobian_mode=" * jacobian_mode_prof, ")")
    flush(stdout)
    t_rhs0 = time_ns()
    sol_prof = clip_gpu_allowscalar() do
        solve(ode_prof, alg_prof;
              save_everystep=false,
              abstol=1.0e-3, reltol=1.0e-3,
              dtmax=1.0e-3,
              maxiters=profile_steps,
              callback=CallbackSet(accepted_step_callback, info_callback))
    end
    rhs_wall_s = (time_ns() - t_rhs0) / 1.0e9
    st = sol_prof.destats
    t_end_prof = sol_prof.t isa AbstractVector ? last(sol_prof.t) : sol_prof.t
    println(">>> PROFILE DONE: wall=", round(rhs_wall_s; digits=3), " s",
            " t_end=", round(t_end_prof; digits=9),
            " retcode=", sol_prof.retcode)
    println(">>> PROFILE STATS (cumulative): nf=", st.nf,
            " njacs=", st.njacs, " nsolve=", st.nsolve,
            " nnonliniter=", st.nnonliniter, " nreject=", st.nreject)
    if rhs_eval_count[] > 0
        print_formulation_timing_report_final!(t_end_prof; label="checkpoint profile")
    end
    close(mold_force_io)
    retraction_diag_io !== nothing && close(retraction_diag_io)
    exit(0)
end

sol = clip_gpu_allowscalar() do
    Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Warn)) do
    if clip_use_explicit_solver()
        explicit_alg, explicit_scheme_name = clip_explicit_euler_algorithm()
        explicit_dt = compression_use_explicit ? compression_explicit_dt :
            (retraction_pre_lift_creep_s > 0.0 ? retraction_explicit_dt_mech_off :
                retraction_explicit_dt_springback)
        explicit_phase = compression_use_explicit ? "compression" : "retraction"
        explicit_ignore_unstable = compression_use_explicit ?
            compression_explicit_ignore_unstable : retraction_explicit_ignore_unstable
        println("\n>>> Entering ODE solve (tspan=", tspan,
                ", ", explicit_phase, " solver=explicit ", explicit_scheme_name,
                ", dt=", explicit_dt,
                compression_use_explicit ? "" :
                    ", freeze T=$(retraction_unfreeze_t_after_punch_lift ? "demold-frozen→unfreeze after punch lift" : (retraction_unfreeze_t_after_eject ? "demold-frozen→unfreeze after eject" : (retraction_explicit_freeze_t ? "always" : "off")))", ")")
        flush(stdout)
        if explicit_ignore_unstable
            solve(ode, explicit_alg;
                  callback=callbacks,
                  save_everystep=false,
                  adaptive=false,
                  dt=explicit_dt,
                  unstable_check=(dt, u, p, t) -> false,
                  maxiters=10_000_000)
        else
            solve(ode, explicit_alg;
                  callback=callbacks,
                  save_everystep=false,
                  adaptive=false,
                  dt=explicit_dt,
                  maxiters=10_000_000)
        end
    else
        integrator_mode = clip_integrator_mode()
        jacobian_strategy = clip_jacobian_strategy()
        use_matrix_free = jacobian_strategy == "matrix_free"
        use_custom_jac = clip_uses_custom_jacobian(jacobian_strategy)
        jacobian_mode = if use_matrix_free
            "matrix_free_jvp"
        elseif jacobian_strategy == "sparse"
            "sparse_finite"
        elseif jacobian_strategy == "colored"
            "colored_finite"
        elseif retraction_cfrp_style
            haskey(ENV, "TP_CLIP_JACOBIAN_MODE") ? clip_jacobian_mode() : "finite"
        else
            clip_jacobian_mode()
        end
        jacobian_autodiff = (use_matrix_free || use_custom_jac) ? nothing :
            (retraction_cfrp_style && !haskey(ENV, "TP_CLIP_JACOBIAN_MODE") ?
                AutoFiniteDiff() : clip_jacobian_autodiff(clip_jacobian_mode()))
        n_ode_dof = clip_ode_reduced[] ? length(ode.u0) :
            length(v0_ode) + length(u0_ode)
        linsolve_default = use_matrix_free ? "gmres" :
            ((retraction_cfrp_style && n_ode_dof >= 500) ? "gmres" : "dense")
        linsolve_solver = clip_linsolve_solver(n_ode_dof;
                                               default_mode=linsolve_default,
                                               matrix_free=use_matrix_free,
                                               jacobian_strategy=jacobian_strategy)
        linsolve_name = lowercase(get(ENV, "TP_CLIP_LINSOLVE", linsolve_default))
        if jacobian_strategy == "sparse" && linsolve_name in ("dense", "rfluf", "lu")
            println(">>> WARNING: TP_CLIP_LINSOLVE=", linsolve_name,
                    " is incompatible with sparse Jacobian (W is SparseMatrixCSC); ",
                    "forcing sparse UMFPACK factorization.")
            flush(stdout)
            linsolve_solver = LinearSolve.UMFPACKFactorization()
            linsolve_name = "sparse"
        elseif jacobian_strategy == "colored" && linsolve_name == "sparse"
            println(">>> WARNING: TP_CLIP_LINSOLVE=sparse is incompatible with colored dense ",
                    "Jacobian; forcing dense LU (RFLU).")
            flush(stdout)
            linsolve_solver = RFLUFactorization()
            linsolve_name = "dense"
        end
        nlsolve = retraction_cfrp_style ? NLNewton() :
            NLNewton(κ=env_float("TP_CLIP_NEWTON_KAPPA", 0.1),
                     max_iter=env_int("TP_CLIP_NEWTON_MAX_ITER", 20),
                     fast_convergence_cutoff=env_float("TP_CLIP_NEWTON_FAST_CUTOFF", 0.9),
                     new_W_dt_cutoff=env_float("TP_CLIP_NEWTON_NEW_W_DT_CUTOFF", 1 / 5),
                     always_new=env_int("TP_CLIP_NEWTON_ALWAYS_NEW", 0) != 0)
        implicit_alg = clip_build_implicit_algorithm(integrator_mode;
                                                     linsolve_solver=linsolve_solver,
                                                     use_matrix_free=use_matrix_free,
                                                     jacobian_autodiff=jacobian_autodiff,
                                                     nlsolve=nlsolve)
        println("\n>>> Entering ODE solve (tspan=", tspan,
                ", implicit solver=", integrator_mode,
                ", jacobian_strategy=", jacobian_strategy,
                ", jacobian=", jacobian_mode,
                ", linsolve=", linsolve_name, ", n_dof=", n_ode_dof,
                clip_jac_reuse_entries ?
                    ", jac_entry_reuse=every_$(clip_jac_reuse_max_steps)_requests" :
                    ", jac_entry_reuse=off",
                env_int("TP_CLIP_ADAPTIVE", 1) == 0 ?
                    ", adaptive=false, fixed_dt=$(env_float("TP_CLIP_FIXED_DT",
                        env_float("TP_CLIP_DT0", 1.0e-6)))" :
                    ", dt0=$(env_float("TP_CLIP_DT0", 1.0e-6)), dtmin=$(env_float("TP_CLIP_DTMIN", 1.0e-12)), dtmax=$(retraction_dtmax)",
                retraction_cfrp_style ? ", style=CFRP" : "", ")")
        flush(stdout)
        # Guard against OrdinaryDiffEq auto-initial-dt producing NaN when the initial RHS
        # has NaNs in a Newton trial (e.g. stiff contact/packing transients). A finite dt0
        # keeps the integrator well-posed; adaptivity will quickly adjust it.
        default_dt0 = (sim_phase == "hold" && hold_checkpoint_restore !== nothing && hold_tlsph_tail_active(hold_checkpoint_restore["t"])) ? 1.0e-4 : 1.0e-6
        dt0 = env_float("TP_CLIP_DT0", default_dt0)
        dtmin0 = env_float("TP_CLIP_DTMIN", 1.0e-12)
        clip_last_synced_integrator_dt[] = -1.0
        clip_integrator_dt[] = dt0
        trial_dt_state[] = dt0
        use_adaptive = env_int("TP_CLIP_ADAPTIVE", 1) != 0
        if use_adaptive
            force_dtmin = env_int("TP_CLIP_FORCE_DTMIN", 0) != 0
            solve(ode, implicit_alg;
                  callback=callbacks,
                  save_everystep=false,
                  abstol=retraction_abstol,
                  reltol=retraction_reltol,
                  dt=dt0,
                  dtmin=dtmin0,
                  dtmax=retraction_dtmax,
                  force_dtmin=force_dtmin,
                  maxiters=10_000_000)
        else
            fixed_dt = env_float("TP_CLIP_FIXED_DT", dt0)
            solve(ode, implicit_alg;
                  callback=callbacks,
                  save_everystep=false,
                  adaptive=false,
                  dt=fixed_dt,
                  maxiters=10_000_000)
        end
    end
    end
end

if formulation_profile_enabled && rhs_eval_count[] > 0
    # `sol.t` is a vector of saved times; report the last time reached.
    print_formulation_timing_report_final!(sol.t[end]; label="final partial window")
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
