# ==========================================================================================
# Full-3D CFK clip molding (coarse-first entry point)
#
# Same physics / solver stack as `stamping_cfk_clip_pseudo2d_implicit.jl`, but:
#   - TP_CLIP_GEOMETRY=full3d  → resolved charge width in y (no plane-strain y lock)
#   - Mirror-symmetric tow/gap layout (centered on charge mid-plane)
#   - Separate checkpoint / VTU prefix: molding_cfk_3d_*
#
# User stepped L-die STLs (same pseudo2d stack, STL packing):
#   stamping_cfk_clip_3d_l_implicit.jl       (full3d y)
#   stamping_cfk_clip_3d_l_thin3d_implicit.jl (thin3d y)
#   CAD: examples/preprocessing/data_user_l_die/{workpiece,mold,punch}.stl
#
# Coarse defaults below (ps = 2.4 mm). Override any TP_CLIP_* env var before launch.
# ==========================================================================================

const _CLIP_FULL3D_DIR = @__DIR__

function _clip_full3d_set_default!(key::String, value::String)
    haskey(ENV, key) || (ENV[key] = value)
    return nothing
end

_clip_full3d_set_default!("TP_CLIP_GEOMETRY", "full3d")
_clip_full3d_set_default!("TP_CLIP_PS_MM", "2.4")
_clip_full3d_set_default!("TP_CLIP_CHARGE_WIDTH_MM", "3.6")
_clip_full3d_set_default!("TP_CLIP_N_PLIES", "3")
_clip_full3d_set_default!("TP_CLIP_SIM_PHASE", "hold")
_clip_full3d_set_default!("TP_CLIP_SAVE_CHECKPOINT", "1")
_clip_full3d_set_default!("TP_CLIP_COMPRESSION_RATIO", "0.975")
_clip_full3d_set_default!("TP_CLIP_VTU_SAVE_DT", "0.05")
_clip_full3d_set_default!("TP_CLIP_SOLIDIFICATION_HOLD_MAX_S", "35")
_clip_full3d_set_default!("TP_CLIP_POST_SOLID_DWELL_S", "0.0")
_clip_full3d_set_default!("TP_CLIP_MU_FRICTION", "0.04")

include(joinpath(_CLIP_FULL3D_DIR, "stamping_cfk_clip_pseudo2d_implicit.jl"))
