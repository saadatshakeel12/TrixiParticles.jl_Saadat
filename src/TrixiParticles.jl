module TrixiParticles

using Reexport: @reexport

using Adapt: Adapt
using Base: @propagate_inbounds
using CSV: CSV
using Dates
using DataFrames: DataFrames, DataFrame
using DelimitedFiles: DelimitedFiles
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect, PresetTimeCallback
using FastPow: @fastpow
using FileIO: FileIO
using ForwardDiff: ForwardDiff
using GPUArraysCore: AbstractGPUArray
using JSON: JSON
using KernelAbstractions: KernelAbstractions, @kernel, @index
using LinearAlgebra: norm, normalize, cross, dot, I, tr, inv, pinv, det, eigen,
                     Symmetric, Diagonal
using MuladdMacro: @muladd
using Polyester: Polyester, @batch
using Printf: @printf, @sprintf
using ReadVTK: ReadVTK
using RecipesBase: RecipesBase, @series
using Random: seed!
using SciMLBase: SciMLBase, CallbackSet, DiscreteCallback, DynamicalODEProblem, u_modified!,
                 get_tmp_cache, set_proposed_dt!, ODESolution, ODEProblem, terminate!,
                 add_tstop!, get_du
@reexport using StaticArrays: SVector
using StaticArrays: @SMatrix, SMatrix, setindex
using Statistics: Statistics
using StrideArraysCore: PtrArray, StaticInt
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!, @notimeit
using TrixiBase: @trixi_timeit, timer, timeit_debug_enabled,
                 disable_debug_timings, enable_debug_timings, TrixiBase
@reexport using TrixiBase: trixi_include, trixi_include_changeprecision
@reexport using PointNeighbors: TrivialNeighborhoodSearch, GridNeighborhoodSearch,
                                PrecomputedNeighborhoodSearch, PeriodicBox,
                                ParallelUpdate, SemiParallelUpdate, SerialUpdate,
                                SerialIncrementalUpdate,
                                FullGridCellList, DictionaryCellList,
                                SerialBackend, PolyesterBackend, ThreadsStaticBackend,
                                ThreadsDynamicBackend, default_backend
using PointNeighbors: PointNeighbors, foreach_point_neighbor, copy_neighborhood_search,
                      @threaded
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes, paraview_collection, vtk_save

# `util.jl` needs to be first because of the macros `@trixi_timeit` and `@threaded`
include("util.jl")
include("general/abstract_system.jl")
include("general/general.jl")
include("setups/setups.jl")
include("schemes/schemes.jl")
# `neighborhood_search.jl` requires the system types to be defined
include("general/neighborhood_search.jl")
# `callbacks.jl` requires the system types to be defined
include("callbacks/callbacks.jl")

# Note that `semidiscretization.jl` depends on the system types and has to be
# included separately. `gpu.jl` in turn depends on the semidiscretization type.
include("general/semidiscretization.jl")
include("general/gpu.jl")
include("preprocessing/preprocessing.jl")
include("io/io.jl")
include("visualization/recipes_plots.jl")

# Module-level stress tensor cache: set by callback, read by structure interactions.
# The legacy `(system_objectid, array)` form remains supported. CPU drivers with
# several independently integrated structures may instead install a dictionary.
const StressTensorArray = AbstractArray{Float64, 3}
const StressTensorCache = Union{Nothing,
                                Tuple{UInt64, StressTensorArray},
                                Dict{UInt64, StressTensorArray}}
const STRESS_TENSOR_CACHE = Ref{StressTensorCache}(nothing)

@inline function stress_tensor_cache_for(system)
    cached = STRESS_TENSOR_CACHE[]
    cached === nothing && return nothing
    system_id = objectid(system)
    if cached isa Tuple
        return cached[1] == system_id ? cached[2] : nothing
    end
    return get(cached, system_id, nothing)
end

export Semidiscretization, semidiscretize, restart_with!
export InitialCondition
export WeaklyCompressibleSPHSystem, EntropicallyDampedSPHSystem, TotalLagrangianSPHSystem,
       TotalLagrangianSPHSystem3d, WallBoundarySystem, DEMSystem, BoundaryDEMSystem, OpenBoundarySystem,
       ImplicitIncompressibleSPHSystem, thermomechanical_loop, thermomechanical_loop3d, eachparticle, update_temperature_sph!,
    update_temperature_sph3d!, elastic_stress3d!, viscous_stress3d!, update_properties!,
    elastic_stress3d_fast!, viscous_stress3d_fast!, elastic_stress3d_trial!, thermal_rhs_sph3d!
export BoundaryZone, InFlow, OutFlow, BidirectionalFlow
export InfoCallback, SolutionSavingCallback, DensityReinitializationCallback,
       PostprocessCallback, StepsizeCallback, UpdateCallback, SteadyStateReachedCallback,
       SplitIntegrationCallback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller,
    TensileArtificialStressMonaghan,
       TransportVelocityAdami, ParticleShiftingTechnique,
       ParticleShiftingTechniqueSun2017, ConsistentShiftingSun2019,
       ContinuityEquationTermSun2019, MomentumEquationTermSun2019
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel,
       SchoenbergQuinticSplineKernel, GaussianKernel, WendlandC2Kernel, WendlandC4Kernel,
       WendlandC6Kernel, SpikyKernel, Poly6Kernel, LaguerreGaussKernel
export StateEquationCole, StateEquationIdealGas, StateEquationAdaptiveCole
export ArtificialViscosityMonaghan, ViscosityAdami, ViscosityMorris, ViscosityAdamiSGS,
       ViscosityMorrisSGS
export DensityDiffusionMolteniColagrossi, DensityDiffusionFerrari, DensityDiffusionAntuono
export tensile_instability_control
export OrganosheetPlyModel, assign_organosheet_ply_ids, same_organosheet_ply,
       organosheet_fiber_energy, organosheet_fiber_pk1,
       organosheet_shear_angle, organosheet_shear_torque,
       organosheet_trellis_energy, organosheet_trellis_pk1,
       organosheet_kirchhoff_stress, coulomb_tangential_velocity_increment
export BoundaryModelMonaghanKajtar, BoundaryModelDummyParticles, AdamiPressureExtrapolation,
       PressureMirroring, PressureZeroing, BoundaryModelCharacteristicsLastiwka,
       BoundaryModelMirroringTafuni, BoundaryModelDynamicalPressureZhang,
       BernoulliPressureExtrapolation, PressureBoundaries
export FirstOrderMirroring, ZerothOrderMirroring, SimpleMirroring
export HertzContactModel, LinearContactModel
export PrescribedMotion, OscillatingMotion2D
export RCRWindkesselModel
export examples_dir, validation_dir
export trixi2vtk, vtk2trixi
export RectangularTank, RectangularShape, SphereShape, ComplexShape
export ParticlePackingSystem, SignedDistanceField
export WindingNumberHormann, WindingNumberJacobson
export VoxelSphere, RoundSphere, reset_wall!, extrude_geometry, load_geometry,
       sample_boundary, planar_geometry_to_face
export SourceTermDamping
export ShepardKernelCorrection, KernelCorrection, AkinciFreeSurfaceCorrection,
       GradientCorrection, BlendedGradientCorrection, MixedKernelGradientCorrection
export nparticles
export available_data, kinetic_energy, total_mass, max_pressure, min_pressure, avg_pressure,
       max_density, min_density, avg_density
export interpolate_line, interpolate_points, interpolate_plane_3d, interpolate_plane_2d,
       interpolate_plane_2d_vtk
export SurfaceTensionAkinci, CohesionForceAkinci, SurfaceTensionMorris,
       SurfaceTensionMomentumMorris
export ColorfieldSurfaceNormal
export SymplecticPositionVerlet

end # module
