@trixi_testset "structure/uniaxial_sandstone_pseudo2d_implicit.jl" begin
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(examples_dir(), "structure",
                                              "uniaxial_sandstone_pseudo2d_implicit.jl"),
                                     particle_spacing=0.01,
                                     specimen_width=0.02,
                                     specimen_height=0.04,
                                     target_engineering_strain=5e-4,
                                     diagnostics_interval=5,
                                     output_dt=5e-4,
                                     postprocess_dt=5e-4,
                                     tspan=(0.0, 5e-4)) [
        r"\[ Info: To create the self-interaction neighborhood search.*\n"
    ]
    @test sol.retcode == ReturnCode.Success
    @test count_rhs_allocations(sol, semi) < 5_000
end
