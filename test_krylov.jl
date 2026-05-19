using OrdinaryDiffEq
using LinearSolve
using LinearAlgebra

function f1!(dv, v, u, p, t)
    dv .= -v .+ u
end
function f2!(du, v, u, p, t)
    du .= v
end
u0 = rand(10)
v0 = rand(10)
prob = DynamicalODEProblem(f1!, f2!, v0, u0, (0.0, 1.0))
sol = solve(prob, TRBDF2(linsolve=KrylovJL_GMRES(), autodiff=AutoFiniteDiff()); abstol=1e-3, reltol=1e-2)
println(sol.retcode)
