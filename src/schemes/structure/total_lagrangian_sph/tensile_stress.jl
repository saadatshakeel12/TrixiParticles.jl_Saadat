@doc raw"""
    TensileArtificialStressMonaghan(; psi=-0.1, exponent=4)

Artificial tensile stress correction for solid SPH to reduce tensile instability.
The correction follows the common Monaghan/Gray form
``(R_a + R_b) f_{ab}^n`` with a user-controlled scaling ``psi`` and exponent ``n``.

# Keywords
- `psi=-0.1`: Scaling for positive principal stresses. Negative values are typically used.
- `exponent=4`: Exponent ``n`` applied to ``f_{ab}``.
"""
struct TensileArtificialStressMonaghan{ELTYPE}
    psi      :: ELTYPE
    exponent :: Int

    function TensileArtificialStressMonaghan(; psi=-0.1, exponent=4)
        new{typeof(psi)}(psi, exponent)
    end
end

@inline function artificial_tensile_stress_tensor(::Nothing, stress_tensor)
    return zero(stress_tensor)
end

@inline function artificial_tensile_stress_tensor(model::TensileArtificialStressMonaghan,
                                                  stress_tensor::SMatrix{NDIMS, NDIMS, ELTYPE}) where {NDIMS, ELTYPE}
    principal = eigen(Symmetric(Matrix(stress_tensor)))

    R_principal = similar(principal.values)
    for i in eachindex(principal.values)
        sigma_i = principal.values[i]
        R_principal[i] = sigma_i > zero(ELTYPE) ? model.psi * sigma_i : zero(ELTYPE)
    end

    R = principal.vectors * Diagonal(R_principal) * principal.vectors'
    return SMatrix{NDIMS, NDIMS, ELTYPE}(R)
end
