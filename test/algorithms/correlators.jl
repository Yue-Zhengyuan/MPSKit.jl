println("
-----------------------------
|   Correlators & Entropy    |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Random

@testset "correlation length / entropy" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^10])
    H = force_planar(transverse_field_ising())
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
    len_crit = correlation_length(ψ; sector = leftunit(ψ))
    entrop_crit = entropy(ψ)

    H = force_planar(transverse_field_ising(; g = 4))
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
    len_gapped = correlation_length(ψ; sector = leftunit(ψ))
    entrop_gapped = entropy(ψ)

    @test len_crit > len_gapped
    @test real(entrop_crit) > real(entrop_gapped)
end

@testset "momentum correlator" begin
    P = ℂ^2
    A = TensorMap(ones(Float64, 2, 1), ℂ^1 ⊗ P, ℂ^1)
    ψ_product = InfiniteMPS([A])
    X = TensorMap(Float64[0 1; 1 0], P, P)
    Z = TensorMap(Float64[1 0; 0 -1], P, P)

    @test only(momentum_correlator(ψ_product, X ⊗ X, X * X, 0.0)) ≈ 0 atol = 1.0e-12
    @test only(momentum_correlator(ψ_product, Z ⊗ Z, Z * Z, 0.37)) ≈ 1
    @test structure_factor(ψ_product, Z ⊗ Z, Z * Z, 0.37) ≈ 1

    Random.seed!(1234)
    D = 2
    ψ = InfiniteMPS(
        [
            TensorMap(randn(ComplexF64, D * dim(P), D), ℂ^D ⊗ P, ℂ^D),
            TensorMap(randn(ComplexF64, D * dim(P), D), ℂ^D ⊗ P, ℂ^D),
        ]
    )
    O₁₂ = X ⊗ Z
    O₁O₂ = X * Z
    O₁, O₂ = MPSKit.decompose_localmpo(MPSKit.add_util_leg(O₁₂))
    O₂_reverse, O₁_reverse = MPSKit.decompose_localmpo(
        MPSKit.add_util_leg(permute(O₁₂, ((2, 1), (4, 3))))
    )
    μ₁ = [expectation_value(ψ, (s,) => X) for s in 1:length(ψ)]
    μ₂ = [expectation_value(ψ, (s,) => Z) for s in 1:length(ψ)]

    """Approximate the connected Fourier sum in a finite real-space window."""
    function truncated_correlator(k; max_cells = 80)
        N_c = length(ψ)
        G = zeros(ComplexF64, N_c, N_c)
        for s in 1:N_c, s′ in 1:N_c
            same_cell = if s == s′
                expectation_value(ψ, (s,) => O₁O₂)
            elseif s < s′
                correlator(ψ, O₁, O₂, s, s′)
            else
                correlator(ψ, O₂_reverse, O₁_reverse, s′, s)
            end
            disconnected = μ₁[s] * μ₂[s′]
            G[s, s′] = cis(-k * (s′ - s)) * (same_cell - disconnected)
            for R in 1:max_cells
                positive = correlator(ψ, O₁, O₂, s, R * N_c + s′) - disconnected
                negative =
                    correlator(ψ, O₂_reverse, O₁_reverse, s′, R * N_c + s) -
                    disconnected
                G[s, s′] += cis(-k * (R * N_c + s′ - s)) * positive
                G[s, s′] += cis(-k * (-R * N_c + s′ - s)) * negative
            end
        end
        return G
    end

    for k in (0.0, 0.37, π)
        G = momentum_correlator(ψ, O₁₂, O₁O₂, k)
        @test G ≈ truncated_correlator(k) atol = 1.0e-9
        @test momentum_correlator(ψ, O₁₂, O₁O₂, k + 2π) ≈ G atol = 1.0e-10
        @test structure_factor(ψ, O₁₂, O₁O₂, k) ≈ sum(G) / length(ψ)
    end

    ks = [π, 0.37, 0.0, 0.37, 0.37 + 2π]
    Gs = momentum_correlator(ψ, O₁₂, O₁O₂, ks)
    @test length(Gs) == length(ks)
    for (G, k) in zip(Gs, ks)
        @test G ≈ momentum_correlator(ψ, O₁₂, O₁O₂, k) atol = 1.0e-10
        @test G ≈ truncated_correlator(k) atol = 1.0e-9
    end
    @test Gs[2] ≈ Gs[4]
    @test Gs[2] ≈ Gs[5] atol = 1.0e-10
    @test structure_factor(ψ, O₁₂, O₁O₂, ks) ≈
        [sum(G) / length(ψ) for G in Gs]

    ks_range = range(-0.2, 0.2; length = 4)
    range_Gs = momentum_correlator(ψ, O₁₂, O₁O₂, ks_range)
    scalar_range_Gs = [momentum_correlator(ψ, O₁₂, O₁O₂, k) for k in ks_range]
    @test all(((G, scalar_G),) -> G ≈ scalar_G, zip(range_Gs, scalar_range_Gs))
    @test only(momentum_correlator(ψ, O₁₂, O₁O₂, [0.23])) ≈
        momentum_correlator(ψ, O₁₂, O₁O₂, 0.23)
    @test isempty(momentum_correlator(ψ, O₁₂, O₁O₂, Float64[]))
    @test isempty(structure_factor(ψ, O₁₂, O₁O₂, Float64[]))

    G_hermitian = momentum_correlator(ψ, Z ⊗ Z, Z * Z, 0.41)
    @test G_hermitian ≈ G_hermitian'

    O₁₂_fermion = f_plus_f_min(Float64, Trivial)
    O₁O₂_fermion = f_num(Float64, Trivial)
    P_fermion = space(O₁O₂_fermion, 1)
    ψ_fermion = InfiniteMPS(P_fermion, P_fermion)
    O₁_fermion, O₂_fermion = MPSKit.decompose_localmpo(
        MPSKit.add_util_leg(O₁₂_fermion)
    )
    O₂_reverse, O₁_reverse = MPSKit.decompose_localmpo(
        MPSKit.add_util_leg(permute(O₁₂_fermion, ((2, 1), (4, 3))))
    )

    """Approximate the fermionic Fourier sum while preserving braided site reversal."""
    function truncated_fermionic_correlator(k; max_cells = 80)
        G = expectation_value(ψ_fermion, (1,) => O₁O₂_fermion)
        for R in 1:max_cells
            G += cis(-k * R) * correlator(ψ_fermion, O₁_fermion, O₂_fermion, 1, R + 1)
            G += cis(k * R) * correlator(ψ_fermion, O₂_reverse, O₁_reverse, 1, R + 1)
        end
        return G
    end

    @test O₁₂_fermion ≠ permute(O₁₂_fermion, ((2, 1), (4, 3)))
    @test only(momentum_correlator(ψ_fermion, O₁₂_fermion, O₁O₂_fermion, 0.41)) ≈
        truncated_fermionic_correlator(0.41) atol = 1.0e-9
    fermion_ks = [0.0, 0.41, π]
    @test only.(
        momentum_correlator(
            ψ_fermion, O₁₂_fermion, O₁O₂_fermion, fermion_ks
        )
    ) ≈ truncated_fermionic_correlator.(fermion_ks) atol = 1.0e-9
end

@testset "expectation value / correlator" begin
    g = 4.0
    ψ = InfiniteMPS(ℂ^2, ℂ^10)
    H = transverse_field_ising(; g)
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))

    @test expectation_value(ψ, H) ≈
        expectation_value(ψ, 1 => -2g * S_x()) + expectation_value(ψ, (1, 2) => -4S_z_S_z())
    Z_mpo = MPSKit.add_util_leg(S_z())
    G = correlator(ψ, Z_mpo, Z_mpo, 1, 2:5)
    G2 = correlator(ψ, S_z_S_z(), 1, 3:2:5)
    @test isapprox(G[2], G2[1], atol = 1.0e-2)
    @test isapprox(last(G), last(G2), atol = 1.0e-2)
    @test isapprox(G[1], expectation_value(ψ, (1, 2) => S_z_S_z()), atol = 1.0e-2)
    @test isapprox(G[2], expectation_value(ψ, (1, 3) => S_z_S_z()), atol = 1.0e-2)

    # the sites have to be ordered `i < j`: every `correlator` method funnels its
    # arguments through the range version, so all three signatures have to throw
    @test_throws ArgumentError correlator(ψ, Z_mpo, Z_mpo, 3, 2)
    @test_throws ArgumentError correlator(ψ, Z_mpo, Z_mpo, 3, 1:2)
    @test_throws ArgumentError correlator(ψ, S_z_S_z(), 3, 2)
    # equal sites are not allowed either
    @test_throws ArgumentError correlator(ψ, Z_mpo, Z_mpo, 2, 2)
end
