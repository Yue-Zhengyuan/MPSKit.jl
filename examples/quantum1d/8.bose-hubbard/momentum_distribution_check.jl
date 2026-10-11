using MPSKit, TensorKit
using TensorKitTensors.BosonOperators: b_min
include(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "models.jl"))
using .ExampleModels
using Random
using Test

Random.seed!(123)

"""Compute the connected momentum distribution by truncating the real-space sum."""
function finite_window_momentum_distribution(
        state::InfiniteMPS, a_op::AbstractTensorMap, ks; npoints::Int
    )
    O₁, O₂ = MPSKit.decompose_localmpo(MPSKit.add_util_leg(a_op' ⊗ a_op))
    correlations = Vector{ComplexF64}(undef, npoints)
    correlations[1] = expectation_value(state, (1,) => a_op' * a_op)
    correlations[2:end] .= correlator(state, O₁, O₂, 1, 2:npoints)
    disconnected = abs2(expectation_value(state, (1,) => a_op))

    return map(ks) do k
        onsite = real(correlations[1] - disconnected)
        tail = sum(1:(npoints - 1)) do r
            return 2 * real(cis(-k * r) * (correlations[r + 1] - disconnected))
        end
        return onsite + tail
    end
end

"""Apply the cosine transform used by the original example to precomputed correlations."""
function example_fourier_transform(correlations, disconnected, ks)
    return map(ks) do k
        return correlations[1] - disconnected + sum(2:length(correlations)) do i
            return 2 * cos(k * i) * (correlations[i] - disconnected)
        end
    end
end

"""Reproduce the finite-window expression currently used in the Bose–Hubbard example."""
function example_momentum_distribution(
        state::InfiniteMPS, a_op::AbstractTensorMap, ks; npoints::Int
    )
    correlations = real.(
        [
            expectation_value(state, (1, i) => a_op' ⊗ a_op) for i in 1:npoints
        ]
    )
    disconnected = abs2(expectation_value(state, (1,) => a_op))
    return example_fourier_transform(correlations, disconnected, ks)
end

cutoff = 4
D = 5
μ, t = 0.5, 0.2
a_op = b_min(; cutoff)
hamiltonian = bose_hubbard_model(; cutoff, U = 1, mu = μ, t)
state = InfiniteMPS(ℂ^(cutoff + 1), ℂ^D)
state, = find_groundstate(
    state, hamiltonian, VUMPS(; tol = 1.0e-8, verbosity = 0, maxiter = 500)
)

ks = range(-0.05, 0.15; length = 500)
npoints = 400

# Warm up the contractions and linear solve before measuring them.
finite_window_momentum_distribution(state, a_op, first(ks):first(ks); npoints = 2)
example_momentum_distribution(state, a_op, first(ks):first(ks); npoints = 2)
momentum_correlator(state, a_op' ⊗ a_op, a_op' * a_op, first(ks))

GC.gc()
finite_window_time = @elapsed finite_window = finite_window_momentum_distribution(
    state, a_op, ks; npoints
)
GC.gc()
finite_window_single_time = @elapsed finite_window_momentum_distribution(
    state, a_op, first(ks):first(ks); npoints
)
GC.gc()
example_window_time = @elapsed example_window = example_momentum_distribution(
    state, a_op, ks; npoints
)
example_correlations = real.(
    [
        expectation_value(state, (1, i) => a_op' ⊗ a_op) for i in 1:npoints
    ]
)
example_disconnected = abs2(expectation_value(state, (1,) => a_op))
GC.gc()
fourier_only_time = @elapsed example_fourier_transform(
    example_correlations, example_disconnected, ks
)
GC.gc()
single_momentum_time = @elapsed momentum_correlator(
    state, a_op' ⊗ a_op, a_op' * a_op, first(ks)
)
GC.gc()
scalar_sum_time = @elapsed scalar_sum = real.(
    [
        only(momentum_correlator(state, a_op' ⊗ a_op, a_op' * a_op, k)) for k in ks
    ]
)
GC.gc()
batched_sum_time = @elapsed infinite_sum = real.(
    only.(
        momentum_correlator(state, a_op' ⊗ a_op, a_op' * a_op, ks)
    )
)

@testset "Bose–Hubbard momentum distribution" begin
    @test finite_window ≈ infinite_sum rtol = 1.0e-7 atol = 1.0e-9
    @test scalar_sum ≈ infinite_sum rtol = 1.0e-9 atol = 1.0e-10
end

println("maximum error with physical displacements: ", maximum(abs, finite_window - infinite_sum))
println("maximum error with the current example indices: ", maximum(abs, example_window - infinite_sum))
println("optimized 400-site window, 500 momenta: ", finite_window_time, " seconds")
println("optimized 400-site window, one momentum: ", finite_window_single_time, " seconds")
println("original 400-site window, 500 momenta: ", example_window_time, " seconds")
println("Fourier transform of precomputed window, 500 momenta: ", fourier_only_time, " seconds")
println("momentum_correlator, one momentum: ", single_momentum_time, " seconds")
println("scalar momentum_correlator calls, 500 momenta: ", scalar_sum_time, " seconds")
println("batched momentum_correlator, 500 momenta: ", batched_sum_time, " seconds")
println("batch speedup over scalar calls: ", scalar_sum_time / batched_sum_time)

"""Return the shortest elapsed time over several serial samples."""
function best_elapsed(f; samples::Int = 5)
    times = Vector{Float64}(undef, samples)
    for sample in eachindex(times)
        GC.gc()
        times[sample] = @elapsed f()
    end
    return minimum(times)
end

println("momentum-count crossover against a freshly constructed 400-site window:")
println("count\tdirect batch\twindow + transform\tprecomputed transform")
for count in (1, 2, 4, 6, 8, 10, 12, 16, 18, 20, 22, 24, 32)
    sample_ks = count == 1 ? [-0.05] : collect(range(-0.05, 0.15; length = count))
    direct_time = best_elapsed() do
        momentum_correlator(state, a_op' ⊗ a_op, a_op' * a_op, sample_ks)
    end
    window_time = best_elapsed() do
        finite_window_momentum_distribution(state, a_op, sample_ks; npoints)
    end
    transform_time = best_elapsed() do
        example_fourier_transform(example_correlations, example_disconnected, sample_ks)
    end
    println(count, '\t', direct_time, '\t', window_time, '\t', transform_time)
end
