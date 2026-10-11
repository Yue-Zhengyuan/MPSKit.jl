"""
    correlator(ψ, O1, O2, i, j)
    correlator(ψ, O12, i, j)

Compute the 2-point correlator `⟨ψ|O1[i]O2[j]|ψ⟩` for inserting `O1` at `i` and `O2` at `j`.
Also accepts ranges for `j`.
The sites must be ordered as `i < j`; other orderings throw an `ArgumentError`.
"""
function correlator end

function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int, j::Int)
    return first(correlator(state, O₁, O₂, i, j:j))
end

function correlator(
        state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int, js::AbstractRange{Int}
    )
    first(js) > i || throw(ArgumentError("i should be smaller than j ($i, $(first(js)))"))
    S₁ = _firstspace(O₁)
    isunitspace(S₁) || throw(ArgumentError("O₁ should start with a trivial leg."))
    S₂ = _lastspace(O₂)
    S₂ == S₁' || throw(ArgumentError("O₂ should end with a trivial leg."))

    G = similar(js, scalartype(state))

    @plansor Vₗ[-1 -2; -3] := state.AC[i][2 3; -3] * removeunit(O₁, 1)[1; 3 -2] *
        conj(state.AC[i][2 1; -1])
    ctr = i + 1

    for (k, j) in enumerate(js)
        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end
        G[k] = @plansor Vₗ[1 2; 4] * state.AR[j][4 5; 6] * removeunit(O₂, 4)[2 3; 5] *
            conj(state.AR[j][1 3; 6])
        ctr = j
    end
    return G
end

function correlator(
        state::AbstractMPS, O₁₂::AbstractTensorMap{<:Any, S, 2, 2}, i::Int, j
    ) where {S}
    O₁, O₂ = decompose_localmpo(add_util_leg(O₁₂))
    return correlator(state, O₁, O₂, i, j)
end

@doc raw"""
    momentum_correlator(ψ, O12, O1O2, k; solver=Defaults.linearsolver)
    momentum_correlator(ψ, O12, O1O2, ks; solver=Defaults.linearsolver)

Compute the connected momentum-space two-point correlator of an [`InfiniteMPS`](@ref).

For an MPS with an ``N_c``-site unit cell, let

```math
g^{ss'}(R) = \left\langle O_{12}[(0,s),(R,s')] \right\rangle,
```

where the physical separation is ``r=R N_c+s'-s``.
This function returns the ``N_c \times N_c`` matrix

```math
G^{ss'}(k) = \sum_{R\in\mathbb{Z}} e^{-ik(RN_c+s'-s)}
\left[g^{ss'}(R)-g^{ss'}_{\mathrm{disc}}\right],
```

with

```math
g^{ss'}_{\mathrm{disc}} = \lim_{|R|\to\infty} g^{ss'}(R).
```

For coincident sites, `O1O2` defines

```math
g^{ss}(0) = \left\langle O_1 O_2(0,s) \right\rangle.
```

The momentum `k` is measured in units of the microscopic lattice spacing and is periodic modulo ``2\pi``.
The disconnected contribution, which produces Bragg peaks at ``k=2\pi m/N_c``, is excluded.
The transfer matrix is assumed to have a unique dominant fixed point.

When `ks` is an `AbstractVector` of momenta, the result is a `Vector` containing one correlator matrix per momentum in input order.
Momentum-independent environments are reused across the batch, and each solve is initialized from the preceding solution, so ordered lists of nearby momenta are most efficient.
An empty momentum vector returns an empty result.
"""
function momentum_correlator(
        state::InfiniteMPS,
        O₁₂::AbstractTensorMap{<:Any, <:Any, 2, 2},
        O₁O₂::AbstractTensorMap{<:Any, <:Any, 1, 1},
        k::Real;
        solver = Defaults.linearsolver,
    )
    return only(momentum_correlator(state, O₁₂, O₁O₂, [k]; solver))
end

function momentum_correlator(
        state::InfiniteMPS,
        O₁₂::AbstractTensorMap{<:Any, <:Any, 2, 2},
        O₁O₂::AbstractTensorMap{<:Any, <:Any, 1, 1},
        ks::AbstractVector{<:Real};
        solver = Defaults.linearsolver,
    )
    T = _momentum_result_type(state, O₁₂, O₁O₂, ks)
    isempty(ks) && return Matrix{T}[]

    data = _prepare_momentum_correlator(state, O₁₂, O₁O₂)
    N_c = length(state)
    results = [Matrix{T}(undef, N_c, N_c) for _ in ks]
    for (index, k) in enumerate(ks), s in 1:N_c, s′ in 1:N_c
        phase = cis(-k * (s′ - s))
        results[index][s, s′] = phase * (
            data.same_cell[s, s′] - data.positive.disconnected[s, s′]
        )
    end

    _add_momentum_tails!(
        results, ks, state, data.positive, data.transfer_regularized, solver, false
    )
    _add_momentum_tails!(
        results, ks, state, data.negative_reversed, data.transfer_regularized, solver, true
    )
    return results
end

"""Prepare all momentum-independent data for a correlator batch."""
function _prepare_momentum_correlator(
        state::InfiniteMPS,
        O₁₂::AbstractTensorMap{<:Any, <:Any, 2, 2},
        O₁O₂::AbstractTensorMap{<:Any, <:Any, 1, 1},
    )
    O₂₁ = permute(O₁₂, ((2, 1), (4, 3)))
    _check_momentum_correlator_spaces(state, O₁₂, O₂₁, O₁O₂)

    O₁, O₂ = decompose_localmpo(add_util_leg(O₁₂))
    O₂_reverse, O₁_reverse = decompose_localmpo(add_util_leg(O₂₁))
    _check_momentum_mpo_pair(O₁, O₂)
    _check_momentum_mpo_pair(O₂_reverse, O₁_reverse)

    N_c = length(state)
    T = promote_type(scalartype(state), scalartype(O₁₂), scalartype(O₁O₂))
    same_cell = Matrix{T}(undef, N_c, N_c)
    for s in 1:N_c, s′ in 1:N_c
        same_cell[s, s′] = if s == s′
            expectation_value(state, (s,) => O₁O₂)
        elseif s < s′
            correlator(state, O₁, O₂, s, s′)
        else
            correlator(state, O₂_reverse, O₁_reverse, s′, s)
        end
    end

    unitcell_transfer = TransferMatrix(state.AR)
    transfer = flip(unitcell_transfer)
    transfer_regularized = flip(
        regularize(unitcell_transfer, l_RR(state), r_RR(state))
    )
    positive = _prepare_momentum_direction(
        state, O₁, O₂, transfer, transfer_regularized
    )
    negative_reversed = _prepare_momentum_direction(
        state, O₂_reverse, O₁_reverse, transfer, transfer_regularized
    )
    return (; same_cell, positive, negative_reversed, transfer_regularized)
end

@doc raw"""
    structure_factor(ψ, O12, O1O2, k; solver=Defaults.linearsolver)
    structure_factor(ψ, O12, O1O2, ks; solver=Defaults.linearsolver)

Compute the connected static structure factor

```math
S(k) = \frac{1}{N_c}\sum_{s,s'=1}^{N_c} G^{ss'}(k),
```

where ``G^{ss'}(k)`` is returned by [`momentum_correlator`](@ref).
The momentum `k` is measured in units of the microscopic lattice spacing.
For an `AbstractVector` `ks`, return one structure factor per momentum in input order.
"""
function structure_factor(
        state::InfiniteMPS,
        O₁₂::AbstractTensorMap{<:Any, <:Any, 2, 2},
        O₁O₂::AbstractTensorMap{<:Any, <:Any, 1, 1},
        k::Real;
        solver = Defaults.linearsolver,
    )
    return only(structure_factor(state, O₁₂, O₁O₂, [k]; solver))
end

function structure_factor(
        state::InfiniteMPS,
        O₁₂::AbstractTensorMap{<:Any, <:Any, 2, 2},
        O₁O₂::AbstractTensorMap{<:Any, <:Any, 1, 1},
        ks::AbstractVector{<:Real};
        solver = Defaults.linearsolver,
    )
    return map(momentum_correlator(state, O₁₂, O₁O₂, ks; solver)) do G
        return sum(G) / length(state)
    end
end

"""Determine the scalar type of momentum-correlator results."""
function _momentum_result_type(state::InfiniteMPS, O₁₂, O₁O₂, ks)
    phase_type = if isempty(ks)
        K = eltype(ks)
        isconcretetype(K) ? typeof(cis(zero(K))) : ComplexF64
    else
        mapreduce(k -> typeof(cis(k)), promote_type, ks)
    end
    return promote_type(
        scalartype(state), scalartype(O₁₂), scalartype(O₁O₂), phase_type
    )
end

"""Validate that the momentum-space operators act on every site in the unit cell."""
function _check_momentum_correlator_spaces(
        state::InfiniteMPS,
        O₁₂::AbstractTensorMap{<:Any, <:Any, 2, 2},
        O₂₁::AbstractTensorMap{<:Any, <:Any, 2, 2},
        O₁O₂::AbstractTensorMap{<:Any, <:Any, 1, 1},
    )
    onsite_spaces = (space(O₁O₂, 1), dual(space(O₁O₂, 2)))
    two_site_spaces = (
        space(O₁₂, 1), space(O₁₂, 2), dual(space(O₁₂, 3)),
        dual(space(O₁₂, 4)), space(O₂₁, 1), space(O₂₁, 2),
        dual(space(O₂₁, 3)), dual(space(O₂₁, 4)),
    )
    for site in 1:length(state)
        P = physicalspace(state, site)
        all(==(P), two_site_spaces) && all(==(P), onsite_spaces) ||
            throw(SpaceMismatch("operators are incompatible with physical space at site $site"))
    end
    return nothing
end

"""Validate the boundary and auxiliary spaces of a decomposed two-site MPO."""
function _check_momentum_mpo_pair(O₁::MPOTensor, O₂::MPOTensor)
    isunitspace(_firstspace(O₁)) ||
        throw(ArgumentError("the first MPO endpoint should start with a trivial leg"))
    isunitspace(_lastspace(O₂)) ||
        throw(ArgumentError("the second MPO endpoint should end with a trivial leg"))
    _lastspace(O₁) == dual(_firstspace(O₂)) ||
        throw(SpaceMismatch("the MPO endpoint auxiliary spaces are incompatible"))
    return nothing
end

"""Construct the source environment at the right boundary of the reference unit cell."""
function _momentum_source(state::InfiniteMPS, O::MPOTensor, site::Int)
    @plansor source[-1 -2; -3] := state.AC[site][2 3; -3] *
        removeunit(O, 1)[1; 3 -2] * conj(state.AC[site][2 1; -1])
    if site < length(state)
        source = source * TransferMatrix(state.AR[(site + 1):length(state)])
    end
    return source
end

"""Close a source at every endpoint while sweeping once through a unit cell."""
function _momentum_closures(state::InfiniteMPS, O::MPOTensor, source)
    T = promote_type(scalartype(state), scalartype(O), scalartype(source))
    closures = Vector{T}(undef, length(state))
    for site in 1:length(state)
        closures[site] = @plansor source[1 2; 4] * state.AR[site][4 5; 6] *
            removeunit(O, 4)[2 3; 5] * conj(state.AR[site][1 3; 6])
        site < length(state) && (source = source * TransferMatrix(state.AR[site]))
    end
    return closures
end

"""Prepare the sources and disconnected terms for one spatial ordering."""
function _prepare_momentum_direction(
        state::InfiniteMPS, O₁::MPOTensor, O₂::MPOTensor, transfer,
        transfer_regularized
    )
    N_c = length(state)
    T = promote_type(scalartype(state), scalartype(O₁), scalartype(O₂))
    disconnected = Matrix{T}(undef, N_c, N_c)
    sources = map(1:N_c) do s
        return _momentum_source(state, O₁, s)
    end
    for s in 1:N_c
        fixedpoint_source = transfer(sources[s]) - transfer_regularized(sources[s])
        disconnected[s, :] .= _momentum_closures(state, O₂, fixedpoint_source)
    end
    return (; sources, endpoint = O₂, disconnected)
end

"""Solve and accumulate one directional tail across a momentum batch."""
function _add_momentum_tails!(
        results, ks, state::InfiniteMPS, direction, transfer, solver, reversed::Bool
    )
    N_c = length(state)
    label = reversed ? "negative" : "positive"
    for s in 1:N_c
        initial = direction.sources[s]
        for (index, k) in enumerate(ks)
            z = cis((reversed ? k : -k) * N_c)
            solution, convhist = linsolve(
                transfer, direction.sources[s], initial, solver, 1, -z
            )
            if convhist.converged == 0
                @warn "momentum correlator failed to converge" momentum = k direction = label sublattice = s normres = convhist.normres
            end
            initial = solution
            closures = _momentum_closures(state, direction.endpoint, solution)
            for s′ in 1:N_c
                row, column = reversed ? (s′, s) : (s, s′)
                phase = cis(-k * (column - row))
                tail = z * (closures[s′] - direction.disconnected[s, s′])
                results[index][row, column] += phase * tail
            end
        end
    end
    return results
end
