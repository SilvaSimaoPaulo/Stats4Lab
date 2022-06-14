"""
Bayesian linear regression for normal distributed residuals and normal distributed target variables
"""
module BayesianRegression
import Distributions as Dists
using LinearAlgebra

"""
	(ϕ::Matrix{<:Real}, m::Vector{<:Real}, S::Matrix{<:Real}, σ²::Real)
Computes the probability `p(y|x)`, where `y = m₁ϕ₁(x₁) + m₂ϕ₂(x₂) + ... + mₖϕₖ(xₖ)` given:
- `ϕ = (ϕ₁(x₁), ..., ϕₖ(xₖ))`
- the initial parameter set `m = (m₁, ..., mₖ)`
- the covariance matrix `S` of the parameters `m` and;
- `σ² = Var[y]`.
"""
function priorPrediction(ϕ::Array{<:Real}, m₀::Vector{<:Real}, S₀::Matrix{<:Real}, σ²::Real)
	ϕᵀ = transpose(ϕ)
	μ = ϕᵀ * m₀
	Σ = ϕᵀ * S₀ * ϕ
	N = Dists.Normal(μ, Σ + σ²)
	return N
end

"""
	(Φ::Array{<:Real}, y::Vector{<:Real}, m₀::Vector{<:Real}, S₀::Array{<:Real}, σ²::Real)
Posterior probability `p(θ|X,Y) = N(θ|mₙ,Sₙ)`, `Sₙ= (S₀⁻¹+σ⁻²ΦᵀΦ)⁻¹`, `mₙ = Sₙ(S₀⁻¹m₀+σ⁻²Φᵀy)`
"""
function posteriorProbability(Φ::Array{<:Real}, y::Vector{<:Real}, m₀::Vector{<:Real}, S₀::Array{<:Real}, σ²::Real)
	S₀⁻¹ = inv(S₀)
	ΦᵀΦ = transpose(Φ) * Φ
	Φᵀy = transpose(Φ) * y
	S₀⁻¹m₀ = S₀⁻¹ * m₀
	Sₙ⁻¹ = S₀⁻¹ + (inv(σ²) * ΦᵀΦ)
	Sₙ = inv(Sₙ⁻¹)
	mₙ = S₀⁻¹m₀ + (inv(σ²) * Φᵀy)
	mₙ = Sₙ * mₙ
	N = Dists.MvNormal(mₙ, Sₙ)
	return N
end

"""
	(ϕ::Array{<:Real}, Φ::Array{<:Real}, y::Vector{<:Real}, m₀::Vector{<:Real}, S₀::Array{<:Real}, σ²::Real)
Posterior prediction
"""
function posteriorPrediction(ϕ::Array{<:Real}, Φ::Array{<:Real}, y::Vector{<:Real}, m₀::Vector{<:Real}, S₀::Array{<:Real}, σ²::Real)
	S₀⁻¹ = inv(S₀)
	ΦᵀΦ = transpose(Φ) * Φ
	Φᵀy = transpose(Φ) * y
	S₀⁻¹m₀ = S₀⁻¹ * m₀
	Sₙ⁻¹ = S₀⁻¹ + (inv(σ²) * ΦᵀΦ)
	Sₙ = inv(Sₙ⁻¹)
	mₙ = S₀⁻¹m₀ + (inv(σ²) * Φᵀy)
	mₙ = Sₙ * mₙ
	μ = transpose(ϕ) * mₙ
	Σ = transpose(ϕ) * Sₙ * ϕ + σ²
	N = Dists.Normal(μ, Σ)
	return N
end


end #BayesianRegression
