module Stats4Lab
export LeastSquares, BayesianRegression, RandomizedTests, ANORE
include("SequenceGenerator.jl")
include("LeastSquares.jl")
include("ANORE.jl")
include("BayesianRegression.jl")
include("RandomizedTests.jl")

"""
	(y::Vector{<:Real}, l=1)

Autocorrelation of a vector *y* with lag *l*
"""
function autocor(y::Vector{<:Real}, l=1)
	N = length(y)
	avg = sum(y) / N
	r = 0.0
	for i in (1+l):N
		r += (y[i] - avg) * (y[i-l] - avg)
	end
	r = r / sum((y .- avg) .^ 2)
	return r
end
precompile(autocor, (Vector{<:Real}, Integer))

end # module
