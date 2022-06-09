
"""
Generates all possible permutations of `nl` alements of type A and `nr` elements of type B using a binary tree. It also generates a binomial experiment of `n` elements.
"""
module SequenceGenerator
export twoFactorPermutations, binomialExperiment

"""
	(sequences::Vector{Vector{Symbol}}, nl::UInt8, nr::UInt8, s::Vector{Symbol}=Symbol[])
Generates all permutations of `N=nl+nr` elements, where `nl` is the number of elements of factor `A` for a given `n` and store the result in `sequences`. The last argument `s` should not be modified, unless you want to complete an initial sequence.
"""
function twoFactorPermutations!(sequences::Vector{Vector{Symbol}}, nl::UInt8, nr::UInt8, s::Vector{Symbol}=Symbol[])
	if nl > 0
		twoFactorPermutations!(sequences, nl - 0b1, nr, vcat(s, :A))
	end
	if nr > 0
		twoFactorPermutations!(sequences, nl, nr - 0b1, vcat(s, :B))
	end
	if nl == 0 && nr == 0
		push!(sequences, s)
	end 
end

"""
	(sequences::Vector{Vector{Symbol}}, n::UInt8, s::Vector{Symbol}=Symbol[])
Generates all binomial experiment sequences for a given `n` and store the result in `sequences`. The last argument `s` should not be modified, unless you want to complete an initial sequence. 
"""
function binomialExperiment!(sequences::Vector{Vector{Symbol}}, n::UInt8, s::Vector{Symbol}=Symbol[])
	if n > 0
		binomialExperiment!(sequences, n - 0x1, vcat(s, :A))
		binomialExperiment!(sequences, n - 0x1, vcat(s, :B))
	else
		push!(sequences, s)
	end
end

end #module SG