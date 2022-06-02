"""
Appplies the randomized tests
"""
module RandomizedTests
import ..SequenceGenerator as SG
import Statistics as Stats

"""
	(A::Vector{<:Real}, B::Vector{<:Real})
Randomized test to compare the means of the samples ``A`` and ``B``. It returns the significance (right tail p-value) assuming the null hypothesis that the difference of the means is zero.
"""
function twoSamplesMeanTest(A::Vector{<:Real}, B::Vector{<:Real})
	nᴬ, nᴮ = UInt8(length(A)), UInt8(length(B))
	μᴬ, μᴮ = Stats.mean(A), Stats.mean(B)
	if μᴬ > μᴮ
		nᴬ, nᴮ =  nᴮ, nᴬ
		A, B = B, A
		μᴬ, μᴮ = μᴮ, μᴬ
	end
	d = μᴮ - μᴬ
	combined = vcat(A, B)
	sequences = Vector{Symbol}[]
	SG.twoFactorPermutations!(sequences, nᴬ, nᴮ)
	N = size(sequences)[1]
	M = 0 #number of randomized samples whose means difference is greater or equal d
	for seq in sequences
		randomμᴬ = 0.
		randomμᴮ = 0.
		for i in 1:length(seq)
			if seq[i] == :A
				randomμᴬ += combined[i]
			else
				randomμᴮ += combined[i]
			end
		end
		if (randomμᴮ / nᴬ - randomμᴬ / nᴮ) >= d
			M += 1
		end
	end
	return M / N
end

"""
	(t::Vector{Tuple{Real, Real}}, d⁰)
Randomized paired test of the pairs *t* whose measurements were performed under the conditions *a* and *b*. It returns the significance (right tail p-value) assuming the null hypothesis that the average difference between *a* and *b* is zero.
"""
function pairedTest(t::Vector{NamedTuple{(:a, :b), Tuple{T, T}}} where T <: Real)
	l = length(t)
	#Compute initial average difference
	d⁰ = sum(t[i][:b] - t[i][:a] for i in 1:l) / l
	sequences = Vector{Symbol}[]
	SG.binomialExperiment!(sequences, UInt8(l))
	N = 2 ^ l
	M = 0
	for i in 1:length(sequences)
		d = 0
		for j in 1:l
			if sequences[i][j] == :A
				d += (t[j][:a] - t[j][:b])
			else
				d += (t[j][:b] - t[j][:a])
			end
		end
		d /= l
		if (d >= d⁰ && d⁰ >= 0) || (d <= d⁰ && d⁰ <= 0)
			M += 1
		end
	end
	return (M - 1) / N
end

end #module
