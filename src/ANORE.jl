"""
**an**alysis **o**f **re**siduals

It contains a struct that stores the residuals ``R`` of a least-squares or loess fit and calculate and stores the plots:
* a scatter plot of ``R`` Vs ``x``, the independent variable;
* a boxplot of ``R``;
* the spread-location (S-L) plot;
* the residuals-fit (R-F) spread plot
"""
module ANORE
import UnicodePlots
import Distributions as Dists
using Printf

"""
	(R::Vector{Real}, x::Vector{<:Real}, yCalc::Vector{<:Real}, xname::String, yname::String)

The constructor has the following arguments:
* The residuals ``R``
* The independent variable ``x``
* The calculated dependent variable ``yCalc``
* The independent variable name ``xname``
* The dependent variable name ``yname``
* A function ``drawPlots`` that plots a vector of plots
"""
struct anore #analysis of residuals
	R::Vector{<:Real}
	plotR::Any
	plotD::Any
	plotSL::Any
	plotRF::Any
	plotN::Any
	drawPlots::Function
	anore(R::Vector{<:Real}, x::Vector{<:Real}, yCalc::Vector{<:Real}, xname::String, yname::String) = begin
		#Residuals
		plotR = UnicodePlots.scatterplot(x, R, title="residuals", xlabel=xname, ylabel="R", marker="∘", width=60, height=18, canvas=UnicodePlots.DotCanvas)

		#S-L plot
		ϵ = sqrt.(abs.(R))
		plotSL = UnicodePlots.scatterplot(yCalc, ϵ, title="S-L plot", xlabel=yname, ylabel="√|R|", marker="∘", width=60, height=18, canvas=UnicodePlots.DotCanvas)

		#R-F plot
		resQuantiles = sort(R)
		nPoints = length(R)
		fValues =  [(i - 0.5) / nPoints for i=1:nPoints]
		fitSpread = yCalc .- Dists.mean(yCalc)
		fitQuantiles = Dists.quantile(fitSpread, fValues)
		ymin = min(minimum(resQuantiles), minimum(fitQuantiles))
		ymax = max(maximum(resQuantiles), maximum(fitQuantiles))
		plotRF = UnicodePlots.scatterplot(fValues, resQuantiles, title="R-F plot", xlabel="f-values", ylabel="quantiles", marker="∘", name = "∘ R", color=:green, ylim=(ymin, ymax), width=60, height=18, canvas=UnicodePlots.DotCanvas)
		UnicodePlots.scatterplot!(plotRF, fValues, fitQuantiles, color=:red, marker="×", name="× "*yname)

		#Normal quantiles
		μ = Dists.mean(R)
		σ = Dists.std(R)
		normQuantiles = Dists.quantile(Dists.Normal(μ, σ), fValues)
		plotN = UnicodePlots.scatterplot(normQuantiles, resQuantiles, title="Normal q-q plot", xlabel="Normal quantiles", ylabel="R", marker="∘", width=60, height=18, canvas=UnicodePlots.DotCanvas)
		UnicodePlots.lineplot!(plotN, 0., 1., color=:red)

		#autocorrelation
		plotD = UnicodePlots.densityplot(R[1:nPoints-1], R[2:end], xlabel="Rᵢ", ylabel="Rᵢ₊₁", title="residuals lag-1 autocorrelation", width=60, height=18)
		
		#Draw the plots
		drawPlots() = begin
			for plt in [plotR, plotD, plotSL, plotRF, plotN]
				print(stdout, "\n")
				print(stdout, plt)
				print(stdout, "\n")
			end
			return nothing
		end
		return new(R, plotR, plotD, plotSL, plotRF, plotN, drawPlots)
	end
end

"""
	(T::Matrix{<:Real}, R::Vector{<:Real}, nParams::Integer)
Lack of fit test for repeated measurements.
* T is a matrix of dependent variable values where each row stores the values obtained under the same conditions;
* R is the vector containing the residuals;
* nParams is the number of parameters.
"""
function lackOfFitTest(T::Matrix{<:Real}, R::Vector{<:Real}, nParams::Integer)
	dims = size(T)
	SR = sum(x ^ 2 for x = R) #Squared sum of residuals
	SE = 0.0 #Pure error contribution to SR
	for i in 1:dims[1]
		for m in 1:(dims[2] - 1)
			for n in (m + 1):dims[2]
				SE += (T[i,n] - T[i,m]) ^ 2 / dims[2]
			end
		end
	end
	SL = SR - SE #Lack of fit error
	NDFR = dims[1] * dims[2] - nParams
	NDFE = dims[1] * (dims[2] - 1)
	NDFL = NDFR - NDFE
	ML = SL / NDFL
	ME = SE / NDFE
	FDistTest =  Dists.ccdf(Dists.FDist(NDFL, NDFE), ML / ME)
	#Printout da ANOVA
	@printf "*** Lack of Fit Test ***\n"
	@printf "SR: squared residuals sum\n"
	@printf "SE: pure error contribution to SR\n"
	@printf "SL: lack of fit contribution to SR\n"
	@printf "NDFR, NDFE, NDFL: residuals, pure error and lack of fit degrees of freedom\n"
	@printf "ML, ME: SL/NDFL and SE/NDFE"
	@printf "significance: right tail p-value of the F distribution F(ML/ME; NDFL, NDFE)\n"
	@printf "------------------------------------------------------------------------------\n"
	@printf "residuals: SR = %3.2E,\tNDFR = %2d,\tSL = %3.2E, NDFL = %2d, ML = %3.2E\n" SR NDFR SL NDFL ML
	@printf "          \t\t\t\t\t\t\t\tSE = %3.2E, NDFE = %2d, ME = %3.2E\n\n" SE NDFE ME
	@printf "significance P(x≥ML/ME;NDFL,NDFE): %4.3f\n" FDistTest
	@printf "------------------------------------------------------------------------------\n"
	@printf "***\n"
	
	return FDistTest
end

end #module ANORE
