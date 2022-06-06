import Stats4Lab.RandomizedTests as RT
import Stats4Lab.LeastSquares as LS
import Distributions as Dists

println("-"^79)
println("RndomizedTests")
println("-"^79)
println("---> twoSamplesMeanTest")

A = 1.0 .* [30, 29, 30, 29]
B = 1.0 .* [32, 31, 31, 30]
d = Dists.mean(B) - Dists.mean(A)
sig = RT.twoSamplesMeanTest(B, A)
println("Significance = $(sig)")

println("\n---> pairedTest")
t = [
	(a=13.2, b=14.0),
	(a=8.2, b=8.8),
	(a=10.9, b=11.2),
	(a=14.3, b=14.2),
	(a=10.7, b=11.8),
	(a=6.6, b=6.4),
	(a=9.5, b=9.8),
	(a=10.8, b=11.3),
	(a=8.8, b=9.3),
	(a=13.3, b=13.6)
]
psig = RT.pairedTest(t)
println("Paired significance = $(psig)")
println("-"^79)

println("-"^79)
println("LeastSquares")
println("-"^79)
println("--->scan") #Weighted mean of two measurements using the scan method
x = [1.3, 1.1]
y = x
σ = [0.1, 0.2]
A = collect(0.9:0.01:1.4)
μ, χ² = LS.scan((x,a)->a, x, y, σ, A)
χ²min = minimum(χ²)
χ²diff = χ² .- χ²min
Δx = [A[i] for i=1:length(A) if χ²diff[i] < 1.0]
Δx = [Δx[1] - μ, Δx[end] - μ]
println("⟨x⟩=$(μ)")
println("Δx = ($(Δx[1]), $(Δx[end]))")
sig = Dists.ccdf(Dists.Chisq(1), χ²min)
println("Significance = $(sig)")

println("\n--->fitLinear (one parameter, unweighted")
x = [1, 2, 3, 4]
y = [6, 6, 10, 30]
f = [1, 4, 9, 16]
σ, b = LS.fitLinear(f, y)
println("Fitted equation: y=($(b) ± $(σ))x²")
bs, s² = LS.scan((x,b)->b*x^2, x, y, ones(4), collect(1.2:0.005:2.0))
println("scan method: y=$(bs)x²")
println("Results after outlier removal.")
x = [2, 3, 4]
y = [6, 10, 30]
f = [4, 9, 16]
σ, b = LS.fitLinear(f, y)
println("Fitted equation: y=($(b) ± $(σ))x²")
bs, s² = LS.scan((x,b)->b*x^2, x, y, ones(3), collect(1.2:0.005:2.0))
println("scan method: y=$(bs)x²")

println("\n--->fitLinear (two parameter, unweighted)")
x1 = [0.34, 0.34, 0.58, 1.26, 1.26, 1.82]
x2 = [0.73, 0.73, 0.69, 0.97, 0.97, 0.46]
y  = [5.75, 4.79, 5.44, 9.09, 8.59, 5.09]
V, A = LS.fitLinear(hcat(x1, x2), y)
println("Parameters: $(A)")
println("Covariance matrix: $(V)")


println("\n--->fitLinear (two parameters, weighted)")
x = [1.1, 1.5, 2.0, 3.1, 4.2, 5.0]
y = [2.0, 2.9, 4.2, 6.0, 8.0, 10.0]
f = [x ones(6)]
V, A = LS.fitLinear(f, y, 100 .* ones(6))
println("Parameters: $(A)")
println("Covariance matrix: $(V)")