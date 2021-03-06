import Stats4Lab.RandomizedTests as RT
import Stats4Lab.LeastSquares as LS
import Stats4Lab.ANORE
import Stats4Lab.BayesianRegression as BayesReg
import Distributions as Dists
using UnicodePlots
using LinearAlgebra

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

println("\n--->scan2D")
x0 = [0.34, 0.34, 0.58, 1.26, 1.26, 1.82]
x1 = [0.73, 0.73, 0.69, 0.97, 0.97, 0.46]
y = [5.75, 4.79, 5.44, 9.09, 8.59, 5.09]
x = [x0 x1]
r1 = collect(1.10:0.005:1.30)
r2 = collect(7.01:0.005:7.21)
A0 = [(u, v) for u=r1, v=r2]
A, s² = LS.scan2D((x,A)->A[1]*x[1]+A[2]*x[2], x, y, ones(6), A0)
print("\n", heatmap(s², xoffset=1.10, yoffset=7.01, xfact=0.005, yfact=0.005), "\n")
println("$(typeof(A))")

println("--->fitLinear (one parameter, weighted)")
L = [0.671, 0.590, 0.522, 0.439, 0.370] .- 0.02
t = [
	8.09 8.06 8.20 8.08 8.22;
	7.65 7.61 7.68 7.57 7.66;
	7.13 7.12 7.06 7.18 7.07;
	6.46 6.48 6.46 6.48 6.45;
	6.00 5.81 6.03 5.89 5.99
] ./ 5
T = Vector{Float64}(vec(transpose(t)))
dims =  size(t)
σT = (0.35 / 5) .* ones(dims[1] * dims[2]) #tempo de reação médio de uma pessoa, ver https://www.laboratoriovirtual.fisica.ufc.br/tempodereacao
L = kron(L, ones(dims[2]))
println("Ajustando os dados...")
x = sqrt.(L)
w = map(x->1.0/x^2, σT)
x = x .- Dists.mean(x)
T = T .- Dists.mean(T)
t = t .- Dists.mean(t)
σk, k = LS.fitLinear(x, T, w)
println("k = $(k) ± $(σk)")

println("--->lackOfFitTest")
TCalc = k .* x
R = T .- TCalc
ANORE.lackOfFitTest(t, R, 1)

println("--->anore")
a = ANORE.anore(R, T, TCalc, "L", "T")
a.drawPlots()
r² = 1 - Dists.var(R) / Dists.var(T)
println("\nr² = $(r²)")

println("-"^79)
println("BayesianRegression")
println("-"^79)
println("--->priorPrediction")
m₀ = zeros(6)
S₀ = Matrix(0.25 * I(6))
ϕ = [x->x ^ i for i=0:5]
X = collect(LinRange(-5, 5, 200))
p = []
for x in X
	Φ = [ϕ[i](x) for i = 1:6]
	push!(p, BayesReg.priorPrediction(Φ, m₀, S₀, 0))
end
v = kron(X, ones(10))
w = vec([rand(p[i], 10)[j] for j=1:10, i=1:200])
plotPolinomials = scatterplot(v, w, xlabel="x", ylabel="y", xlim=(-4.5, 4.5), ylim=(-5, 5), width=60, height=18)
print("\n", plotPolinomials, "\n")

println("\n---posteriorPrediction")
println("Fits a 3rd degree polynomial in a simulated sample (sine function + gaussian noisy).")
x = collect(0.0:0.1:1.0)
y = [sin(2π*x[i]) + 0.25 * randn() for i = 1:11]
f = [x->1, x->x, x->x^2, x->x^3]
Φ = [f[j](x[i]) for i = 1:11, j = 1:4] #design matrix
m₀ = [0, 10, -30, 21] #initial parameters got from GoogleSheets
S₀ = Matrix(I(4))
σ² = 0.5
upperLimit = Float64[]
lowerLimit = Float64[]
expected = Float64[]
posteriori = []
for i in 1:length(x)
	local ϕ = [1, x[i], x[i] ^ 2, x[i] ^ 3]
	N = BayesReg.posteriorPrediction(ϕ, Φ, y, m₀, S₀, σ²)
	push!(posteriori, N)
	push!(upperLimit, Dists.quantile(N, 0.84))
	push!(lowerLimit, Dists.quantile(N, 0.16))
	push!(expected, Dists.mean(N))
end
plotN = scatterplot(x, y, marker="∘", xlabel="x", ylabel="y", canvas=DotCanvas, width=54, height=21)
lineplot!(plotN, x, upperLimit, color=:red, name="68% CL")
lineplot!(plotN, x, lowerLimit, color=:red)
lineplot!(plotN, x, expected, color=:blue, name="expected")