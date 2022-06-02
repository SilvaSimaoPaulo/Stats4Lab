import Stats4Lab.RandomizedTests as RT
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