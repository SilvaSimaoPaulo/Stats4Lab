import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("test/runTests.jl")