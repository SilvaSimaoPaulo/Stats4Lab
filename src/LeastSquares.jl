
module LeastSquares
using LinearAlgebra

export χ², scan, fitLinear, fitGaussNewton! # 0800 724 2102 Itaú consignado, protocolo 79642878

"""
	(f::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Any) 
Auxiliary function to compute ``χ²=Σ(yᵢ-f(xᵢ,A))²/σ²``
"""
χ²(f::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Any) =
	sum(((y[i] - f(x[i], A)) / σ[i]) ^ 2 for i=1:length(x))

"""
	(f::Function, x::Matrix{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Any)
Auxiliary function to compute ``χ²=Σ(yᵢ-f(xᵢ,A))²/σ²`` for more than one variable
"""
χ²(f::Function, x::Matrix{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Any) = 
	sum(((y[i] - f(x[i,:], A)) / σ[i]) ^ 2 for i=1:length(y))

"""
	(f::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Vector{<:Real})
Perform a *χ²* scan in the range *A* given the function *f* of *x* and one parameter *A*, such that ``y=f(x,a)+ϵ``. It returns the the best *A* that minimizes ``χ²`` and the complete ``χ²`` matrix.
"""
function scan(f::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Vector{<:Real})
	s² = [χ²(f, x, y, σ, p) for p=A]
	s²min, k = findmin(s²)
	return A[k], s²
end

"""
	(f::Function, x::Array{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Matrix{Tuple{T, T}} where T <: Real)
Perform a *χ²* scan in the matrix of parameters *A* given the function *f* and the vectors *x* and *y* such that ``y=f(x,A)+ϵ``. It returns the the best *A* that minimizes ``χ²`` and the complete ``χ²`` matrix.
"""
function scan2D(f::Function, x::Array{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Matrix{Tuple{T, T}} where T <: Real)
	#dims = size(A)
	s² = [χ²(f, x, y, σ, p) for p=A]
	s²min, index = findmin(s²)
	return A[index[1], index[2]], s²
end

"""
	(f::Matrix{<:Real}, y::Vector{<:Real})
Computes the least-squares fit of ``y = a1 × f1 + a2 × f2 + ... + ap × fp`` where ``a1, a2, ..., ap`` are parameters and ``f1, f2, ..., fp`` functions of one or more independent variables.
* ``f`` is the design matrix;
* ``y`` is the target variable vector;
The covariance matrix ``M`` and the parameters vector ``A`` are returned.
"""
function fitLinear(f::Matrix{<:Real}, y::Vector{<:Real})
	dims = size(f)
	p = dims[2] #Número de parâmetros de ajuste
	#Calcula os termos da equação matricial MA=B, ver eq 12.7 de Vuolo, J. H. Fundamentos da teoria de erros, 2ed, 1996
	M = transpose(f) * f
	B = transpose(f) * y
	M = Symmetric(M)
	if p == 2 #Simple rule. LAPACK crashes in fitGauss! if the regression does not converge
		C =[M[2, 2] (-M[1, 2]); (-M[2, 1]) M[1, 1]] ./ (M[1, 1] * M[2, 2] - M[1, 2] ^ 2)
		A = C * B
		return C, A
	end
	C = inv(M) #Matriz de covariância
	A = M\B #O operador \ resolve automaticamente o sistema MA=B
	return C, A
end
precompile(fitLinear, (Matrix, Vector))

"""
	(f::Matrix{<:Real}, y::Vector{<:Real}, w::Vector{<:Real})
Same as above, with weights ``w``.
"""
function fitLinear(f::Matrix{<:Real}, y::Vector{<:Real}, w::Vector{<:Real})
	dims = size(f)
	for i=1:dims[1]
		sqrtw = sqrt(w[i])
		for j=1:dims[2]
			f[i,j] *= sqrtw
		end
		y[i] *= sqrtw
	end
	fitLinear(f, y)
end
precompile(fitLinear, (Matrix, Vector, Vector))

"""
	(x::Vector{<:Real}, y::Vector{<:Real})
The same as above, but for only one parameter.
"""
function fitLinear(x::Vector{<:Real}, y::Vector{<:Real})
	σ² = 1 / sum(x .^ 2)
	return sqrt(σ²), σ² * sum(y .* x)
end
precompile(fitLinear, (Vector, Vector))

"""
	(x::Vector{<:Real}, y::Vector{<:Real}, w::Vector{<:Real})
The same as above, with weights ``w``.
"""
function fitLinear(x::Vector{<:Real}, y::Vector{<:Real}, w::Vector{<:Real})
	x = x .* sqrt.(w)
	y = y .* sqrt.(w)
	fitLinear(x, y)
end
precompile(fitLinear, (Vector, Vector, Vector))

"""
	(flist::Vector{Function}, x::Vector{<:Real}, y::Vector{<:Real}, w::Vector{<:Real}, A::Vector{<:Real})
Non linear fit using Gauss-Newton method
"""
function fitGaussNewton!(flist::Vector{Function}, x::Vector{<:Real}, y::Vector{<:Real}, w::Vector{<:Real}, A::Vector{<:Real})
	nParams = length(A)
	nPoints = length(y)
	f = flist[1]
	∂f = flist[2:end]
	f0 = Vector{Float64}(undef, nPoints)
	∂fx = Matrix{Float64}(undef, nPoints, nParams) #Calculated partial derivatives
	R = Vector{Float64}(undef, nPoints) #Residuals
	C = Matrix{Float64}(undef, nParams, nParams) #Covariance matrix
	for i in 1:nPoints
		f0[i] = f(x[i], A)
	end
	R = y .- f0
	χ² = sum((R .* w) .^ 2)
	error = 1.0
	counter = 0
	while error > 0.01
		χ²old = χ²
		for i in 1:nPoints
			for j in 1:nParams
				∂fx[i, j] = ∂f[j](x[i], A)
			end
		end
		C, δA = fitLinear(∂fx, R, w)
		A = A .+ δA
		for i in 1:nPoints
			f0[i] = f(x[i], A)
		end
		R = y .- f0
		χ² = sum((R .* w) .^ 2)
		error = abs(χ² - χ²old) / χ²
		counter += 1
		#print(stderr, "Iteration $(counter): χ² = $(χ²), error = $(error)\n")
		if counter > 10
			print(stderr, "fitGauss! did not converged after $(counter) iterations\n")
			break
		end
	end
	counter > 10 && return nothing, nothing
	in(0, isfinite.(A)) && return nothing, nothing
	return C, A
end
precompile(fitGaussNewton!, (Vector, Vector, Vector, Vector, Vector))

"""
	(flist::Vector{Function}, x::Vector{<:Real}, y::Vector{<:Real}, A::Vector{<:Real})
Non linear fit using Gauss-Newton method
"""
function fitGaussNewton!(flist::Vector{Function}, x::Vector{<:Real}, y::Vector{<:Real}, A::Vector{<:Real})
	w = ones(length(y))
	return fitGaussNewton!(flist, x, y, w, A)
end
precompile(fitGaussNewton!, (Vector, Vector, Vector, Vector))

end #End of module LeastSquares