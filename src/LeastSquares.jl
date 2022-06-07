
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
	(f::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, mParam1::Real, mParam2::Real, N::Integer)
Perform a *χ²* scan for one parameter *a* in the range *mParam1* to *mParam2* using *N* points given function *f* and the vectors *x* and *y* such that ``y=f(x,a)+ϵ``. It returns the the best *a* that minimizes ``χ²`` and the complete *a* and ``χ²`` curve as a matrix with the *a* values in the first row and the ``χ²`` in the second one.
"""
function scan(f::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Vector{<:Real})
	s² = [χ²(f, x, y, σ, p) for p=A]
	s²min, k = findmin(s²)
	return A[k], s²
end

"""
	(f::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, mParam1::Real, mParam2::Real, N::Integer)
Perform a *χ²* scan for one parameter *a* in the range *mParam1* to *mParam2* using *N* points given function *f* and the vectors *x* and *y* such that ``y=f(x,a)+ϵ``. It returns the the best *a* that minimizes ``χ²`` and the complete *a* and ``χ²`` curve as a matrix with the *a* values in the first row and the ``χ²`` in the second one.
"""
function scan2D(f::Function, x::Array{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, A::Matrix{Tuple{T, T}} where T <: Real)
	#dims = size(A)
	s² = [χ²(f, x, y, σ, p) for p=A]
	s²min, index = findmin(s²)
	return A[index[1], index[2]], s²
end

"""
(f::Matrix{<:Real}, y::Vector{<:Real})
Faz o ajuste de uma função y = a1 × f1 + a2 × f2 + ... + ap × fp que seja linar nos parâmetros a1, a2, ..., ap, ainda que as funções f1, f2, ..., fp de uma ou mais variáveis independentes não seja linear.
* f é uma matriz de dimensões n×p na qual cada coluna é constituída dos valores de fp calculadas para cada um dos n valores da(s) variável(is) independente(s);
* y é um vetor contendo os valores da variável dependente e;
São retornados a matriz de covariância M e o vetor A com os valores dos parâmetros ajustados.
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
(f::Matrix{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real})
Faz o ajuste de uma função y = a1 × f1 + a2 × f2 + ... + ap × fp que seja linar nos parâmetros a1, a2, ..., ap, ainda que as funções f1, f2, ..., fp de uma ou mais variáveis independentes não seja linear.
* f é uma matriz de dimensões n×p na qual cada coluna é constituída dos valores de fp calculadas para cada um dos n valores da(s) variável(is) independente(s);
* y é um vetor contendo os valores da variável dependente e;
* w é um vetor com os pesos a serem usados, normalmente o inverso do quadrado das incertezas.
São retornadas a matriz de covariância M e o vetor A com os valores dos parâmetros ajustados.
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
The same as above, but for only one parameter.
"""
function fitLinear(x::Vector{<:Real}, y::Vector{<:Real})
	σ² = 1 / sum(x .^ 2)
	return sqrt(σ²), σ² * sum(y .* x)
end
precompile(fitLinear, (Vector, Vector))

"""
The same as above, with weights.
"""
function fitLinear(x::Vector{<:Real}, y::Vector{<:Real}, w::Vector{<:Real})
	x = [x[i] * sqrt(w[i]) for i=1:nPoints]
	y = [y[i] * sqrt(w[i]) for i=1:nPoints]
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