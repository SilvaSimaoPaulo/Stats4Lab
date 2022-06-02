
"""
As principais funções deste módulo são a scan, que ajusta uma função y=f(x) de um parâmetro p e uma variável real a um conjunto de dados por meio de uma varredura (scan) em diversos valores de p buscando o que minimiza a soma χ², e a função fitFinear, que ajusta os dados a uma função y=a+bx pelo método dos mínimos quadrados.
"""
module LeastSquares
using LinearAlgebra

export χ², scan, fitLinear, fitGauss!

"""
(mFunc::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, mParam::Real)
Calcula χ² para os dados y, com respectivas incertezas σ, e os valores teóricos calculados pela função mFunc a partir de x e dos parâmetros mParams.
"""
function χ²(mFunc::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, mParams)
	soma = 0.0
	for i in length(x)
		soma += ((y[i] - mFunc(x[i], mParams)) / σ[i]) ^ 2
	end
	return soma
end

"""
(x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, mFunc, mParam1::Real, mParam2::Real, N::Integer)
Retorna o parâmetro bestParam através de uma varredura sobre N valores do parâmetro a ser estimado no intervalo entre mParam1 e mParam2, escolhendo o valor para o qual χ²(x, y, σ, mFunc, mParam) é mínimo.
"""
function scan(mFunc::Function, x::Vector{<:Real}, y::Vector{<:Real}, σ::Vector{<:Real}, mParam1::Real, mParam2::Real, N::Integer)
	if mParam1 > mParam2 #Garante mParam1 < mParam2
		tmp = mParam1
		mParam1 = mParam2
		mParam2 = tmp
	end
	
	params = LinRange(mParam1, mParam2, N)
	χ²dif = [χ²(mFunc, x, y, σ, p) for p=params]
	χ²min, bestParam = findmin(χ²dif)
	χ²dif = χ²dif .- χ²min
	paramInf = paramSup = bestParam
	for j in 2:N
		if (χ²dif[j-1] > 1.0) && (χ²dif[j] < 1.0)
			paramInf = (params[j-1] + params[j]) * 0.5
		end
		if (χ²dif[j-1] < 1.0) && (χ²dif[j] > 1.0)
			paramSup = (params[j-1] + params[j]) * 0.5
		end
	end
	errInf = bestParam - paramInf
	errSup = paramSup - bestParam
	return bestParam, errInf, errSup, χ²min
end

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
	p = dims[2] #Número de parâmetros de ajuste
	M = Matrix{Float64}(undef, p, p)
	B = Vector{Float64}(undef, p)
	#Calcula os termos da equação matricial MA=B, ver eq 12.7 de Vuolo, J. H. Fundamentos da teoria de erros, 2ed, 1996
	for i in 1:p
		M[i, i] = sum(f[:,i] .^ 2 .* w)
		for j in (i+1):p
			M[i,j] = sum(f[:,i] .* f[:,j] .* w)
		end
		B[i] = sum(y .* f[:,i] .* w)
	end
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
precompile(fitLinear, (Matrix, Vector, Vector))

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
The same as above, but for only one parameter.
"""
function fitLinear(x::Vector{<:Real}, y::Vector{<:Real}, w::Vector{<:Real})
	a = y ./ x
	media = sum(a .* w) / sum(w)
	σa = inv(sqrt(sum(w)))
	return σa, media
end
precompile(fitLinear, (Vector, Vector, Vector))

"""
The same as above, but without weights.
"""
function fitLinear(x::Vector{<:Real}, y::Vector{<:Real})
	a = y ./ x
	return sum(a) / length(a)
end
precompile(fitLinear, (Vector, Vector))

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