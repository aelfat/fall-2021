using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables
using SMM
using DataFramesMeta
using Pipe,Distributions


function wrapper()
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1


# 1)
bols = inv(X'*X)*X'*y
println(bols)

function ols_gmm(beta, X, y)
    g = y .- X*beta
    # I defined in LinearAlgebra
    J = g'*I*g
    return J
end
α̂_optim = optimize(a -> ols_gmm(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
println(α̂_optim.minimizer)

# the answers are very close
# α̂_optim.minimizer.-bols |> println

#2)
#a) changed alpha_start to alpha_true to get the result quickly
include("ps2question5.jl")

#b)
alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,
            -2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]

function gmm_logit(alpha, X, y)

    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j])
        dem .+= num[:,j]
    end

    P = num./repeat(dem,1,J)

    loglike = -sum( bigY.*log.(P) )
    g = @pipe (bigY .- P) |> reshape(_,(J*N, 1))
    J = g'*I*g
    return J[1,1]
end

alpha_hat_optim = optimize(a -> gmm_logit(a, X, y), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)

#c)
alpha_rand = rand(6*size(X,2))

alpha_hat_optim = optimize(a -> gmm_logit(a, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)

# I got different etimates which means that  the function isn't globally concave

#4)

N=size(X,1)
K=size(X,2)
J = length(unique(y))
Xrand = rand(N,K)
beta = rand(J)
# f)
#https://github.com/JuliaStats/Distributions.jl/blob/master/src/univariate/continuous/gumbel.jl
epsilon = rand(Gumbel(),N,J)
YY = zeros(N)
for i in 1:N
YY[i] = argmax(X[i]*beta[:] + epsilon[i,:])
end

# Q4

MA = SMM.parallelNormal() # Note: this line may take up to 5 minutes to execute
dc = SMM.history(MA.chains[1])
dc = dc[dc[:accepted].==true, :]
println(describe(dc))

# Q5
function ols_smm(θ, X, y, D)
	N=size(X,1)
	K=size(X,2)
	J = length(unique(y))
    β = θ[1:end-1]
    σ = θ[end]
    if length(β)==1
        β = β[1]
    end
    bigY = zeros(N,J)
	for j=1:J
            bigY[:,j] = y.==j
    end
	bigY2 = zeros(N,J)

    # N+1 moments in both model and data
    gmodel = zeros(N+1,D)
    # data moments are just the y vector itself
    # and the variance of the y vector
    gdata  = vcat(y,var(y))
    #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####
    # This is critical!                   #
    Random.seed!(1234)                    #
    # You must always use the same ε draw #
    # for every guess of θ!               #
    #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
    # simulated model moments
    for d=1:D
        ε = σ*randn(N)
        ỹ = X*β .+ ε
        gmodel[1:end-1,d] = ỹ
        gmodel[  end  ,d] = var(ỹ)
    end
    # criterion function
    err = vec(gdata .- mean(gmodel; dims=2))
    # weighting matrix is the identity matrix
    # minimize weighted difference between data and moments
    J = err'*I*err
    return J
end
return nothing
end
wrapper()
