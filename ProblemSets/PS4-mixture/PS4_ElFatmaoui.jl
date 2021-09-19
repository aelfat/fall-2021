
using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions,
FreqTables, Optim, HTTP, GLM, ForwardDiff

function PS4()
    # includes both the mlogit and lgwt functions
    include("functions.jl")

    #:::::::::::::::::::
    # question 1   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::::::::::::::::

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
    df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code



    startvals = [0.04037443688915296, 0.2439931891755024, -1.571319982008478,
    0.04332546800040571, 0.1468547553146026, -2.95910294680261, 0.1020574214746393,
     0.7473078162016722, -4.120049525011452, 0.037562865451966576, 0.6884890953253581,
     -3.6557702948516075, 0.0204543204452073, -0.3584022910222397, -4.376928748361732,
      0.1074636880063156, -0.5263752170984033, -6.199198099540245, 0.11688249286984193,
       -0.28705670558923196, -5.322248251070095, 1.30747948066456]

    # mlogit_with_Z included in the functions.jl script
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)

           # run the optimizer
           theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=1))

           theta_hat_mle_ad = theta_hat_optim_ad.minimizer
           # evaluate the Hessian at the estimates
           H  = Optim.hessian!(td, theta_hat_mle_ad)
           theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
              println(
              DataFrame(Row =  1:length(theta_hat_mle_ad), Coef = theta_hat_mle_ad, se = theta_hat_mle_ad_se)
              )

    #:::::::::::::::::::
    # question 2   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::::::::::::::::
# they make more sense here as they are positive

    #:::::::::::::::::::
    # question 3   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::::::::::::::::
    # define distribution
    d = Normal(0,1) # mean=0, standard deviation=1

    # get quadrature nodes and weights for 7 grid points
    nodes, weights = lgwt(7,-4,4)

    # now compute the integral over the density and verify it's 1
    sum(weights.*pdf.(d,nodes))

    # now compute the expectation and verify it's 0
    sum(weights.*nodes.*pdf.(d,nodes))

    #:::::::::::::::::::
    # question 4   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::::::::::::::::

    # b)
    #::::
    # define distribution
    d = Normal(0,2) # mean=0, standard deviation=1
    nstd = 5*sqrt(2)
    # get quadrature nodes and weights for 7 grid points
    nodes, weights = lgwt(7,-nstd,nstd)

    # now compute the expectation of x^2 where x~N(0,2)
    println(
    sum(weights.*nodes.*nodes.*pdf.(d,nodes))
    )
    # get quadrature nodes and weights for 10 grid points
    nodes, weights = lgwt(10,-nstd,5*nstd)
    println(
    sum(weights.*nodes.*nodes.*pdf.(d,nodes))
    )

    # The quadrature nodes and weights for 10 grid points yields better approximation as it is very close to 4, the variance

    # c)
    #::::
    NumObs = 10^6
    x = rand(Normal(0,2),NumObs)
    x_sq = x.*x

    # pdf of N(0,2)
          function f(y)
           1/(2*sqrt(2*pi)) * exp(-0.5 * (y/2)^2)
           end
    w = (2*nstd)/NumObs
    fuct1 = map(y -> f(y), x) .* x_sq
    MonteCarloEst1 = w * sum(fuct1) # result very close to 4
    println(MonteCarloEst1)

    fuct2 = map(y -> f(y), x) .* x
    MonteCarloEst2 = w * sum(fuct2) # result very close to 0
    println(MonteCarloEst2)

    fuct3 = map(y -> f(y), x)
    MonteCarloEst3 = w * sum(fuct3) #result Not too close to one;close if Numobs = 10^8
    println(MonteCarloEst3)
    # we can also just use pdf() to get the same results
    # pdf(d::MultivariateDistribution, x::AbstractArray)

    println(sum(
     w .*
    (x_sq .* pdf.(Normal(0,2),x))
    ))

    println(sum(
    w .*
    (x .* pdf.(Normal(0,2),x))
    ))

    println(sum(
    w .*
    pdf.(Normal(0,2),x)
    ))



    # D = 10^6 yields better approximation

    # D)
    #:::::
    # both methods are numerical methods for integral approximation; one can be mapped to the other

    #:::::::::::::::::::
    # question 5   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::::::::::::::::

    # using Gauss-Legendre quadrature
    function mixedlogit(theta, gridPoints, X, Z, y)

            alpha = theta[1:end-1]
            gamma = theta[end]
            K = size(X,2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)
            for j=1:J
                bigY[:,j] = y.==j
            end
            bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

            T = promote_type(eltype(X),eltype(theta))
            num   = zeros(T,N,J)
            dem   = zeros(T,N)

     # define distribution
     d = Normal(0,1) # mean=0, standard deviation=1
     # get quadrature nodes and weights for 7 grid points
     nodes, weights = lgwt(gridPoints,-4,4)

            for j=1:J
                num[:,j] = sum(
                (exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)).*
                weights .* pdf.(d,nodes)
                )
                dem .+= num[:,j]
            end

            P = num./repeat(dem,1,J)

            loglike = -sum( bigY.*log.(P) )

            return loglike
        end



        # using Monte Carlo
        function mixedlogit(theta, NumObs, X, Z, y)

                alpha = theta[1:end-1]
                gamma = theta[end]
                K = size(X,2)
                J = length(unique(y))
                N = length(y)
                bigY = zeros(N,J)
                for j=1:J
                    bigY[:,j] = y.==j
                end
                bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

                T = promote_type(eltype(X),eltype(theta))
                num   = zeros(T,N,J)
                dem   = zeros(T,N)

         # define distribution
         x = rand(Normal(0,1),NumObs) # mean=0, standard deviation=1
         #nstd = 4
         w = (8)/NumObs

                for j=1:J
                    num[:,j] = sum(
                    (exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)).*
                    w .* pdf.(Normal(0,1),x)
                    )
                    dem .+= num[:,j]
                end

                P = num./repeat(dem,1,J)

                loglike = -sum( bigY.*log.(P) )

                return loglike
            end
end

#:::::::::::::::::::
# question 6   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::
PS4()
