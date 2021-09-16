
#Worked with Bill and Aleeze

    using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions,
    FreqTables, Optim, HTTP, GLM

    #:::::::::::::::::::
    # question 1   :::::
    #:::::::::::::::::::

function ps3()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
    df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8) 
    y = df.occupation


    K = size(X,2)+1
    J = length(unique(y))

        function mlogit(Beta, X, Z,y)
            
            K = size(X,2)+1
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)

                for j=1:J
                    bigY[:,j] = y.==j
                end

            bigBeta = [reshape(Beta,K,J-1) zeros(K)]
            bigZ = zeros(N,J)

                for j=1:J
                    bigZ[:,j] = Z[:,j]-Z[:,J]
                end

            num = zeros(N,J)
            dem = zeros(N)

                for j=1:J
                    XZ = cat(X,bigZ[:,j],dims=2)
                    num[:,j] = exp.(XZ*bigBeta[:,j])
                    dem .+= num[:,j]
                end

            P = num./repeat(dem,1,J)
            
            loglike = -sum( bigY.*log.(P) )
            
                return loglike
        end
    
        startHere0 = rand((J-1)*K)
        startNearSol = 1.0000001 * [0.04885858084422522, 0.0009042465857047558, -2.144317825388887,
         0.4425753938680137, 0.04448332173290848, 0.7276857878249758, -3.1545819831597375, -0.028422550899673403,
          0.09091530580736103, -0.13094629047687026, -4.1213637842532025, 0.2578758066405763, 0.025242867291607733,
           0.8513538201292878, -4.026561140941997, -0.5232455580421997, 0.04201112040602136, -0.5535189061525868, 
           -4.471599137048158, -0.97027896459939, 0.0856777779538767, -1.1140765374638752, -6.954753655923485, 
           -0.3570730023566607, 0.08253331780409119, -0.5940641835229424, -5.325234363329876, -0.8111789371715972]

    mlogit_hat_optim = optimize(b-> mlogit(b,X,Z, y), startNearSol, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=200))
    println(mlogit_hat_optim.minimizer)

    #:::::::::::::::::::
    # question 2   :::::
    #:::::::::::::::::::

 # source: https://stats.idre.ucla.edu/stata/output/multinomial-logistic-regression/
    # the standard interpretation of the multinomial logit is that for a unit change in the predictor 
    # variable, the logit of outcome m relative to the referent group is expected to change by its respective
    # parameter estimate (which is in log-odds units) given the variables in the model are held constant. 

    # in our example it is the logit of occupation relative to a referenced occupation is expected to change by 
    # the estimated parametr, ceteris paribus.


    #:::::::::::::::::::
    # question 3   :::::
    #:::::::::::::::::::
    function nested_logit(alpha, X, Z, y, nesting_structure)

        beta = alpha[1:end-3]
        lambda = alpha[end-2:end-1]
        gamma = alpha[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)

            for j=1:J
                bigY[:,j] = y.==j
            end
        bigBeta = [repeat(beta[1:K],1,length(nesting_structure[1])) repeat(beta[K+1:2K],1,length(nesting_structure[2])) zeros(K)]

        T = promote_type(eltype(X),eltype(alpha))
        num   = zeros(T,N,J)
        lidx  = zeros(T,N,J) 
        dem   = zeros(T,N)

        for j=1:J
            if j in nesting_structure[1]
                lidx[:,j] = exp.( (X*bigBeta[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[1] )
            elseif j in nesting_structure[2]
                lidx[:,j] = exp.( (X*bigBeta[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[2] )
            else
                lidx[:,j] = exp.(zeros(N))
            end
        end

        for j=1:J
            if j in nesting_structure[1]
                num[:,j] = lidx[:,j].*sum(lidx[:,nesting_structure[1][:]];dims=2).^(lambda[1]-1)
            elseif j in nesting_structure[2]
                num[:,j] = lidx[:,j].*sum(lidx[:,nesting_structure[2][:]];dims=2).^(lambda[2]-1)
            else
                num[:,j] = lidx[:,j]
            end

            dem .+= num[:,j]
        end

        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end
        nesting_structure = [[1 2 3], [4 5 6 7]]
        startvals = [2*rand(2*size(X,2)).-1; 1; 1; .1]

        td2 = TwiceDifferentiable(alpha -> nested_logit(alpha, X, Z, y, nesting_structure), startvals; autodiff = :forward)
        α_hat_nlogit = optimize(td2, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=200))
        α_hat_nlogit_ad = α_hat_nlogit.minimizer

        H  = Optim.hessian!(td2, α_hat_nlogit_ad)
        α_hat_nlogit_ad_se = sqrt.(diag(inv(H)))

        println([α_hat_nlogit_ad α_hat_nlogit_ad_se])

        return nothing
    end

        ps3()