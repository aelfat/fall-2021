# ---------------------
# Ahmed El Fatmaoui
# Homework # 2
#----------------------


        using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions,
              FreqTables, Optim, HTTP, GLM

    function how2()
        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 1
        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        
        f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
        minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
        startval = rand(1)   # random starting value
        result = optimize(minusf, startval, BFGS())


        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 2
     

        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body,DataFrame)

        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.married.==1


        function ols(beta, X, y)
            ssr = (y.-X*beta)'*(y.-X*beta)
            return ssr
        end

        beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
        println(beta_hat_ols.minimizer)

        
        bols = inv(X'*X)*X'*y
        df.white = df.race.==1
        bols_lm = lm(@formula(married ~ age + white + collgrad), df)


        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 3
        #:::::::::::::::::::::::::::::::::::::::::::::::::::

        # di[Xi * alpha] − log(1+exp(Xi * alpha)) 
        function logit(alpha, X, d)
            loglike =    - sum([
            d[i] * (X * alpha)[i] - log(1 + exp((X * alpha)[i]))
            for i = 1:size(X,1)
            ])

            return loglike
        end


        beta_hat_loglikhd = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
        println(beta_hat_loglikhd.minimizer)



        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 4
        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # see Lecture 3 slides for example
        α̂ = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
        println(α̂)

        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 5
        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        
        freqtable(df, :occupation) # note small number of obs in some occupations
        df = dropmissing(df, :occupation)
        df[df.occupation.==10,:occupation] .= 9
        df[df.occupation.==11,:occupation] .= 9
        df[df.occupation.==12,:occupation] .= 9
        df[df.occupation.==13,:occupation] .= 9
        freqtable(df, :occupation) # problem solved

        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.occupation


        JJ =  size(freqtable(df, :occupation),1) # number of unique occupatins
        NN = size(X,1)  # number of observations (individuals i)


        function mlogit(alpha, X, d)
            C = [i==j ? 1 : 0 for i in d , j in 1:(JJ-1)]

            log_mat = [log(exp.(X[i,:]' * alpha[:,j]) / (1+sum(exp.(X[i,:]'*alpha)))) for i in 1:NN ,j in 1:(JJ-1)]

            return -sum(C.*log_mat)

        end

        beta_mnlog = optimize(b -> mnlogit(b, X, y), rand(size(X,2),JJ-1), LBFGS(), Optim.Options(g_tol=1e-7, iterations=100_000, show_trace=true))
            println(beta_mnlog.minimizer)


    end 

how2()


