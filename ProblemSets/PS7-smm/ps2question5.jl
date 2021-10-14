#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, y)

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

    return loglike
end

alpha_zero = zeros(6*size(X,2))
alpha_rand = rand(6*size(X,2))
alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
alpha_start = alpha_true.*rand(size(alpha_true))
println(size(alpha_true))
alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)
