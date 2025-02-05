function lgwt(N0::Integer,a::Real=-1,b::Real=1)
    # This script is for computing definite integrals using Legendre-Gauss
    # Quadrature. Computes the Legendre-Gauss nodes and weights on an interval
    # [a,b] with truncation order N
    #
    # Suppose you have a continuous function f(x) which is defined on [a,b]
    # which you can evaluate at any x in [a,b]. Simply evaluate it at all of
    # the values contained in the x vector to obtain a vector f. Then compute
    # the definite integral using sum(f.*w);
    #
    # Written by Greg von Winckel - 02/25/2004
    # First adapted to Julia by Tyler Ransom on 08-04-2015
    # Updated on 08-17-2020

    N  = N0-1
    N1 = N+1
    N2 = N+2

    xu = range(-1,stop=1,length=N1)

    # Initial guess
    y = cos.((2*(0:N) .+ 1)*pi/(2*N .+ 2))  .+  ( 0.27/N1 ) .* sin.( pi .* xu .* N/N2 )

    # Legendre-Gauss Vandermonde Matrix
    L  = zeros(N1,N2)

    # Derivative of LGVM
    Lp = zeros(N1,N2)

    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method

    y0 = 2


    vareps = 2e-52
    # Iterate until new points are uniformly within epsilon of old points


    i = 0
    tracker=0
    it_max=10
    while (norm(y.-y0,Inf)>vareps && tracker<=it_max)
        d=norm(y.-y0,Inf)

        L[:,1]  .= 1
        Lp[:,1] .= 0

        L[:,2] .= y

        for k=2:N1
            L[:,k+1] = ( (2*k-1)*y .* L[:,k] .- (k-1)*L[:,k-1] )/k
        end

        Lp = (N2)*( L[:,N1] .- y .* L[:,N2] )./(1 .- y.^2)
        y0 = y
        y  = y0 - L[:,N2]./Lp
        if norm(y.-y0,Inf)==d
            tracker+=1
        end
        i+=1

    end

    # Linear map from[-1,1] to [a,b]
    x = (a.*(1 .- y) .+ b .* (1 .+ y))./2

    # Compute the weights
    w=(b-a)./((1 .- y.^2).*Lp.^2)*(N2/N1)^2

    return x,w
end

#--------------------------------------------------

# Dr. Ransom Function for mlogit from PS3 solution
function mlogit_with_Z(theta, X, Z, y)

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
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end

        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end
