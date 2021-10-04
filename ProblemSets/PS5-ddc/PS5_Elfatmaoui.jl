using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

function PS5()
# read in function to create state transitions for dynamic model
include("create_grids.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# load in the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body,DataFrame)
df1 = CSV.read(HTTP.get(url).body,DataFrame)



# create bus id variable
df = @transform(df, bus_id = 1:size(df,1))

#---------------------------------------------------
# reshape from wide to long (must do this twice be-
# cause DataFrames.stack() requires doing it one
# variable at a time)
#---------------------------------------------------
# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
#u1(x1t,b)=θ0+θ1x1t +θ2b
alpha = glm(@formula(Y ~ Branded + Odometer), df_long, Binomial(), LogitLink())
println(alpha)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body,DataFrame)

Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

# [print(string(":", x, ", ")) for x in names(df)]
colNames1 = [:Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11,
:Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20]
Odo = Matrix(df[:,colNames1])


colNames2 = [:Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7, :Xst8, :Xst9, :Xst10, :Xst11,
 :Xst12, :Xst13, :Xst14, :Xst15, :Xst16, :Xst17, :Xst18, :Xst19, :Xst20]
Xst = Matrix(df[:,colNames2])

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3b: generate state transition matrices
#:::::::::::::::::::::::::::::::::::::::::::::::::::
zval,zbin,xval,xbin,xtran = create_grids()

rowIndex = size(xtran)[1]
outIndx = length(unique(df.Branded))
TT = size(Y)[2]

# future value
FV = zeros(rowIndex, outIndx, TT+1)



	@views @inbounds function myfun(alpha)

		    # Loop backwards over t from T +1 to 1
		    for t in TT:-1:1
		        #Loop over the two possible brand states {0, 1}
		        for b in 0:1:1
		            #Loop over the possible permanent route usage states (i.e. from 1 to zbin)
		            for z in 1:1:zbin
		                #Loop over the possible odometer states (i.e. from 1 to xbin)
		                for x in 1:1:xbin
		                    row = x + (z-1) * xbin
		                    v1 = alpha[1] +  alpha[2]*xval[x] + alpha[3]*b + xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
		                    v0 = xtran[1+(z-1)*xbin,:]' * FV[(z-1)*xbin+1:z*xbin,b+1,t+1]

							FV[row,b+1,t]=0.9*log(exp(v1) + exp(v0))
		                end
		            end
		        end
		    end

#(d) Construct the log likelihood using the future value terms from the previous step and only
#using the observed states in the data. This will entail a for loop over buses and time periods.
		BusNum = size(Y)[1]
		loglike = 0
				for i in BusNum
					for t in 1:TT
					  row1 = 1 + (df.Zst[i]-1)*xbin   #Zst from the data rather than a hypothetical value z.
		              row0 = Xst[i,t] + (df.Zst[i]-1)*xbin #x + (z-1)*xbin as in v1t above, except we substitute :Xst and :Zst for x and z.

		              u = coef(alpha_hat_glm)[1] + coef(alpha_hat_glm)[2]*Xst[i,t] + coef(alpha_hat_glm)[3]*df.Branded[i] +
					  (xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,df.Branded[i]+1,t+1] 
		              P = exp(u)/(1+exp(u))
		              loglike += Y[i,t]log(P)+(Y[i,t]-1)log(1-P)
					end
				end
				return -loglike

	end

	optim = optimize(a -> myfun(a),coef(alpha) , LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    println(optim.minimizer)

end

PS5()
