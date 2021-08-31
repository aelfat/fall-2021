
############################
#### Homework No 1 #########
#### Ahmed El Fatmaoui #####
############################

#cd("/Users/ahmedelfatmaoui/Documents/MetricsIII/HOW1")
#---------------
#-----Q1--------
#---------------

using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions, Kronecker

function q1()
    #a)
         Random.seed!(1234)
         
         A = rand(Uniform(-5, 10),10,7)
         B = rand(Normal(-2, 15),10,7)
         C = [A[1:5,1:5] B[1:5,6:7]]
         D = (A .<= 0) .* A
    #d-g)
         elm = size(A)
         unqElm = unique(D)
         E = reshape(B,70,1)
         E2 = vec(B)
         F =  reshape([A B], 10, 7, 2)
         F = permutedims(F, [3, 1, 2])
         G = kronecker(B,C)
        # G2 = kronecker(C,F)
    #g-k)
    save("matrixpractice.jld", "A", A, "B", B, "C", C,"D", D, "E", E, "F", F, "G", G )
    save("firstmatrix.jld", "A", A, "B", B, "C", C,"D", D )
    CSV.write("Cmatrix.csv",DataFrame(C,:auto))
    CSV.write("Dmatrix.dat",DataFrame(D,:auto), delim = ' ')
    
     return A, B, C, D
    
end

A, B, C, D = q1()

#---------------
#-----Q2--------
#---------------

function q2(A, B, C)
    #a)
  AB = [A[i,j]*B[i,j] for i in 1:10, j in 1:7]
  
  #b)
  Cprime =  [C[i,j] for i in 1:5, j in 1:7 if -5 <= C[i,j] <= 5]
  
  C[-5 .<= C .<= 5]
  
  #c)
  N, K, T = 15169, 6, 5
  
  X = zeros(N,K,T);
  
  x5 = rand(Binomial(20,0.6),N)
  x6 = rand(Binomial(20,0.5),N) 
      for k in 1:T 
  
          X[:,1,k] = ones(N)
  
          if rand(1)[1] <= 75*(6-k)/5
              X[:,2,k] = ones(N)
          end
  
          X[:,3,k] = rand(Normal(15 + k - 1, 5*(k-1)),N)
  
          X[:,4,k] = rand(Normal(pi*(6-k)/3, 1/exp(1)),N)
  
          # let columns 1, 5 and 6 remain stationary over time.
          X[:,5, k] = x5
          X[:,6, k] = x6
  
      end
  
  # d)
  β = zeros(K,T);
  β[1,:] = [1:0.25:2;];
  β[2,:] = [log(j) for j in 1:T];
  β[3,:] = [-sqrt(j) for j in 1:T];
  β[4,:] = [exp(j) - exp(j+1) for j in 1:T];
  β[5,:] = [j for j in 1:T];
  β[6,:] = [j/3 for j in 1:T];

  β  
      
  # e) 

 Y = [X[:,:,t] * β[:,t] + rand(Normal(0,0.36),N,1)  for t in 1:T] 
  
end
  
q2(A,B,C)


#---------------
#-----Q3--------
#---------------
df = CSV.read("nlsw88.csv", DataFrame);
function q3()
    # a)
    df = CSV.read("nlsw88.csv", DataFrame);
    # ismissing(df)
    # [print(string(":", x, ", ")) for x in names(df)]

        # b)
        NotMarrPer = 1 - sum(df[:,"married"])/size(df)[1];
        GradPer = sum(df[:,"collgrad"])/size(df)[1];

        println("Percent not married = ", round(NotMarrPer, digits = 4) * 100,
        "\nCollege graduate percentage = ", round(GradPer, digits =4 ) * 100)

            # c)  note: tabulate command is undefined
            # source of freq. tables:
            # https://stackoverflow.com/questions/34654489/rs-table-function-in-julia-for-dataframes

            prop(freqtable(df.race))

                # d) 

                intrq = (describe(df,:q75)[:,2] - describe(df,:q25)[:,2])
                summarystats = [describe(df,:nmissing, :mean, :median, :std, :min, :max, :nunique) intrq]  
                rename(summarystats, :x1 => :itrq)

                    # e)
                    ## Cross tab between industry and occupation
                    freqtable(df.industry, df.occupation)

                        # f)
                        # source: https://dataframes.juliadata.org/stable/man/split_apply_combine/

                        q1 = df[:, [:race, :occupation, :wage]];
                        gdf = groupby(q1, [:race, :occupation]);
                        combine(gdf, :wage => mean)

end

q3()

#---------------
#-----Q4--------
#---------------
function q4()
        fmat = load("firstmatrix.jld")
        #keys(fmat)
        #fmat["B"]

        # a-e)
        function matrixops(A, B)
            # element by element product of inputs, A and b
            # input matrices product 
            # input matrices sum
                    if(size(A) != size(B)) 
                    println("inputs must have the same size") 
                    else
                        return println(
                            "A .* B = ",  A .* B,
                            "\n\nA' * B = ", A' * B,
                            "\n\nA + B = ",  A + B
                        )
                    end
        end

        matrixops(A, B)

        # f)
        # we get the error message back
        matrixops(C, D)

        # g)

        matrixops(
            convert(Array,df.ttl_exp ),
            convert(Array,df.wage )
        )
end
    q4()