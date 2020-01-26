module GPEmulator

"""
Module: GP
-------------------------------------
Packages required: Statistics,
                   Distributions,
                   LinearAlgebra,
                   GaussianProcesses,
                   (Optim)
                   ScikitLearn,
-------------------------------------
Idea: To create an emulator (GP). 
      Include functions to optimize the emulator
      and to make predictions of mean and variance
      uses ScikitLearn (or GaussianProcesses.jl 
-------------------------------------
Exports: GPObj
         optimize_hyperparameters
         predict
         emulate
         extract
-------------------------------------
"""
# import CES modules
include("EKI.jl")
include("Truth.jl")

# packages
using Statistics 
using Distributions
using LinearAlgebra
using GaussianProcesses
using ScikitLearn
using Optim

@sk_import gaussian_process : GaussianProcessRegressor
@sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)
const sk = ScikitLearn
#using Interpolations #will use this later (faster)
#using PyCall
#py=pyimport("scipy.interpolate")

# exports
export GPObj
export predict
export emulate
export extract

#####
##### Structure definitions
#####

#structure to hold inputs/ouputs kernel type and whether we do sparse GP
struct GPObj
    inputs::Matrix{Float64}
    data::Matrix{Float64}
    models::Vector
    package::String
end

function GPObj(inputs, data, package)
    if package == "gp_jl"
        models = Any[]
        outputs = convert(Matrix{Float64}, data')
        inputs = convert(Matrix{Float64}, inputs')

        for i in 1:size(outputs, 1)
            # Zero mean function
            kmean = MeanZero() 
            # Construct kernel:
            # Sum kernel consisting of Matern 5/2 ARD kernel and Squared 
            # Exponential kernel 
            len2 = 1.0
            var2 = 1.0
            kern1 = SE(len2, var2)
            kern2 = Matern(5/2, [0.0, 0.0, 0.0], 0.0)
            lognoise = 0.5
            # regularize with white noise
            white = Noise(log(2.0))
            # construct kernel
            kern = kern1 + kern2 + white

            # inputs: N_param x N_samples
            # outputs: N_data x N_samples
            m = GPE(inputs, outputs[i, :], kmean, kern, sqrt(lognoise))
            optimize!(m)
            push!(models, m)
        end
        
        GPObj(inputs, outputs, models, package)
        
    elseif package == "sk_jl"
        
        len2 = ones(size(inputs, 2))
        var2 = 1.0

        varkern = ConstantKernel(constant_value=var2,
                                 constant_value_bounds=(1e-05, 1000.0))
        rbf = RBF(length_scale=len2, length_scale_bounds=(1.0, 1000.0))
                 
        white = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-05, 10.0))
        kern = varkern * rbf + white
        models = Any[]

        outputs = convert(Matrix{Float64}, data')
        inputs = convert(Matrix{Float64}, inputs)
        
        for i in 1:size(outputs,1)
            out = reshape(outputs[i,:], (size(outputs, 2), 1))
            
            m = GaussianProcessRegressor(kernel=kern,
                                         n_restarts_optimizer=10,
                                         alpha=0.0, normalize_y=true)
            ScikitLearn.fit!(m, inputs, out)
            if i==1
                println(m.kernel.hyperparameters)
                print("Completed training of: ")
            end
            print(i,", ")
            push!(models, m)
        end

        GPObj(inputs, outputs, models, package)

    else 
        println("use package sk_jl or gp_jl")
    end
end # function GPObj

function predict(gp::GPObj, new_inputs::Array{Float64}; 
                 prediction_type="y")
    
    # predict data (type "y") or latent function (type "f")
    # column of new_inputs gives new parameter set to evaluate gp at
    M = length(gp.models)
    mean = Array{Float64}[]
    var = Array{Float64}[]
    # predicts columns of inputs so must be transposed
    if gp.package == "gp_jl"
        new_inputs = convert(Matrix{Float64}, new_inputs')
    end    
    for i=1:M
        if gp.package == "gp_jl"
            if prediction_type == "y"
                mu, sig2 = predict_y(gp.models[i], new_inputs)
                push!(mean, mu)
                push!(var, sig2) 

            elseif prediction_type == "f"
                mu, sig2 = predict_f(gp.models[i], new_inputs')
                push!(mean, mu)
                push!(var, sig2) 

            else 
                println("prediction_type must be string: y or f")
                exit()
            end

        elseif gp.package == "sk_jl"
            
            mu, sig = gp.models[i].predict(new_inputs, return_std=true)
            sig2 = sig .* sig
            push!(mean, mu)
            push!(var, sig2) 

        end
    end

    return mean, var

end # function predict


function emulate(u_tp::Array{Float64, 2}, g_tp::Array{Float64, 2}, 
                 gppackage::String)

    gpobj = GPObj(u_tp, g_tp, gppackage) # construct the GP based on data

    return gpobj
end


function extract(truthobj::TruthObj, ekiobj::EKIObj, N_eki_it::Int64)
    
    yt = vcat(truthobj.solution.u'...)[end,:]
    yt_cov = truthobj.cov
    yt_covinv = inv(yt_cov)
    
    # Note u[end] does not have an equivalent g
    u_tp = ekiobj.u[end-N_eki_it:end-1] # N_eki_it x [N_ens x N_param]
    g_tp = ekiobj.g[end-N_eki_it+1:end] # N_eki_it x [N_ens x N_data]

    # u does not require reduction, g does:
    # g_tp[j] is jth iteration of ensembles 
    u_tp = cat(u_tp..., dims=1) # [(N_eki_it x N_ens) x N_param]
    g_tp = cat(g_tp..., dims=1) # [(N_eki_it x N_ens) x N_data]

    return yt, yt_cov, yt_covinv, u_tp, g_tp
end

end # module GPEmulator
