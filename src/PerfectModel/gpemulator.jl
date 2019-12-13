module GPEMULATOR
"""
Module: GPEMULATOR
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
Exports: optimize_hyperparameters
         predict
-------------------------------------
"""

#packages
using Statistics 
using Distributions
using LinearAlgebra
using GaussianProcesses
using ScikitLearn
@sk_import gaussian_process : GaussianProcessRegressor
@sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)
#sk = ScikitLearn
using Optim
#using Interpolations #will use this later (faster)
#using PyCall
#py=pyimport("scipy.interpolate")
#imports

#exports
export GPemulator
export predict, optimize_hyperparameters


#function __init__()
#end

#####################################################################
#
#Structure definitions
#
#################################################################### 
#structure to hold inputs/ouputs kernel type and whether we do sparse GP
struct GPemulator
    inputs::Matrix{Float64}
    data::Matrix{Float64}
    models::Vector
    package::String
end

#####################################################################
#
#Function definitions
#
#################################################################### 

function GPemulator(inputs,data,package)
    

    if package == "gp_jl"
        #Create the GP object(s)
        #mean:
        mean=MeanZero()
        #kernel - lengthscale(squared), variance
        len2=ones(size(inputs,2))
        var2=1.0
        kern=SEArd(len2,var2)
        #likelihood
        #lik=GaussLik(1.0)
        lognoise=0.5
        
        #regularize with white noise
        white=Noise(log(2.0))
        
        kern=kern+white
        
        models=Any[]
        println(size(inputs))
        println(size(data))
        flush(stdout)
        outputs=convert(Matrix{Float64},data')
        inputs=convert(Matrix{Float64},inputs')
        
        #priors
        priorVec=fill(LogNormal(),length(len2)+1+1)
        for i=1:size(outputs,1)
            #m=GP(inputs,data,mean,kernel,likelihood)
            #inputs param dim x pts in R^2
            #data[i,:] pts x 1 in R
            m=GPE(inputs,outputs[i,:],mean,kern,log(sqrt(lognoise)))
            println(m.kernel)
            set_priors!(m.kernel,priorVec)
            #set_priors!(m.lik,Normal())
            #println(m.kernel.priors)
            #println(m.lik.priors)
            flush(stdout)
            push!(models,m)
            
        end
        
        GPemulator(inputs,outputs,models,package)
        
    elseif package == "sk_jl"
        
        len2=ones(size(inputs,2))
        var2=1.0

        varkern=ConstantKernel(constant_value=var2,constant_value_bounds=(1e-05,10000.0))
        rbf=RBF(length_scale=len2,length_scale_bounds=(1.0,10000.0))
                 
        white=WhiteKernel(noise_level=1.0,noise_level_bounds=(1e-05,10.0))
        kern=varkern*rbf+white
        models=Any[]
        
        #need python style indexing
        #outputs=[convert(Array{Float64},data[i,:]) for i in size(data,2)]
        #inputs=[convert(Array{Float64},inputs[i,:]) for i in size(inputs,2)]
        
        outputs=convert(Matrix{Float64},data')
        inputs=convert(Matrix{Float64},inputs)
        println(size(outputs[1,:]))
        println(size(inputs))
        
        for i=1:size(outputs,1)
            out=reshape(outputs[i,:],(size(outputs,2),1))
            
            m=GaussianProcessRegressor(kernel=kern,
                                       n_restarts_optimizer=10,
                                       alpha=0.0)
            ScikitLearn.fit!(m,inputs,out)
            if i==1
                println(m.kernel.hyperparameters)
                print("Completed training of: ")
            end
            print(i,", ")
            
            #println(m.kernel.hyperparameters)
            flush(stdout)
            push!(models,m)
            #println(m)
            #flush(stdout)
        end

        GPemulator(inputs,outputs,models,package)

    else 
        println("use package sk_jl or gp_jl")
    end
end
#export function definitions

function optimize_hyperparameters(gp::GPemulator)
    if gp.package == "gp_jl"
        for i=1:length(gp.models) 
            optimize!(gp.models[i])
            println(gp.models[i].kernel)
            flush(stdout)
        end
    elseif gp.package == "sk_jl"
        println("optimization already accounted for with fit!")
        flush(stdout)
    end
end

function predict(gp::GPemulator, new_inputs::Array{Float64};prediction_type="y")
    
    
    #predict data (type "y") or latent function (type "f")
    #column of new_inputs gives new parameter set to evaluate gp at
    M=length(gp.models)
    mean=Array{Float64}[]
    var=Array{Float64}[]
    #predicts columns of inputs so must be transposed
    if gp.package=="gp_jl"
        new_inputs=convert(Matrix{Float64},new_inputs')
    end    
    for i=1:M
        if gp.package=="gp_jl"
            if prediction_type == "y"
                mu,sig2=predict_y(gp.models[i],new_inputs)
                #mu,sig2=predict_y(gp.models[i],new_inputs)
                push!(mean,mu)
                push!(var,sig2) 

            elseif prediction_type == "f"
                mu,sig2=predict_f(gp.models[i],new_inputs')
                push!(mean,mu)
                push!(var,sig2) 

            else 
                println("prediction_type must be string: y or f")
                exit()
            end

        elseif gp.package=="sk_jl"
            
            mu,sig = gp.models[i].predict(new_inputs,return_std=true)
            sig2=sig.*sig
            push!(mean,mu)
            push!(var,sig2) 

        end
    end

    return mean,var

end






























end
