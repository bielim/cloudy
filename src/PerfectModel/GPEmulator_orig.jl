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
# import Cloudy modules
include("/home/melanie/cloudy/src/PerfectModel/EKI.jl")
using ..EKI
include("/home/melanie/cloudy/src/PerfectModel/Truth.jl")
using ..Truth

# packages
using Statistics 
using Distributions
using LinearAlgebra
using GaussianProcesses
using ScikitLearn

#@sk_import gaussian_process : GaussianProcessRegressor
# @sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)
#sk = ScikitLearn
using Optim
#using Interpolations #will use this later (faster)
#using PyCall
#py=pyimport("scipy.interpolate")
#imports

#exports
export GPObj
export predict
export optimize_hyperparameters
export emulate
export extract


#####
##### Structure definitions
#####
#structure to hold inputs/ouputs kernel type and whether we do sparse GP
struct GPObj
    inputs::Array{Float64, 2}
    data::Array{Float64, 2}
    models::Vector
    package::String
end

#####
##### Function definitions
#####

function GPObj(inputs, data, package)

    if package == "gp_jl"
        #Create the GP object(s)
        #mean:
        mean = MeanZero()
        #kernel - lengthscale(squared), variance
        len2 = ones(size(inputs, 2))
        var2 = 1.0
        kern = SEArd(len2,var2)
        #likelihood
        #lik=GaussLik(1.0)
        lognoise = 0.5
        
        #regularize with white noise
        white = Noise(log(2.0))
        
        kern = kern + white
        
        models = Any[]
        println(size(inputs))
        println(size(data))
        flush(stdout)
        outputs = convert(Matrix{Float64}, data')
        inputs = convert(Matrix{Float64}, inputs')
        
        #priors
        priorVec = fill(LogNormal(), length(len2)+1+1)
        for i in 1:size(outputs, 1)
            #m=GP(inputs,data,mean,kernel,likelihood)
            #inputs param dim x pts in R^2
            #data[i,:] pts x 1 in R
            m = GPE(inputs, outputs[i,:], mean, kern, log(sqrt(lognoise)))
            println(m.kernel)
            set_priors!(m.kernel, priorVec)
            #set_priors!(m.lik,Normal())
            #println(m.kernel.priors)
            #println(m.lik.priors)
            flush(stdout)
            push!(models, m)
            
        end
        
        GPObj(inputs, outputs, models, package)
        
    elseif package == "sk_jl"
        
        len2 = ones(size(inputs,2))
        var2 = 1.0

        varkern = ConstantKernel(constant_value=var2,
                                 constant_value_bounds=(1e-05,10000.0))
        rbf = RBF(length_scale=len2, length_scale_bounds=(1.0,10000.0))
                 
        white = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-05,10.0))
        kern = varkern * rbf + white
        models = Any[]
        
        #need python style indexing
        #outputs=[convert(Array{Float64},data[i,:]) for i in size(data,2)]
        #inputs=[convert(Array{Float64},inputs[i,:]) for i in size(inputs,2)]
        
        outputs = convert(Matrix{Float64}, data')
        inputs = convert(Matrix{Float64}, inputs)
        println(size(outputs[1,:]))
        println(size(inputs))
        
        for i in 1:size(outputs,1)
            out = reshape(outputs[i,:], (size(outputs, 2), 1))
            
            m = GaussianProcessRegressor(kernel=kern,
                                         n_restarts_optimizer=10,
                                         alpha=0.0)
            ScikitLearn.fit!(m, inputs, out)
            if i==1
                println(m.kernel.hyperparameters)
                print("Completed training of: ")
            end
            print(i,", ")
            
            #println(m.kernel.hyperparameters)
            flush(stdout)
            push!(models, m)
            #println(m)
            #flush(stdout)
        end

        GPObj(inputs, outputs, models, package)

    else 
        println("use package sk_jl or gp_jl")
    end
end #function GPObj

function optimize_hyperparameters(gp::GPObj)
    if gp.package == "gp_jl"
        for i in 1:length(gp.models) 
            optimize!(gp.models[i])
            println(gp.models[i].kernel)
            flush(stdout)
        end
    elseif gp.package == "sk_jl"
        println("optimization already accounted for with fit!")
        flush(stdout)
    end
end #function optimize_hyperparameters

function predict(gp::GPObj, new_inputs::Array{Float64}; 
                 prediction_type="y")
    
    #predict data (type "y") or latent function (type "f")
    #column of new_inputs gives new parameter set to evaluate gp at
    M = length(gp.models)
    mean = Array{Float64}[]
    var = Array{Float64}[]
    #predicts columns of inputs so must be transposed
    if gp.package == "gp_jl"
        new_inputs = convert(Matrix{Float64},new_inputs')
    end    
    for i=1:M
        if gp.package == "gp_jl"
            if prediction_type == "y"
                mu,sig2 = predict_y(gp.models[i],new_inputs)
                #mu,sig2=predict_y(gp.models[i],new_inputs)
                push!(mean,mu)
                push!(var,sig2) 

            elseif prediction_type == "f"
                mu, sig2 = predict_f(gp.models[i],new_inputs')
                push!(mean,mu)
                push!(var,sig2) 

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

end #function predict


function emulate(u_tp::Array{Float64, 2}, g_tp::Array{Float64, 2}, 
                 gppackage::String)

    gpobj=GPObj(u_tp, g_tp, gppackage)#construct the GP based on data
    if gppackage=="gp_jl"
        optimize_hyperparameters(gpobj)
    end
                 
    #=
    #test GP train it on some points and test on others:
    utrain=u_tp[enssize*6+1:end-2*enssize,:]
    gtrain=g_tp[enssize*6+1:end-2*enssize,:]
    utest=u_tp[end-enssize+1:end,:]
    gtest=g_tp[end-enssize+1:end,:]
    gpobj=GPObj(utrain,gtrain,gppackage)#construct the GP based on data
    
    if gppackage=="gp_jl"
        optimize_hyperparameters(gpobj)
    end


    #unfortunately can't save GP like this with the sklearn package 
    #as the model is a pointer
    #@save datdir*"gp.jld2" gpobj
    
    #plot graph of gp prediction (if using whole domain
    if dtype == "none"
        #@ true params need to be pts x params 
        utrue=[0.7 log(7200)]
        xplt=truthobj.lats
        y_pred,y_predvar = predict(gpobj,utrue)
        y_pred=cat(y_pred...,dims=2)
        y_predvar=cat(y_predvar...,dims=2)
        y_predsd=sqrt.(y_predvar)
        #y_pred=1x96
        println(size(y_predvar))
        println(y_predvar)
        #backend
        gr()
        
        plot(xplt,y_pred[1:nlats],
             ribbon=(2*y_predsd[1:nlats],2*y_predsd[1:nlats]),
             legend=false,
             dpi=300,
             left_margin=50px,
             bottom_margin=50px)
        xlabel!("latitudes")
        ylabel!("relative humidity")
        title!("Relative humidity")
        savefig(outdir*"rhumplot_gp.png")
        
        plot(xplt,y_pred[nlats+1:2*nlats],
             ribbon=(2*y_predsd[nlats+1:2*nlats],2*y_predsd[nlats+1:2*nlats]),             
             legend=false,
             dpi=300,
             left_margin=50px,
             bottom_margin=50px)
        xlabel!("latitudes")
        ylabel!("precipitation")
        title!("Daily precipitation")
        savefig(outdir*"precipplot_gp.png")
        
        #ylim_min=0.9*minimum(y_pred[2*nlats+1:3*nlats])
        #ylim_max=1.1*maximum(y_pred[2*nlats+1:3*nlats])
        
        
        plot(xplt,y_pred[2*nlats+1:3*nlats],
             ribbon=(2*y_predsd[2*nlats+1:3*nlats],2*y_predsd[2*nlats+1:3*nlats]),
             #ylims = (ylim_min,ylim_max),
             legend=false,
             dpi=300,
             left_margin=50px,
             bottom_margin=50px)
        xlabel!("latitudes")
        ylabel!("probability of extreme precipitation")
        title!("Extreme precipitation probability")
        savefig(outdir*"extplot_gp.png")
    end
    =#
              
    return gpobj
end

function extract(truthobj::TruthObj, ekiobj::EKIObj, N_eki_it::Int64)
    
    yt = vcat(truthobj.solution.u'...)[end,:]
    yt_cov = truthobj.cov
    yt_covinv = inv(yt_cov)
    
    #note u[end] does not have an equivalent g
    u_tp = ekiobj.u[end-N_eki_it:end-1]#it x [enssize x N_param]
    g_tp = ekiobj.g[end-N_eki_it+1:end]#it x [enssize x N_data]

    #u does not require reduction, g does:
    #g_tp[j] is jth iteration of ensembles 
    u_tp = cat(u_tp..., dims=1) #[(it x ens) x N_param]
    g_tp = cat(g_tp..., dims=1) #[(it x ens) x N_data

    return yt, yt_cov, yt_covinv, u_tp, g_tp
end

end #module GPEmulator
