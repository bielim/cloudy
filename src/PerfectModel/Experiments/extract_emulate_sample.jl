include("../src/file_sys.jl")
include("../src/gpemulator.jl")
include("../src/mcmc.jl")
include("../src/eki.jl")
include("../src/truth.jl")
#1 Extracts the correct designs from the user,
#2 Emulate - trains an GP emulator on the avaliable training points restricted to the design data
#3 Sample - Runs an MCMC algorithm (modified for application on GP)   

using JLD2
using Random
using Statistics
using Distributions
using LinearAlgebra
using .GPEMULATOR
using .MCMC
using .EKI
using Distributed
using Plots
using Plots.PlotMeasures

#Functions: LOTS OF THEM SO SCROLL DOWN TO SCRIPT
function get_des_idx(dtype::String,grpi::Int64,dgrp_size::Int64,nlats::Int64,dpts_size::Int64,len::Int64)
    if dtype == "distinct"
        dmin=((grpi-1)*dgrp_size+1)
        dmax=grpi*dgrp_size
    elseif dtype == "overlap"
        dmin=grpi
        dmax=grpi+dgrp_size-1
    elseif dtype == "none"
        dmin=1
        dmax=nlats
    end

    #create design indices for group i
    des_ind_tmp=collect(dmin:dmax)
    des_ind=[(j-1).*dpts_size .+ des_ind_tmp for j=1:len]
    return cat(des_ind...,dims=1)
end

function extract(dtype::String,grpi::Int64,dgrp_size::Int64,nlats::Int64,dpts_size::Int64,truthobj::TruthObj,ekiobj::EKIObj)
    
    #first get the design indices 
    des_ind=get_des_idx(dtype,grpi,dgrp_size,nlats,dpts_size,length(truthobj.data_names))
    

    yt=truthobj.mean[des_ind]
    yt_cov=truthobj.cov[des_ind,des_ind]
    yt_covinv=inv(yt_cov)
    #draw a random sample (same for all designs)
    yt_sample=truthobj.sample[sample_ind]
    yt_sample=yt_sample[des_ind]
    
    #restrict input-output pairs
    #note u[end] does not have an equivalent g
    u_tp=ekiobj.u[end-eki_it_train:end-1]#it x [enssize x param]
    g_tp=ekiobj.g[end-eki_it_train+1:end]#it x [enssize x dat]
    enssize=size(u_tp[1],1)

    #u does not require reduction, g does:
    #g_tp[j] is jth iteration of ensembles (size 100 x 96)
    #g_tp[j][:,des_ind] gives the restriction of data to des_ind for all ensembles
    g_tp=[g_tp[j][:,des_ind] for j=1:length(g_tp)]
    #g_tp still a vector of matrices [ 100 x len(des_ind)]
    u_tp=cat(u_tp...,dims=1)#[(itxens) x param]
    g_tp=cat(g_tp...,dims=1)#[(itxens) x len(des_ind)]

    return des_ind,yt,yt_cov,yt_covinv,yt_sample,u_tp,g_tp
end

function emulate(u_tp::Matrix{Float64},g_tp::Matrix{Float64},gppackage::String)

    gpobj=GPemulator(u_tp,g_tp,gppackage)#construct the GP based on data
    if gppackage=="gp_jl"
        optimize_hyperparameters(gpobj)
    end
                 
    #=
    #test GP train it on some points and test on others:
    utrain=u_tp[enssize*6+1:end-2*enssize,:]
    gtrain=g_tp[enssize*6+1:end-2*enssize,:]
    utest=u_tp[end-enssize+1:end,:]
    gtest=g_tp[end-enssize+1:end,:]
    gpobj=GPemulator(utrain,gtrain,gppackage)#construct the GP based on data
    
    if gppackage=="gp_jl"
        optimize_hyperparameters(gpobj)
    end
    

    #test predictions
    gtest_pred,gtest_predvar= predict(gpobj,utest)
    gtest_pred=cat(gtest_pred..., dims=2)
    err=(gtest-gtest_pred)*yt_covinv*(gtest-gtest_pred)'
    l2err=sqrt.(sum(sum(err.^2)))
    println("hparam error in predictions")
    println(l2err)

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


function find_mcmc_step!(mcmc_test::MCMCObj,gpobj::GPemulator)
    step=mcmc_test.step[1]
    mcmc_accept=false
    doubled=false
    halved=false
    countmcmc=0
    
    println("Begin step size search")  
    println("iteration 0; current parameters ", mcmc_test.param')
    flush(stdout)
    it=0
    while mcmc_accept == false 
        
        param = convert(Matrix{Float64},mcmc_test.param')
        #test predictions param' is 1x2
        gp_pred,gp_predvar= predict(gpobj,param)
        gp_pred=cat(gp_pred..., dims=2)
        gp_predvar=cat(gp_predvar..., dims=2)
        
        mcmc_sample(mcmc_test,vec(gp_pred),vec(gp_predvar))
        it+=1
        if it % 2000 == 0
            countmcmc+=1
            acc_ratio=accept_ratio(mcmc_test)
            println("iteration ", it, "; acceptance rate = ", acc_ratio, ", current parameters ", param)
            flush(stdout)
            if countmcmc == 20 
                println("failed to choose suitable stepsize in ", countmcmc, "iterations")
                exit()
            end
            it=0
            if doubled && halved
                step*=0.75
                reset_with_step!(mcmc_test,step)
                doubled=false
                halved=false
            elseif acc_ratio<0.15
                step*=0.5
                reset_with_step!(mcmc_test,step)
                halved=true
            elseif acc_ratio>0.35
                step*=2.0
                reset_with_step!(mcmc_test,step)
                doubled=true
            else
                mcmc_accept=true
            end
            if mcmc_accept == false
                println("new step size: ", step)
                flush(stdout)
            end
        end
           
    end
    return mcmc_test.step[1]
end

function sample_posterior!(mcmc::MCMCObj,gpobj::GPemulator, max_iter::Int64)

    println("iteration 0; current parameters ", mcmc.param')
    flush(stdout)
   
    for mcmcit=1:max_iter
        param = convert(Matrix{Float64},mcmc.param')
        #test predictions param' is 1x2
        gp_pred,gp_predvar= predict(gpobj,param)
        gp_pred=cat(gp_pred..., dims=2)
        gp_predvar=cat(gp_predvar..., dims=2)
        
        mcmc_sample(mcmc,vec(gp_pred),vec(gp_predvar))
   
        if mcmcit % 10000 == 0
            acc_ratio=accept_ratio(mcmc)
            println("iteration ", mcmcit ," of ", max_iter, "; acceptance rate = ", acc_ratio, ", current parameters ", param)
            flush(stdout)
        end
    end

end



#A "design" corresponds to a subset of indexing of data. i.e this script will be provided with an array 
dpts_size=parse(Int64,ARGS[1])
dgrp_size=parse(Int64,ARGS[2])
dtype=ARGS[3]
nlats=parse(Int64,ARGS[4])
pardir=ARGS[5]
outdir=ARGS[6]


datdir=outdir*"eki_ltruth/"
#Load truth
@load datdir*"truth.jld2" truthobj
#Load EKI object (for input/output pairs)
@load datdir*"eki.jld2" ekiobj

#seed for random sample from truth
#Random.seed!(1234)
sample_ind=randperm!(collect(1:length(truthobj.sample)))[1]
#package Gaussian processes "gp_jl", Scikitlearn "sk_jl"
gppackage="sk_jl"

#design
des_ind=Array{Int64,1}
eki_it_train=5 #no. layers of eki to train on

if dtype == "distinct"
    ngrp=dpts_size/dgrp_size
elseif dtype == "overlap"
    ngrp=dpts_size-dgrp_size+1
elseif dtype == "none"
    ngrp=1
    @assert nlats == dpts_size #assert the design pts are just the latitudes
else
    println("unrecognised design type")
    exit()
end

println("number of groups")
println(ngrp)

#@distributed for grpi=1:ngrp
#for grpi=1:ngrp    
for grpi=12:18
    println(" ")
    println("design: ", grpi)
    ##############################################################
    #
    #----Extract----
    #
    #a) create design extraction indices
    #a) Load data and parameters from EKI
    #b) pair up input/output data and restrict with design indices
    ##############################################################

    des_ind,yt,yt_cov,yt_covinv,yt_sample,u_tp,g_tp=extract(dtype,grpi,dgrp_size,nlats,dpts_size,truthobj,ekiobj)
    println(des_ind)

    ############################################################
    #
    #----Emulate----
    #
    #a) Create GP emulator based on input/output
    #b) Optimize hyperparameters of GP
    ############################################################
   
    gpobj=emulate(u_tp,g_tp,gppackage)

    ###################################################################
    #
    #----Sample----
    #a) Create the priors (loaded from EKI priors)
    #b) load truth sample for use in sampling
    #c) Create MCMC object from GP, prior
    #d) run MCMC sampling algorithm on GP to obtain samples of posterior
    ####################################################################

    #call ekiobj.param_names
    param_names=ekiobj.unames
    
    #eventually, call ekiobj.priors (priors created from param_names)
    
    #rhbm prior
    if "rhbm_N" in param_names
        rhbm_mean=0.6       
        rhbm_sd=0.35
        rprior=Dict("distribution" => "normal","mean" => rhbm_mean, "sd" => rhbm_sd)
        
        #rhbm_N=rand(TruncatedNormal(0.6,0.35,0,1) , ens_size)
    elseif "rhbm_U" in param_names
        rhbm_min=0.
        rhbm_max=1.
        rprior=Dict("distribution" => "uniform", "min" => rhbm_min, "max" => rhbm_max)
        
        #  rhbm_U=rand(Uniform(0,1) , ens_size)
    end

    #logtau prior
    if "logtau" in param_names
        tau_mean=12*3600#6000
        tau_sd=12*3600#7200
        
        tmp=log(tau_sd/tau_mean)^2 +1
        logtau_mean=log(tau_mean) - 0.5*tmp
        logtau_sd=sqrt(tmp)
        
        tprior = Dict("distribution" => "normal", "mean" => logtau_mean, "sd" => logtau_sd)
        #logtau=rand(Normal(logtau_mean,logtau_sd) , ens_size)    
    end
    
    prior=[rprior,tprior]
    println(typeof(prior))

    #initial values
    
    u0=vec(mean(u_tp,dims=1))
    println("initial parameters: ", u0)
    println("prior for rhum ",rprior)
    println("prior for logtau ",tprior)

    #sample truth from
    #yt_sample (already done)

    #MCMC parameters    
    burnin=0
    mcmc_alg="rwm"
    step=0.1
    #Choose a suitable step size for MCMC
    mcmc_test= MCMCObj(yt_sample,yt_cov,prior,step,u0,5000,mcmc_alg,burnin)
    new_step=find_mcmc_step!(mcmc_test,gpobj)
    

    #Begin MCMC
    println("Begin MCMC - with step size ", new_step)
    flush(stdout)
    #reset parameters 
    burnin=5000
    max_iter=100_000
    mcmc= MCMCObj(yt_sample,yt_cov,prior,new_step,u0,max_iter,mcmc_alg,burnin)
    println("check u0",u0)
    println("check mcmc u0",mcmc.param)
    sample_posterior!(mcmc,gpobj,max_iter)
    
    #save final MCMC object
    @save datdir*"mcmc_"*string(grpi)*".jld2" mcmc
    
    posterior=get_posterior(mcmc)      
    post_mean= mean(posterior,dims=1)
    post_cov= cov(posterior,dims=1)
    println("post_mean")
    println(post_mean)
    println("post_cov")
    println(post_cov)
    println("D util")
    println(det(inv(post_cov)))
    println(" ")

end



