module CES

using Cloudy.KernelTensors
using Cloudy.CDistributions
using Cloudy.Sources

using JLD2 # saving and loading Julia arrays
using Distributions  # probability distributions and associated functions
using DelimitedFiles 
using Sundials
using StatsBase
using LinearAlgebra
using DifferentialEquations
using Random
using Statistics 
using Plots
using POMDPModelTools
using GaussianProcesses
#using ScikitLearn
using Optim

#@sk_import gaussian_process : GaussianProcessRegressor
#@sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)
#sk = ScikitLearn
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
export EKIObj
export construct_initial_ensemble
export compute_error
export run_cloudy
export run_cloudy_ensemble
export update_ensemble!
export TruthObj
export run_cloudy_truth
export MCMCObj
export mcmc_sample! 
export accept_ratio 
export reset_with_step!
export get_posterior
export find_mcmc_step!
export sample_posterior!



#####
##### Structure definitions
#####

# Structure to organize the "truth"

struct TruthObj
    distr_init::CDistributions.Distribution{Float64}
    solution::ODESolution
    yt::Array{Float64, 1}
    cov::Array{Float64, 2}
    data_names::Vector{String}
end

function TruthObj(distr_init::CDistributions.Distribution{Float64}, 
                  solution::ODESolution, cov::Array{Float64, 2},
                  data_names::Vector{String})

    yt = vcat(solution.u'...)[end, :]

    TruthObj(distr_init, solution, yt, cov, data_names)

end

#####
##### Functions definitions
#####

function check_re(M::Array{Float64, 1}, t::Float64, re_crit::Float64,
                  dist::CDistributions.Distribution{Float64})
    dist_temp = CDistributions.update_params_from_moments(dist, M)
    re_crit - CDistributions.moment(dist_temp, 1.0) / CDistributions.moment(dist_temp, 2.0/3.0) < 0
end


function run_cloudy_truth(kernel::KernelTensor{Float64}, 
                          dist::CDistributions.Distribution{Float64}, 
                          moments::Array{Float64}, 
                          tspan::Tuple{Float64, Float64}, 
                          data_names::Vector{String})

    # Numerical parameters
    tol = 1e-7

    ##
    ## ODE: Step moments forward in time
    ##

    # moments_init is the initial condition for the ODE problem
    n_moments = length(moments)
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = CDistributions.moment(dist, convert(Float64, mom))
    end
    #println(moments_init)

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src_coalescence(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)

    # Solve the ODE
    sol = solve(prob, CVODE_BDF(), reltol=tol, abstol=tol, save_everystep=true)
    #solcov = 0.1^2*(cov(vcat(sol.u'...)))
    # Here we are artificially turning the value of 0 for the variance of the first 
    # moment (a result of mass conservation) into a small value epsilon. Otherwise
    # the matrix is singular, and attempts to invert it will result in errors
    #eps = 0.1
    #solcov[2, 2] = eps
    # TO DO: Find a way to estimate a reasonable covariance structure of 
    # the noise
    Random.seed!(1234)
    A = rand(Distributions.Normal(0, 2.0), n_moments, n_moments) 
    solcov = A * A'
    println("solcov: $solcov")
    #solmean = vec(mean(vcat(sol.u'...), dims=1))
    #solcov = Diagonal(A * A')
    #solcov = 0.1^2 * Matrix{Float64}(I, n_moments, n_moments)

    return TruthObj(dist, sol, solcov, data_names)
end


function run_cloudy_truth(kernel::KernelTensor{Float64}, 
                          dist::CDistributions.Distribution{Float64}, 
                          moments::Array{Float64, 1}, re_crit::Float64,
                          data_names::Vector{String})

    # Numerical parameters
    tol = 1e-7

    ##
    ## ODE: Step moments forward in time
    ##

    # moments_init is the initial condition for the ODE problem
    n_moments = length(moments)
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = CDistributions.moment(dist, convert(Float64, mom))
    end
    #println(moments_init)

    tspan = (0., 100.)
    condition(M, t, integrator) = check_re(M, t, re_crit, dist)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src_coalescence(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)

    # Solve the ODE
    #sol = solve(prob, CVODE_BDF(), reltol=tol, abstol=tol, callback=cb)
    #solcov = cov(vcat(sol.u'...))
    Random.seed!(1234)
    A = rand(Distributions.Normal(0.0, 2.0), n_moments, n_moments) 
    solcov = A * A'
    # TO DO: Find a way to estimate a reasonable covariance structure of 
    # the noise
   # A = rand(Distributions.Uniform(-5.0, 5.0), n_moments, n_moments) 
    #cov = Diagonal(A * A')

    # This i
    solmean = vec(mean(vcat(sol.u'...), dims=1))

    return TruthObj(dist, sol, solmean, solcov, data_names)
end


"""
Module: EKI
-------------------------------------
Packages required: LinearAlgebra, 
                   Statistics,
                   Distributions
-------------------------------------
Idea: To construct an object to perform Ensemble Kalman updates
      It also measures errors to the truth to assess convergence
-------------------------------------
Exports: EKIObject
         update_ensemble!
         compute_error
         construct_initial_ensemble
         run_cloudy
         run_cloudy_ensemble
-------------------------------------
"""



#####
#####  Structure definitions
#####

# structure to organize data
struct EKIObj
     u::Vector{Array{Float64, 2}}
     unames::Vector{String}
     g_t::Vector{Float64}
     cov::Array{Float64, 2}
     N_ens::Int64
     g::Vector{Array{Float64, 2}}
     error::Vector{Float64}
end

#####
##### Function definitions
#####

# outer constructors
function EKIObj(parameters::Array{Float64, 2}, parameter_names::Vector{String},
                t_mean, t_cov::Array{Float64, 2})
    
    #covariance regularization
    #delta=0.0
    #t_cov= t_cov + delta*maximum(diag(t_cov))*Matrix{Float64}(I,size(t_cov)[1],size(t_cov)[2])
    # ensemble size
    N_ens = size(parameters)[1]
    # parameters
    u = Array{Float64, 2}[] # array of Matrix{Float64}'s
    push!(u, parameters) # insert parameters as at end of array (in this case just 1st entry)
    # observations
    g = Vector{Float64}[]
    # error store
    error = []
    
    EKIObj(u, parameter_names, t_mean, t_cov, N_ens, g, error)
end


"""
    construct_initial_ensemble(priors, N_ens::Int64)
  
Constructs the initial parameters, by sampling N_ens samples from specified 
prior distributions.
"""
function construct_initial_ensemble(N_ens::Int64, priors; rng_seed=42)
    # priors is an array of tuples, each of which contains the priors for all
    # parameters of the distribution they specify. Priors of length >1 (i.e, 
    # more than one tuple of parameters) mean that the distribution is a 
    # mixture.
    
    #priors_flattened = collect(Iterators.flatten(priors))
    #N_params = length(priors_flattened)
    N_params = length(priors)
    params = zeros(N_ens, N_params)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_params
        #prior_i = priors_flattened[i]
        prior_i = priors[i]
        if !(typeof(prior_i) == Deterministic{Float64})
            params[:, i] = rand(prior_i, N_ens)
        else
            # We are dealing with a Dirac Delta distribution (meaning that the 
            # corresponding parameter is treated as a constant). 
            # Distributions.jl does not have an implementation of a Delta
            # distribution, but POMDPModelTools does, so we have to use
            # POMDPModelTools' sampler here
            params[:, i] = fill(POMDPModelTools.rand(prior_i), N_ens)
        end
    end
    
    #write_ens(param_dir, param_names, param, "ens")
    return params
end # function construct_initial_ensemble

function compute_error(eki)
    meang = dropdims(mean(eki.g[end], dims=1), dims=1)
    diff = eki.g_t - meang 
    X = eki.cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(eki.error, newerr)
end


function run_cloudy_ensemble(kernel::KernelTensor{Float64}, 
                             dist::CDistributions.Distribution{Float64},
                             params::Array{Float64, 2}, 
                             moments::Array{Float64, 1}, config) 
    # config: either a tuple defining the time interval over which the 
    # size distribution is stepped forward in time, or a scalar definiing
    # the critical effective radius at which the integration is stopped
    
    # If dist is a Gamma Distribution, none of the parameters (n, θ, k) 
    # must be negative. For the time being we just set any negative values
    # to a small positive value. 
    # TODO: This is a temporary solution - there are probably better ways to
    # deal with this. E.g., another option would be to prevent this problem
    # already when the noise gets added in the EnKI.
    # Also the parameter check currently only applies to Gamma distributions,
    # but eventually validity of parameters needs to be ensured for all
    # distribution types.
#    if typeof(dist) == CDistributions.Gamma{Float64}
#        epsilon = 1e-5
#        params[params.<=0.] .= epsilon
#    end
    N_ens = size(params, 1) # params is N_ens x N_params
    n_moments = length(moments)
    g_ens = zeros(N_ens, n_moments)

    Random.seed!(42)

    for i in 1:N_ens
        # generate the initial distribution
        dist = CDistributions.update_params(dist, params[i, :])
        # run cloudy with this initial distribution
        g_ens[i, :] = run_cloudy(kernel, dist, moments, config)
    end

    return g_ens
end # function run_coudy_ensemble

        

"""
run_cloudy(kernel, dist, moments, re_crit)

- `kernel` - is the collision-coalescence kernel that determines the evolution 
           of the droplet size distribution
- `dist` - is a mass distribution function
- `re_crit` - is the critical cloud drop effective radius (in μm), i.e., the 
            effective radius (re) at which the run is stopped. According
            to Rosenfeld et al. 2012, "The Roles of Cloud Drop Effective 
            Radius and LWP in Determining Rain Properties in Marine 
            Stratocumulus" (doi: 10.1029/2012GL052028), rain is initiated
            when re near cloud top is around 12-14μm.

TODO: In the current implementation, re is something that's proportional to the 
effective radius, not the effective radius itself. (The reason is that we work 
with mass distributions, not with size distributions, and re is defined as the
third moment of the size distribution divided by the second moment of the 
size distribution, which is proportional to the first moment of the mass 
distribution divided by the 2/3rd moment of the mass distribution.)

Returns the moments at the time `re_crit` is reached, the time when that 
happened, and the covariance matrix of the moments
"""
function run_cloudy(kernel::KernelTensor{Float64}, 
                    dist::CDistributions.Distribution{Float64}, 
                    moments::Array{Float64, 1}, re_crit::Float64)

    # Numerical parameters
    tol = 1e-7

    ######
    ###### ODE: Step moments forward in time
    ######

    # Make sure moments are up to date. mom0 is the initial condition for the 
    # ODE problem
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = CDistributions.moment(dist, convert(Float64, mom))
    end

    # Define time interval for solving the ODE problem. 
    # Note: The integration will be stopped when re == re_crit, so tspan[1] is 
    # just an upper bound on the integration time period (ensures termination
    # even if re never equals re_crit)
    tspan = (0., 1000.)
    # Stop condition for ODE solver: re == re_crit
    condition(M, t, integrator) = check_re(M, t, re_crit, dist, integrator)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src_coalescence(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)

    # Solve the ODE
    sol = solve(prob, Tsit5(), alg_hints=[:stiff], reltol=tol, abstol=tol, 
                callback=cb)
    println("t final:")
    println(sol.t[end])
    if !(sol.t[end] < tspan[2])
        println("Haven't reached re_crit at t=$tspan[2]")
    end
    #@assert sol.t[end] < tspan[2]
    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]
    time = sol.t[end]
        
    return moments_final
    
end # function run_cloudy


"""
run_cloudy(kernel, dist, moments, tspan)

- `kernel` - is the collision-coalescence kernel that determines the evolution 
           of the droplet size distribution
- `dist` - is a mass distribution function
- `tspan` - is a tuple definint the time interval over which cloudy is run
"""
function run_cloudy(kernel::KernelTensor{Float64}, 
                    dist::CDistributions.Distribution{Float64}, 
                    moments::Array{Float64, 1}, 
                    tspan=Tuple{Float64, Float64})
  
    # Numerical parameters
    tol = 1e-7

    ######
    ###### ODE: Step moments forward in time
    ######

    # Make sure moments are up to date. mom0 is the initial condition for the 
    # ODE problem
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = CDistributions.moment(dist, convert(Float64, mom))
    end

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src_coalescence(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)
    # Solve the ODE
    sol = solve(prob, Tsit5(), alg_hints=[:stiff],reltol=tol, abstol=tol)
    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]
    time = tspan[2]
        
    return moments_final
end # function run_cloudy


function update_ensemble!(eki, g)

    u = eki.u[end]

    #dim1 is ensemble size
    #dim2 is param size / data size
    u_bar = fill(0.0, size(u)[2])
    g_bar = fill(0.0, size(g)[2])
    
    cov_ug = fill(0.0, size(u)[2], size(g)[2])
    cov_gg = fill(0.0, size(g)[2], size(g)[2])

    # update means/covs with new param/observation pairs u, g
    for j = 1:eki.N_ens

        u_ens = u[j, :]
        g_ens = g[j, :]

        # add to mean
        u_bar += u_ens
        g_bar += g_ens
        
        #add to cov
        cov_ug += u_ens * g_ens'
        cov_gg += g_ens * g_ens'
    end
   
    u_bar = u_bar / eki.N_ens
    g_bar = g_bar / eki.N_ens
    cov_ug = cov_ug / eki.N_ens - u_bar * g_bar'
    cov_gg = cov_gg / eki.N_ens - g_bar * g_bar'

    # update the parameters (with additive noise too)
    #noise = rand(MvNormal(zeros(size(g)[2]), eki.cov), eki.N_ens) # n_data * n_iter
    noise = zeros(size(g)[2], eki.N_ens)
#    println("size of noise")
#    println(size(noise))
#    println("eki.g_t")
#    println(eki.g_t)
#    println("eki.cov")
#    println(eki.cov)
    y = (eki.g_t .+ noise)' # add g_t (n_moments) to each column of noise (n_moments x N_ens), then transp. into N_ens x n_moments
    tmp = (cov_gg + eki.cov) \ (y - g)' # n_moments x n_moments \ [N_ens x n_moments - N_ens x n_moments]' --> tmp is n_moments x N_ens
    u += (cov_ug * tmp)' # N_ens x n_params  + [2x96 * 96*100]'
    println("cov_gg: $cov_gg")
    println("eki.cov: $(eki.cov)")
    println("cov_ug: $cov_ug")
    println("(cov_ug * tmp)': $((cov_ug * tmp)')")
    
    # store new parameters (and observations)
    push!(eki.u, u) # 100 x 2.
    push!(eki.g, g) # 100 x 96

    compute_error(eki)

end # function update_ensemble!
 
function check_re(M::AbstractArray, t::Float64, re_crit::Float64,
                  dist::CDistributions.Distribution{Float64}, integrator)
    #update_params!(dist, M)
    #println("time $(integrator.t)")
    #display(plot!(integrator,vars=(0, 3),legend=false))
    dist_temp = CDistributions.update_params_from_moments(dist, M)
    println("check:")
    println(CDistributions.moment(dist_temp, 1.0)/CDistributions.moment(dist_temp, 2/3))
    re_crit - CDistributions.moment(dist_temp, 1.0) / CDistributions.moment(dist_temp, 2/3) < 0
end


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

#####
##### Structure definitions 
###### #structure to hold inputs/ouputs kernel type and whether we 
##### do sparse GP struct GPObj inputs::Array{Float64, 2} 
##### data::Array{Float64, 2} models::Vector package::String end #####
##### Function definitions
#####

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
        println("size outputs: $(size(outputs))") #3x200
        println("size inputs: $(size(inputs))") #3x200

        for i in 1:size(outputs, 1)
            # Zero mean function
            meank = MeanZero() 
            # Construct kernel:
            # Sum kernel consisting of Matern 5/2 ARD kernel and Squared 
            # Exponential Iso kernel 
            len2 = 1.0
            var2 = 1.0
            kern1 = SE(len2, var2)
            kern2 = Matern(5/2, [0.0, 0.0, 0.0], 0.0)
            lognoise = 0.5
            #regularize with white noise
            white = Noise(log(2.0))
            # construct kernel
            kern = kern1 + kern2 + white

            #priors on GP hyperparameters
            #priorVec = fill(LogNormal(), 3)

            #inputs param dim x pts in R^2
            #data[i,:] pts x 1 in R
          

            #inputs param dim x pts in R^2
            #data[i,:] pts x 1 in R
            m = GPE(inputs, outputs[i,:], meank, kern, sqrt(lognoise))
            optimize!(m)
            u_true = [3650.0 4.06852 2.0]
            println("prediction on u_true:")
            println(predict_y(m, u_true'))
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
        
        #need python style indexing
        #outputs=[convert(Array{Float64},data[i,:]) for i in size(data,2)]
        #inputs=[convert(Array{Float64},inputs[i,:]) for i in size(inputs,2)]
        
        outputs = convert(Matrix{Float64}, data')
        inputs = convert(Matrix{Float64}, inputs)
        #println(size(outputs[1,:]))
        #println(size(inputs))
        
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
        new_inputs = convert(Matrix{Float64}, new_inputs')
    end    
    for i=1:M
        if gp.package == "gp_jl"
            if prediction_type == "y"
                mu, sig2 = predict_y(gp.models[i], new_inputs)
                push!(mean, mu)
                push!(var, sig2) 

            elseif prediction_type == "f"
                mu, sig2 = predict_f(gp.models[i],new_inputs')
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

end #function predict


function emulate(u_tp::Array{Float64, 2}, g_tp::Array{Float64, 2}, 
                 gppackage::String)

    gpobj = GPObj(u_tp, g_tp, gppackage)#construct the GP based on data
#    if gppackage=="gp_jl"
#        optimize_hyperparameters(gpobj)
#    end

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

"""
Module: MCMC
-------------------------------------
Packages required: LinearAlgebra, 
                   Statistics,
                   Distributions
-------------------------------------
Idea: To construct a simple MCMC object which, given a prior distribution,
      will update in a Metropolis-Hastings fashion with respect to a quadratic
      likelihood. It also computes acceptance ratios
-------------------------------------
Exports: MCMCObj
         mcmc_sample
         accept_ratio
         mcmc_sample
         reset_with_step!
         get_posterior
         find_mcmc_step!
         sample_posterior!
-------------------------------------
"""
#####
##### Structure definitions
#####

# structure to organize MCMC parameters and data
struct MCMCObj
    truth_sample::Vector{Float64}
    truth_cov::Array{Float64, 2}
    truth_covinv::Array{Float64, 2}
    proposal_cov::Array{Float64, 2}
    prior::Array
    step::Array{Float64}
    burnin::Int64
    param::Vector{Float64}
    posterior::Array{Float64, 2}
    log_posterior::Array{Union{Float64,Nothing}}
    iter::Array{Int64}
    accept::Array{Int64}
    algtype::String
end


#####
##### Function definitions
#####

#outer constructors
function MCMCObj(truth_sample::Vector{Float64}, truth_cov::Array{Float64, 2}, 
                 proposal_cov::Array{Float64, 2}, priors::Array, step::Float64, 
                 param_init::Vector{Float64}, max_iter::Int64, 
                 algtype::String, burnin::Int64)
    
    #first row is param_init 
    posterior = zeros(max_iter + 1, length(param_init))
    posterior[1, :] = param_init
    param = param_init
    log_posterior = [nothing]
    iter = [1]
    truth_covinv = inv(truth_cov)
    accept = [0]
    if algtype != "rwm"
        println("only random walk metropolis 'rwm' is implemented so far")
        sys.exit()
    end
    MCMCObj(truth_sample, truth_cov, truth_covinv, proposal_cov, priors, [step], 
            burnin, param, posterior, log_posterior, iter, accept, algtype)

end

function reset_with_step!(mcmc::MCMCObj, step::Float64)
    #reset to beginning with new stepsize
    mcmc.step[1] = step
    mcmc.log_posterior[1] = nothing
    mcmc.iter[1] = 1
    mcmc.accept[1] = 0
    mcmc.posterior[2:end, :] = zeros(size(mcmc.posterior[2:end, :]))
    mcmc.param[:] = mcmc.posterior[1, :]
end


#exported functions
function get_posterior(mcmc::MCMCObj)
    return mcmc.posterior[mcmc.burnin+1:end, :]
end

function mcmc_sample!(mcmc::MCMCObj, g::Vector{Float64}, gvar::Vector{Float64})
    if mcmc.algtype == "rwm"
        log_posterior = log_likelihood(mcmc, g, gvar) + log_prior(mcmc)
    end

    if mcmc.log_posterior[1] isa Nothing #do an accept step.
        mcmc.log_posterior[1] = log_posterior - log(0.5) #this makes p_accept = 0.5
    end    
    #else
    p_accept = exp(log_posterior - mcmc.log_posterior[1])
    #println("log post of current position: $log_posterior")
    #println("log post of previous position: $(mcmc.log_posterior[1])")

    if p_accept > rand(Distributions.Uniform(0, 1))
        mcmc.posterior[1 + mcmc.iter[1], :] = mcmc.param
        mcmc.log_posterior[1] = log_posterior
        mcmc.accept[1] = mcmc.accept[1] + 1
    else 
        mcmc.posterior[1 + mcmc.iter[1], :] = mcmc.posterior[mcmc.iter[1], :]
    end
    mcmc.param[:] = proposal(mcmc)[:]#performed by sampling about posterior[1+mcmc.iter[1],:]
    mcmc.iter[1] = mcmc.iter[1] + 1
  
end # function mcmc_sample!

function accept_ratio(mcmc::MCMCObj)
    return convert(Float64, mcmc.accept[1]) / mcmc.iter[1]
end


function log_likelihood(mcmc::MCMCObj, g::Vector{Float64}, 
                        gvar::Vector{Float64})
    log_rho = [0.0]
    if gvar == nothing
        diff = g - mcmc.truth_sample
        log_rho[1] = -0.5 * diff' * mcmc.truth_covinv * diff
    else
      total_cov = Diagonal(gvar) .+ mcmc.truth_cov
      total_cov = mcmc.truth_cov
      total_cov_inv = inv(total_cov)
      diff = g - mcmc.truth_sample
      log_rho[1] = -0.5 * diff' * total_cov_inv * diff - 0.5 * log(det(total_cov))
    end
    return log_rho[1]
end


function log_prior(mcmc::MCMCObj)
    log_rho = [0.0]
    # Assume independent priors for each parameter
    # TODO: This currently doesn't work for Mixture ditributions yet. priors = mcmc.prior[1]
    # extracts the priors of all parameters that define the first (primitive) components of
    # a distribution which is potentially a mixture of distributions. 
    priors = mcmc.prior
    for (param, prior_dist) in zip(mcmc.param, priors)
      log_rho[1] += logpdf(prior_dist, param) # get distrubtion at current parameter values
#        else
#            println("we have a deterministic distribution")
#            if param == Deterministic.val
#                log_rho[1] += 0.
#            else
#                log_rho[1] -= 100_000.
#            end
#       end
    end

    return log_rho[1]
end


function proposal(mcmc::MCMCObj)

    variances = ones(length(mcmc.param))
#    for (idx, prior) in enumerate(mcmc.prior)
#        variances[idx] = var(prior)
#    end

    if mcmc.algtype == "rwm"
        #prop_dist = MvNormal(mcmc.posterior[1 + mcmc.iter[1], :], (mcmc.step[1]) * Diagonal(variances))
        prop_dist = MvNormal(zeros(length(mcmc.param)), (mcmc.step[1]^2) * Diagonal(variances))
        #prop_dist = MvNormal(zeros(length(mcmc.param)), (mcmc.step[1]^2) * mcmc.proposal_cov)
    end
    sample = mcmc.posterior[1 + mcmc.iter[1], :] .+ rand(prop_dist)

    for (idx, prior) in enumerate(mcmc.prior)
        while !insupport(prior, sample[idx])
            println("not in support - resampling")
            sample[:] = mcmc.posterior[1 + mcmc.iter[1], :] .+ rand(prop_dist)
        end
    end
            
    return sample
end


function find_mcmc_step!(mcmc_test::MCMCObj, gpobj::GPObj)
    step = mcmc_test.step[1]
    mcmc_accept = false
    doubled = false
    halved = false
    countmcmc = 0
    
    println("Begin step size search")  
    println("iteration 0; current parameters ", mcmc_test.param')
    flush(stdout)
    it = 0
    while mcmc_accept == false 
        
        param = convert(Array{Float64, 2}, mcmc_test.param')
        #test predictions param' is 1x2
        gp_pred, gp_predvar = predict(gpobj, param)
        gp_pred = cat(gp_pred..., dims=2)
        gp_predvar = cat(gp_predvar..., dims=2)
        
        mcmc_sample!(mcmc_test, vec(gp_pred), vec(gp_predvar))
        it += 1
        if it % 2000 == 0
            countmcmc += 1
            acc_ratio = accept_ratio(mcmc_test)
            println("iteration ", it, "; acceptance rate = ", acc_ratio, 
                    ", current parameters ", param)
            flush(stdout)
            if countmcmc == 20 
                println("failed to choose suitable stepsize in ", countmcmc, 
                        "iterations")
                exit()
            end
            it = 0
            if doubled && halved
                step *= 0.75
                reset_with_step!(mcmc_test, step)
                doubled = false
                halved = false
            elseif acc_ratio < 0.15
                step *= 0.5
                reset_with_step!(mcmc_test, step)
                halved = true
            elseif acc_ratio>0.35
                step *= 2.0
                reset_with_step!(mcmc_test, step)
                doubled = true
            else
                mcmc_accept = true
            end
            if mcmc_accept == false
                println("new step size: ", step)
                flush(stdout)
            end
        end
           
    end

    return mcmc_test.step[1]
end # function find_mcmc_step!


function sample_posterior!(mcmc::MCMCObj, gpobj::GPObj, max_iter::Int64)

    println("iteration 0; current parameters ", mcmc.param')
    flush(stdout)
   
    for mcmcit in 1:max_iter
        param = convert(Array{Float64, 2}, mcmc.param')
        #test predictions param' is 1x2
        gp_pred, gp_predvar = predict(gpobj, param)
        gp_pred = cat(gp_pred..., dims=2)
        gp_predvar = cat(gp_predvar..., dims=2)
        
        mcmc_sample!(mcmc, vec(gp_pred), vec(gp_predvar))
   
        if mcmcit % 1000 == 0
            acc_ratio = accept_ratio(mcmc)
            println("iteration ", mcmcit ," of ", max_iter, 
                    "; acceptance rate = ", acc_ratio, 
                    ", current parameters ", param)
            flush(stdout)
        end
    end
end # function sample_posterior! 

end # module CES
