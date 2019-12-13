module EKI
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
         update_ensemble
         compute_error
         construct_initial_ensemble
         run_cloudy
         run_cloudy_ensemble
-------------------------------------
"""

#packages
using Statistics 
using Plots
using Distributions
using Cloudy.CDistributions
using Cloudy.Sources
using Cloudy.KernelTensors
using LinearAlgebra
using DifferentialEquations
using POMDPModelTools
#using Interpolations #will use this later (faster)
#using PyCall
#py=pyimport("scipy.interpolate")

# exports
export EKIObj
export construct_initial_ensemble
export compute_error
export run_cloudy
export run_cloudy_ensemble
export update_ensemble!



####
#### Structure definitions
####

#structure to organize data
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
#
#outer constructors
function EKIObj(parameters::Array{Float64, 2},
                parameter_names::Vector{String},
                t_mean::Vector{Float64},
                t_cov::Array{Float64, 2})
    
    #covariance regularization
    #delta=0.0
    #t_cov= t_cov + delta*maximum(diag(t_cov))*Matrix{Float64}(I,size(t_cov)[1],size(t_cov)[2])
    # ensemble size
    N_ens = size(parameters)[1]
    # parameters
    u = Array{Float64, 2}[] #array of Matrix{Float64}'s
    push!(u, parameters) #insert parameters as at end of array(in this case just 1st entry)
    # observations
    g = Vector{Float64}[]
    
    # error store
    error=[]
    
    EKIObj(u, parameter_names, t_mean, t_cov, N_ens, g, error)
end


"""
    construct_initial_ensemble(priors, N_ens::Int64)
  
Constructs the initial parameters, by sampling N_ens samples from specified 
prior distributions. dist
"""
function construct_initial_ensemble(N_ens::Int64, priors)
    # priors is an array of tuples, each of which contains the priors for all
    # parameters of the distribution they specify. Priors of length >1 (i.e, 
    # more than one tuple of parameters) mean that the distribution is a 
    # mixture.
    
    priors_flattened = collect(Iterators.flatten(priors))
    N_params = length(priors_flattened)
    params = zeros(N_ens, N_params)
    for i in 1:N_params
        prior_i = priors_flattened[i]
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
end

function compute_error(eki::EKIObj)
    meang = dropdims(mean(eki.g[end], dims=1), dims=1)
    diff = eki.g_t - meang 
    X = eki.cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(eki.error, newerr)
end

function run_cloudy_ensemble(kernel, dist, params::Array{Float64, 2}, moments, 
                             config) 
    # config: either a tuple defining the time interval over which the 
    # size distribution is stepped forward in time, or a scalar definiing
    # the critical effective radius at which the integration is stopped
    #TODO: config should be something like *args in python - the idea is to
    #call  run_cloudy with whatever variables config contains, and multiple
    #dispatch will then ensure the "right" run_cloudy is executed

    N_ens = size(params, 1) # params is N_ens x N_params
    n_moments = length(moments)
    g_ens = zeros(N_ens, n_moments)

    for i in 1:N_ens
        # generate the initial distribution
        dist = CDistributions.update_params(dist, params[i, :])
        # run cloudy with this initial distribution
        g_ens[i, :] = run_cloudy(kernel, dist, moments, config)
    end

    return g_ens
end

        

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
function run_cloudy(kernel::KernelTensor{FT}, 
                    dist::CDistributions.Distribution{FT}, 
                    moments::Array{FT, 1}, re_crit::FT) where {FT <: Real}

    # Numerical parameters
    tol = 1e-8

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
    tspan = (0., 100.)
    # Stop condition for ODE solver: re == re_crit
    condition(M, t, integrator) = check_re(M, t, re_crit, dist, integrator)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src_coalescence(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)

    # Solve the ODE
    sol = solve(prob, Tsit5(), reltol=tol, abstol=tol, callback=cb)
    @assert sol.t[end] < tspan[2]
    time = sol.t
    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]
    time = sol.t[end]
    #covariance = cov(vcat(sol.u'...))
        
    return moments_final
    
end


"""
run_cloudy(kernel, dist, moments, tspan)

- `kernel` - is the collision-coalescence kernel that determines the evolution 
           of the droplet size distribution
- `dist` - is a mass distribution function
- `tspan` - is a tuple definint the time interval over which cloudy is run
"""
function run_cloudy(kernel::KernelTensor{FT}, 
                    dist::CDistributions.Distribution{FT}, 
                    moments::Array{FT, 1}, 
                    tspan=Tuple{FT, FT}) where {FT <: Real}
  
    # Numerical parameters
    tol = 1e-8

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
    sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)
    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end,:]
    time = tspan[2]
    #covariance = cov(vcat(sol.u'...))
        
    return moments_final
end


function update_ensemble!(eki::EKIObj, g)

    u = eki.u[end]

    #dim1 is ensemble size
    #dim2 is param size / data size
    u_bar = fill(0.0,size(u)[2])
    g_bar = fill(0.0,size(g)[2])
    
    cov_ug = fill(0.0,size(u)[2],size(g)[2])
    cov_gg = fill(0.0,size(g)[2],size(g)[2])
    # update means/covs with new param/observation pairs u, g
    for j = 1:eki.N_ens

        u_ens = u[j,:]
        g_ens = g[j,:]
        #add to mean
        u_bar += u_ens
        g_bar += g_ens
        
        #add to cov
        cov_ug += u_ens*g_ens'
        cov_gg += g_ens*g_ens'
    end
   
    u_bar = u_bar/eki.N_ens
    g_bar = g_bar/eki.N_ens
    cov_ug = cov_ug / eki.N_ens - u_bar*g_bar'
    cov_gg = cov_gg / eki.N_ens - g_bar*g_bar'
    #update the parameters (with additive noise too)
    noise = rand(MvNormal(zeros(size(g)[2]), eki.cov), eki.N_ens)#96x100
    y = (eki.g_t .+ noise)' #add g_t (96) to each column of noise (96x100), then transp. into 100 x 96
    tmp = (cov_gg + eki.cov) \ (y-g)' #96 x 96 \ [100 x 96 - 100 x 96]' = 96 x 100
    
    u += (cov_ug * tmp)' # 100x2 + [2x96 * 96*100]'
    
    #store new parameters (and observations)
    push!(eki.u, u)#100 x 2.
    push!(eki.g, g)#100 x 96

    compute_error(eki)

end #function update_ensemble
 
function check_re(M::AbstractArray, t::T, re_crit::T,
                  dist::CDistributions.Distribution{T}, integrator) where{T<:Real}
    #update_params!(dist, M)
    println("time $(integrator.t)")
    display(plot!(integrator,vars=(0, 3),legend=false))
    re_crit - CDistributions.moment(dist, 3.0) / CDistributions.moment(dist, 2.0) < 0
end

end #module EKI