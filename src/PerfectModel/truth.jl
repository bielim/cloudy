# Import Cloudy modules
using Cloudy.KernelTensors
import Cloudy.CDistributions
using Cloudy.Sources

# Other modules
using JLD2 # saving and loading Julia arrays
using Random
using Distributions  # probability distributions and associated functions
using DelimitedFiles 
using StatsBase
using LinearAlgebra
using DifferentialEquations

export TruthObj
export check_re
export run_cloudy_truth


struct TruthObj
    distr_init::CDistributions.Distribution{Float64}
    solution::ODESolution
    mean::Vector{Float64}
    cov::Array{Float64, 2}
    data_names::Vector{String}
end


function check_re(M::Array{T, 1}, t::T, re_crit::T,
                  dist::CDistributions.Distribution{T}) where{T<:Real}
    # update_params!(dist, M)
    re_crit - CDistributions.Distributions.moment(dist, 3.0) / CDistributions.Distributions.moment(dist, 2.0) < 0
    end


function run_cloudy_truth(kernel::KernelTensor{T}, 
                          dist::CDistributions.Distribution{T}, 
                          tspan::Tuple{T,T}, 
                          data_names::Vector{String}) where {T <: Real}

    # Numerical parameters
    tol = 1e-8

    ######
    ###### ODE: Step moments forward in time
    ######

    # Make sure moments are up to date. moments_init is the 
    # initial condition for the ODE problem
    n_moments = length(fieldnames(typeof(dist)))
    moments_init = fill(NaN, n_moments)
    for i in 1:n_moments
      moments_init[i] = CDistributions.moment(dist, convert(Float64,i))
    end
    println(moments_init)
    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src_coalescence(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)

    # Solve the ODE
    sol = solve(prob, Tsit5(), reltol=tol, abstol=tol, save_everystep=true)
    # solcov = cov(vcat(sol.u'...))
    # TO DO: Find a way to estimate a reasonable covariance structure of 
    # the noise
    A = rand(Distributions.Uniform(-2, 2), n_moments, n_moments) 
    cov = A * A'
    solmean = vec(mean(vcat(sol.u'...), dims=1))

    return TruthObj(dist, sol, solmean, cov, data_names)
end
