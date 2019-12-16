module Truth

# Import Cloudy modules
using Cloudy.KernelTensors
using Cloudy.CDistributions
using Cloudy.Sources

# packages
using JLD2 # saving and loading Julia arrays
using Distributions  # probability distributions and associated functions
using DelimitedFiles 
using StatsBase
using LinearAlgebra
using DifferentialEquations

export TruthObj
export run_cloudy_truth



#####
##### Structure definitions
#####

# Structure to organize the "truth"

struct TruthObj
    distr_init::CDistributions.Distribution{Float64}
    solution::ODESolution
    mean::Vector{Float64}
    cov::Array{Float64, 2}
    data_names::Vector{String}
end


#####
##### Functions definitions
#####

function check_re(M::Array{Float64, 1}, t::Float64, re_crit::Float64,
                  dist::CDistributions.Distribution{Float64})
    # update_params!(dist, M)
    re_crit - CDistributions.Distributions.moment(dist, 3.0) / CDistributions.Distributions.moment(dist, 2.0) < 0
end


function run_cloudy_truth(kernel::KernelTensor{Float64}, 
                          dist::CDistributions.Distribution{Float64}, 
                          tspan::Tuple{Float64, Float64}, 
                          data_names::Vector{String})

    # Numerical parameters
    tol = 1e-8

    ##
    ## ODE: Step moments forward in time
    ##

    # moments_init is the initial condition for the ODE problem
    n_moments = length(fieldnames(typeof(dist)))
    moments_init = fill(NaN, n_moments)
    for i in 1:n_moments
      moments_init[i] = CDistributions.moment(dist, convert(Float64, i))
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
    A = rand(Distributions.Uniform(-2.0, 2.0), n_moments, n_moments) 
    cov = A * A'
    solmean = vec(mean(vcat(sol.u'...), dims=1))

    return TruthObj(dist, sol, solmean, cov, data_names)
end

end # module Truth
