module Truth

# Import Cloudy modules
using Cloudy.KernelTensors
using Cloudy.CDistributions
using Cloudy.Sources

# packages
using JLD2 # saving and loading Julia arrays
using Distributions  # probability distributions and associated functions
using DelimitedFiles 
using Sundials
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
    solcov = 0.1^2*Diagonal(diag(cov(vcat(sol.u'...))))
    solcov = convert(Array, solcov)
    # Here we are artificially turning the value of 0 for the variance of the first 
    # moment (a result of mass conservation) into a small value epsilon. Otherwise
    # the matrix is singular, and attempts to invert it will result in errors
    eps = 0.1
    solcov[2,2] = eps
    println(typeof(solcov))  
    # TO DO: Find a way to estimate a reasonable covariance structure of 
    # the noise
    #A = rand(Distributions.Uniform(-2.0, 2.0), n_moments, n_moments) 
    #cov = A * A'
    solmean = vec(mean(vcat(sol.u'...), dims=1))
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
    sol = solve(prob, CVODE_BDF(), reltol=tol, abstol=tol, callback=cb)
    solcov = cov(vcat(sol.u'...))
    # TO DO: Find a way to estimate a reasonable covariance structure of 
    # the noise
   # A = rand(Distributions.Uniform(-5.0, 5.0), n_moments, n_moments) 
    #cov = Diagonal(A * A')

    solmean = vec(mean(vcat(sol.u'...), dims=1))

    return TruthObj(dist, sol, solmean, solcov, data_names)
end


end # module Truth
