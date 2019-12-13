module MCMC
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
         sample
         accept_ratio
-------------------------------------
"""

#packages
using Statistics 
using Distributions
using LinearAlgebra

#exports
export MCMCObj
export mcmc_sample,accept_ratio,reset_with_step!,get_posterior



#####################################################################
#
#Structure definitions
#
#################################################################### 
#structure to organise MCMC parameters and data
struct MCMCObj
    truth_sample::Vector{Float64}
    truth_cov::Matrix{Float64}
    truth_covinv::Matrix{Float64}
    prior::Array
    step::Array{Float64}
    burnin::Int64
    param::Vector{Float64}
    posterior::Matrix{Float64}
    log_posterior::Array{Union{Float64,Nothing}}
    iter::Array{Int64}
    accept::Array{Int64}
    algtype::String
end

#####################################################################
#
#Function definitions
#
#################################################################### 

#outer constructors
function MCMCObj(truth_sample::Vector{Float64},truth_cov::Matrix{Float64},prior::Array,step::Float64,param_init::Vector{Float64}, max_iter::Int64, algtype::String,burnin::Int64)
    
    #first row is param_init 
    posterior=zeros(max_iter+1,length(param_init))
    posterior[1,:]=param_init
    param=param_init
    log_posterior=[nothing]
    iter=[1]
    truth_covinv= inv(truth_cov)
    accept=[0]
    if algtype != "rwm"
        println("only random walk metropolis 'rwm' is implemented so far")
        sys.exit()
    end
    MCMCObj(truth_sample,truth_cov,truth_covinv,prior,[step],burnin,param,posterior,log_posterior,iter,accept,algtype)


end

function reset_with_step!(self::MCMCObj,step::Float64)
    #reset to beginning with new stepsize
    self.step[1]=step
    self.log_posterior[1]=nothing
    self.iter[1]=1
    self.accept[1]=0
    self.posterior[2:end,:] =zeros(size(self.posterior[2:end,:]))
    self.param[:]=self.posterior[1,:]
    println(self.param)
end


#exported functions
function get_posterior(self::MCMCObj)
    return self.posterior[self.burnin+1:end,:]
end

function mcmc_sample(self::MCMCObj, g::Vector{Float64}, gvar::Vector{Float64})
    if self.algtype == "rwm"
        log_posterior = log_likelihood(self,g,gvar) + log_prior(self)
    end

    if self.log_posterior[1] isa Nothing #do an accept step.
        self.log_posterior[1]=log_posterior - log(0.5) #this makes p_accept = 0.5
    end    
    #else
    p_accept = exp(log_posterior - self.log_posterior[1])
    if p_accept>rand(Uniform(0,1))
        self.posterior[1+self.iter[1],:] = self.param
        self.log_posterior[1] = log_posterior
        self.accept[1]=self.accept[1]+1        
    else 
        self.posterior[1+self.iter[1],:] = self.posterior[self.iter[1],:]            
    end
    self.param[:]=proposal(self)[:]#performed by sampling about posterior[1+self.iter[1],:]
    self.iter[1]=self.iter[1]+1
  
end


function accept_ratio(self::MCMCObj)
    return convert(Float64,self.accept[1])/self.iter[1]
end


#other functions
function log_likelihood(self::MCMCObj,g::Vector{Float64},gvar::Vector{Float64})
    #we either use the true covariance (if we have an evaluation of true model)
    #or we use a supplied covariance (if we say have a supplied GP mean/covariance)
    log_rho=[0.0]
    if gvar == nothing
        diff= g-self.truth_sample
        log_rho[1]= -0.5 * diff' * self.truth_covinv * diff
    else
        gcov_inv=Diagonal(1.0 ./ gvar)
        log_gpfidelity = -0.5*sum(log.(gvar))
        diff = g-self.truth_sample
        log_rho[1]= -0.5 * diff' * gcov_inv * diff + log_gpfidelity
    end
    return log_rho[1]
end

function log_prior(self::MCMCObj)
    log_rho=[0.0]
    #assume independent priors for each parameter
    for (param,prior) in zip(self.param,self.prior)
        
        if prior["distribution"] == "uniform"
            dist=Uniform(prior["min"],prior["max"])
        elseif prior["distribution"] == "normal"
            dist=Normal(prior["mean"],prior["sd"])
        else
            println("distribution not implemented, see src/mcmc.jl for distributions")
            sys.exit()
        end
        log_rho[1]+=logpdf(dist,param)#get distrubtion at current parameter values
    end

    return log_rho[1]
    
end

function proposal(self::MCMCObj)
    var=zeros(length(self.param))
    unidx=zeros(length(self.prior))
    for (idx,prior) in enumerate(self.prior)
        if prior["distribution"] == "uniform"
            var[idx] = (1.0/12.0)*(prior["max"]-prior["min"])^2
            unidx[idx]=1
        elseif prior["distribution"] == "normal"
            var[idx] = prior["sd"]^2
        end
    end
    if self.algtype == "rwm"
        prop_dist= MvNormal(self.posterior[1+self.iter[1],:],(self.step[1]^2)*Diagonal(var))
    end
    sample=rand(prop_dist)
    for (idx,prior) in enumerate(self.prior)
        if unidx[idx]>0 #technically this step is only guaranteeing if one of the distributions is unif, not multiples
            while sample[idx]<prior["min"] || sample[idx]>prior["max"]
                sample[:]=rand(prop_dist)
            end
        end
        
    end
                
    return sample

end



end
