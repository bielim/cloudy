#TODO: Need "distribution type object" that specifies the particle size
# distribution. This includes the type(s) of the distribution (e.g., "normal"), 
# the parameters of the distribution(s)can e.g. be a list of dicts
include("../src/PerfectModel/eki.jl")
include("../src/PerfectModel/truth.jl")

#imports
using .EKI
using JLD2

#Script:
# run 1:  creates and save the initial ensemble in pardir
#
# run 2:  extracts truth observation from state, saves in datdir, 
#         creates and saves EKI object in datdir,
#         extracts ens observation from ens data from pardir,
#         performs EKI step (with ens data) to update parameters in pardir,
#
# run 3+: loads EKI object, 
#         extracts ens observation from ens data form pardir,
#         performs EKI step (with ens data) to update parameters in pardir
#




####
#### Inputs and setup
####

#global inputs

#input parameters from command line
ens_size = parse(Int64,ARGS[1]) 
eki_iter = parse(Int64,ARGS[2])
nlats = parse(Int64, ARGS[3]) 
res = ARGS[4]
pardir = ARGS[5]  

########################

#file system
homedir=split(pwd(),"/test")[1]*"/"
outdir=homedir*"output/"
if ~isdir(outdir)
    mkdir(outdir)
end
truth_dir_base="eki_"*res*"truth" #name of truth
datdir=outdir*"eki_"*res*"truth/"

#data variables 
data_fieldnames=["ps","temp","sphum","convection_rain","condensation_rain"]
precip_units = 86400.0 # s/day
batch=30 #days/batch
truth_init=1500 #days after which we collect truth
ext_percentile=0.9

#ensemble variables
#ens_size=100
param_names=["rhbm_U","logtau"]
data_names=["rhum","precip","extreme"]
#pardir=homedir*res*"design/"
#THE BELOW MUST MATCH TMPDIR in run_gcm(insert from .../fms_tmp/${exp_name}_${res}${design_name})
design_name="design_ens_jl"

#First run through code, create initial ensemble then leave

if ~isdir(pardir)
    #initial ensemble:
    mkdir(pardir)

    #on first run of code - create initial ensemble
    construct_initial_ensemble(ens_size,param_names,pardir)
    #runs on transpose
    exit()
end


#Subsequent run through code, run the truth
#Save mean, covariance, the latitudes, and some sample batched data.
#---writes to file
if ~isdir(datdir)
    mkdir(datdir)

    #create truth statistics of the truth
    truth_mean,truth_cov,truth_full,lats,threshold = construct_truth_statistics(truth_dir_base,
                                                                                          data_fieldnames,
                                                                                          precip_units,
                                                                                          ext_percentile,
                                                                                          nlats,
                                                                                          truth_init,
                                                                                          batch)


    truth_mean=dropdims(truth_mean,dims=2)
    
    #create truth object
    truthobj=TruthObj(truth_mean,truth_cov,truth_full,data_names,lats)

    #construct the EK{I,S} object
    initial_parameters = read_ens(pardir*"/","ens")#paramsize x enssize matrix
    idlog=findall(x->x=="logtau",param_names)
    initial_parameters[idlog,:]=log.(initial_parameters[idlog,:])
    initial_parameters=copy(transpose(initial_parameters))#enssize x paramsize
    ###############
    #runs on transpose (enssize x param/datasize)
    
    #truth_mean=dropdims(truth_mean,dims=2)
    ekiobj=EKIObj(initial_parameters,param_names,truth_mean,truth_cov) #for now this works based on enssize x param/datasize
   
    #save the EKI object
    @save datdir*"eki.jld2" ekiobj
    @save datdir*"truth.jld2" truthobj
    @save datdir*"threshold.jld2" threshold
else
    @load datdir*"eki.jld2" ekiobj 
    @load datdir*"threshold.jld2" threshold
end

#create ensemble observed data from the GCM runs on the parameters
g_ens,lats = construct_ensemble_statistics(design_name,
                                           data_fieldnames,
                                           precip_units,
                                           ext_percentile,
                                           threshold,
                                           nlats,
                                           res,
                                           batch)

#Then update the parameters with EK{I,S} based on this ensemble data.

#ekiobj still works based on ens x param but the other operations work on param x ens so we need to transpose
g_ens=copy(transpose(g_ens))# ens x param size 
update_ensemble(ekiobj,g_ens)

uwrit=copy(transpose(ekiobj.u[end]))
gwrit=copy(transpose(ekiobj.g[end]))
#write to file #param/data size x ens size
#use copy, or it tries to send a transpose object into write function

write_ens(pardir*"/",param_names,uwrit,"ens")
write_ens(pardir*"/",param_names,gwrit,"dat")

open(datdir*"eki_error","a") do file
    writedlm(file,ekiobj.error[end])
end


#save the EKI object
@save datdir*"eki.jld2" ekiobj
#@save datdir*"threshold.jld2" threshold
