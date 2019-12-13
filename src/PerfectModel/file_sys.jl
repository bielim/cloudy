#includes
using DelimitedFiles
"""

A set of function to clean up searching the filesystem

"""

#Truth:
"""
    find_truth(base_name::AbstractString)

Finds (and date orders) directories containing the truth GCM run "base_name" as defined in the run script for the truth
Returns an array of strings of the directory paths
"""
function find_truth(base_name::String)
    truth_path = split(pwd(),"fms-idealized")[1]*"fms_tmp/"*base_name
    truth_dir=filter!(e->e∉["mppnccombine.ifc", "exe.fms"],readdir(truth_path)) 
    if length(truth_dir)>1
        error("must be only one truth type in "*truth_path)
    end
    truth_path*="/"*truth_dir[1]*"/output/combine/"
    truth_dirs=readdir(truth_path)
    sort!(truth_dirs,by = x->parse(Int64,split(split(x,"day")[2],"h")[1]))#puts it in numerical order by date
    truth_list=truth_path.*truth_dirs.*"/"

    return truth_list
end


#Ensemble
"""
    find_ens(design_name::String,batch::Int64date=0)
    
Finds (and ensemble orders) directories containing the ensemble GCM runs for "design_name" at the most recent date (one may choose other integer indices for earlier dates(index as "end-date"))
Returns and array of strings of the directory paths, also returns number of batches in a file
"""
function find_ens(name::String,res::String,batch::Int64,date=0)
    #base_name=split(split(pwd(),"exp/")[2],"/")[1]*"_"*res*name 
    base_name=split(pwd(),"/")[end-1]*"_"*res*name
    ens_path = split(pwd(),"fms-idealized")[1]*"fms_tmp/"*base_name
    rens=readdir(ens_path)
    sort!(rens,by = x->parse(Int64,split(x,"_")[3]))#sort by ensemble number
    ens_pathlist=ens_path*"/".*rens.*"/"
    ens_pathlist.*="output/combine/"#these are the dirs with all ens runs in 
    rens=readdir(ens_pathlist[1])#as all the same for each ens member
    sort!(rens,by = x->parse(Int64,split(split(x,"day")[2],"h")[1]))#puts it in numerical order by date
    ens_pathlist.*=rens[end-date].*"/"#picks only the most recent month run
    #at this point we note that our spin up and our ensemble runs are in different files.
    #finally obtain the number of days in a batch
    batch_per_file= (parse(Int64,split(split(rens[2],"day")[2],"h")[1]) - parse(Int64,split(split(rens[1],"day")[2],"h")[1]))/batch
    batch_per_file= convert(Int64,batch_per_file)     
    
    return ens_pathlist,batch_per_file
end

#writing ensemble 
"""
        write_ens(dir::String,param_names::Array{String,1},params::Array{Float64,},prefix::String)

writes a set of files in dir, named ens_* for each ensemble member containing param names and associated values
"""
function write_ens(dir::String,param_names::Array{String,1},params::Array{Float64,},prefix::String)
    #NOTE THIS FUNCTION TAKES param/data x enssize size arrays. prints them to ens size files.
    pn=param_names
    p=params
    if prefix == "ens"
        #get index of logtau, and exponentiate
        idlog=findall(x->x=="logtau",pn)
        
        if length(idlog)>0
            idlog=idlog[1]#as idlog is still an array
            #println(idlog)
            #pn[idlog]="tau"
            p[idlog,:]=exp.(p[idlog,:])
            
        end
    end

    J=size(p)[2] #ensemble size
    for j=1:J
        fname =dir*"/"*prefix*"_"*string(j)
        writeout(fname,[pn],[p[:,j]])
    end

    
end

"""
    writeout(fname::String, names::Array{String,1}, vals::Array{Float64,1})

Opens file fname, if it exists, enters values. If it doesnt exist it first adds value names.
"""
function writeout(fname::String,names::Array{String,1},vals::Array{Float64,1})
    #if file doesnt exist, create it and write param names 
    if ~isfile(fname)
        open(fname,"w") do file
            writedlm(file,names)
        end   
    end
    #when file exists write values
    open(fname,"a") do file
        writedlm(file,vals)
    end
end

"""
    writeout(fname::String, names::Array{String,1}, vals::Array{Float64,1})

Opens file fname, if it exists, enters values(as array of arrays, it writes rows). If it doesnt exist it first adds value names.
"""
function writeout(fname::String,names::Array{Array{String,1},1},vals::Array{Array{Float64,1},1})
    #if file doesnt exist, create it and write param names 
    if ~isfile(fname)
        open(fname,"w") do file
            writedlm(file,names)
        end   
    end
    #when file exists write values
    open(fname,"a") do file
        writedlm(file,vals)
    end
end

function read_ens(dirname::String,prefix::String; from_end=0)
    parameters=Vector{Float64}[]
    fnames=dirname.*readdir(dirname)
    fnames=filter(x->occursin(prefix*"_",x),fnames)
    sort!(fnames,by = x->parse(Int64,split(x,"_")[end]))#puts it in numerical order by ensemble number
          
    for f in fnames
        tmp=readdlm(f,Float64,skipstart=1)[end-from_end,:]#skips 1st line (string entries)
        push!(parameters,tmp)
    end
    return hcat(parameters...) # turns 100-array of 2-arrays into the (2 x ens_size) matrix
    #return parameters
    
end

function read_file(fname::String)  
    return read_file(fname,Float64,0)#skips 1st line (string entries)
end
    

function read_file(fname::String,skip_lines::Int64)  
    content=readdlm(fname,Float64,skipstart=skip_lines)#skips 1st line (string entries)
    return dropdims(content,dims=1)
end
   
