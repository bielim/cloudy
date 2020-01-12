module LogTrans


export transform
export back_transform


function transform(𝐱)
    eps = 1e-6
    min_val = minimum(𝐱)
    shift = min_val - eps 
    return transform(𝐱, shift), shift
end


function transform(𝐱, shift)
    if any(𝐱 .+ 1.0 .- shift <= 1.0)
        println(𝐱)
        throw(DomainError)
    end
    log.(𝐱 .+ 1.0 .- shift)
end


function back_transform(𝐱; shift=0)
    exp.(𝐱) .+ shift .- 1.0
end

end #module LogTrans




