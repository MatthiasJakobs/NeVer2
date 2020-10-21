using DelimitedFiles
using MIPVerify
using Gurobi
using JuMP
using Images

function get_max_index(
    x::Array{<:Real, 1})::Integer
    return findmax(x)[2]
end

"""
We needed to rescale our data to [0, 1] and viceversa, therefore we highjacked the Rescale layer: mean
should be the max value of the data and std should be the min value of the data.
"""
struct Rescale <: Layer
    mean::Real
    std::Real
end

function Base.show(io::IO, p::Rescale)
    print(io, "Rescale(mean: $(p.mean), std: $(p.std))")
end


function apply(p::Rescale, x::Array{<:MIPVerify.JuMPReal})
    output = x.*(p.mean - p.std) .+ p.std
    return output
end

(p::Rescale)(x::Array{<:MIPVerify.JuMPReal}) = apply(p, x)

net_name = ARGS[1]
img_name = ARGS[2]
epsilon =  parse(Float64, ARGS[3])
targeted = ARGS[4]

layers = []
param_dict = MIPVerify.matread(net_name * ".mat")
cfg_dict = MIPVerify.matread(net_name * "_cfg.mat")

aux_image = readdlm(img_name, '\t', Float64, '\n')
real_image = aux_image[1:784]
target = Int(aux_image[785]) + 1

min = minimum(real_image)
max = maximum(real_image)

real_image = (real_image .- min) ./ (max - min)


push!(layers, Rescale(max, min))

open(net_name * "_cfg.txt") do file

    for ln in eachline(file)

        if occursin("fc", ln)

            in_ch = cfg_dict[ln * "/in"]
            out_ch = cfg_dict[ln * "/out"]
            fc = get_matrix_params(param_dict, ln, (in_ch, out_ch))
            push!(layers, fc)

        elseif occursin("relu", ln)

            push!(layers, ReLU())

        end

    end
end

nn = Sequential(layers, net_name)

class_output = (real_image |> nn |> get_max_index)

if (targeted == "True") & (target == class_output)
    print("ERROR: if the adversarial test is targeted, target should be different from the real output label.")
    exit(1)
elseif (targeted == "False") & (target != class_output)
    print("ERROR: if the adversarial test is untargeted, target should be equals to the real output label.")
    exit(1)
end

if (targeted == "True")
    d = MIPVerify.find_adversarial_example(nn, real_image, target, GurobiSolver(OutputFlag=0),
                                           cache_model=false, invert_target_selection=false, norm_order=Inf,
                                           pp = MIPVerify.LInfNormBoundedPerturbationFamily(epsilon), tolerance=0.001);
else
    d = MIPVerify.find_adversarial_example(nn, real_image, target, GurobiSolver(OutputFlag=0),
                                           cache_model=false, invert_target_selection=true, norm_order=Inf,
                                           pp = MIPVerify.LInfNormBoundedPerturbationFamily(epsilon), tolerance=0.001);
end

print(d[:SolveStatus])
print(d[:PerturbedInput])

if (d[:SolveStatus] == "Optimal")
    exit(0)
else
    exit(1)
end