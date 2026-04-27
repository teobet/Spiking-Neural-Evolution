push!(LOAD_PATH, "src/")

using SpikingNeuralEvolution
using Plots

function XOR(inputs::Vector{Bool}) 
    f = reduce(xor, inputs)
    return [f, !f]
end

histories = Evolve(XOR, 0x005, true)

exit(0)