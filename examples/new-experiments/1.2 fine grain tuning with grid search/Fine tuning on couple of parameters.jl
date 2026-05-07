push!(LOAD_PATH, "../../../src/")

using SpikingNeuralEvolution

using DataFrames
using Statistics
using Distributed
using Random

# Add workers (use all CPU cores except one)
if nprocs() == 1
    addprocs(Sys.CPU_THREADS)
end

# Distribute code to all workers
@everywhere begin
    push!(LOAD_PATH, "../../../src/")
    using SpikingNeuralEvolution
    using Statistics

    n = UInt16(5)

    function XOR(inputs::Vector{Bool})
	    return [reduce(xor, inputs)]
    end
end

# Extract previous 1000 results to rerun with 1000 iterations instead of 100, from "1.0 digest.csv"
# convert "i | float float float float float float float float float float uint16" to (float float float float float float float float)
file = readlines(open("1.1 digest.txt", "r"))
combinations = [[parse(Float64, element) for element in (Vector{String}(split(split(strip(line), "│")[2]))[1:8])] for line in file[1:15]]
tuned_combinations = []
for combination in combinations
    for new in [-1.0, -0.005, 0.0, 0.005, 1.0]
        for remove in [-1.0, -0.005, 0.0, 0.005, 1.0]
            push!(tuned_combinations, [combination[1] + new, combination[2] + remove, combination[3:8]...])
            push!(tuned_combinations, [combination[1:2]...,combination[3] + new, combination[4] + remove, combination[5:8]...])
            push!(tuned_combinations, [combination[1:4]...,combination[5] + new, combination[6] + remove, combination[7:8]...])
            push!(tuned_combinations, [combination[1:6]...,combination[7] + new, combination[8] + remove])
        end
    end
end

# Function to run a single evolution experiment
@everywhere function run_evolution_experiment(params::Union{Vector{Float64}, Vector{Float64}})
    new_layer, remove_layer, new_neuron, remove_neuron, new_rule, remove_rule, new_input, remove_input = params

    evolution_parameters = EvolutionParameters(
        UInt32(15),    # How many simulations
        UInt32(1000),    # How many iterations per simulation, raised to 1000
        UInt32(80),    # Min number of random population
        UInt32(120),   # Max number of random population
        UInt16(1),     # Min number of random hidden layers
        UInt16(3),     # Max number of random hidden layers
        SpikingNeuralEvolution.MaxIterations, # Stopping criteria
        1.0,    # % of examples
        1.0     # Threshold
    )

    mutation_probabilities = MutationProbabilities(
        new_layer,      # New layer
        remove_layer,   # Remove layer
        new_neuron,     # New neuron
        remove_neuron,  # Remove neuron
        new_rule,       # Add rule
        remove_rule,    # Remove rules
        new_input,      # Add input line
        remove_input    # Remove input line
    )

    histories = Evolve(XOR, n, evolution_parameters, mutation_probabilities)

    final_values = []
    how_many_successful = 0

    for key in keys(histories)
        push!(final_values, histories[key][2][end])
        if histories[key][2][end] == 1
            how_many_successful += 1
        end
    end

    return (new_layer, remove_layer, new_neuron, remove_neuron,
        new_rule, remove_rule, new_input, remove_input,
        mean(final_values), maximum(final_values), how_many_successful)
end

# Run experiments in parallel using pmap
println("Starting fitting with $(nworkers()) workers on previous 1000 parameter combinations...")
results_list = pmap(run_evolution_experiment, tuned_combinations)

# Convert results to DataFrame
results = DataFrame(
    NewLayer=Float64[],
    RemoveLayer=Float64[],
    NewNeuron=Float64[],
    RemoveNeuron=Float64[],
    NewRule=Float64[],
    RemoveRule=Float64[],
    NewInput=Float64[],
    RemoveInput=Float64[],
    MeanFitness=Float64[],
    MaxFitness=Float64[],
    SuccessfulSimulations=UInt16[]
)

for result in results_list
    push!(results, result)
end

sort!(results, :MeanFitness, rev=true)

println("\n=== 1000 Iterations Results ===")
println("Total experiments: $(nrow(results))")
println("\nTop 1000 results by MeanFitness:")
println(results[1:min(1000, nrow(results)), :])

exit(0)
