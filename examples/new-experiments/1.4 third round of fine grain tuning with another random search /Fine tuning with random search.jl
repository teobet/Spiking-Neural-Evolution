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

# convert "i | float float float float float float float float float float uint16" to (float float float float float float float float)
file = readlines(open("1.3 digest.txt", "r"))
combinations = [[parse(Float64, element) for element in (Vector{String}(split(split(strip(line), "│")[2]))[1:8])] for line in file[1:45]]
tuned_combinations = []
for combination in combinations
    for _ in 1:100
        new_layer = max(0.0, combination[1] + ((rand() * 0.02) - 0.01))  # Randomly adjust by up to ±0.01
        remove_layer = max(0.0, combination[2] + ((rand() * 0.02) - 0.01))
        new_neuron = max(0.0, combination[3] + ((rand() * 0.02) - 0.01))
        remove_neuron = max(0.0, combination[4] + ((rand() * 0.02) - 0.01))
        new_rule = max(0.0, combination[5] + ((rand() * 0.02) - 0.01))
        remove_rule = max(0.0, combination[6] + ((rand() * 0.02) - 0.01))
        new_input = max(0.0, combination[7] + ((rand() * 0.02) - 0.01))
        remove_input = max(0.0, combination[8] + ((rand() * 0.02) - 0.01))
        push!(tuned_combinations, [new_layer, remove_layer, new_neuron, remove_neuron, new_rule, remove_rule, new_input, remove_input])
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

# Add previous file results to the list
splitted_strings = [Vector{String}(split(split(strip(line), "│")[2])) for line in file]
for i in 1:length(splitted_strings)
    push!(results_list, Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, UInt16}(
        [
        parse(Float64, splitted_strings[i][1]),  # new_layer
        parse(Float64, splitted_strings[i][2]),  # remove_layer
        parse(Float64, splitted_strings[i][3]),  # new_neuron
        parse(Float64, splitted_strings[i][4]),  # remove_neuron
        parse(Float64, splitted_strings[i][5]),  # new_rule
        parse(Float64, splitted_strings[i][6]),  # remove_rule
        parse(Float64, splitted_strings[i][7]),  # new_input
        parse(Float64, splitted_strings[i][8]),  # remove_input
        parse(Float64, splitted_strings[i][9]),  # mean_fitness
        parse(Float64, splitted_strings[i][10]), # max_fitness
        parse(UInt16, splitted_strings[i][11]) # successful_simulations
        ]
    ))
end
    
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
