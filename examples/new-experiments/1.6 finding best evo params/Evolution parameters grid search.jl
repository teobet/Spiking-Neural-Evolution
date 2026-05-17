push!(LOAD_PATH, "../../../src/")

using SpikingNeuralEvolution

using DataFrames
using Statistics
using Distributed
using Random

# Add workers (use all CPU cores)
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



evolution_parameters_combinations = Vector{Union{UInt32, UInt16}}()

for population in [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    for min_hidden in [1, 2, 3, 4, 5]..
        for max_hidden in [min_hidden, min_hidden + 1, min_hidden + 2, min_hidden + 3, min_hidden + 4]
            push!(evolution_parameters_combinations, [
                UInt32(15),    # How many simulations
                UInt32(1000),    # How many iterations per simulation
                UInt32(population),    # Min number of random population
                UInt32(population),   # Max number of random population
                UInt16(min_hidden),     # Min number of random hidden layers
                UInt16(max_hidden),      # Max number of random hidden layers
                SpikingNeuralEvolution.MaxIterations, # Stopping criteria
                1.0,    # % of examples
                1.0     # Threshold
            ])
        end
    end
end

# Function to run a single evolution experiment
@everywhere function run_evolution_experiment(params::Vector{Union{UInt32, UInt16}})

    evolution_parameters = EvolutionParameters(
        params...
    )

    # using the empirically best mutation probabilities found in the 1.4 experiment
    mutation_probabilities = MutationProbabilities(
        0.193638,
        0.031221,
        0.139078,
        0.00377863,
        0.0812008,
        0.124834,
        0.10922,
        0.0766559
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

    return (
        UInt32(params[3]),  # Population
        UInt16(params[5]),  # LayersMin
        UInt16(params[6]),  # LayersMax
        mean(final_values),
        maximum(final_values),
        how_many_successful
    )
end

# Run experiments in parallel using pmap
println("Starting fitting with $(nworkers())")
results_list = pmap(run_evolution_experiment, evolution_parameters_combinations)

# Convert results to DataFrame
results = DataFrame(
    Population=UInt32[],
    LayersMin=UInt16[],
    LayersMax=UInt16[],
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
println("\nTop results by MeanFitness:")
println(results[1:min(1000, nrow(results)), :])

exit(0)
