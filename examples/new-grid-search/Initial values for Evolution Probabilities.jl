push!(LOAD_PATH, "../../../src/")

using SpikingNeuralEvolution


using Plots
using DataFrames
using Statistics

# For speed
using ThreadSafeDicts

n = UInt16(5)

function XOR(inputs::Vector{Bool})
    return reduce(xor, inputs)
end


# Grid search

results = DataFrame(NewLayer=Float64[], RemoveLayer=Float64[], NewNeuron=Float64[], RemoveNeuron=Float64[], NewRule=Float64[], RemoveRule=Float64[], NewInput=Float64[], RemoveInput=Float64[], MeanFitness=Float64[], MaxFitness=Float64[], SuccessfulSimulations=UInt16[])
results_ts = ThreadSafeDict()

new_layer = 0.0
remove_layer = 0.0
new_neuron = 0.0
remove_neuron = 0.0
new_rule = 0.0
remove_rule = 0.0
new_input = 0.0
remove_input = 0.0

test_range = [0.00, 0.016, 0.033, 0.05, 0.066, 0.083, 0.10, 0.116, 0.133, 0.15, 0.166, 0.183, 0.20]

global_counter = 1

Threads.@threads for new_input in test_range
    Threads.@threads for remove_input in test_range
        Threads.@threads for new_rule in test_range
            Threads.@threads for remove_rule in test_range
                Threads.@threads for new_neuron in test_range
                    Threads.@threads for remove_neuron in test_range
                        Threads.@threads for new_layer in test_range
                            Threads.@threads for remove_layer in test_range
                                evolution_parameters = EvolutionParameters(
                                    UInt32(15),    # How many simulations
                                    UInt32(50),  # How many iterations per simulation
                                    UInt32(80),    # Min number sof random population
                                    UInt32(120),    # Max number of random population
                                    UInt16(1),   # Min number of random hidden layers
                                    UInt16(3),    # Max number of random hidden layers
                                    SpikingNeuralEvolution.MaxIterations, # Stopping criteria
                                    1.0,    # % of examples
                                    1.0     # Threshold
                                )
                                mutation_probabilities = MutationProbabilities(
                                    new_layer,  # New layer
                                    remove_layer,  # Remove layer
                                    new_neuron,  # New neuron
                                    remove_neuron,  # Remove neuron
                                    new_rule,  # Add rule
                                    remove_rule,  # Remove rules
                                    new_input,   # Add input line
                                    remove_input   # Remove input line
                                )

                                histories = Evolve(XOR, n, evolution_parameters, mutation_probabilities)

                                # I create an array that will contain all the final final_values
                                # of fitness for each simulation

                                final_values = []

                                # This var indicates the iterations required to achieve (if this happens)
                                # the threshold value of fitness (in case not specified, 1)

                                how_many_successful = 0

                                for key in keys(histories)
                                    push!(final_values, histories[key][2][end])

                                    if histories[key][2][end] == 1
                                        how_many_successful += 1
                                    end

                                end

                                #push!(results, (lay_min, lay_max, population, mean(final_values), maximum(final_values), mean_iterations_of_max_fitness / how_many_successful, iteration_of_max_fitness))
                                results_ts[string(new_layer)*string(remove_layer)*string(new_neuron)*string(remove_neuron)*string(new_rule)*string(remove_rule)*string(new_input)*string(remove_input)] = (new_layer,remove_layer,new_neuron,remove_neuron,new_rule,remove_rule,new_input,remove_input, mean(final_values), maximum(final_values), how_many_successful)

                                println("[$global_counter/815730721] $new_layer - $remove_layer - $new_neuron - $remove_neuron - $new_rule - $remove_rule - $new_input - $remove_input: done, " * string(how_many_successful) * "/" * string(evolution_parameters.simulations))
                                global_counter += 1
                            end
                        end
                    end
                end
            end
        end
    end
end

for key in keys(results_ts)
    push!(results, results_ts[key])
end

sort!(results, :MeanFitness, rev=true)

println(results)

"""
Mutations on layers:
 Row │ NewLayer  RemoveLayer  NewNeuron  RemoveNeuron  NewRule  RemoveRule  NewInput  RemoveInput  MeanFitness  MaxFitness  SuccessfulSimulations
     │ Float64   Float64      Float64    Float64       Float64  Float64     Float64   Float64      Float64      Float64     UInt16
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     0.2          0.05        0.0           0.0      0.0         0.0       0.0          0.0     0.827083     0.875                        0
   2 │     0.1          0.1         0.0           0.0      0.0         0.0       0.0          0.0     0.8125       0.875                        0
   3 │     0.15         0.2         0.0           0.0      0.0         0.0       0.0          0.0     0.810417     0.875                        0
   4 │     0.1          0.2         0.0           0.0      0.0         0.0       0.0          0.0     0.808333     0.875                        0
   5 │     0.1          0.0         0.0           0.0      0.0         0.0       0.0          0.0     0.808333     0.875                        0
   6 │     0.2          0.15        0.0           0.0      0.0         0.0       0.0          0.0     0.808333     0.875                        0
   7 │     0.2          0.2         0.0           0.0      0.0         0.0       0.0          0.0     0.804167     0.875                        0
   8 │     0.2          0.0         0.0           0.0      0.0         0.0       0.0          0.0     0.8          0.875                        0
   9 │     0.15         0.05        0.0           0.0      0.0         0.0       0.0          0.0     0.8          0.875                        0
  10 │     0.15         0.0         0.0           0.0      0.0         0.0       0.0          0.0     0.8          0.84375                      0
  11 │     0.2          0.1         0.0           0.0      0.0         0.0       0.0          0.0     0.797917     0.875                        0
  12 │     0.05         0.2         0.0           0.0      0.0         0.0       0.0          0.0     0.795833     0.875                        0
  13 │     0.15         0.15        0.0           0.0      0.0         0.0       0.0          0.0     0.79375      0.84375                      0
  14 │     0.0          0.1         0.0           0.0      0.0         0.0       0.0          0.0     0.79375      0.84375                      0
  15 │     0.05         0.0         0.0           0.0      0.0         0.0       0.0          0.0     0.79375      0.875                        0
  16 │     0.1          0.05        0.0           0.0      0.0         0.0       0.0          0.0     0.791667     0.875                        0
  17 │     0.05         0.15        0.0           0.0      0.0         0.0       0.0          0.0     0.791667     0.875                        0
  18 │     0.1          0.15        0.0           0.0      0.0         0.0       0.0          0.0     0.789583     0.84375                      0
  19 │     0.15         0.1         0.0           0.0      0.0         0.0       0.0          0.0     0.789583     0.875                        0
  20 │     0.0          0.05        0.0           0.0      0.0         0.0       0.0          0.0     0.785417     0.875                        0
  21 │     0.05         0.1         0.0           0.0      0.0         0.0       0.0          0.0     0.785417     0.90625                      0
  22 │     0.05         0.05        0.0           0.0      0.0         0.0       0.0          0.0     0.777083     0.84375                      0
  23 │     0.0          0.15        0.0           0.0      0.0         0.0       0.0          0.0     0.770833     0.875                        0
  24 │     0.0          0.0         0.0           0.0      0.0         0.0       0.0          0.0     0.770833     0.8125                       0
"""

"""

"""