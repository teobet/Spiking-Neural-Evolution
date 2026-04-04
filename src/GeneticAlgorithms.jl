"""
GeneticAlgorithms

A module providing functions to perform procedured based on
Genetic Algorithms (GA) on circuits composed of SN P-System neurons

"""
module GeneticAlgorithms

    # Libraries required
    include("SNPCircuit.jl")
    using .SNPCircuit

    include("Utils.jl")
    using .Utils

    using Random

    # Structs and functions that are exported to the user
    export Crossover, 
        Mutation,
        Fitness,
        CleanCircuit,
        CrossoverNew,
        MutationNew,
        FitnessNew,
        CleanCircuitNew,
        ProportionateSelection



    """
    Crossover(circuit1, circuit2)

    Performs a crossover between two circuits. In particular, a slicing
    index n is randomly sampled, and the first n layers of the first circuit
    are considered. These layers are attached to the last 1 - m layers of the
    second circuit, creating the first child.
    In the same way the last 1 - n layers of the first circuit are attached to
    the first m layer of the second one, creating the second child 

    # Arguments
    - `circuit1::CircuitOfNeurons`: The first parent.
    - `circuit2::CircuitOfNeurons`: The second parent.

    # Returns
    - `CircuitOfNeurons`: The new individual obtained by performing the crossover`.
    """

    #TODO extend considering multiple outputs
    function CrossoverNew(circuit1::CircuitOfNeurons, circuit2::CircuitOfNeurons, examples::Vector{Vector{Bool}}, labels::Vector{Bool})
        # If the parent circuits are not eligible for the crossover (less than 2 layers),
        # the following 2 lines prevent errors from being raised, then returning both of the parents as is
        child1 = circuit1
        child2 = circuit2

        if (NumberOfLayers(circuit1) > 2 && NumberOfLayers(circuit2) > 2)
            # Between p \in [0.4,0.6] is taken from the first parent and 1 - p
            # is taken from the second one
            slice = 0.4 + ((0.6 - 0.4) * rand())
            n = UInt16(floor(slice * NumberOfLayers(circuit1)))   
            m = NumberOfLayers(circuit2) - UInt16(floor((1 - slice) * NumberOfLayers(circuit2)))

            # The offsprings will contain the layers [1:n] from the first
            # circuit, and the layers [m:end] from the second circuit and
            child1 = CircuitOfNeurons(append!(
                deepcopy(circuit1.layers[1:n]), 
                deepcopy(circuit2.layers[m:end])))

            child2 = CircuitOfNeurons(append!(
                deepcopy(circuit2.layers[1:m]), 
                deepcopy(circuit1.layers[n:end])))
        end

        child1 = CleanCircuitNew(child1, UInt16(length(examples[1])), UInt16(length(labels[1])))
        child2 = CleanCircuitNew(child2, UInt16(length(examples[1])), UInt16(length(labels[1])))

        # We chose the child circuit with the highest fitness
        if Fitness(child1, examples, labels) >= Fitness(child2, examples, labels)
            return child1
        else
            return child2
        end
    end

    function Crossover(circuit1::CircuitOfNeurons, circuit2::CircuitOfNeurons, examples::Vector{Vector{Bool}}, labels::Vector{Bool})
    # Between 30% and 70% taken from the first parent
    slice = rand(3:7) / 10

    n = Int64(floor(slice * NumberOfLayers(circuit1)))

    if n == 0
        n = 1
    end

    # And 1 - n taken from the second parent
    m = NumberOfLayers(circuit2) - Int64(floor((1 - slice) * NumberOfLayers(circuit2)))

    if m == NumberOfLayers(circuit2)
        m = NumberOfLayers(circuit2) - 1
    end

    # The merged circuit will contain the layers [1:n] from the first
    # circuit, and the layers [m:end] from the second circuit
    individual = CircuitOfNeurons(append!(
        deepcopy(circuit1.layers[1:n]),
        deepcopy(circuit2.layers[m:end])))

    # Notice that this procedure can break some connection between neurons, therefore
    # a 'cleaning' procedure is executed in the middle layer in order to remove empty input lines
    individual = CleanCircuit(individual, UInt16(length(examples[1])), UInt16(length(labels[1])))

    return individual
end

    """
    Mutation(circuit, inputs, probabilities)

    Performs a mutation on a circuit. The implemented possible mutations 
    are the following:

    - Adds a new random layer
    - Removes an existing random layer
    - Adds a new random neuron in a random layer
    - Removes an existing random neuron from a random layer
    - Adds a new random rule in a random neuron
    - Removes an existing random rule from a random neuron
    - Adds a new random input lines in a random neuron
    - Removes an existing random rule from a random neuron

    New mutations are possible and easily implementable.

    # Arguments
    - `circuit::CircuitOfNeurons`: The circuit to be mutated.
    - `inputs::UInt16`: The number of inputs for the circuit.
    - `probabilities::MutationProbabilities`: A set of probabilities for the possible mutations.

    # Returns
    - `CircuitOfNeurons`: A new individual obtained by performing the crossover`.
    """
    function MutationNew(circuit::CircuitOfNeurons, inputs::UInt16, probabilities=MutationProbabilities(0.01, 0.08, 0.03, 0.2, 0.005, 0.08, 0.01, 0.01))
        # Adds a new layer
        if rand() < probabilities.new_layer
            
            # For circuit of exactly 2 layers, the second one (output layer) will always be selected
            index = rand(2:NumberOfLayers(circuit))
            
            # We add a layer with a number of neuron between n and n + 2, where n is 
            # the number of neuron in the layer before the one selected, according
            # to the generation of a random circuit (`SNPCircuit.GenerateRandomCircuit`) 
            previous_neurons = NumberOfNeurons(circuit.layers[index - 1])
            n_neurons = rand(previous_neurons:previous_neurons + 2)

            neurons = Vector{Neuron}()
    
            for _ in 1:n_neurons
                push!(neurons, GenerateRandomNeuron(NumberOfNeurons(circuit.layers[index - 1])))
            end
                
            insert!(circuit.layers, index, LayerOfNeurons(neurons))
        end
        
        # Remove a random hidden layer (if the number of layers is greater than two)
        if rand() < probabilities.remove_layer
            if NumberOfLayers(circuit) > 2
                deleteat!(circuit.layers, rand(2:(NumberOfLayers(circuit) - 1)))
            end
        end
        
        # All the mutations on the neurons are tested for each layer
        for i in 1:NumberOfLayers(circuit)
            layer = circuit.layers[i]
                
            # Addition and deletion of neurons are not performed on the last layer
            if i != NumberOfLayers(circuit)
                # Insert a new random neuron  
                if rand() < probabilities.new_neuron
                    if i == 1
                        previous_inputs = inputs
                    else
                        previous_inputs = NumberOfNeurons(circuit.layers[i-1])
                    end

                    # This block handles the case in which all the neurons of the previous
                    # layer are deleted
                    if previous_inputs != 0
                        neuron = GenerateRandomNeuron(previous_inputs)
                        insert!(layer.neurons, rand(1:NumberOfNeurons(layer)), neuron)
                    end
                end
                
                # Remove a random neuron
                if rand() < probabilities.remove_neuron
                        index_remove = rand(1:NumberOfNeurons(layer))
                        deleteat!(layer.neurons, index_remove)
                end
            end

            # The mutations for the rules and the input lines are tested for each neuron
            for j in 1:NumberOfNeurons(layer)
            
                # Remove a random rule
                if rand() < probabilities.remove_rule
                    if !isempty(circuit.layers[i].neurons[j].rules)
                        index_remove = rand(1:length(circuit.layers[i].neurons[j].rules))
                        deleteat!(circuit.layers[i].neurons[j].rules, index_remove)
                    end
                end  
                   
                # Add a new random rule
                if rand() < probabilities.new_rule
                    if i == 1
                        previous_inputs = inputs
                    else
                        previous_inputs = NumberOfNeurons(circuit.layers[i - 1])
                    end

                    # This block identifies the rules that can be added to the neuron
                    rules_candidates = Vector{UInt16}()
                    for k in 0:previous_inputs
                        if !(k in circuit.layers[i].neurons[j].rules)
                            push!(rules_candidates, k)
                        end
                    end

                    if !isempty(rules_candidates)
                        rule = rules_candidates[rand(1:length(rules_candidates))]
                        push!(circuit.layers[i].neurons[j].rules, rule)
                    end 
                end  

                # Add a new random input line
                if rand() < probabilities.new_input_line
                    if i == 1
                        previous_inputs = inputs
                    else
                        previous_inputs = NumberOfNeurons(circuit.layers[i - 1])
                    end

                    # This block identifies the input lines to which the neuron can connect to
                    input_lines_candidates = Vector{UInt16}()
                    for k in 1:previous_inputs
                        if !(k in circuit.layers[i].neurons[j].input_lines)
                            push!(input_lines_candidates, k)
                        end
                    end

                    if !isempty(input_lines_candidates)
                        input_line = input_lines_candidates[rand(1:length(input_lines_candidates))]
                        push!(circuit.layers[i].neurons[j].input_lines, input_line)
                    end
                end

                # Remove a random input line
                if rand() < probabilities.remove_input_line
                    if !isempty(circuit.layers[i].neurons[j].input_lines)
                        index_remove = rand(1:length(circuit.layers[i].neurons[j].input_lines))
                        deleteat!(circuit.layers[i].neurons[j].input_lines, index_remove)
                    end
                end
            end     
        end     

        return circuit
    end

    function Mutation(circuit::CircuitOfNeurons, inputs::UInt16, probabilities=MutationProbabilities(0.01, 0.08, 0.03, 0.2, 0.005, 0.08, 0.01))
        # Adds a new layer
        if rand() < probabilities.new_layer

            if length(circuit.layers) == 2
                # TODO check questo due e commenta sta roba
                index = 2
            else
                index = rand(2:length(circuit.layers))
            end

            n_neurons = length(circuit.layers[index].neurons)

            neurons = Vector{Neuron}()

            for n_neuron in 1:n_neurons
                push!(neurons, GenerateRandomNeuron(NumberOfNeurons(circuit.layers[index-1])))
            end

            insert!(circuit.layers, index, LayerOfNeurons(neurons))
        end

        # Remove a random layer (if the number of layers is greater than two)
        if rand() < probabilities.remove_layer
            if NumberOfLayers(circuit) > 2
                deleteat!(circuit.layers, rand(2:(length(circuit.layers) - 1)))
            end
        end

        # All the mutations on the neurons are tested for each layer
        for i in 1:length(circuit.layers)-1
            layer = circuit.layers[i]

            # Insert a new random neuron  
            if rand() < probabilities.new_neuron
                if i == 1
                    neuron = GenerateRandomNeuron(inputs)
                    insert!(layer.neurons, rand(1:length(layer.neurons)), neuron)
                else
                    neuron = GenerateRandomNeuron(NumberOfNeurons(circuit.layers[i-1]))
                    insert!(layer.neurons, rand(1:length(layer.neurons)), neuron)
                end
            end

            # Remove a random neuron
            if rand() < probabilities.remove_neuron
                if length(layer.neurons) > 2
                    index_remove = rand(1:length(layer.neurons))
                    deleteat!(layer.neurons, index_remove)
                end
            end

            # The mutations for the rules are tested for each neuron
            for j in 1:NumberOfNeurons(circuit.layers[i])

                # Remove a random rule
                if rand() < probabilities.remove_rule
                    if length(circuit.layers[i].neurons[j].rules) > 2
                        index_remove = rand(1:length(circuit.layers[i].neurons[j].rules))
                        deleteat!(circuit.layers[i].neurons[j].rules, index_remove)
                    end
                end

                # Add a new random rule
                if rand() < probabilities.new_rule
                    #TODO migliora sta merda
                    added = false
                    while added == false
                        rule = rand(1:100)
                        if (rule in circuit.layers[i].neurons[j].rules) == false
                            push!(circuit.layers[i].neurons[j].rules, rule)
                            added = true
                        end
                    end
                end

                # Sample a new set of random input lines
            if rand() < probabilities.new_input_line
                    if i == 1
                        previous_inputs = inputs
                    else
                        previous_inputs = NumberOfNeurons(circuit.layers[i-1])
                    end

                    possible_input_lines = randperm(length(collect(1:previous_inputs)))

                    num_input_lines = rand(1:length(possible_input_lines))

                    circuit.layers[i].neurons[j].input_lines = collect(1:previous_inputs)[possible_input_lines[1:num_input_lines]]
                end
            end

        end

        return circuit
    end

    """
    CleanCircuit(c, inputs)

    Performs a cleaning operation of a given circuit. After performing
    a crossover or a mutation, some input lines of some neurons can be
    invalid (e.g., connected to a neuron that is not there anymore) or 
    entire layers can be empty. This function removes all the invalid 
    input lines and empty layers.

    # Arguments
    - `c::CircuitOfNeurons`: The considered circuit.
    - `inputs::UInt16`: The number of inputs of the circuit.

    # Returns
    - `Circuit`: A new circuit with no invalid input lines`.
    """

    # TODO(cumentate) 
    function CleanCircuitNew(c::CircuitOfNeurons, inputs::UInt16, outputs::UInt16)
        # Cloning the input circuit for convenience reasons
        circuit = deepcopy(c)

        i = 1
        while i <= NumberOfLayers(circuit)
            if i == 1
                previous_inputs = inputs
            else
                previous_inputs = NumberOfNeurons(circuit.layers[i - 1])
            end

            j = 1
            while j <= NumberOfNeurons(circuit.layers[i])
                k = 1
                while k <= length(circuit.layers[i].neurons[j].input_lines)
                    if circuit.layers[i].neurons[j].input_lines[k] > previous_inputs
                        deleteat!(circuit.layers[i].neurons[j].input_lines, k)
                    else
                        k = k + 1
                    end
                end

                if isempty(circuit.layers[i].neurons[j].input_lines)
                    deleteat!(circuit.layers[i].neurons, j)
                else
                    j = j + 1
                end
            end

            # If the clean process has removed a neuron of the last output layer, this block
            # adds as many random neurons as needed
            if i == NumberOfLayers(circuit) && (NumberOfNeurons(circuit.layers[i]) < outputs)
                for _ in NumberOfNeurons(circuit.layers[i]):outputs
                    push!(circuit.layers[i].neurons, GenerateRandomNeuron(previous_inputs))
                end
            end

            if isempty(circuit.layers[i].neurons)
                if i == 1
                    # If the evolution process has progressively decreased the number of neuron
                    # in the first (input) layer emptying it, we keep the choises made by the algorithgm,
                    # avoiding a complete new initialisation of the layer (chosing between _inputs_ 
                    # and _inputs_ + 2), adding just one random neuron
                    push!(circuit.layers[i].neurons, GenerateRandomNeuron(previous_inputs))
                else
                    # For any other layer we just delete it
                    deleteat!(circuit.layers, i)
                    i = i - 1
                end
            end
            i = i + 1
        end

        # TEST
        for j in 1:NumberOfLayers(circuit)

            if j == 1
                previous_inputs = inputs
            else
                previous_inputs = NumberOfNeurons(circuit.layers[j-1])
            end

            for k in 1:NumberOfNeurons(circuit.layers[j])
                for h in 1:length(circuit.layers[j].neurons[k].input_lines)
                    if circuit.layers[j].neurons[k].input_lines[h] > previous_inputs
                        throw(error("Found mismatch in input lines number AFTER CLEANING"))
                        exit()
                    end
                end
            end
        end
        # TEST

        return circuit
    end

function CleanCircuit(c::CircuitOfNeurons, inputs::UInt16, outputs::UInt16)

    # Cloning the input circuit for convenience reasons
    circuit = deepcopy(c)

    # Cleaning the first layer
    for i in 1:NumberOfNeurons(circuit.layers[1])
        circuit.layers[1].neurons[i].input_lines = filter(x -> x <= inputs, circuit.layers[1].neurons[i].input_lines)
    end

    # Cleaning the hidden layers
    for i in 2:NumberOfLayers(circuit)-1
        j = 1
        while j <= NumberOfNeurons(circuit.layers[i])

            # Enumerating all the input lines that does not connect to any previous neuron
            unconnected = Differences(NumberOfNeurons(circuit.layers[i-1]),
                circuit.layers[i].neurons[j].input_lines)


            if length(unconnected) == length(circuit.layers[i].neurons[j].input_lines)
                # If all the input lines are invalid, the whole neuron is removed
                deleteat!(circuit.layers[i].neurons, j)
            else
                # Otherwise, only the invalid lines are removed
                circuit.layers[i].neurons[j].input_lines =
                    setdiff(circuit.layers[i].neurons[j].input_lines, unconnected)

                j += 1

            end
        end

        # If, after the clean, the layer has no neurons, adds a random neuron
        if NumberOfNeurons(circuit.layers[i]) == 0
            if i == 1
                push!(circuit.layers[i].neurons, GenerateRandomNeuron(inputs))
            else
                push!(circuit.layers[i].neurons, GenerateRandomNeuron(NumberOfNeurons(circuit.layers[i-1])))
            end
        end
    end

    # The same cleaning operation is performed at the output layer
    # TODO: Può essere inglobato sopra togliendo il -1?
    for i in 1:NumberOfNeurons(circuit.layers[end])
        unconnected = Differences(NumberOfNeurons(circuit.layers[end-1]),
            circuit.layers[end].neurons[i].input_lines)

        circuit.layers[end].neurons[i].input_lines = setdiff(circuit.layers[end].neurons[i].input_lines, unconnected)

        if length(circuit.layers[end].neurons[i].input_lines) < 2
            circuit.layers[end].neurons[i].input_lines = 1:NumberOfNeurons(circuit.layers[end-1])
        end
    end

    #TODO Considera il caso in cui il layer rimane senza neuroni

    return circuit
end

    """
    Fitness(circuit, examples, labels)

    Computes the fitness value of a circuit with respect to a set of examples and
    relative labels. This value represents the number of correct labels computed by the
    circuit. The value of fitness is between 0 (all wrong) and 1 (all correct).

    # Arguments
    - `circuit::CircuitOfNeurons`: The considered circuit.
    - `examples::Vector{Vector{Bool}}`: The set of examples, typically a set of 2^n combinations
    - `labels::Vector{Bool}`: The corresponding set of labels, one for each example

    # Returns
    - `Float64`: The value of fitness, between 0 and 1`.
    """

    #TODO extend considering multiple outputs
    function Fitness(circuit::CircuitOfNeurons, examples::Vector{Vector{Bool}}, labels::Vector{Bool})
        correct = 0

        # The circuit is cloned so that, at the end of evaluation, spikes are reset (TODO migliorabile)
        circuit_clone = deepcopy(circuit) 

        for i in 1:length(examples)
            evaluation = EvaluateCircuitOfNeurons(circuit_clone, examples[i])[1]
            if evaluation == labels[i]
                correct += 1 
            end
        end

        return correct / length(examples)
    end

    """
    ProportionateSelection(fitness)

    Returns the index of the selected individual, according to a sequence of fitness
    values. The larger the fitness of an individual, the larger the probability to
    be selected by the algorithm

    # Arguments
    - `fitness::Vector{Float64}`: The set of fitness values of each circuit.

    # Returns
    - `Float64`: The value of fitness, between 0 and 1.
    """
    function ProportionateSelection(fitness::Vector{Float64})
        # Normalizing the set of fitnesses
        normalized = sort(fitness) ./ sum(fitness)

        # Computing the cumulative probabilities
        cumulative_probabilities = cumsum(reverse(normalized))

        # Sampling an item from the cumulative probabilities, the larger the
        # fitness, the larger the probability to be chosen
        selected_index = findfirst(x -> x > rand(), cumulative_probabilities)

        return selected_index
    end

end #Module