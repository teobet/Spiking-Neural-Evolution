function CleanCircuit(c::CircuitOfNeurons, inputs::UInt16, outputs::UInt16)
    # Cloning the input circuit for convenience reasons
    circuit = deepcopy(c)

    i = 1
    while i <= NumberOfLayers(circuit)
        if i == 1
            previous_inputs = inputs
        else
            previous_inputs = NumberOfNeurons(circuit.layers[i-1])
        end

        j = 1
        while j <= NumberOfNeurons(circuit.layers[i])
            invalid_input_lines = setdiff(circuit.layers[i].neurons[j].input_lines, collect(1:previous_inputs))
            clean_input_lines = setdiff(circuit.layers[i].neurons[j].input_lines, invalid_input_lines)

            if isempty(clean_input_lines)
                deleteat!(circuit.layers[i].neurons, j)
            else
                circuit.layers[i].neurons[j].input_lines = clean_input_lines
                j = j + 1
            end

        end

        
        if isempty(circuit.layers[i].neurons)
            if i == 1
                # If the evolution process has progressively decreased the number of neuron
                # in the first layer (input layer), we keep the choises made by the algorithgm,
                # avoiding a complete new initialisation of the layer (chosing between _inputs_ 
                # and _inputs_ + 2), adding just one random neuron
                push!(circuit.layers[i].neurons, GenerateRandomNeuron(previous_inputs))
                i = i + 1
            elseif i == NumberOfLayers(circuit)
                # If the output layer loses all the neurons after the mutations,
                # we add _outputs_ random neurons in the last layer, keeping the circuit
                # faithful to the number of outputs we except the function f to have
                for _ in 1:outputs
                    push!(circuit.layers[i].neurons, GenerateRandomNeuron(previous_inputs))
                end
                i = i + 1
            else
                # For any other layer we just delete it
                deleteat!(circuit.layers, i)
            end
        end
    end
    
    return circuit
end