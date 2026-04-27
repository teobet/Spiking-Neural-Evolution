"""
    SNPCircuit

A module providing structs and functions to handle a 
circuit composed of Spiking Neural P-Systems (SN P-System) based
neurons

"""
module SNPCircuit

    # Library required when generating a random network
    using Random
    

    # Structs and functions that are exported to the user
    export Neuron, 
           LayerOfNeurons, 
           CircuitOfNeurons, 
           EvaluateCircuitOfNeurons, 
           GenerateRandomCircuit, 
           GenerateRandomNeuron,
           NumberOfLayers, 
           NumberOfNeurons
           
    """
    Neuron
       
    A struct representing a SN P-System neuron
       
    # Fields
    - `rules::Vector{UInt16}`: The set of rules of the neuron.
    - `input_lines::Vector{UInt16}`: The set of input lines connected to the neuron
    - `number_of_spikes::UInt32`: The number of spikes contained in the neuron
    """
    mutable struct Neuron
        rules::Vector{UInt16}
        input_lines::Vector{UInt16}
        number_of_spikes::UInt32
    end


    """
    LayerOfNeurons
       
    A layer of neurons is just a list of neurons, each containing
    its own rules, its input lines and its own number of spikes. Each
    neuron takes its inputs from the previous layers; similarly, the output
    values computed will be taken by the neurons in the next layer, hence it
    is not necessary to explicitly duplicate the output values produced by
    the neurons.
       
    # Fields
    - `neurons::Vector{Neuron}`: The sequence of neurons
    """
    mutable struct LayerOfNeurons
        neurons::Vector{Neuron}
    end

    
    """
    CircuitOfNeurons
       
    A circuit of neurons is just a sequence of layers
       
    # Fields
    - `layers::Vector{LayerOfNeurons}`: The sequence of layers
    """
    mutable struct CircuitOfNeurons
        layers::Vector{LayerOfNeurons}
    end

    # TODO Function,Var --> function,var (secondo convenzione)
    """
    NumberOfLayers(circuit)

    Returns the number of layers of a given circuit

    # Arguments
    - `circuit::CircuitOfNeurons`: The considered circuit.

    # Returns
    - `Int`: The number of layers of the circuit`.
    """
    function NumberOfLayers(circuit::CircuitOfNeurons)
        return UInt16(length(circuit.layers))
    end


    """
    NumberOfNeurons(layer)

    Returns the number of neurons of a given layer

    # Arguments
    - `layer::LayerOfNeurons`: The considered layer.

    # Returns
    - `Int`: The number of neurons of the layer`.
    """
    function NumberOfNeurons(layer::LayerOfNeurons)
        return UInt16(length(layer.neurons))
    end


    """
    EvaluateNeuron(sigma, input_vector)

    Given an input vector, compute the output and the new internal
    state of the neuron.

    # Arguments
    - `sigma::Neuron`: The considered neuron.
    - `input_vector::Vector{Bool}`: The boolean vector fed to the neuron.

    # Returns
    - `Bool`: true if the neuron fires a spike, false otherwise`.
    """
    function EvaluateNeuron(sigma::Neuron, input_vector::Vector{Bool})
        # Compute the number of input spikes
        TotalNumberOfSpikes::UInt64 = sigma.number_of_spikes
        for i in sigma.input_lines
            if input_vector[i]
                TotalNumberOfSpikes += 1
            end
        end
        # Checks whether the corresponding rule exists
        if TotalNumberOfSpikes in sigma.rules
            sigma.number_of_spikes = 0
            return true
        else
            # INFO perché le conserva?
            sigma.number_of_spikes = TotalNumberOfSpikes
            return false
        end
    end

    """
    EvaluateLayerOfNeurons(layer, input_vector)

    Given an input vector, compute the output and the new internal
    state of each neuron in the layer.

    # Arguments
    - `layer::LayerOfNeurons`: The considered layer.
    - `input_vector::Vector{Bool}`: The boolean vector fed to each neuron.

    # Returns
    - `Vector{Bool}`: a vector of output spikes`.
    """
    function EvaluateLayerOfNeurons(layer::LayerOfNeurons, input_vector::Vector{Bool})
        output_vector::Vector{Bool} = []
        for neuron in layer.neurons
            push!(output_vector, EvaluateNeuron(neuron, input_vector))
        end
        return output_vector
    end

    
    """
    EvaluateCircuitOfNeurons(circuit, input_vector)

    Given the input vector of the first layer, compute the output and the new 
    internal state of each neuron in every layer of the circuit.

    # Arguments
    - `circuit::CircuitOfNeurons`: The considered circuit.
    - `input_vector::Vector{Bool}`: The boolean vector fed to the first layer.

    # Returns
    - `Vector{Bool}`: a vector of output spikes of the circuit`.
    """
    function EvaluateCircuitOfNeurons(circuit::CircuitOfNeurons, input_vector::Vector{Bool})
        output_vector::Vector{Bool} = input_vector
        for layer in circuit.layers
            output_vector = EvaluateLayerOfNeurons(layer, output_vector)
        end
        return output_vector
    end


    """
    GenerateRandomCircuit(inputs, outputs, min_layers, max_layers)

    This function generates a random circuit with a given set of parameters.
    In particular, the generated circuit will take as input vectors of
    boolean values of size _inputs_ and will return a vector of _outputs_
    values. It will consist of an input layer, a random value between 
    _min_layers_ and _max_layers_ of hidden layers, and an output layer.

    # Arguments
    - `inputs::UInt16`: The number of inputs fed to the circuit.
    - `outputs::UInt16`: The number of output spikes of the circuit.
    - `min_layers::UInt16`: The minimum number of hidden layers (1 by default). 
    - `max_layers::UInt16`: The maximum number of hidden layers (2 by default).

    # Returns
    - `Vector{Bool}`: a vector of output spikes of the circuit`.
    """
    function GenerateRandomCircuit(inputs::UInt16, outputs::UInt16, min_layers::UInt16 = 0x0001, max_layers::UInt16 = 0x0002)
        # This variable refers to the number of hidden layers
        n_layers = rand(min_layers:max_layers)
        
        layers = Vector{LayerOfNeurons}()
        
        # Hidden layers
        for i in 1:n_layers
            # We assume that, when generating a random network, each layer will contain 
            # x \in {n, n + 1, n + 2} neurons, where n is the number of neurons of the previous layer
            if i == 1
                previous_inputs = inputs
            else
                previous_inputs = NumberOfNeurons(layers[i - 1])
            end

            n_neurons = rand(previous_inputs:(previous_inputs + 2))

            neurons = Vector{Neuron}()

            for _ in 1:n_neurons
                # Each neuron has random {rules, input_lines, number_of_spikes}
                push!(neurons, GenerateRandomNeuron(previous_inputs)) 
            end
            
            push!(layers, LayerOfNeurons(neurons))

        end
                

        # After adding the hidden layers, each circuit must have an output
        # layer, which will contain _outputs_ neurons

        neurons = Vector{Neuron}()
                
        for _ in 1:outputs
            push!(neurons, GenerateRandomNeuron(NumberOfNeurons(layers[end])))
        end

        push!(layers, LayerOfNeurons(neurons))

        return CircuitOfNeurons(layers)
    end
       
    """
    GenerateRandomNeuron(inputs)

    This function generates a random neuron given a number of
    inputs (i.e., the number of neurons of the previous layer).
    The values of input_lines and rules are randomly assigned.

    In particular, all possible input_lines are enumerated, then a 
    subset (or the whole set) of them is randmoly chosen. The same happens
    for the rules.

    # Arguments
    - `inputs::UInt16`: The number of inputs of the neuron.

    # Returns
    - `Neuron`: the randmoly generated neuron`.
    """
    function GenerateRandomNeuron(inputs::UInt16)
        # Permuting the enumeration of all the possible input lines and selecting a (sub)set of them
        possible_input_lines = randperm(inputs)
        num_input_lines = rand(1:length(possible_input_lines))
        possible_input_lines = possible_input_lines[1:num_input_lines]
            
        # The same is performed on the rules, also considering (a0 -> a), therefore the `.- 1``
        possible_rules = randperm(inputs + 1) .- 1
        num_rules = rand(1:length(possible_rules))
        possible_rules = possible_rules[1:num_rules]

        return Neuron(possible_rules, possible_input_lines, 0)
    end

    """
    Base.show(io, neuron)

    Prints the content of a neuron

    # Arguments
    - `io::IO`: The I/O stream to render to.
    - `neuron::Neuron`: The neuron to be printed

    # Examples
    ```julia
    println(neuron)
    ```
    """
    function Base.show(io::IO, neuron::Neuron)
        # Prints the general informations of the neuron
        print(io, "    Neuron\n")

        # Proceedes with printing the rules of the neuron
        print(io, "      Rules: {")
        for i in 1:length(neuron.rules)
            print(io, string(sort(neuron.rules)[i]))
            if i < length(neuron.rules)
                print(io, ", ")
            end
        end

        # Finally, prints the input lines to which the neuron is connected
        print(io, "}\n      Input lines: {")
        for i in 1:length(neuron.input_lines)
            print(io, string(sort(neuron.input_lines)[i]))
            if i < length(neuron.input_lines)
                print(io, ", ")
            end
        end
        print(io, "}\n")
    end
    
    """
    Base.show(io, layer)

    Prints the content of a layer of neurons

    # Arguments
    - `io::IO`: The I/O stream to render to.
    - `layer::LayerOfNeurons`: The layer to be printed

    # Examples
    ```julia
    println(layer)
    ```
    """
    function Base.show(io::IO, layer::LayerOfNeurons)
        # Prints the general informations of the layer
        print(io, "  Layer (" * string(NumberOfNeurons(layer)) * " neurons)\n")

        # Proceedes with printing the inner content of the neurons
        for neuron in layer.neurons
            print(io, neuron) 
        end
    end
    
    """
    Base.show(io, circuit)

    Prints the content of a circuit

    # Arguments
    - `io::IO`: The I/O stream to render to.
    - `circuit::CircuitOfNeurons`: The circuit to be printed

    # Examples
    ```julia
    print(circuit)
    ```
    """
    function Base.show(io::IO, circuit::CircuitOfNeurons)
        # Prints the general informations of the circuit
        println(io, "Circuit (" * string(NumberOfLayers(circuit)) * " layers)")

        # Proceedes with printing the inner content of the layers
        for layer in circuit.layers
            print(io, layer)
        end
    end

end #Module