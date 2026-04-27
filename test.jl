inputs = 3

examples = Vector{Vector{Bool}}()
labels = Vector{Vector{Bool}}()

function f(input::Vector{Bool})
    return reverse(input)[1:2]
end

combinations = reverse.(Iterators.product(fill(0:1, inputs)...))[:]

for j in 1:Int(round((2^inputs) * 1))
    combination = [bitstring(i)[end] == '1' for i in combinations[j]]

    println(combination)

    push!(examples, combination)
    push!(labels, f(combination))
end

print("Examples: ")
println(examples)
print("Labels: ")
println(labels)

exit(0)