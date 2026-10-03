import math

# Inputs
inputs = [2.0, 3.0]

# Two neurons, each with two weights
weights = [
    [0.5, -0.5],
    [-0.3, 0.8]
]

# One bias per neuron
biases = [0.1, 0.2]

# Process each neuron
outputs = []

for i in range(2):

    z = (
        inputs[0] * weights[i][0]
        + inputs[1] * weights[i][1]
        + biases[i]
    )

    output = 1 / (1 + math.exp(-z))

    outputs.append(output)

    print(f"Neuron {i + 1}:")
    print("Weighted sum:", z)
    print("Output:", output)

print("\nLayer outputs:", outputs)