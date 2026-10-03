import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def layer_forward(inputs, weights, biases):

    outputs = []

    for i in range(len(weights)):

        z = biases[i]

        for j in range(len(inputs)):
            z += inputs[j] * weights[i][j]

        output = sigmoid(z)

        outputs.append(output)

    return outputs


# Inputs
inputs = [2.0, 3.0]

# Three neurons
weights = [
    [0.5, -0.5],
    [-0.3, 0.8],
    [0.2, 0.4]
]

biases = [0.1, 0.2, -0.1]

# Run the layer
result = layer_forward(inputs, weights, biases)

print("Layer outputs:", result)






# First layer
hidden = layer_forward(inputs, weights, biases)

# Second layer
weights2 = [
    [0.7, -0.2, 0.5],
    [-0.4, 0.6, 0.3]
]

biases2 = [0.1, -0.1]

output = layer_forward(hidden, weights2, biases2)

print("Hidden layer:", hidden)
print("Final output:", output)
