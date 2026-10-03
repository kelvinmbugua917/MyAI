
import math

# 1. Define the inputs
inputs = [2.0, 3.0]

# 2. Define the weights
weights = [0.5, -0.5]

# 3. Define the bias
bias = 0.1

# 4. Calculate the weighted sum
z = sum(x * w for x, w in zip(inputs, weights)) + bias

# 5. Apply the sigmoid activation function
output = 1 / (1 + math.exp(-z))

# 6. Display the results
print("Inputs:", inputs)
print("Weights:", weights)
print("Bias:", bias)
print("Weighted sum:", z)
print("Neuron output:", output)


# 7. Define the expected answer
target = 1.0

# 8. Calculate prediction error
error = target - output

# 9. Calculate squared loss
loss = error ** 2

# 10. Display the training information
print("Target:", target)
print("Error:", error)
print("Loss:", loss)




# STEP 3: Numerical derivative experiment

def calculate_loss(w2):

    z = (
        inputs[0] * weights[0]
        + inputs[1] * w2
        + bias
    )

    prediction = 1 / (1 + math.exp(-z))

    loss = (target - prediction) ** 2

    return loss


# Current weight
current_w2 = weights[1]

# Small change
h = 0.01

# Calculate loss at the current weight
current_loss = calculate_loss(current_w2)

# Calculate loss after increasing the weight
new_loss = calculate_loss(current_w2 + h)

# Estimate the derivative
derivative = (new_loss - current_loss) / h

print("\n--- DERIVATIVE EXPERIMENT ---")
print("Current weight:", current_w2)
print("Current loss:", current_loss)
print("New weight:", current_w2 + h)
print("New loss:", new_loss)
print("Estimated derivative:", derivative)




# STEP 4: Learning rate experiment

def simple_loss(w):
    return (w - 2) ** 2


def simple_gradient(w):
    return 2 * (w - 2)


initial_weight = 0.0

learning_rates = [0.1, 0.5, 1.1]

print("\n--- LEARNING RATE EXPERIMENT ---")

for learning_rate in learning_rates:

    gradient = simple_gradient(initial_weight)

    new_weight = (
        initial_weight
        - learning_rate * gradient
    )

    new_loss = simple_loss(new_weight)

    print("\nLearning rate:", learning_rate)
    print("Gradient:", gradient)
    print("Updated weight:", new_weight)
    print("New loss:", new_loss)



# # STEP 5: Automatic weight update

# learning_rate = 0.1
# current_w2 = weights[1]

# print("\n--- AUTOMATIC WEIGHT UPDATE ---")

# for step in range(50):
#     current_loss = calculate_loss(current_w2)

#     h = 0.01
#     new_loss = calculate_loss(current_w2 + h)

#     derivative = (new_loss - current_loss) / h

#     # Gradient descent
#     current_w2 = current_w2 - learning_rate * derivative

#     if (step + 1) % 10 == 0:
#         print(
#             "Step:", step + 1,
#             "| Weight:", current_w2,
#             "| Loss:", calculate_loss(current_w2)
#         )


# final_z = inputs[0] * weights[0] + current_w2 * inputs[1] + bias

# final_prediction = 1 / (1 + math.exp(-final_z))

# print("\n--- FINAL TRAINED RESULT ---")
# print("Final weight:", current_w2)
# print("Final weighted sum:", final_z)
# print("Final prediction:", final_prediction)
# print("Target:", target)
# print("Final loss:", calculate_loss(current_w2))



# STEP 6: Training all weights and bias

learning_rate = 0.1

w1 = weights[0]
w2 = weights[1]
b = bias

print("\n--- FULL NEURON TRAINING ---")

for step in range(100):

    # Forward propagation
    z = inputs[0] * w1 + inputs[1] * w2 + b

    prediction = 1 / (1 + math.exp(-z))

    loss = (target - prediction) ** 2

    # Backpropagation
    dL_dprediction = 2 * (prediction - target)

    dprediction_dz = prediction * (1 - prediction)

    dL_dz = dL_dprediction * dprediction_dz

    # Gradients for every parameter
    dw1 = dL_dz * inputs[0]
    dw2 = dL_dz * inputs[1]
    db = dL_dz

    # Update all parameters
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b = b - learning_rate * db

    if (step + 1) % 20 == 0:
        print(
            "Step:", step + 1,
            "| Loss:", loss,
            "| Prediction:", prediction
        )

print("\n--- TRAINED PARAMETERS ---")
print("Weight 1:", w1)
print("Weight 2:", w2)
print("Bias:", b)
print("Final prediction:", prediction)
print("Target:", target)

