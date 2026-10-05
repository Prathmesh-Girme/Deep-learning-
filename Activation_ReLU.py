#Activation function
import numpy as np


def ReLu(z):
    return max(0, z)


def Marvellous_Neuron_Forward(inputs, weights, bias):
    print("Inputs are (X):", inputs)
    print("Weights are (W):", weights)
    print("Bias (b):", bias)

    z = 0

    for i in range(len(inputs)):
        z = z + (inputs[i] * weights[i])

    z = z + bias

    print("Weighted sum + Bias:", z)

    output = ReLu(z)

    print("ReLU Output:", output)

    return output


def main():
    print("---- Marvellous Neural Network ----")

    inputs = [1.0, 2.0, 3.0]

    weights = [0.6, 0.4, -0.2]

    bias = 0.5

    Marvellous_Neuron_Forward(inputs, weights, bias)


if __name__ == "__main__":
    main()

