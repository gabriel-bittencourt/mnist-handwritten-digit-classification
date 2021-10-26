"""
  Handwritten digit classification using a 3-layers Neural Network
"""

import numpy as np
import pandas as pd
import scipy.special


# Sigmoid function
def sigmoid(x):
    return scipy.special.expit(x)


# Rescale data
def scale_data(data, new_min, new_max):
    return np.interp(data, (data.min(), data.max()), (new_min, new_max))


class NeuralNetwork:

    def __init__(self, sizes, learning_rate=0.2):

        # sizes format: array[input nodes, hidden nodes, output nodes]
        self.sizes = sizes
        self.learning_rate = learning_rate

        # The weights are initialized randomly, using normal distribution
        # X ~ N(0, 1/sqrt(number of upcoming links))
        self.weights = [np.random.normal(0, pow(y, -0.5), size=(y, x))
                        for x, y in zip(sizes[0:], sizes[1:])]

    # Train the neural network
    def train(self, training_dataset, epochs=1):

        for e in range(epochs):

            for data in training_dataset:

                label = data[0]
                values = data[1:]

                # Rescale input data
                values = scale_data(values, 0.01, 1)
                values = np.array(values, ndmin=2).T

                # Targets array
                target_array = np.zeros(self.sizes[-1]) + 0.01
                target_array[label] = 0.99
                target_array = np.array(target_array, ndmin=2).T

                # Feeding forward (input layer -> hidden layer)
                hidden_in = np.dot(self.weights[0], values)
                hidden_out = sigmoid(hidden_in)

                # Feeding forward (hidden layer -> output layer)
                final_layer_in = np.dot(self.weights[1], hidden_out)
                final_layer_output = sigmoid(final_layer_in)

                # Errors (back propagation)
                output_error = target_array - final_layer_output
                hidden_error = np.dot(self.weights[1].T, output_error)

                # Update weights (hidden -> output)
                self.weights[1] += np.dot(
                    output_error * final_layer_output * (1 - final_layer_output),
                    hidden_out.T
                ) * self.learning_rate

                # Update weights (input -> hidden)
                self.weights[0] += np.dot(
                    hidden_error * hidden_out * (1 - hidden_out),
                    values.T
                ) * self.learning_rate

    # Return the neural network prediction for a given data
    def query(self, data):

        data = np.array(data, ndmin=2).T

        # Feeding forward

        hidden_in = np.dot(self.weights[0], data)
        hidden_out = sigmoid(hidden_in)

        output_in = np.dot(self.weights[1], hidden_out)
        output_out = sigmoid(output_in)

        return output_out

    # Return the neural network accuracy for a given test dataset
    def score(self, testing_dataset):

        score = []

        for data in testing_dataset:
            target = data[0]
            test_data = scale_data(data[1:], 0.01, 1)

            output = self.query(test_data)

            label = np.argmax(output)

            if label == target:
                score.append(1)
            else:
                score.append(0)

        accuracy = np.asarray(score).sum() / len(score)

        return accuracy


df = pd.read_csv('./datasets/mnist_train.csv', header=None)
training_data = np.array(df)

layers = [784, 100, 10]
lr = 0.2
n_epochs = 1


n = NeuralNetwork(layers, lr)

n.train(training_data, n_epochs)

df = pd.read_csv('./datasets/mnist_test.csv', header=None)
testing_data = np.array(df)

acc = n.score(testing_data)
print("Accuracy: ", acc)
