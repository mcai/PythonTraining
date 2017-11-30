import numpy as np


# Neural network
class NeuralNetwork:
    # Initialize the neural network.
    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.learning_rate = learning_rate

        self.activation_function = sigmoid

        self.weights_inputs_hidden = np.random.normal(0, np.power(self.num_hidden, -0.5), (self.num_hidden, self.num_inputs))
        self.weights_hidden_outputs = np.random.normal(0, np.power(self.num_outputs, -0.5), (self.num_outputs, self.num_hidden))

    # Train the neural network.
    def train(self, inputs, targets):
        inputs = inputs.T
        targets = targets.T

        hidden_in = np.matmul(self.weights_inputs_hidden, inputs)
        hidden_out = self.activation_function(hidden_in)

        outputs_in = np.matmul(self.weights_hidden_outputs, hidden_out)
        outputs_out = self.activation_function(outputs_in)

        outputs_errors = targets - outputs_out
        hidden_errors = np.matmul(self.weights_hidden_outputs.T, outputs_errors)

        self.weights_hidden_outputs += np.multiply(np.multiply(self.learning_rate * outputs_errors, outputs_out), 1 - outputs_out) * hidden_out.T
        self.weights_inputs_hidden += np.multiply(np.multiply(self.learning_rate * hidden_errors, hidden_out), 1 - hidden_out) * inputs.T

    # Test the neural network.
    def test(self, inputs):
        inputs = inputs.T

        hidden_in = np.matmul(self.weights_inputs_hidden, inputs)
        hidden_out = self.activation_function(hidden_in)

        outputs_in = np.matmul(self.weights_hidden_outputs, hidden_out)
        outputs_out = self.activation_function(outputs_in)

        return outputs_out.T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
