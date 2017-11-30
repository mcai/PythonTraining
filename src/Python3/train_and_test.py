from .neural_network import NeuralNetwork
import numpy as np


def train_and_test(train_data, test_data, num_inputs, num_hidden, num_outputs, learning_rate, num_epochs):
    nn = NeuralNetwork(num_inputs, num_hidden, num_outputs, learning_rate)

    print('Training..')

    for epoch in range(int(num_epochs)):
        for i in range(np.size(train_data, 0)):
            inputs = train_data[[i], 1:] / 255.0 * 0.99 + 0.01

            targets = np.zeros((1, num_outputs)) + 0.01
            targets[0, int(train_data[i, 0])] = 0.99

            nn.train(inputs, targets)

    print('Testing..')

    scores = np.zeros((np.size(test_data, 0), 1))

    for i in range(np.size(test_data, 0)):
        inputs = test_data[[i], 1:] / 255.0 * 0.99 + 0.01

        targets = np.zeros((1, num_outputs)) + 0.01
        targets[0, int(test_data[i, 0])] = 0.99

        outputs = nn.test(inputs)

        output = np.argmax(outputs)
        target = np.argmax(targets)

        print('Output: {}, target: {}, output == target: {}\n'.format(output, target, output == target))

        scores[i, 0] = (output == target)

    accuracy = np.mean(scores)

    print('Number of hidden nodes: {}, learning rate: {}, number of epochs: {}, accuracy: {}\n'.format(num_hidden, learning_rate, num_epochs, accuracy))

    return accuracy

