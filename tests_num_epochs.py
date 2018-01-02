import numpy as np
import matplotlib.pyplot as plt

from train_and_test import train_and_test

print('Loading data..')

mnist_train_data = np.loadtxt('data/mnist_train.csv', delimiter=',')
mnist_test_data = np.loadtxt('data/mnist_test.csv', delimiter=',')

num_inputs = 784
num_hidden = 100
num_outputs = 10

learning_rate = 0.1

num_epochs_range = np.array([1, 2, 3, 5, 6, 7, 10, 15, 20])

accuracies = np.array([0.0] * len(num_epochs_range))

for i in range(np.size(num_epochs_range)):
    num_epochs = num_epochs_range[i]
    accuracy = train_and_test(mnist_train_data, mnist_test_data, num_inputs, num_hidden, num_outputs, learning_rate, num_epochs)

    accuracies[i] = accuracy

best_accuracy, best_accuracy_index = np.max(accuracies), np.argmax(accuracies)
best_num_epochs = num_epochs_range[best_accuracy_index]

print('Best number of epochs: {}, best accuracy: {}\n'.format(best_num_epochs, best_accuracy))

plt.plot(num_epochs_range, accuracies, '-o')
plt.title('Number of Epochs vs. Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.savefig('tests_num_epochs.jpg')
plt.show()
