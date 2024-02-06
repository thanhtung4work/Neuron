from numpy import dot, array, tanh, mean

from module import Neuron, mse


def numeric_derivative(f, x, c=0, method="central", h=1e-6):
    if method == "central":
        return (f(x + h, c) - f(x - h, c)) / (2 * h)
    elif method == "forward":
        return (f(x + h, c) - f(x, c)) / h
    elif method == "backward":
        return (f(x, c) - f(x - h, c)) / h
    return 0


def train(neuron: Neuron, inputs, labels, epochs, lr=0.001):
    for i in range(epochs):
        pred = neuron.forward(inputs)
        error = mse(pred, labels)

        # multiply the error by input and then 
        # by gradient of tanh function to calculate
        # the adjustment needs to be made in weights
        adjustment = dot(inputs.T, error * numeric_derivative(mse, pred, labels))
        if i % 1000 == 0: 
            print(f"epoch {i}, loss: {mean(error)}")

        neuron.weights -= adjustment * lr

if __name__ == "__main__":
    my_neuron = Neuron()
    print("Before training: \n", my_neuron.weights)

    train_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_outputs = array([[0, 1, 1, 0]]).T
 
    train(my_neuron, train_inputs, train_outputs, 10000)

    print("After training: \n", my_neuron.weights)