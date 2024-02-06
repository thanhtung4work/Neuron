from numpy import exp, array, random, dot, tanh

class Neuron:
    def __init__(self):
        random.seed(69)
        self.weights = random.random((3, 1)) * 2 -1
    
    def backward(self, x):
        return 1.0 - tanh(x) ** 2
    
    def forward(self, x):
        return tanh(dot(x, self.weights))