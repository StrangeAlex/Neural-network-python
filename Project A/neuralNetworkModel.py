import numpy as np
np.random.seed(1)

class NeuralNetwork():
    def __init__(self, learningRate, bias, neurons):

        #initialize amount of neurons
        self.neurons = neurons

        #initialize learning rate
        self.learningRate = learningRate

        #initializing weights and putting random values to them
        self.weights = []
        for i in range(len(self.neurons)-1):
            self.weights.append(np.random.normal(0.0, pow(self.neurons[i+1], 0.5),
            (self.neurons[i+1], self.neurons[i])))

        #The same thing but for bias neurons
        self.bias = bias
        self.biasWeights = []
        for i in range(len(self.neurons)-1):
            self.biasWeights.append(np.random.normal(0.0, pow(self.neurons[i+1], 0.5) * self.bias,
            (self.neurons[i+1], 1)))


        #Take sigmoid as activation function
        self.actFunc = lambda x: 1 / (1 + np.exp(-x))
        self.derivativeActFunc = lambda x: x * (1 - x)

        pass

    #Training NN
    def train(self, inputsList, targetsList):

        #transforming outputs into 2D array
        inputLayer = np.array(inputsList, ndmin = 2).T
        targets = np.array(targetsList, ndmin = 2).T

        #Running forward propagation
        layers = []
        layers.append(inputLayer)
        for i in range(len(self.neurons)-1):
            layers.append(self.actFunc(np.dot(self.weights[i], layers[i]) + self.biasWeights[i]))

        #Creating errors array and then reversing it so it comes from first layer to last
        errors = []
        for i in range(len(self.neurons)-1):
            if i == 0: #output error is just targets - outputs
                errors.append(targets - layers[len(layers)-1])
            else:
                errors.append(np.dot(self.weights[len(self.weights)-i].T, errors[i-1]))
        errors.reverse()

        #updating weights
        for i in range(len(self.neurons)-1):
            gradient = np.dot((errors[len(errors)-1-i] * self.derivativeActFunc(layers[len(layers)-1-i])), np.transpose(layers[len(layers)-i-2]))
            delta = self.learningRate * gradient
            self.weights[len(self.weights)-1-i] += delta


        if self.bias != 0:
            for i in range(len(self.neurons)-1):
                biasGradient = errors[len(errors)-1-i] * self.derivativeActFunc(layers[len(layers)-1-i])
                biasDelta = self.learningRate * biasGradient
                self.biasWeights[len(self.weights)-1-i] += biasDelta
        pass


    #Testing NN
    def query(self, inputsList):

        #transforming input data into 2D array
        inputLayer = np.array(inputsList, ndmin = 2).T

        layers = []
        layers.append(inputLayer)

        #running forward propagation for all layers
        for i in range(len(self.neurons)-1):
            layers.append(self.actFunc(np.dot(self.weights[i], layers[i]) + self.biasWeights[i]))

        #returning exactly what we need
        return layers[len(layers)-1]
