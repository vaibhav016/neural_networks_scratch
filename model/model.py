from math import exp
from random import random, seed


class Model:
    def __init__(self):
        self.model = list()
        self.previous_input_shape = 0

    def add(self, **kwargs):
        seed(1)
        previous_layer_shape = kwargs.get('input_shape') if kwargs.get('input_shape') else self.previous_input_shape
        number_of_neurons_current_layer = kwargs['number_of_neurons']
        hidden_layer = [{'weights': [random() for i in range(previous_layer_shape + 1)]} for i in
                        range(number_of_neurons_current_layer)]
        self.previous_input_shape = number_of_neurons_current_layer
        self.model.append(hidden_layer)

    def activation(self, weights, input):
        sum = weights[-1]
        for i in range(len(weights) - 1):
            sum = sum + weights[i] * input[i]

        return 1.0 / (1.0 + exp(-sum))

    def forward_propagation(self, data_row, network):
        input = data_row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                neuron['output'] = self.activation(neuron['weights'], input)
                new_inputs.append(neuron['output'])
            input = new_inputs
        return input

    def transfer_derivative(self, output):
        # sigmoid function is used so its derivative is this.
        return output * (1.0 - output)

    def backward_propagate_error(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']
