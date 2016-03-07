#!/usr/bin/python3

#Basically we have a set of inputs x, xdot, x2dot, theta thetadot theta2 dot
#we need to find a set of weights that would give us the best uptime for the ball
#So we instatntiate the neural network
#every n (say 10ms) time we sample where is the pendulum
#we pass the data to the nn, the neural network evolves a set of weights,
# applies them to the data and produces an output to give to the pendulum

import numpy
import math

def StepActivation(value, threshold):
    if threshold < value: return 1
    return 0


def SigmoidActivation(value, threshold):
    try:

        return 1 / (1 + numpy.exp((-value)/threshold))
    except Exception as ex:
        print('Exception {0}, value = {0}, threshold = {1}'.format(ex, value, threshold))




class Neuron(object):
    def __init__(self, weights=5, activation=SigmoidActivation):
        self.weights = numpy.random.randn(weights)
        self.activation = activation


    def output(self, inputs, activation=None):
        #First let's make sure things match
        if len(inputs) != (len(self.weights) -1):
            raise Exception('Number of inputs {0} to weights {1} mismatch'.format(inputs, self.weights-1))

        #in case we change the activation function
        if activation is not None:
            self.activation = activation

        try:
            _total = 0
            for n in range(len(inputs)):
                _total += inputs[n] * self.weights[n]
        except:
            print('Error \n   {3} \n    {4} \nN is {0}, inputs:{1}, weights={2}'.format(n, len(inputs), len(self.weights), self.weights, inputs))


        _threshold = self.weights[-1] # the last weight used for the bias

        return self.activation(_total, _threshold)



class NeuronLayer(object):
    #initialized the layer with defaults if necessary
    def __init__(self, neurons=3, weights=5, activation=SigmoidActivation):
        self.neurons = [Neuron(weights, activation) for n in range(neurons)]
        self.number_of_neurons = neurons
        self.weights_per_neuron = weights
        self.activation = activation


    def get_outputs(self, inputs):
        retval = []
        for neuron in self.neurons:
            retval.append(neuron.output(inputs, self.activation))

        return retval







class NEvoNetwork (object):

    def __init__(self, inputs=3, outputs=1, hiddenlayers=1,  hiddenneurons=3, inputweights=3, activation=SigmoidActivation ):
        self.layers = []
        self.layers.append(NeuronLayer(inputs, inputweights+1, activation))

        _layerweights = inputs
        for n in range(hiddenlayers):
            # creates hidden layers. each layer adds one weight
            self.layers.append(NeuronLayer(hiddenneurons, _layerweights + 1, activation))
            _layerweights = hiddenneurons

        self.layers.append(NeuronLayer(outputs, _layerweights +1, activation))


    def get_outputs(self, inputs):
        #Layer 0 is the input layet
        output = self.layers[0].get_outputs(inputs)
        #print('0 layer output {0} \n      {1}'.format(output,inputs))

        #now iterates through the rest of the layers
        for n in range(1, len(self.layers)):
            output = self.layers[n].get_outputs(output)
            #print('layer {1} output {0}'.format(output, n))

        return output


    def get_weights(self):
        retval = []
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.weights:
                   retval.append(weight)

        return retval


    #def set_weights(self, weights):
    #    n=0
    #    for layer in self.layers:
    #        for neuron in layer.neurons:
    #            for weight in neuron.weights:
    #                weight = weights[n]
    #                n +=1

    def set_weights(self, weights):
        try:
            n=0
            for layer in range(len(self.layers)):
                for neuron in range(len(self.layers[layer].neurons)):
                    for weight in range(len(self.layers[layer].neurons[neuron].weights)):
                        self.layers[layer].neurons[neuron].weights[weight] = weights[n]
                        n +=1
        except Exception as ex:
            print('Error setting weights {0}, {1}'.format(ex, weights))