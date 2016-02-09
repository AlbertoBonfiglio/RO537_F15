#!/usr/bin/python3

#Basically we have a set of inputs x, xdot, x2dot, theta thetadot theta2 dot
#we need to find a set of weights that would give us the best uptime for the ball
#So we instatntiate the neural network
#every n (say 10ms) time we sample where is the pendulum
#we pass the data to the nn, the neural network evolves a set of weights,
# applies them to the data and produces an output to give to the pendulum




class NEvoNetwork (object):

    def __init__(self, inputs=6, hidden=0, outputs=1):
        self.inputs = [0 for n in range(inputs)]
        self.ihweights = self.get_matrix(inputs, hidden)

        self.hidden = [0 for n in range(hidden)]
        self.hbias = [0 for n in range(hidden)]
        self.houtputs = [0 for n in range(hidden)]
        self.howeights = self.get_matrix(inputs, hidden)


        self.howeights = self.get_matrix(inputs, hidden)
        self.outputs = [0 for n in range(outputs)]




    def get_matrix(self, rows=1, columns=1):
        return [[0 for j in range(columns)] for i in range(rows)]

