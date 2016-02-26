#!/usr/bin/python3


from fractions import Fraction
from math import sin, cos


import matplotlib.pyplot as plt
import numpy as np

from HW5.classes.controller import Controller
from HW5.classes.invertedpendulum import InvertedPendulum
from HW5.classes.GeneticAlgorithm import Population
from HW5.classes.NeuralNetwork import NEvoNetwork, NeuronLayer, Neuron

def main():
    pendulum = InvertedPendulum()
    for n in np.arange(-200, 200, 10):
        cart, theta, forces = pendulum.applyforcea(u=n, tmax=2.5, timeslice=0.001)
        x, y = transform(theta)
        showGraph(x, y, cart, 0.001, "Relative motion of cart and pendulum u={0}".format(n))


def transform(theta):
    r = 1
    x = []
    y = []
    for n in range(int(len(theta))):
        # since we placed theta=0 up vertically we need to shift
        # the axis 90 degrees counterclockwise (-pi/2)
        # so x becomes sin(t) and y cos(t)
        y.append(r*cos(theta[n]))
        x.append(r*sin(theta[n]))

    return x, y


def showGraph(x, y, cart, timeslice=0.01, caption=""):
    #TODO make sure the x axis reflects the time it takes to drop

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False,
                             tight_layout=True, figsize=(9, 4.5))
    fig.suptitle(caption, fontsize=18, fontweight='bold')

    ax = axes[0]
    ax.set_title('Pendulum')
    ax.plot(x, y)
    ax.spines['left'].set_position(('axes', 0.0))
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('axes', 0.0))
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ticks = np.arange(min(x), max(x) * timeslice)
    labels = range(ticks.size)
    ax.set_xticks(ticks, labels)
    #ax.xlabel('seconds')

    x1 = []
    for n in range(len(cart)):
        x1.append(x[n] + cart[n])

    ax = axes[1]
    ax.set_title('Pendulum respect cart')
    ax.plot(x1, y)
    ax.spines['left'].set_position(('axes', 0.0))
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('axes', 0.0))
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks(ticks, labels)
    #ax.xlabel('seconds')

    plt.show()


def nnmain():
    contr = Controller()
    contr.start()

    while contr.isRunning():
        pass

    genome = [0,0,0,0,0,0]
    pop = Population(genome, 100)
    fit = pop.individuals[3].fitness()


def NNTest():
    try:

        x = NEvoNetwork(inputs=5, outputs=2, hiddenlayers=3,  hiddenneurons=5, inputweights=5)
        y = x.get_outputs([2, 4, 2, 8, 2.56])
        print(y)
    except Exception as ex:
        print(ex)



if __name__ == '__main__':
    NNTest()

    main()

    nnmain()



