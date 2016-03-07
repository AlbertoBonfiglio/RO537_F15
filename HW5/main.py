#!/usr/bin/python3


from fractions import Fraction
from math import sin, cos


import matplotlib.pyplot as plt
import numpy as np

from HW5.classes.controller import Controller
from HW5.classes.invertedpendulum import InvertedPendulum, State
from HW5.classes.GeneticAlgorithm import Population
from HW5.classes.NeuralNetwork import NEvoNetwork, NeuronLayer, Neuron



def pendulumTest(timeslice=0.001, tmax=100):
    pendulum = InvertedPendulum()
    for n in np.arange(-200, 200, 100):
        states, time = pendulum.time_to_ground(u=n, tmax=tmax, timeslice=timeslice)
        n=0
        for state in states:
            force = pendulum.get_force(state)
            print('Force : {}'.format(force))
            n+=1
            if n > 60: break

        theta = (state.theta for state in states)
        cart =  (state.x for state in states)
        x, y = transform(theta)

       # showGraph(x, y, cart, 0.001, "Relative motion of cart and pendulum u={0}".format(n))


def transform(theta):
    r = 1
    x = []
    y = []
    for n in theta:
        # since we placed theta=0 up vertically we need to shift
        # the axis 90 degrees counterclockwise (-pi/2)
        # so x becomes sin(t) and y cos(t)
        y.append(r*cos(n))
        x.append(r*sin(n))

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
    i = 0
    for n in cart:
        x1.append(x[i] + n)
        i += 1

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


def nnmain(timeslice=0.001, tmax=0.2):
    #step 0: set up Neural network
    #step 1: Start pendulum with random force
    #step 2: at n milliseconds take state
    #step 3: pass state to genetic algorithm
    #step 4: GA runs n epochs and evaluates what gives the best output
    #step 5: GA passes weights to NN to evaluate a force to apply to pendulum
    #Step 6: NN applies force to pendulum
    #Step 7: Goto step 2

    pendulum = InvertedPendulum()
    NN = NEvoNetwork(inputs=6, outputs=1, hiddenlayers=1,  hiddenneurons=10, inputweights=6)
    ga = Population(pendulum, NN)

    force = np.random.randint(-200, 200)
    states, time = pendulum.time_to_ground(u=force, tmax=tmax, timeslice=timeslice)
    end_state = states[-1]

    ga.create(end_state)

    ga.evolve(epochs=50)



def NNTest():
    try:
        x = NEvoNetwork(inputs=6, outputs=1, hiddenlayers=2,  hiddenneurons=15, inputweights=6)
        w = x.get_weights()
        w[1] = 666
        x.set_weights2(w)
        w1= x.get_weights()
        print(w)
        y = x.get_outputs([2, 4, 2, 8, 2.56, 3])
        print(y)
    except Exception as ex:
        print(ex)


def gaTest():
    genome = np.zeros(6)

    ga = Population(genome, size=100)

    parents = [ga.individuals[4], ga.individuals[45]]

    x1, x2 = ga.crossover(parents)

    induhviduals = ga.evolve(125)
    print('done')


if __name__ == '__main__':
    # in the simulation world it should not fall because the simulation is multi threaded

    #start with the 0 values then pass the alues every 100ms to the NN
    # the number of weights should be the same as the inputs in the network for all layers

    #NNTest()

    #gaTest()


    #pendulumTest(0.01)
    #pendulumTest(0.001)
    #pendulumTest(0.0001)

    #pendulumTest(0.001, 0.001*200)

    nnmain()



