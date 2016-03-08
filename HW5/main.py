#!/usr/bin/python3


import time as tm
from math import sin, cos, pi

import matplotlib.pyplot as plt
import numpy as np

from HW5.classes.NeuralNetwork import NEvoNetwork, TanhActivation
from HW5.classes.invertedpendulum import InvertedPendulum
from HW5.classes.GeneticAlgorithm import Population



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
    NN = NEvoNetwork(inputs=6, outputs=1, hiddenlayers=1,  hiddenneurons=8, inputweights=6, activation=TanhActivation)
    ga = Population(pendulum, NN)

    force = -50 # np.random.randint(-100, -10)
    states, time = pendulum.time_to_ground(u=force, tmax=tmax, timeslice=timeslice)
    end_state = states[-1]
    print('Force={1:3f} -Theta={0:4f}'.format(end_state.theta, force))

    theta_array = []
    for n in range(0, 100):
        ga.create(end_state, size=100, fitness_func=time_in_threshold)
        ga.create(end_state, size=100, fitness_func=angle_from_zero)

        t0 = tm.time()
        ga.evolve(epochs=75)
        t1 = tm.time()
        print('Evolutionj in {0}'.format(t0-t1))

        t0 = tm.time()
        induhvidual = ga.getFittestIndividual()
        NN.set_weights(induhvidual.alleles)
        force = NN.get_outputs([end_state.x, end_state.xdot, end_state.x2dot, end_state.theta, end_state.thetadot, end_state.theta2dot])[0] * 1000
        states, time = pendulum.time_to_ground(u=force, initialstate=end_state, tmax=0.2, timeslice=timeslice)
        t1 = tm.time()
        print('state in {0}'.format(t0-t1))
        end_state = states[-1]
        theta_array.append(end_state.theta)
        print('Force={1:3f} -Theta={0:4f} -Fitness={2:3f}'.format(end_state.theta, force, induhvidual.fitness_score))


def nnmain2(timeslice=0.001, tmax=0.05):
    #step 0: set up Neural network
    #step 1: Start pendulum with random force
    #step 2: at n milliseconds take state
    #step 3: pass state to genetic algorithm
    #step 4: GA runs n epochs and evaluates what gives the best output
    #step 5: GA passes weights to NN to evaluate a force to apply to pendulum
    #Step 6: NN applies force to pendulum
    #Step 7: Goto step 2

    pendulum = InvertedPendulum()
    NN = NEvoNetwork(inputs=6, outputs=1, hiddenlayers=1,  hiddenneurons=8, inputweights=6, activation=TanhActivation)
    ga = Population(NN=NN, size=100)

    force = -50 # np.random.randint(-100, -10)
    initial_state, time = pendulum.get_State(u=force, tmax=tmax, timeslice=timeslice)
    print('Force={1:3f} -Theta={0:4f}'.format(initial_state[-1].theta, force))

    master_array =[]

    ga.create(size=60)

    threshold =((-pi/2), (pi/2))
    for n in range(0, 100):
        for induhvidual in ga.individuals:
            NN.set_weights(induhvidual.alleles)
            theta_array =[]
            theta_array += initial_state
            state = []
            state += initial_state

            airborne = True
            while airborne:
                force = NN.get_outputs([state[-1].x, state[-1].xdot, state[-1].x2dot, state[-1].theta, state[-1].thetadot, state[-1].theta2dot])[0] * 100
                state, time = pendulum.get_State(u=force, initialstate=state[-1], tmax=tmax, timeslice=timeslice)

                theta_array += state
                #print('Force = {0}, Theta = {1}'.format(force, state.theta))
                if state[-1].theta < threshold[0] or state[-1].theta > threshold[1]:
                    airborne = False
                else:
                    induhvidual.set_fitness(1)

            master_array.append(state)

        print('Best fitness score = {0}'.format(ga.getFittestIndividual().get_fitness(), n))
        ga.evolve(epochs=1)




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

    #nnmain()
    nnmain2()



