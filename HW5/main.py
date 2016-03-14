#!/usr/bin/python3


import time as tm
from math import sin, cos, pi, degrees

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from HW5.classes.NeuralNetwork import NEvoNetwork, TanhActivation
from HW5.classes.invertedpendulum import InvertedPendulum, State
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





def run_controller(trials=10, epochs =250, timeslice=0.002, tmax=0.2, threshold =((-pi/2), (pi/2))):
    #step 0: set up Neural network
    #step 1: Start pendulum with random force
    #step 2: at n milliseconds take state
    #step 3: pass state to genetic algorithm
    #step 4: GA runs n epochs and evaluates what gives the best output
    #step 5: GA passes weights to NN to evaluate a force to apply to pendulum
    #Step 6: NN applies force to pendulum
    #Step 7: Goto step 2

    #TODO run multiple simulations, collect avg data and error and produce graphs
    MAX_REWARD = 1000
    pendulum = InvertedPendulum()
    NN = NEvoNetwork(inputs=6, outputs=1, hiddenlayers=1,  hiddenneurons=12, inputweights=6, activation=TanhActivation)
    ga = Population(NN=NN, size=30)

    main_array = []
    reward_array =[]


    for trial in range(0, trials):
        ga.create(size=30)
        force = np.random.randint(-5, 5)
        while force == 0:
            force = np.random.randint(-5, 5)

        initial_state, time = pendulum.get_State(u=force, tmax=tmax, timeslice=timeslice)
        print('Force={1:3f} -Theta={0:4f}'.format(initial_state[-1].theta, force))

        reward_array = []
        population_array = []
        for epoch in range(0, epochs):
            for induhvidual in ga.individuals:
                NN.set_weights(induhvidual.alleles)
                state = []
                state.append(initial_state[-1])

                airborne = True
                while airborne:
                    force = NN.get_outputs([state[-1].x, state[-1].xdot, state[-1].x2dot, state[-1].theta, state[-1].thetadot, state[-1].theta2dot])[0] * 5
                    state, time = pendulum.get_State(u=force, initialstate=state[-1], tmax=tmax, timeslice=timeslice)

                    if state[-1].theta < threshold[0] or state[-1].theta > threshold[1]:
                        airborne = False
                    else:
                        induhvidual.set_fitness(1)

                    if induhvidual.get_fitness() >= MAX_REWARD: break

            reward_array.append(ga.getFittestIndividual().get_fitness())
            population_array.append(ga.getPopulationFitness())
            print('Trial: {0} - Epoch {1} --> Best fitness score = {2}, - Pop Fitness = {3}'
                  .format(trial, epoch, ga.getFittestIndividual().get_fitness(), ga.getPopulationFitness()))
            ga.evolve(epochs=1)

        main_array.append((reward_array, population_array))

    return main_array


def plot_controller_run(data, trials, epochs, threshold):
    performace_array = []
    for n in range(len(data[0][0])):
        fitness_list = [item[0][n] for item in data]
        pop_list = [item[1][n] for item in data]

        performance = (fitness_list, np.mean(fitness_list), stats.sem(fitness_list)/2, np.std(fitness_list)/2,
                       pop_list, np.mean(pop_list), stats.sem(pop_list)/2, np.std(pop_list)/2)

        performace_array.append(performance)

    y = [item[1] for item in performace_array]
    yerr = [item[2] for item in performace_array]
    x = np.arange(len(y))

    y1 = [item[5] for item in performace_array]
    y1err = [item[6] for item in performace_array]

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].set_title("Average Performance of Neuro-Evolutionary controller \n "
                       "over n={0} trials, {1:.2f}>theta<{2:.2f} "
                       .format(trials, degrees(threshold[0]), degrees(threshold[1])))
    axarr[0].errorbar(x, y, yerr=yerr, label='Best Fitness')
    axarr[0].legend(loc="upper left", shadow=True, fancybox=True)
    axarr[0].set_ylabel('Fitness')

    axarr[1].errorbar(x, y1, yerr=y1err, label='Avg Pop. Fitness')
    axarr[1].legend(loc="upper left", shadow=True, fancybox=True)
    axarr[1].set_ylabel('Fitness')

    plt.xlabel('Epochs (e={0})'.format(epochs))

    plt.show()





if __name__ == '__main__':

    #pendulumTest(0.01)
    #pendulumTest(0.001)
    #pendulumTest(0.0001)

    #pendulumTest(0.001, 0.001*200)

    #nnmain()
    trials= 15
    epochs= 150
    threshold =((-pi), (pi))
    data = run_controller(trials, epochs, threshold=threshold)
    plot_controller_run(data, trials, epochs, threshold)


