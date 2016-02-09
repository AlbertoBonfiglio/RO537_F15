#!/usr/bin/python3


from fractions import Fraction
from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np
from classes.controller import Controller
from classes.invertedpendulum import InvertedPendulum

from HW5.classes.population import Population


def main():
    #neuralNet = NeuralNetwork()

    pendulum = InvertedPendulum()
    for n in np.arange(-0.5, 10, 0.5):
        cart, theta = pendulum.applyforce(u=n, tmax=2.5, timeslice=0.01)
        x, y = transform(theta)
        showGraph(x, y, cart, 0.01, "Relative motion of cart and pendulum u={0}".format(n))


def nnmain():
    contr = Controller()
    contr.start()

    while contr.isRunning():
        pass

    genome = [0,0,0,0,0,0]
    pop = Population(genome, 100)
    fit = pop.individuals[3].fitness()


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
    ax.xlabel('seconds')

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
    ax.xlabel('seconds')

    plt.show()





def x1():

    x=np.arange(-10.0,10.0,0.1)
    y=np.arctan(x)

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ax.plot(x,y,'b.')

    y_pi   = y/np.pi
    unit   = 0.25
    y_tick = np.arange(-0.5, 0.5+unit, unit)

    y_label = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$+\frac{\pi}{4}$",   r"$+\frac{\pi}{2}$"]
    ax.set_yticks(y_tick*np.pi)
    ax.set_yticklabels(y_label, fontsize=20)

    y_label2 = [r"$" + format(r, ".2g")+ r"\pi$" for r in y_tick]
    ax2 = ax.twinx()
    ax2.set_yticks(y_tick*np.pi)
    ax2.set_yticklabels(y_label2, fontsize=20)

    plt.show()


def create_pi_labels(a, b, step):

    max_denominator = int(1/step)
    # i added this line and the .limit_denominator to solve an
    # issue with floating point precision
    # because of floating point precision Fraction(1/3) would be
    # Fraction(6004799503160661, 18014398509481984)

    values = np.arange(a, b+step/10, step)
    fracs = [Fraction(x).limit_denominator(max_denominator) for x in values]
    ticks = values*np.pi

    labels = []

    for frac in fracs:
        if frac.numerator==0:
            labels.append(r"$0$")
        elif frac.numerator<0:
            if frac.denominator==1 and abs(frac.numerator)==1:
                labels.append(r"$-\pi$")
            elif frac.denominator==1:
                labels.append(r"$-{}\pi$".format(abs(frac.numerator)))
            else:
                labels.append(r"$-\frac{{{}}}{{{}}} \pi$".format(abs(frac.numerator), frac.denominator))
        else:
            if frac.denominator==1 and frac.numerator==1:
                labels.append(r"$\pi$")
            elif frac.denominator==1:
                labels.append(r"${}\pi$".format(frac.numerator))
            else:
                labels.append(r"$\frac{{{}}}{{{}}} \pi$".format(frac.numerator, frac.denominator))

    return ticks, labels

if __name__ == '__main__':
    #main()

    nnmain()



