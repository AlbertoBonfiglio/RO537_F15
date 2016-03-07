#!/usr/bin/python3

from random import uniform, randint, shuffle
from math import radians
import copy

import numpy

from HW5.classes.invertedpendulum import InvertedPendulum, State
from HW5.classes.NeuralNetwork import NEvoNetwork

def time_to_ground(pendulum, state=None, force=0):
    #in this instance we calculate fitness based on how long the pendulum stays up
    #returns milliseconds
    states, time = pendulum.time_to_ground(u=force, initialstate=state)
    return time


def time_in_threshold(pendulum, state=None, force=0):
    #in this instance we calculate fitness based on how long the pendulum stays within
    # plus or minus the threshold
    #Basically we count how much time the pendulum stays within the threshold
    t1 = radians(-20),
    t2 = radians(20),
    states, time = pendulum.time_in_threshold(u=force, initialstate=state, threshold=(t1,t2))
    return time


class Individual (object):

    def __init__(self, alleles, pendulum, state, NN):
        self.pendulum = pendulum
        self.state = state
        #these are the actual weights we are evolving
        self.alleles = alleles
        self.fitness_score = 0
        self.inputs = [self.state.x, self.state.xdot, self.state.x2dot, self.state.theta, self.state.thetadot, self.state.theta2dot]
        self.NN = NN


    def calculate_fitness(self, func=None):
        #TODO find a way to return a specific fitness value from different

        if func == None:
            func = time_to_ground

        #this should calculate a force to apply
        self.NN.set_weights(self.alleles)
        output = self.NN.get_outputs(self.inputs)[0] * 1000

        #Now we apply the force and see how it pefrorms to give a fitness
        self.fitness_score = func(self.pendulum, self.state, output)








class Population (object):


    def __init__(self, pendulum, NN, state=None, fitness_func=None, size=100):
        if NN is None: raise Exception('NN not initialized')
        if pendulum is None: raise Exception('Pendulum not initialized')
        self.size = size
        self.genome = len(NN.get_weights())
        self.crossover_rate = 0.4
        self.mutation_rate = 0.3
        self.weightmax = 1
        self.pendulum = pendulum
        self.state = state
        self.NN = NN
        self.individuals = []
        self.fitness_function = fitness_func
        if state is not None: self.create(state, size, fitness_func)


    def create(self, state, size=100, fitness_func=None):
        self.fitness_function = fitness_func
        self.size = size

        induhviduals = []
        for n in range(self.size):
            alleles = [uniform(-self.weightmax, self.weightmax) for n in range(self.genome)]
            shuffle(alleles)
            induhvidual = Individual(alleles, self.pendulum, state, self.NN)
            induhvidual.calculate_fitness(fitness_func)
            induhviduals.append(induhvidual)

        self.individuals = induhviduals

        return induhviduals


    def evolve(self, epochs:50, fitness_treshold=0.2):
        if len(self.individuals) == 0: raise Exception("Population ont initialized")

        try:
            population = self.individuals

            for e in range(epochs):
                individuals = self.__getFittest(population, fitness_treshold)
                induhviduals = []
                new_population = self.size-len(individuals)

                for n in range(0, new_population, 2):
                    parent1 = self.select(individuals)
                    parent2 = self.select(individuals, parent1)

                    #crossover
                    #child1, child2 = self.crossover([parent1, parent2])

                    #non crossover
                    child1 = copy.deepcopy(parent1)
                    child2 = copy.deepcopy(parent2)

                    mutant1 = self.mutate(child1)
                    mutant2 = self.mutate(child2)

                    mutant1.calculate_fitness(self.fitness_function)
                    mutant2.calculate_fitness(self.fitness_function)

                    induhviduals.append(mutant1)
                    induhviduals.append(mutant2)

                population = individuals + induhviduals

            self.individuals = population
            return population
        except Exception as ex:
            print('Evolve', ex, e, n)


    def select(self, chromosomes, excluded=None):
        try:
            localcopy = copy.copy(chromosomes)
            if excluded is not None:
                localcopy.remove(excluded)

            sum_fitness = sum([chromosome.fitness_score for chromosome in localcopy])
            choice = uniform(0, sum_fitness)
            current_score = 0
            for chromosome in localcopy:
                current_score += chromosome.fitness_score
                if current_score > choice:
                    return chromosome

            #in case can't decide
            return localcopy[randint(0, len(localcopy)-1)]

        except Exception as ex:
            print('Select', ex)


    def crossover(self, parents, pivot=None):
        try:
            if uniform(0, 1) > self.crossover_rate:
                child0 = Individual(parents[0].alleles, self.pendulum, state=parents[0].state, NN=self.NN)
                child1 = Individual(parents[1].alleles, self.pendulum, state=parents[0].state, NN=self.NN)

                return child0, child1

            if pivot is None:
                pivot = randint(0, self.genome)

            alleles0 = parents[0].alleles[0:pivot] + parents[1].alleles[pivot:]
            alleles1 = parents[0].alleles[pivot:] + parents[1].alleles[0:pivot]

            child0 = Individual(alleles0, self.pendulum, state=parents[0].state, NN=self.NN)
            child1 = Individual(alleles1, self.pendulum, state=parents[0].state, NN=self.NN)

            return child0, child1
        except Exception as ex:
            print('Crossover -', ex)

    def mutate(self, chromosome):
        try:
            retval = copy.copy(chromosome)
            for n in range(len(chromosome.alleles)):
                if uniform(0, 1) < self.mutation_rate:
                    retval.alleles[n] = uniform(-self.weightmax, self.weightmax)

            return retval
        except Exception as ex:
            print('Mutate -', ex)



    def __getFittest(self, population, n=0.2):
        induhviduals = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        return induhviduals[0: int(len(population)*n)]


    def getFittestIndividual(self, population=None):
        if population is None:
            population = self.individuals
        induhviduals = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        return induhviduals[0]


    def getPopulationFitness(self, population=None):
        if population is None:
            population = self.individuals

        return sum([chromosome.fitness_score for chromosome in population])


    def check_alleles(self, alleles):
        for a in alleles:
            if a is None:
                raise Exception('Error setting up alleles')


    def isDifferent(self, chromosome1, chromosome2):
        for n in range(len(chromosome1.alleles)):
            if chromosome1.alleles[n] != chromosome2.alleles[n]:
                print('has mutated {0} --> {1}'.format(chromosome1.alleles[n], chromosome2.alleles[n]))
                return True

        return False