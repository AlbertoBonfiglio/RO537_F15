#!/usr/bin/python3

from random import uniform, randint, shuffle, choice
from math import radians, pi
import copy
import time

import numpy

from HW5.classes.invertedpendulum import InvertedPendulum, State
from HW5.classes.NeuralNetwork import NEvoNetwork

class Individual (object):

    def __init__(self, alleles):
        #these are the actual weights we are evolving
        self.alleles = alleles
        self.fitness_score = 0

    def set_fitness(self, score):
        self.fitness_score += score

    def get_fitness(self):
        return self.fitness_score


class Population (object):

    def __init__(self, NN, size=50):
        if NN is None: raise Exception('NN not initialized')
        self.size = size
        self.genome = len(NN.get_weights())
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.weightmax = 1
        self.NN = NN
        self.individuals = []


    def create(self,  size=50):
        self.size = size

        induhviduals = []
        for n in range(self.size):
            alleles = [uniform(-self.weightmax, self.weightmax) for n in range(self.genome)]
            shuffle(alleles)
            induhvidual = Individual(alleles)
            induhviduals.append(induhvidual)

        self.individuals = induhviduals

        return induhviduals


    def evolve(self, epochs:50, fitness_treshold=0.3):
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
                    child1, child2 = self.crossover([parent1, parent2])

                    #non crossover
                    #child1 = copy.deepcopy(parent1)
                    #child2 = copy.deepcopy(parent2)

                    mutant1 = self.mutate(child1)
                    mutant2 = self.mutate(child2)

                    #self.isDifferent(mutant1, parent1)
                    #self.isDifferent(mutant2, parent2)

                    induhviduals.append(mutant1)
                    induhviduals.append(mutant2)

                population = individuals + induhviduals

                self.reset_fitness(population)

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
            fitchoice = uniform(0, sum_fitness)
            current_score = 0
            for chromosome in localcopy:
                current_score += chromosome.fitness_score
                if current_score > fitchoice:
                    return chromosome

            #in case can't decide
            return choice(localcopy)

        except Exception as ex:
            print('Select', ex)


    def select2(self, chromosomes, excluded=None):
        try:
            localcopy = copy.copy(chromosomes)
            if excluded is not None:
                localcopy.remove(excluded)

            #in case can't decide
            return choice(localcopy)

        except Exception as ex:
            print('Select', ex)




    def crossover(self, parents, pivot=None):
        try:
            if uniform(0, 1) > self.crossover_rate:
                child0 = Individual(parents[0].alleles)
                child1 = Individual(parents[1].alleles)

                return child0, child1

            if pivot is None:
                pivot = randint(0, self.genome)

            alleles0 = parents[0].alleles[0:pivot] + parents[1].alleles[pivot:]
            alleles1 = parents[0].alleles[pivot:] + parents[1].alleles[0:pivot]

            child0 = Individual(alleles0)
            child1 = Individual(alleles1)

            return child0, child1
        except Exception as ex:
            print('Crossover -', ex)


    def mutate(self, chromosome):
        try:
            retval = []
            for n in range(len(chromosome.alleles)):
                if uniform(0, 1) < self.mutation_rate:
                    retval.append(uniform(-self.weightmax, self.weightmax))
                else:
                    retval.append(chromosome.alleles[n])

            return Individual(alleles=retval)

        except Exception as ex:
            print('Mutate -', ex)

    def reset_fitness(self, population):
        if population is None:
            population = self.individuals
        for induhvidual in population:
            induhvidual.fitness_score = 0

        return population


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