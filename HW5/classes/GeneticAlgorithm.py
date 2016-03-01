#!/usr/bin/python3

from random import uniform, randint
from HW5.classes.invertedpendulum import InvertedPendulum


class Individual (object):

    def __init__(self, alleles, pendulum, threshold=0.1):
        self.threshold = threshold
        self.pendulum = pendulum
        #these are the actual weights we are evolving
        self.alleles = alleles
        self.fitness_score = 0



    #TODO find a way to return a specific fitness value from different
    #functions e.g. 0.0 to 1.0
    def calculate_fitness(self, func=None):

        if func == None:
            func = self.time_to_ground
        self.fitness_score = func()


    #in this instance we calculate fitness based on how long the pendulum stays up
    #returns milliseconds
    def time_to_ground(self):
        #TODO perform calulatoin
        time = self.pendulum.applyforce2(u=10)

        return time

    #in this instance we calculate fitness based on how long the pendulum stays within
    # plus or minus the threshold
    #returns milliseconds
    def time_to_threshold(self):
        #TODO perform calulatoin
        return 0



class Population (object):


    def __init__(self, genome, M=10, m=1, l=1, size=100):
        self.M = M
        self.m = m
        self.l = l
        self.size = size
        self.genome = genome
        self.crossover_rate = 0.3
        self.mutation_rate = 0.3
        self.pendulum = InvertedPendulum(M, m, l)
        self.individuals = self.create(self.pendulum, size)


    def create(self, pendulum, size=100, fitness_func=None):
        self.fitness_function = fitness_func

        induhviduals = []
        for n in range(self.size):
            alleles = [uniform(-10, 10) for n in range(len(self.genome))]
            induhvidual = Individual(alleles, pendulum)
            induhvidual.calculate_fitness(fitness_func)
            induhviduals.append(induhvidual)

        return induhviduals


    def evolve(self, epochs:50, fitness_treshold=0.2):
        try:
            population = self.individuals

            for e in range (epochs):
                individuals = self.__getFittest(population, fitness_treshold)
                induhviduals = []
                new_population = self.size-len(individuals)

                for n in range(0, new_population, 2):
                    parent1 = self.select(individuals)
                    parent2 = self.select(individuals)

                    child1, child2 = self.crossover([parent1, parent2])
                    child1.calculate_fitness(self.fitness_function)
                    child1.calculate_fitness(self.fitness_function)
                    induhviduals.append(self.mutate(child1))
                    induhviduals.append(self.mutate(child2))

                population = individuals + induhviduals
                print('epoch: {0} - fitness {1}'.format(e, self.getPopulationFitness(population)))


            self.individuals = population
            return population
        except Exception as ex:
            print(ex, e, n)


    def select(self, chromosomes):
        try:
            sum_fitness  = sum([chromosome.fitness_score for chromosome in chromosomes])
            choice = uniform(0, sum_fitness)
            current_score = 0
            for chromosome in chromosomes:
                current_score += chromosome.fitness_score
                if current_score > choice:
                    return chromosome

            #in case can't decide
            return chromosomes[randint(0, len(chromosomes)-1)]

        except Exception as ex:
            print('select', ex)


    def crossover(self, parents, pivot=None):
        if uniform(0, 1) > self.crossover_rate:
            child0 = parents[0]
            child1 = parents[1]
            return child0, child1

        if pivot is None:
            pivot = randint(0, len(self.genome))

        alleles0 = parents[0].alleles[0:pivot] + parents[1].alleles[pivot:]
        alleles1 = parents[0].alleles[pivot:] + parents[1].alleles[0:pivot]

        child0 = Individual(alleles0, self.pendulum)
        child1 = Individual(alleles1, self.pendulum)

        return child0, child1



    def mutate(self, chromosome):
        for n in chromosome.alleles:
            if uniform(0, 1) < self.mutation_rate:
                n = uniform(-10, 10)

        return chromosome


    def __getFittest(self, population, n=0.2):
        induhviduals = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        return induhviduals[0: int(len(population)*n)]


    def getPopulationFitness(self, population=None):
        if population is None:
            population = self.individuals

        return  sum([chromosome.fitness_score for chromosome in population])



