#!/usr/bin/python3

from random import uniform, randint
from HW5.classes.invertedpendulum import InvertedPendulum, State
from HW5.classes.NeuralNetwork import NEvoNetwork

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
            func = self.time_to_ground

        #this should calculate a force to apply
        self.NN.set_weights(self.alleles)
        output = self.NN.get_outputs(self.inputs)[0] * 1000


        #Now we apply the force and see how it pefrorms to give a fitness
        self.fitness_score = func(output)
        #print("force = {0}, score = {1}".format(output, self.fitness_score))


    def time_to_ground(self, force=0):
    #in this instance we calculate fitness based on how long the pendulum stays up
    #returns milliseconds
        #TODO perform calulatoin
        states, time = self.pendulum.time_to_ground(u=force, initialstate=self.state)

        return time


    def time_to_threshold(self):
    #in this instance we calculate fitness based on how long the pendulum stays within
    # plus or minus the threshold
    #returns milliseconds
        #TODO perform calulation
        #Basically we count how much time the pendulum stays within the threshold

        return 0





class Population (object):


    def __init__(self, pendulum, NN, state=None, fitness_func=None, size=100):
        if NN is None: raise Exception('NN not initialized')
        if pendulum is None: raise Exception('Pendulum not initialized')
        self.size = size
        self.genome = len(NN.get_weights())
        self.crossover_rate = 0.3
        self.mutation_rate = 0.3
        self.pendulum = pendulum
        self.state = state
        self.NN = NN
        self.individuals = []
        self.fitness_function = fitness_func
        if state is not None: self.create(state, size, fitness_func)


    def create(self, state, size=100, fitness_func=None):
        self.fitness_function = fitness_func

        induhviduals = []
        for n in range(self.size):
            alleles = [uniform(-1, 1) for n in range(self.genome)]
            self.check_alleles(alleles)
            induhvidual = Individual(alleles, self.pendulum, state, self.NN)
            induhvidual.calculate_fitness(fitness_func)
            induhviduals.append(induhvidual)

        self.individuals = induhviduals
        print('Success')
        return induhviduals


    def evolve(self, epochs:50, fitness_treshold=0.2):
        if len(self.individuals) == 0: raise Exception("Population ont initialized")

        try:
            print('Start evo')
            population = self.individuals

            for e in range (epochs):
                individuals = self.__getFittest(population, fitness_treshold)
                induhviduals = []
                new_population = self.size-len(individuals)

                for n in range(0, new_population, 2):
                    parent1 = self.select(individuals)
                    parent2 = self.select(individuals)

                    child1, child2 = self.crossover([parent1, parent2])

                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    child1.calculate_fitness(self.fitness_function)
                    child1.calculate_fitness(self.fitness_function)

                    induhviduals.append(child1)
                    induhviduals.append(child2)

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
            pivot = randint(0, self.genome)

        alleles0 = parents[0].alleles[0:pivot] + parents[1].alleles[pivot:]
        alleles1 = parents[0].alleles[pivot:] + parents[1].alleles[0:pivot]

        child0 = Individual(alleles0, self.pendulum, state=parents[0].state, NN=self.NN)
        child1 = Individual(alleles1, self.pendulum, state=parents[0].state, NN=self.NN)

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


    def check_alleles(self, alleles):
        for a in alleles:
            if a is None:
                raise Exception('Error setting up alleles')



