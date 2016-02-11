#!/usr/bin/python3

from random import uniform

from HW5.classes.invertedpendulum import InvertedPendulum


class Individual (object):

    def __init__(self, genome, M=10, m=1, l=1, threshold=0.1):
        self.threshold = threshold
        #there are the physics constants for each individual
        self.M = M
        self.m = m
        self.l = l

        #this is the actual data we are evolving weights for
        self.genome = genome

        #these are the actual weights we are evolving
        self.alleles = [uniform(-10, 10) for n in range(len(genome))]
        self.pendulum = InvertedPendulum(M, m, l)



    def fitness(self, func=None):
        if func == None:
            func = self.time_to_ground()
        return func()


    #in this instance we calculate fitness based on how long
    #the pendulum stays up
    #returns milliseconds
    def time_to_ground(self):
        #TODO perform calulatoin
        time = self.pendulum.applyforce2(u=10)


        return 0

    #in this instance we calculate fitness based on how long
    #the pendulum stays within plus or minus the threshold
    #returns milliseconds
    def time_to_threshold(self):
        #TODO perform calulatoin
        return 0