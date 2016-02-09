#!/usr/bin/python3
from HW5.classes.individual import Individual

class Population (object):


    def __init__(self, genome, M=10, m=1, l=1, size=100):
        self.M = M
        self.m = m
        self.l = l
        self.size = size
        self.genome = genome
        self.individuals = self.create(size)

        print(self.individuals)

    def create(self, size=100):
        return [Individual(self.genome, self.M, self.m, self.l) for n in range(self.size)]


    def evolve(self, epochs:1000):
        pass



