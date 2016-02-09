from __future__ import division

import sys, random
import time

from math import sqrt
from pickle import *
from common import DrawHelper


#Individuals
class Individual:
    score = 0
    length = 30
    separator = ' '

    def __init__(self, chromosome = None, length = 30):
        self.length = length
        self.score = 0  # set during evaluation
        self.chromosome = chromosome or self._makechromosome()


    def _makechromosome(self):
        "makes a chromosome from randomly selected alleles."
        chromosome = []
        lst = [i for i in xrange(self.length)]
        for i in xrange(self.length):
            choice = random.choice(lst)
            lst.remove(choice)
            chromosome.append(choice)
        return chromosome


    def evaluate(self, matrix = None, optimum = None):
        self.score = self.get_alleles_score(matrix, self.chromosome)


    def get_alleles_score(self, matrix, sequence):
        #Returns the total length of the alleles """
        total = 0
        num_alleles = len(sequence)
        for i in range(num_alleles):
            j = (i + 1) % num_alleles
            allele_i = sequence[i]
            allele_j = sequence[j]
            total += matrix[allele_i, allele_j]
        return total


    def crossover2(self, other):
        left, right = self._pickpivots()
        p1 = Individual(None, self.length)
        p2 = Individual(None, self.length)

        c1 = [ c for c in self.chromosome \
               if c not in other.chromosome[left:right + 1]]
        p1.chromosome = c1[:left] + other.chromosome[left:right + 1] + c1[left:]

        c2 = [ c for c in other.chromosome \
               if c not in self.chromosome[left:right + 1]]
        p2.chromosome = c2[:left] + self.chromosome[left:right + 1] + c2[left:]

        return p1, p2

    def crossover(self, other):
        left = self._pickpivot()
        right = len(other.chromosome)
        p1 = Individual(None, self.length)
        p2 = Individual(None, self.length)

        c1 = [ c for c in self.chromosome \
               if c not in other.chromosome[left:right + 1]]
        p1.chromosome = c1[:left] + other.chromosome[left:right + 1]

        c2 = [ c for c in other.chromosome \
               if c not in self.chromosome[left:right + 1]]
        p2.chromosome = c2[:left] + self.chromosome[left:right + 1]

        return p1, p2


    def mutate(self):
        "swap two element"
        left, right = self._pickpivots()
        temp = self.chromosome[left]
        self.chromosome[left] = self.chromosome[right]
        self.chromosome[right] = temp

    def _pickpivots(self):
        left = random.randint(0, self.length - 2)
        right = random.randint(left, self.length - 1)
        return left, right

    def _pickpivot(self):
        return random.randint(0, self.length - 1)

    def __repr__(self):
        "returns string representation of self"
        return '<%s chromosome="%s" score=%s>' % \
               (self.__class__.__name__,
                self.separator.join(map(str, self.chromosome)), self.score)

    def copy(self):
        twin = self.__class__(self.chromosome[:])
        twin.score = self.score
        return twin

    def __cmp__(self, other):
        return cmp(self.score, other.score)


class Environment:
    size = 0

    def __init__(self, matrix = None, alleles=None, population = None, size = 100, maxgenerations = 1000,\
                 newindividualrate = 0.15, crossover_rate = 0.9,\
                 mutation_rate = 0.05, file = ''):

        self.matrix = matrix
        self.alleles = alleles
        self.size = size
        self.population = self._makepopulation()
        self.maxgenerations = maxgenerations
        self.newindividualrate = newindividualrate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        for individual in self.population:
            individual.evaluate(self.matrix)
        self.generation = 0
        self.minscore = sys.maxint
        self.minindividual = None
        self.ChangeLog = []
        self.bestOverall = None;
        self.file = file


    def _makepopulation(self):
        return [Individual(None, len(self.alleles)) for i in range(0, self.size)]


    def __getRateOfChange(self):
        if len(self.ChangeLog) < 10: #allows at least 10 iterations
            return 1

        lastSet = self.ChangeLog[-10:]
        x =(lastSet[8] - lastSet[9])
        y = lastSet[9]
        delta =  x / y
        return delta


    def run(self, runCounter):
        # right now we use a max iteration.
        # One option is to put a stop when the improvement in score goes below a pct
        __drawer = DrawHelper()
        __targetLog = None
        __line = ''
        __fileName = 'genetic_run_' + str(runCounter) + '[' + self.file + ']'
        __targetLog = open(__fileName +  '.txt', 'w')

        start = time.time()
        __line = __line + 'Genetic run started at:' + str(start) + '\n'

        rnd = random.random()

        _deltaTreshold = -0.0001
        _loopsWithNoChange = 0
        self._getBestScore()
        i=0

        solutions_count = len(self.population)
        #while (self.__getRateOfChange() ):
        for i in range(1, self.maxgenerations + 1):
            i = i + 1

            print "Generation no:" + str(i)

            _previousScore = self.minindividual.score #use a delta of change to stop looping indefinetly

            crossover = False

            # gets a population of high scores by sorting based on score
            # then removing the percentage of
            from operator import attrgetter
            __sortedlist = sorted(self.population, key = attrgetter('score'), reverse=True)
            __numberofchildren = int(len(self.population) * self.newindividualrate)

            solutions_count = solutions_count + (__numberofchildren *2) # counting generated not actually added

            #drop the least fit
            for unfit in range(0, __numberofchildren):
                dropped = __sortedlist[unfit]
                self.population.remove(dropped)

            #selects parents from the remainin population
            __parentList = random.sample(self.population, __numberofchildren *2)
            __childrenList = []
            for parent in range(0, __numberofchildren):
                __parentA = __parentList[parent]
                __parentB = __parentList[parent + 5]

                child1, child2 = __parentA.crossover(__parentB)
                child1.evaluate(self.matrix)
                child2.evaluate(self.matrix)

                __childrenList.append(child1)
                __childrenList.append(child2)

            #sorts the children based on their fitness
            __sortedchildren = sorted(__childrenList, key = attrgetter('score'))

            #if crossover:
            #    self._crossOver()

            #now mutates a random number of population
            __numberofmutants = int(len(self.population) * self.mutation_rate)
            __mutantList = random.sample(self.population, __numberofmutants)
            for mutants in range(0, __numberofmutants):
                __mutantList[mutants].mutate()
                __mutantList[mutants].evaluate(self.matrix)
            solutions_count = solutions_count + __numberofmutants

            # finally adds the fittest children to the populatiom
            for child in range(0, __numberofchildren):
                self.population.append(__sortedchildren[child])

            #bring in immigrants and grow the population!
            __modulus = i % int(random.randint(100, 200))
            if __modulus == 0:
                _immigrantList = [Individual(None, len(self.alleles)) for imm in range(0, random.randint(0, 4))]
                for immigrant in range (0, len(_immigrantList)):
                    self.population.append(_immigrantList[immigrant])
                solutions_count = solutions_count + len(_immigrantList)

            # shuffles the population
            random.shuffle(self.population)

            self._getBestScore()
            if len(self.ChangeLog) == 0:
                self.ChangeLog.append(self.minindividual.score)

            elif (self.ChangeLog[len(self.ChangeLog)-1 ] <> self.minindividual.score): # do not append oif no change
                self.ChangeLog.append(self.minindividual.score)

            if (self.bestOverall is None):
                self.bestOverall = self.minindividual.copy()
            else:
                if not (self.minindividual.score >= self.bestOverall.score):
                    self.bestOverall = self.minindividual.copy()

            __line = __line +  str(self.minindividual.score) + '\n'

            #print "current = ", self.minindividual.score
            #print "overall  = ", self.bestOverall.score
            #print "population = ", len(self.population)
        #end loop


        __line = __line + 'Solutions Count:' + str(solutions_count) + '\n'
        __line = __line + 'Best score:' + str(self.bestOverall.score) + '\n'

        __line =__line +  'Best tour:' + str(self.bestOverall) + '\n'

        __line = __line + 'Final Population:' + str(len(self.population)) + '\n'

        __line = __line + 'Genetic run time:' + str(time.time() - start) + '\n'

        __targetLog.write(__line)
        __targetLog.close()
        __drawer.write_tour_to_img1(self.alleles, self.bestOverall.chromosome, '%s: %f'%(self.file, self.bestOverall.score), file(__fileName + '.png','w'))

        print self.minindividual

        # end of Run() **********

    def _getBestScore(self):
        for j in range(0, len(self.population)): # finds the best individual (could be optimized but )
            self.population[j].evaluate(self.matrix)
            curscore = self.population[j].score
            if curscore < self.minscore:
                self.minscore = curscore
                self.minindividual = self.population[j]


    def _getDelta(self, score, previous):
        return ((score - previous) / previous) * (-1)


    def _crossOver(self):
        rnd = random.random()
        if rnd < self.crossover_rate:
            children = []
            newindividual = int(self.newindividualrate * self.size / 2)
            for i in range(0, newindividual):
                selected1 = self._selectrank()
                selected2 = self._selectrank()
                parent1 = self.population[selected1]
                parent2 = self.population[selected2]
                child1, child2 = parent1.crossover(parent2)
                child1.evaluate(self.matrix)
                child2.evaluate(self.matrix)
                children.append(child1)
                children.append(child2)

            totalreplaced = 0
            totalscore = 0
            for k in range(0, self.size):
                totalscore += self.population[k].score

            for i in range(0, newindividual):
                # Loops through all the population and replce with child
                # if the parent score is above the randomly generated score
                randscore = random.random()
                addscore = 0
                for j in range(0, self.size):
                    addscore += (self.population[j].score / totalscore)
                    if addscore >= randscore:
                        #if self.population[j].score > children[i].score :
                        self.population[j] = children[i]
                        totalscore = totalscore - self.population[j].score + children[i].score
                        totalreplaced = totalreplaced +1
                        break

            print "Total individuals replaced ", totalreplaced, newindividual


    def _select(self):
        totalscore = 0
        for i in range(0, self.size):
            totalscore += self.population[i].score

        randscore = random.random() * (self.size - 1)
        addscore = 0
        selected = 0
        for i in range(0, self.size):
            addscore += (1 - self.population[i].score / totalscore)
            if addscore >= randscore:
                selected = i
                break

        return selected


    def _selectrank(self, choosebest = 0.9):
        self.population.sort()
        if random.random() < choosebest:
            return random.randint(0, self.size * self.newindividualrate)
        else:
            return random.randint(self.size * self.newindividualrate, self.size - 1)


    def _printpopulation(self):
        for i in range(0, self.size):
            print "Individual ", i, self.population[i]