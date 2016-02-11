#!/usr/bin/python3

from HW5.classes.invertedpendulum import InvertedPendulum

class Controller (object):

    def __init__(self):
        self.running_flag = False
        self.pendulum = InvertedPendulum()


    def isRunning(self):
        return self.running_flag

    def start(self, impulse=1, freq=0.01):
        if self.running_flag: raise Exception('Already running')

        self.frequency = freq
        self.running_flag = True
        self.pendulum.applyforce(u=impulse)


    def stop(self):
        self.running_flag = False
