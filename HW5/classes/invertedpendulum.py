#!/usr/bin/python3

from math import sin, cos, pi, degrees
from scipy import arange

g = 9.8 #gravity acceleration constant (m/s**2)


class State(object):
    def __init__(self, x = 0, xdot = 0, x2dot = 0, theta = 0, thetadot = 0, theta2dot = 0):
        self.x = x
        self.xdot = xdot
        self.x2dot = x2dot
        self.theta = theta
        self.thetadot = thetadot
        self.theta2dot = theta2dot




# eq1 --> (M+m)x2dot - ml sinΘ Θdot^2 + ml cosΘ Θ2dot = u
#         (M+m)x2dot = u +  (ml sinΘ Θdot^2) - (ml cosΘ Θ2dot)
#         x2dot = (u +  (ml sinΘ Θdot^2) - (ml cosΘ Θ2dot)) / (M+m)

# eq2 --> m x2dot cosΘ + ml Θ2dot = mg sinΘ
#         l Θ2dot = g sinΘ - x2dot cosΘ +
#         Θ2dot = ( g sinΘ - x2dot cosΘ ) / l

class InvertedPendulum (object):

    def __init__(self, M=10, m=1, l=1,  verbose=False):
        self.M = M   # kg mass of cart
        self.m = m   # kg mass of pendulum
        self.l = l   # lenght of pendulum arm

        self.state = State()

        self.verbose = verbose



    def __getLinearAccelleration(self, u=1, theta=0, thetadot=0, theta2dot=0):
        mass = (self.M + self.m)
        ml = self.m * self.l
        numerator = u + (ml * sin(theta) * (thetadot**2)) - (ml * cos(theta) * theta2dot)
        x2dot = numerator / mass
        return x2dot


    def __getAngularAccelleration(self, x2dot=0, theta=0):
        Θ2dot = ((g * sin(theta)) - (x2dot * cos(theta))) / self.l
        return Θ2dot


    def __getLinearAccelleration2(self, u=1, theta=0, thetadot=0):
        # Calculates the linear acceleration
        a = u + \
            (self.m * self.l * sin(theta) * thetadot**2) - \
            (self.m * g * cos(theta) * sin(theta))
        b = self.M + self.m - (self.m * cos(theta)**2)

        x2dot = a/b
        return x2dot


    def __getAngularAccelleration2(self, u=1, theta=0, thetadot=0):
        M = self.M + self.m
        a = (u * cos(theta)) - (M * g * sin(theta)) + (self.m * self.l * (cos(theta) * sin(theta)) * thetadot)
        b = (self.m * self.l * (cos(theta)**2)) - (M * self.l)

        Θ2dot = a / b
        return Θ2dot


    def __getInstantaneousVelocity(self, xdot, x2dot, seconds):
        try:
            return xdot + (x2dot * seconds)
        except ZeroDivisionError:
            return xdot


    def __getInstAngularVelocity(self, thetadot, theta2dot, seconds):
        try:
            return thetadot + (theta2dot * seconds)
        except ZeroDivisionError:
            return thetadot


    def __getLinearVelocity(self, xdot, x2dot, t):
        try:
            return xdot + (x2dot * t)
        except ZeroDivisionError:
            return xdot


    def __getAngularVelocity(self, thetadot, theta2dot, t):
        try:
            return thetadot + (theta2dot * t)
        except ZeroDivisionError:
            return thetadot



    def apply_force(self, u=1, initialstate=None, threshold=(-(pi/2), (3*pi/2)), tmax=10, timeslice=0.001):
        try:
            stateArray = []

            if initialstate is None: initialstate = State()

            x = initialstate.x
            xdot = initialstate.xdot
            x2dot = initialstate.x2dot
            theta = initialstate.theta
            thetadot = initialstate.thetadot
            theta2dot = initialstate.theta2dot

            for t in arange(timeslice, tmax, timeslice):
                x2dot = self.__getLinearAccelleration(u, theta, thetadot, theta2dot)
                theta2dot = self.__getAngularAccelleration(x2dot, theta)

                xdot = self.__getInstantaneousVelocity(xdot, x2dot, tmax)
                thetadot = self.__getInstAngularVelocity(thetadot, theta2dot, tmax)

                x = x + (xdot * timeslice)
                theta = theta + (thetadot * timeslice)

                s = State(x, xdot, x2dot, theta, thetadot, theta2dot)
                stateArray.append(s)

                u = 0 # after first pass the force is 0 as it is an impulse

                #limits to 180 degrees excursion
                if theta <= threshold[0] or theta >= threshold[1]:
                    break

            return stateArray, t

        except Exception as ex:
            print(ex)







#endregion



    def time_to_ground(self, u=1, initialstate=None, threshold=(-(pi/2), (3*pi/2)), tmax=10, timeslice=0.001):
        state, time = self.apply_force( u, initialstate, threshold, tmax, timeslice)

        return state, time








