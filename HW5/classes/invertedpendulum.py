#!/usr/bin/python3

from math import sin, cos, pi, degrees, radians
from scipy import arange

g = 9.81 #gravity acceleration constant (m/s**2)


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


    def apply_force(self, u=1, initialstate=None, threshold=(-(pi/2), (pi/2)), tmax=10, timeslice=0.001):
        try:
            stateArray = []
            impulse  = u

            if initialstate is None: initialstate = State()

            x = initialstate.x
            xdot = initialstate.xdot
            x2dot = initialstate.x2dot
            theta = initialstate.theta
            thetadot = initialstate.thetadot
            theta2dot = initialstate.theta2dot

            #x2dot = self.__getLinearAccelleration(u, theta, thetadot, theta2dot)
            #theta2dot = self.__getAngularAccelleration(x2dot, theta)

            n=0
            for t in arange(0, tmax, timeslice):
                x2dot = self.__getLinearAccelleration(u, theta, thetadot, theta2dot)
                theta2dot = self.__getAngularAccelleration(x2dot, theta)

                xdot = self.__getLinearVelocity(xdot, x2dot, timeslice)
                thetadot = self.__getAngularVelocity(thetadot, theta2dot, timeslice)


                x = x + (xdot * timeslice)
                theta = theta + (thetadot * timeslice)

                s = State(x, xdot, x2dot, theta, thetadot, theta2dot)
                stateArray.append(s)

                #u = 0 # after first pass the force is 0 as it is an impulse

                n += 1
                #limits to n degrees excursion
                if impulse >=0:
                    if theta <= threshold[0]:
                        break
                elif impulse < 0:
                    if theta >= threshold[1]:
                        break

            return stateArray, timeslice * n

        except Exception as ex:
            print(ex)


    def apply_force2(self, u=1, initialstate=None, threshold=(-(pi/2), (pi/2)), tmax=5, timeslice=0.001):
        try:
            stateArray = []
            impulse  = u

            if initialstate is None: initialstate = State()

            x = initialstate.x
            xdot = initialstate.xdot
            x2dot = initialstate.x2dot
            theta = initialstate.theta
            thetadot = initialstate.thetadot
            theta2dot = initialstate.theta2dot

            n=0
            for t in arange(0, tmax, timeslice):
                if theta >= -(pi/2) and theta <= (pi/2):
                    x2dot = self.__getLinearAccelleration(u, theta, thetadot, theta2dot)
                    theta2dot = self.__getAngularAccelleration(x2dot, theta)

                    xdot = self.__getLinearVelocity(xdot, x2dot, timeslice)
                    thetadot = self.__getAngularVelocity(thetadot, theta2dot, timeslice)

                    x = x + (xdot * timeslice)
                    theta = theta + (thetadot * timeslice)

                    s = State(x, xdot, x2dot, theta, thetadot, theta2dot)
                    stateArray.append(s)

                    #u = 0 # after first pass the force is 0 as it is an impulse

                    #limits to n degrees excursion
                    if theta >= threshold[0] and theta <= threshold[1]:
                        n += 1

                    #if it exceeds the boundaries stop iterating
                    if n > 0 and (theta < threshold[0] or theta > threshold[1]):
                        #print('threshold exceeded {0}'.format(theta))
                        break

                else:
                    break

            return stateArray, timeslice * n

        except Exception as ex:
            print(ex)


    def apply_force3(self, u=1, initialstate=None, tmax=0.2, timeslice=0.001):
        try:
            if initialstate is None: initialstate = State()

            x = initialstate.x
            xdot = initialstate.xdot
            x2dot = initialstate.x2dot
            theta = initialstate.theta
            thetadot = initialstate.thetadot
            theta2dot = initialstate.theta2dot

            n=0
            for t in arange(0, tmax, timeslice):
                if theta >= -(pi/2) and theta <= (pi/2):
                    x2dot = self.__getLinearAccelleration(u, theta, thetadot, theta2dot)
                    theta2dot = self.__getAngularAccelleration(x2dot, theta)

                    xdot = self.__getLinearVelocity(xdot, x2dot, timeslice)
                    thetadot = self.__getAngularVelocity(thetadot, theta2dot, timeslice)

                    x = x + (xdot * timeslice)
                    theta = theta + (thetadot * timeslice)

                else:
                    break

            return theta

        except Exception as ex:
            print(ex)


    def get_force(self, state=None):
        #Eq = (M+m)x2dot - ml sinΘ Θdot^2 + ml cosΘ Θ2dot = u

        total_mass = self.M + self.m
        ml = self.m * self.l

        retval = (total_mass* state.x2dot) - \
                 (ml * sin(state.theta) *(state.thetadot** 2)) + \
                 (ml * cos(state.theta) * state.theta2dot)

        return retval


    def time_to_ground(self, u=1, initialstate=None, threshold=(-(pi/2), (pi/2)), tmax=10, timeslice=0.001):
        state, time = self.apply_force(u, initialstate, threshold, tmax, timeslice)
        return state, time


    def time_in_threshold(self, u=1, initialstate=None, threshold=(-(pi/2), (pi/2)), tmax=10, timeslice=0.001):
        state, time = self.apply_force2(u, initialstate, threshold, tmax, timeslice)
        return state, time


    def smallest_angle(self, u=1, initialstate=None,  tmax=0.2, timeslice=0.001):
        theta = self.apply_force(u, initialstate, tmax, timeslice)
        return theta

    def get_State(self, u=1, initialstate=None, threshold=(-(pi/2), (pi/2)), tmax=10, timeslice=0.001):
        try:
            stateArray = []
            impulse  = u

            if initialstate is None: initialstate = State()

            x = initialstate.x
            xdot = initialstate.xdot
            x2dot = initialstate.x2dot
            theta = initialstate.theta
            thetadot = initialstate.thetadot
            theta2dot = initialstate.theta2dot

            #x2dot = self.__getLinearAccelleration(u, theta, thetadot, theta2dot)
            #theta2dot = self.__getAngularAccelleration(x2dot, theta)

            n=0
            for t in arange(0, tmax, timeslice):
                x2dot = self.__getLinearAccelleration(u, theta, thetadot, theta2dot)
                theta2dot = self.__getAngularAccelleration(x2dot, theta)

                xdot = self.__getLinearVelocity(xdot, x2dot, timeslice)
                thetadot = self.__getAngularVelocity(thetadot, theta2dot, timeslice)

                x = x + (xdot * timeslice)
                theta = theta + (thetadot * timeslice)

                s = State(x, xdot, x2dot, theta, thetadot, theta2dot)
                stateArray.append(s)

                #u = 0 # after first pass the force is 0 as it is an impulse

                n += 1
                #limits to n degrees excursion
                if impulse >=0:
                    if theta <= threshold[0]:
                        break
                elif impulse < 0:
                    if theta >= threshold[1]:
                        break

            return stateArray, n

        except Exception as ex:
            print(ex)


