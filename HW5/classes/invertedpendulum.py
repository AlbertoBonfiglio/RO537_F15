#!/usr/bin/python3

# eq1 --> (M+m)x2dot - ml sinΘ Θdot^2 + ml cosΘ Θ2dot = u
# eq2 --> m x2dot cosΘ + ml Θ2dot = mg sinΘ --> x2d cos0 + l 02dot - g sin0 = 0



from math import sin, cos, pi, degrees
from scipy import arange

g = 9.8 #gravity acceleration constant (m/s**2)


class Forces(object):
    def __init__(self):
        self.x = 0
        self.xdot = 0
        self.x2dot = 0
        self.theta = 0
        self.thetadot = 0
        self.theta2dot = 0



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

        self.x = 0
        self.xdot = 0
        self.x2dot = 0
        self.theta = 0
        self.thetadot = 0
        self.theta2dot = 0

        self.verbose = verbose

#region private methods
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

#endregion

#region public methods


    def applyforce(self, u=1, tmax=10, timeslice=0.01):
        try:
            xArray = []
            thetaArray = []
            forcearray = []

            x = 0
            xdot = 0
            theta = 0
            thetadot = 0

            x2dot = self.__getLinearAccelleration2(u, theta, thetadot)
            theta2dot = self.__getAngularAccelleration(x2dot, theta)

            xdot = self.__getInstantaneousVelocity(xdot, x2dot, tmax)
            thetadot = self.__getInstAngularVelocity(thetadot, theta2dot, tmax)

            for t in arange(timeslice, tmax, timeslice):
                f = Forces()

                x = x + (xdot * timeslice)
                theta = theta + (thetadot * timeslice)

                print('Position -> {0}'.format(x))
                print('Theta -> {0} - {1}'.format(theta, degrees(theta)))

                xArray.append(x)
                thetaArray.append(theta)

                f.x = x
                f.xdot = xdot
                f.x2dot = x2dot
                f.theta = theta
                f.thetadot = thetadot
                f.theta2dot = theta2dot
                forcearray.append(f)

                #limits to 180 degrees excursion
                if theta <= -(pi/2) or theta >= (3*pi/2):
                    break

            return xArray, thetaArray, forcearray
        except Exception as ex:
            print(ex)


    def applyforcea(self, u=1, tmax=10, timeslice=0.01):
        try:
            xArray = []
            thetaArray = []
            forcearray = []

            x = 0
            xdot = 0
            theta = 0
            thetadot = 0
            theta2dot = 0

            time = 0
            for t in arange(timeslice, tmax, timeslice):
                time += 1
                f = Forces()
                x2dot = self.__getLinearAccelleration(u, theta, thetadot, theta2dot)
                theta2dot = self.__getAngularAccelleration(x2dot, theta)

                xdot = self.__getInstantaneousVelocity(xdot, x2dot, tmax)
                thetadot = self.__getInstAngularVelocity(thetadot, theta2dot, tmax)

                x = x + (xdot * timeslice)
                theta = theta + (thetadot * timeslice)

                #print('Position -> {0}'.format(x))
                #print('Theta -> {0} - {1}'.format(theta, degrees(theta)))

                xArray.append(x)
                thetaArray.append(theta)

                f.x = x
                f.xdot = xdot
                f.x2dot = x2dot
                f.theta = theta
                f.thetadot = thetadot
                f.theta2dot = theta2dot
                forcearray.append(f)

                u = 0 # after first pass the force is 0 as it is an impulse

                #limits to 180 degrees excursion
                if theta <= -(pi/2) or theta >= (3*pi/2):
                    break

            print('it took {0} of {1} timeslices'.format(time, timeslice))
            return xArray, thetaArray, forcearray

        except Exception as ex:
            print(ex)


    #Returns the time it takes to hit the ground
    def applyforce2(self, u=1, x=0, xdot=0, x2dot=0, theta=0, thetadot=0, theta2dot=0, tmax=10, timeslice=0.01):
        try:
            _x = x
            _xdot = xdot
            _theta = theta
            _thetadot = thetadot

            _x2dot = self.__getLinearAccelleration(u, _theta, _thetadot)
            _theta2dot = self.__getAngularAccelleration(_x2dot, _theta)

            #_xdot = self.__getInstantaneousVelocity(_xdot, _x2dot, tmax)
            #_thetadot = self.__getInstAngularVelocity(_thetadot, _theta2dot, tmax)

            for t in arange(timeslice, tmax, timeslice):
                _xdot += (_x2dot * timeslice)
                _x = _x + (_xdot * timeslice)

                _thetadot += (_theta2dot * timeslice)
                _theta = _theta + (_thetadot * timeslice)

                if self.verbose == True:
                    print('Position -> {0}'.format(_x))
                    print('Theta -> {0} - {1}'.format(theta, degrees(_theta)))

                #limits to 180 degrees excursion
                if _theta <= -(pi/2) or _theta >= (3*pi/2):
                    break

            return t
        except Exception as ex:
            print(ex)

#endregion


#region 'Deprecated Functions'


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


#endregion
