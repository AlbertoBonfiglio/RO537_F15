#!/usr/bin/python3
from math import exp

def softmaxnaive(self, values):
    div = 0
    for i in range(len(values)):
        div = div + exp(values[i])
    result = [0 for i in range(len(values))]
    for i in range(len(values)):
        result[i] = exp(values[i]) / div
    return result


  def softmax(self, values):
    m = max(values)
    scale = 0
    for i in range(len(values)):
        scale = scale + (exp(values[i] - m))
    result = [0 for i in range(len(values))]
    for i in range(len(values)):
        result[i] = exp(values[i] - m) / scale
    return result