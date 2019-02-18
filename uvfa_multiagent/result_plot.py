#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as pyplot

if __name__ == '__main__':
    coordinate = np.load('coordinate.npy')
    coordinate = coordinate[:,2]
    distance = np.load('distance.npy')
    pyplot.rcParams["font.size"] = 15
    pyplot.plot(coordinate, distance)
    pyplot.xlabel('zcoordinate')
    pyplot.ylabel('distance')
    pyplot.tight_layout()
    pyplot.show()
