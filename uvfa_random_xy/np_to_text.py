#!/usr/bin/env python
# coding: utf-8
import numpy as np

if __name__ == '__main__':
    coordinate = np.load('coordinate.npy')
    x = coordinate[:,0]
    x = x[:, np.newaxis]
    y = coordinate[:,1]
    y = y[:, np.newaxis]
    coordinate = np.concatenate([x, y], axis = 1)
    distance = np.load('distance.npy')
    distance = distance[:, np.newaxis]
    data = np.concatenate([coordinate, distance], axis = 1)
    np.savetxt('data.csv', data)
