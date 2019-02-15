#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

if __name__ == '__main__':
    coordinate = np.load('coordinate.npy')
    distance = np.load('distance.npy')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coordinate[:,0], coordinate[:,1], coordinate[:,2], c=distance)
    plt.colorbar(sc)
    plt.show()
