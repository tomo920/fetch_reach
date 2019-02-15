#!/usr/bin/env python
# coding: utf-8
import numpy as np
from matplotlib import pyplot
episode = np.load('episode.npy')
total_reward = np.load('total_reward.npy')
loss_q = np.load('loss_q.npy')
latest_buffer = np.load('latest_buffer.npy')
pyplot.rcParams["font.size"] = 15
pyplot.plot(episode, total_reward)
pyplot.xlabel('episode')
pyplot.ylabel('total reward')
pyplot.tight_layout()
pyplot.show()
pyplot.plot(episode, loss_q)
pyplot.show()
