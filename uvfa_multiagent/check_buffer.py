import numpy as np
from matplotlib import pyplot
episode = np.load('episode.npy')
total_reward = np.load('total_reward.npy')
goal_list = np.load('goal_list.npy')
print(goal_list)
print(total_reward)
pyplot.plot(episode, total_reward)
pyplot.show()
