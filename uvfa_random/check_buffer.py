import numpy as np
from matplotlib import pyplot
episode = np.load('episode.npy')
total_reward = np.load('total_reward.npy')
loss_q = np.load('loss_q.npy')
latest_buffer = np.load('latest_buffer.npy')
goal_list = np.load('goal_list.npy')
print(goal_list)
print(total_reward)
pyplot.plot(episode, total_reward)
pyplot.show()
pyplot.plot(episode, loss_q)
pyplot.show()