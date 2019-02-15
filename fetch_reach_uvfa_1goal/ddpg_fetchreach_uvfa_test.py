#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import math
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from ddpg import Actor, actor_learning_rate
from enviroment import Enviroment, action_high, action_low

threshold = 0.01
timestep_limit = 100

class ddpg_fetchreach_test:
    def __init__(self):
        params = np.load(sys.argv[1])
        params = params[1]
        w1 = params[0]
        b1 = params[1]
        w2 = params[2]
        b2 = params[3]
        w3 = params[4]
        b3 = params[5]
        self.actor = Actor(w1, b1, w2, b2, w3, b3, actor_learning_rate, action_high, action_low)
        self.env = Enviroment()
        self.goal = np.array([0.3, 0.2, 0.15])
        rospy.Subscriber('/joint_states', JointState, self.get_state)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.get_position)

    def get_state(self, state):
        self.state = state.position

    def get_position(self, link_states):
        self.position = link_states.pose[-1].position

    def train(self):
        state = self.state
        inputs = np.concatenate([state, self.goal])
        action = self.actor.action(inputs)
        self.env._step(action)
        next_state = self.state
        position = self.position
        position = np.array([position.x, position.y, position.z])
        distance = math.sqrt(np.sum((self.goal-position)**2))
        if distance <= threshold:
            self.done = 1
        reward = -100.0 * distance
        print("reward: %f" % reward)
        done = self.done
        self.total_reward += reward

    def _reset(self):
        self.done = 0
        self.total_reward = 0.0
        self.env._reset()

def main():
    df = ddpg_fetchreach_test()
    episode = []
    total_reward = []
    rospy.Rate(0.08).sleep()
    for num_episode in range(200):
        if not rospy.is_shutdown():
            print("episode: %d" % num_episode)
            episode.append(num_episode)
            df._reset()
            for num_timesteps in range(timestep_limit):
                print("time_steps: %d" % num_timesteps)
                df.train()
                if (df.done or num_timesteps == timestep_limit-1):
                    print("Episode finish---time steps: %d" % num_timesteps)
                    print("total reward: %d" % df.total_reward)
                    total_reward.append(df.total_reward)
                    break

if __name__ == '__main__':
    try:
        if not rospy.is_shutdown():
            main()
            #rospy.spin()
    except rospy.ROSInterruptException:
        pass
