#!/usr/bin/env python
# coding: utf-8

#fetchreach for one fixedgoal by uvfa

import numpy as np
import math
from matplotlib import pyplot
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from ddpg import Agent, Buffer
from enviroment import Enviroment, action_high, action_low

high_range = np.array([0.5, 0.5, 0.5])
low_range = np.array([-0.5, -0.5, 0.0])
goal_high = np.array([0.337, 0.337, 0.15])
goal_low = np.array([-0.337, -0.337, 0.15])
threshold = 0.01
timestep_limit = 10
replay_num = 4
optimazation_num = 40
use_her = False

class ddpg_fetchreach:
    def __init__(self):
        self.agent = Agent(7, 7, 3, action_high, action_low, goal_high, goal_low)
        self.buffer = Buffer(10000)
        self.env = Enviroment()
        self.restriction = 2.0
        self.goal_list = []
        rospy.Subscriber('/joint_states', JointState, self.get_state)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.get_position)

    def get_state(self, state):
        self.state = state.position

    def get_position(self, link_states):
        self.position = link_states.pose[-1].position

    def take_step(self):
        state = self.state
        action = self.agent.choose_action(state, self.goal, self.restriction)
        self.env._step(action)
        next_state = self.state
        position = self.position
        position = np.array([position.x, position.y, position.z])
        done, reward = self.check_goal(position, self.goal)
        self.done = done
        self.total_reward += reward
        self.buffer.store([state, self.goal, action, next_state, reward, done, position])

    def train(self):
        qloss = 0.0
        for _ in range(optimazation_num):
            loss = self.agent.train(self.buffer.transitions)
            qloss += loss
        return qloss / optimazation_num

    def _reset(self):
        self.done = 0
        self.total_reward = 0.0
        self.restriction += 0.05
        self.env._reset()
        self.goal = self.agent.choose_goal(self.state)
        print(self.goal)
        self.goal_list.append(self.goal)

    def check_goal(self, position, goal):
        distance = math.sqrt(np.sum((goal-position)**2))
        if distance <= threshold:
            done = 1
        else:
            done = 0
        return done, -100.0 * distance

    def save(self, str):
        self.agent.save(str)

def main():
    df = ddpg_fetchreach()
    episode = []
    total_reward = []
    loss_q = []
    rospy.Rate(0.08).sleep()
    for num_episode in range(2000):
        print("episode: %d" % num_episode)
        df._reset()
        episode.append(num_episode)
        for num_timesteps in range(timestep_limit):
            print("time_steps: %d" % num_timesteps)
            df.take_step()
            if (df.done or num_timesteps == timestep_limit-1):
                qloss = df.train()
                loss_q.append(qloss)
                print("Episode finish---time steps: %d" % num_timesteps)
                print("total reward: %d" % df.total_reward)
                total_reward.append(df.total_reward)
                np.save('episode', episode)
                np.save('total_reward', total_reward)
                np.save('loss_q', loss_q)
                df.save('latest_params')
                np.save('goal_list', df.goal_list)
                np.save('latest_buffer', df.buffer.transitions)
                break

if __name__ == '__main__':
    try:
        if not rospy.is_shutdown():
            main()
            #rospy.spin()
    except rospy.ROSInterruptException:
        pass
