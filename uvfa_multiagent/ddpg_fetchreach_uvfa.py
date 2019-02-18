#!/usr/bin/env python
# coding: utf-8

#fetchreach for one fixedgoal by uvfa

import numpy as np
import random
import math
from matplotlib import pyplot
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from ddpg import Agent, Buffer
from enviroment import Enviroment, action0_high, action0_low, action1_high, action1_low, action2_high, action2_low, action3_high, action3_low, action4_high, action4_low, action5_high, action5_low, action6_high, action6_low

high_range = np.array([0.5, 0.5, 0.5])
low_range = np.array([-0.5, -0.5, 0.0])
threshold = 0.01
timestep_limit = 10
replay_num = 4
optimazation_num = 40
z = 0.15

class MultiAgent:
    def __init__(self):
        self.agent0 = Agent(10, 1, action0_high, action0_low)
        self.agent1 = Agent(10, 1, action1_high, action1_low)
        self.agent2 = Agent(10, 1, action2_high, action2_low)
        self.agent3 = Agent(10, 1, action3_high, action3_low)
        self.agent4 = Agent(10, 1, action4_high, action4_low)
        self.agent5 = Agent(10, 1, action5_high, action5_low)
        self.agent6 = Agent(10, 1, action6_high, action6_low)
        self.agent_list = []
        self.buffer0 = Buffer(10000)
        self.buffer1 = Buffer(10000)
        self.buffer2 = Buffer(10000)
        self.buffer3 = Buffer(10000)
        self.buffer4 = Buffer(10000)
        self.buffer5 = Buffer(10000)
        self.buffer6 = Buffer(10000)
        self.buffer_list = []

    def choose_action(self, inputs, restriction):
        action = []
        for agent in self.agent_list:
            action.append(agent.choose_action(inputs, restriction))
        return action

    def store(self, transition, goal):
        actions = transition[1]
        state = transition[0]
        next_state = transition[2]
        reward = transition[3]
        done = transition[4]
        inputs = np.concatenate([state, goal])
        next_inputs = np.concatenate([next_state, goal])
        for i, _buffer in enumerate(self.buffer_list):
            action = actions[i]
            _buffer.store([inputs, action, next_inputs, reward, done])

    def train(self):
        for agent, _buffer in zip(self.agent_list, self.buffer_list):
            agent.train(_buffer.transitions)

class ddpg_fetchreach:
    def __init__(self):
        self.multiagent = MultiAgent()
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
        inputs = np.concatenate([state, self.goal])
        action = self.multiagent.choose_action(inputs, self.restriction)
        self.env._step(action)
        next_state = self.state
        position = self.position
        #next_inputs = np.concatenate([next_state, self.goal])
        position = np.array([position.x, position.y, position.z])
        self.position_list.append(position)
        done, reward = self.check_goal(position, self.goal)
        self.done = done
        self.total_reward += reward
        transition = [state, action, next_state, reward, done]
        self.transition_list.append(transition)

    def train(self):
        for t in range(len(self.transition_list)):
            transition = self.transition_list[t]
            self.multiagent.store(transition, self.goal)
        for _ in range(optimazation_num):
            self.multiagent.train()

    def _reset(self):
        self.done = 0
        self.total_reward = 0.0
        self.restriction += 0.025
        self.goal = self.set_goal()
        print(self.goal)
        self.goal_list.append(self.goal)
        self.env._reset()
        self.position_list = []
        self.transition_list = []

    def set_goal(self):
        x = np.random.uniform(0.0, 0.5)
        y = np.random.uniform(0.0, 0.5)
        goal = np.array([x, y, z])
        #confirm goal whether in work range
        while np.sum(goal**2) > 0.25:
            x = np.random.uniform(0.0, 0.5)
            y = np.random.uniform(0.0, 0.5)
            goal = np.array([x, y, z])
        return goal

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
    rospy.Rate(0.08).sleep()
    for num_episode in range(6000):
        print("episode: %d" % num_episode)
        df._reset()
        episode.append(num_episode)
        for num_timesteps in range(timestep_limit):
            print("time_steps: %d" % num_timesteps)
            df.take_step()
            if (df.done or num_timesteps == timestep_limit-1):
                df.train()
                print("Episode finish---time steps: %d" % num_timesteps)
                print("total reward: %d" % df.total_reward)
                total_reward.append(df.total_reward)
                np.save('episode', episode)
                np.save('total_reward', total_reward)
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
