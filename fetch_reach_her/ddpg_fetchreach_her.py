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
threshold = 0.01
timestep_limit = 300
replay_num = 4
optimazation_num = 40
goal1 = np.array([0.3, 0.2, 0.15])
goal2 = np.array([-0.2, -0.15, 0.35])
goal3 = np.array([0.1, -0.4, 0.1])
use_her = True

class ddpg_fetchreach:
    def __init__(self):
        self.agent = Agent(10, 7, action_high, action_low)
        self.buffer = Buffer(10000)
        self.env = Enviroment()
        self.restriction = 0.0
        self.goal = goal1
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
        action = self.agent.choose_action(inputs, self.restriction)
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
            state = transition[0]
            action = transition[1]
            next_state = transition[2]
            reward = transition[3]
            done = transition[4]
            inputs = np.concatenate([state, self.goal])
            next_inputs = np.concatenate([next_state, self.goal])
            self.buffer.store([inputs, action, next_inputs, reward, done])
            self.goal_list.append(self.goal)
            if use_her:
                for k in range(replay_num):
                    # hindsight strategy is future
                    p = np.random.randint(t, len(self.transition_list))
                    pseudo_goal = self.position_list[p]
                    done, reward = self.check_goal(self.position_list[t], pseudo_goal)
                    inputs = np.concatenate([state, pseudo_goal])
                    next_inputs = np.concatenate([next_state, pseudo_goal])
                    self.buffer.store([inputs, action, next_inputs, reward, done])
                    self.goal_list.append(pseudo_goal)
        qloss = 0.0
        for _ in range(optimazation_num):
            loss = self.agent.train(self.buffer.transitions)
            qloss += loss
        return qloss / optimazation_num

    def _reset(self, episode):
        self.done = 0
        self.total_reward = 0.0
        self.restriction += 0.33
        if episode%3 == 0:
            self.goal = goal1
        elif episode%3==1:
            self.goal = goal2
        else:
            self.goal = goal3
        print(self.goal)
        self.env._reset()
        self.position_list = []
        self.transition_list = []

    def set_goal(self):
        goal = np.random.uniform(low_range, high_range)
        #confirm goal whether in work range
        while np.sum(goal**2) > 0.25:
            goal = np.random.uniform(low_range, high_range)
        return goal

    def check_goal(self, position, goal):
        distance = math.sqrt(np.sum((goal-position)**2))
        if distance <= threshold:
            done = 1
            reward = 0
        else:
            done = 0
            reward = -1
        return done, reward

    def save(self, str):
        self.agent.save(str)

def main():
    df = ddpg_fetchreach()
    episode = []
    total_reward = []
    loss_q = []
    rospy.Rate(0.08).sleep()
    for num_episode in range(600):
        print("episode: %d" % num_episode)
        df._reset(num_episode)
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
                np.save('latest_buffer', df.buffer.transitions)
                np.save('goal_list', df.goal_list)
                break

if __name__ == '__main__':
    try:
        if not rospy.is_shutdown():
            main()
            #rospy.spin()
    except rospy.ROSInterruptException:
        pass
