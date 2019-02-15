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
timestep_limit = 200
replay_num = 4
optimazation_num = 40
goal1 = [0.3, 0.2, 0.15]
goal2 = [0.3, -0.3, 0.15]
goal_input1 = np.concatenate([goal1, [0.0, 0.0, 0.0]])
goal_input2 = np.concatenate([[0.0, 0.0, 0.0], goal2])

class ddpg_fetchreach:
    def __init__(self):
        self.agent = Agent(13, 7, action_high, action_low)
        self.buffer1 = Buffer(10000) #for goal1
        self.buffer2 = Buffer(10000) #for goal2
        self.env = Enviroment()
        self.restriction = 0.0
        self.goal = goal1
        self.goal_input = goal_input1
        self.goalnum = 1
        rospy.Subscriber('/joint_states', JointState, self.get_state)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.get_position)

    def get_state(self, state):
        self.state = state.position

    def get_position(self, link_states):
        self.position = link_states.pose[-1].position

    def take_step(self):
        state = self.state
        inputs = np.concatenate([state, self.goal_input])
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
            inputs = np.concatenate([state, self.goal_input])
            next_inputs = np.concatenate([next_state, self.goal_input])
            if self.goalnum == 1:
                self.buffer1.store([inputs, action, next_inputs, reward, done])
            else:
                self.buffer2.store([inputs, action, next_inputs, reward, done])
        qloss = 0.0
        for _ in range(optimazation_num):
            loss = self.agent.train(self.buffer1.transitions, self.buffer2.transitions)
            qloss += loss
        if self.goalnum == 1:
            self.goalnum = 2
        else:
            self.goalnum = 1
        return qloss / optimazation_num

    def _reset(self):
        self.done = 0
        self.total_reward = 0.0
        self.restriction += 0.5
        print(self.goalnum)
        if self.goalnum == 1:
            self.goal = goal1
            self.goal_input = goal_input1
        else:
            self.goal = goal2
            self.goal_input = goal_input2
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
    for num_episode in range(200):
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
                np.save('latest_buffer1', df.buffer1.transitions)
                np.save('latest_buffer2', df.buffer2.transitions)
                break

if __name__ == '__main__':
    try:
        if not rospy.is_shutdown():
            main()
            #rospy.spin()
    except rospy.ROSInterruptException:
        pass
