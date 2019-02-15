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

threshold = 1e-4
timestep_limit = 100
goal1 = np.array([0.3, 0.2, 0.15])
goal2 = np.array([-0.2, -0.15, 0.35])
goal3 = np.array([0.1, -0.4, 0.1])

class ddpg_fetchreach_test:
    def __init__(self):
        params = np.load(sys.argv[1])
        self.coordinate_list = []
        self.distance_list = []
        params = params[1]
        w1 = params[0]
        b1 = params[1]
        w2 = params[2]
        b2 = params[3]
        w3 = params[4]
        b3 = params[5]
        self.actor = Actor(w1, b1, w2, b2, w3, b3, actor_learning_rate, action_high, action_low)
        self.env = Enviroment()
        rospy.Subscriber('/joint_states', JointState, self.get_state)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.get_position)

    def get_state(self, state):
        self.state = state.position

    def get_position(self, link_states):
        self.position = link_states.pose[-1].position

    def step(self):
        state = self.state
        inputs = np.concatenate([state, self.goal])
        action = self.actor.action(inputs)
        self.env._step(action)
        position = self.position
        position = np.array([position.x, position.y, position.z])
        return math.sqrt(np.sum((position-self.goal)**2))

    def _reset(self, goal):
        self.goal = goal
        self.done = 0
        self.env._reset()

    def check_goal(self, goal):
        if np.sum(goal**2) <= 0.25 and goal[2] > 0:
            return 1
        else:
            return 0

    def _test(self, testgoal):
        x = testgoal[0]
        y = testgoal[1]
        z = testgoal[2]
        #change x
        while(1):
            x -= 0.02
            goal = np.array([x, y, z])
            if self.check_goal(goal) == 0:
                break
            if math.sqrt(np.sum((goal-testgoal)**2)) > 0.5:
                break
            self._reset(goal)
            print(goal)
            self.coordinate_list.append(goal)
            before_distance = 100
            for i in range(100):
                distance = self.step()
                if abs(before_distance - distance) <= threshold:
                    break
                before_distance = distance
            self.distance_list.append(distance)
        x = testgoal[0]
        while(1):
            x += 0.02
            goal = np.array([x, y, z])
            if self.check_goal(goal) == 0:
                break
            if math.sqrt(np.sum((goal-testgoal)**2)) > 0.5:
                break
            self._reset(goal)
            print(goal)
            self.coordinate_list.append(goal)
            before_distance = 100
            for i in range(100):
                distance = self.step()
                if abs(before_distance - distance) <= threshold:
                    break
                before_distance = distance
            self.distance_list.append(distance)
        x = testgoal[0]
        #change y
        while(1):
            y -= 0.02
            goal = np.array([x, y, z])
            if self.check_goal(goal) == 0:
                break
            if math.sqrt(np.sum((goal-testgoal)**2)) > 0.5:
                break
            self._reset(goal)
            print(goal)
            self.coordinate_list.append(goal)
            before_distance = 100
            for i in range(100):
                distance = self.step()
                if abs(before_distance - distance) <= threshold:
                    break
                before_distance = distance
            self.distance_list.append(distance)
        y = testgoal[1]
        while(1):
            y += 0.02
            goal = np.array([x, y, z])
            if self.check_goal(goal) == 0:
                break
            if math.sqrt(np.sum((goal-testgoal)**2)) > 0.5:
                break
            self._reset(goal)
            print(goal)
            self.coordinate_list.append(goal)
            before_distance = 100
            for i in range(100):
                distance = self.step()
                if abs(before_distance - distance) <= threshold:
                    break
                before_distance = distance
            self.distance_list.append(distance)
        y = testgoal[1]
        #change z
        while(1):
            z -= 0.02
            goal = np.array([x, y, z])
            if self.check_goal(goal) == 0:
                break
            if math.sqrt(np.sum((goal-testgoal)**2)) > 0.5:
                break
            self._reset(goal)
            print(goal)
            self.coordinate_list.append(goal)
            before_distance = 100
            for i in range(100):
                distance = self.step()
                if abs(before_distance - distance) <= threshold:
                    break
                before_distance = distance
            self.distance_list.append(distance)
        z = testgoal[2]
        while(1):
            z += 0.02
            goal = np.array([x, y, z])
            if self.check_goal(goal) == 0:
                break
            if math.sqrt(np.sum((goal-testgoal)**2)) > 0.5:
                break
            self._reset(goal)
            print(goal)
            self.coordinate_list.append(goal)
            before_distance = 100
            for i in range(100):
                distance = self.step()
                if abs(before_distance - distance) <= threshold:
                    break
                before_distance = distance
            self.distance_list.append(distance)

    def threegoal_test(self):
        for goal in [goal1, goal2, goal3]:
            self._reset(goal)
            print(goal)
            self.coordinate_list.append(goal)
            before_distance = 100
            for i in range(100):
                distance = self.step()
                if abs(before_distance - distance) <= threshold:
                    break
                before_distance = distance
            self.distance_list.append(distance)
            self._test(goal)

def main():
    df = ddpg_fetchreach_test()
    rospy.Rate(0.05).sleep()
    df.threegoal_test()
    np.save('coordinate', df.coordinate_list)
    np.save('distance', df.distance_list)

if __name__ == '__main__':
    try:
        if not rospy.is_shutdown():
            main()
            #rospy.spin()
    except rospy.ROSInterruptException:
        pass
