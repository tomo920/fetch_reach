#!/usr/bin/env python
# coding: utf-8
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

action0_high = 2.97
action0_low = -2.97
action1_high = 1.57
action1_low = -1.57
action2_high = 1.57
action2_low = -1.57
action3_high = 0.04
action3_low = -2.81
action4_high = 1.51
action4_low = -2.77
action5_high = 1.57
action5_low = -1.57
action6_high = 2.97
action6_low = -2.96
action_high = np.array([action0_high, action1_high, action2_high, action3_high,
                        action4_high, action5_high, action6_high])
action_low = np.array([action0_low, action1_low, action2_low, action3_low,
                        action4_low, action5_low, action6_low])

joint_names = ["crane_x7_shoulder_fixed_part_pan_joint",
                "crane_x7_shoulder_revolute_part_tilt_joint",
                "crane_x7_upper_arm_revolute_part_twist_joint",
                "crane_x7_upper_arm_revolute_part_rotate_joint",
                "crane_x7_lower_arm_fixed_part_joint",
                "crane_x7_lower_arm_revolute_part_joint",
                "crane_x7_wrist_joint"]
t = 1.0 #10.0
 
class Enviroment:
    def __init__(self):
        rospy.init_node('ddpg_fetchreach')
        self.command_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size = 10)
        self.jt = JointTrajectory()
        self.jt.joint_names = joint_names
        self.rate = rospy.Rate(0.67)
        #self.rate = rospy.Rate(10.0) #0.67 1.0
        #damy
        self._reset()
        #self._reset()

    def _step(self, action):
        self.set_command(action)
        self.command_pub.publish(self.jt)
        self.jt.points = []
        self.rate.sleep()

    def set_command(self, action):
        #self.jt = JointTrajectory()
        #self.jt.joint_names = joint_names
        p = JointTrajectoryPoint()
        p.positions = action
        p.time_from_start = rospy.Duration.from_sec(t)
        self.jt.points.append(p)

    def _reset(self):
        reset_state = np.random.uniform(action_low, action_high)
        self._step(reset_state)
        #rospy.Rate(0.67).sleep()
