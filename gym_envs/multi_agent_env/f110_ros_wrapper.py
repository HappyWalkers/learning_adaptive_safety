import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from typing import List
import numpy as np
import time


rclpy.init()

class F110ROSWrapper(Node):
    '''
    This class is a wrapper around the F110 gym environment.
    It's used to interact with the gym_bridge or the real car through ROS.
    It should provide interfaces that a environment should provide so the caller can use it as a gym environment.
    
    It should provide step and reset functions.
    The step function should take a action from caller, send the action to the ROS topic, wait for the next observation, and return the observation.
    The reset function should take a pose from caller, waiting for the enviroment to be reset by human, and return the observation.

    For the reinforcement training, the caller may send actions for multiple agents. 
    In this case, this wrapper should send the actions to the ROS topic for each agent, 
    and also receive the observations from the ROS topic for each agent.
    For the reinforcement inference, the caller may only send actions for the ego agent.
    '''
    def __init__(
            self,
            num_agents = 2,
            ego_idx = 0,
            odom_topic_list = ['/ego_racecar/odom', '/opp_racecar/odom'],
            scan_topic_list = ['/scan', '/opp_scan'],
            drive_topic_list = ['/drive', '/opp_drive'],
            reset_topic_list = ['/initialpose', '/goal_pose'],
            reset_done_topic_list = ['/reset_done', '/opp_reset_done'],
            time_step = 0.01,
            params = None,
            ):
        super().__init__('f110_ros_wrapper')

        assert num_agents > 0, "The number of agents should be greater than 0."
        assert num_agents == len(odom_topic_list) == len(scan_topic_list) == len(drive_topic_list), "The number of topics should be equal to the number of agents."

        self.num_agents = num_agents
        self.ego_idx = ego_idx
        self.time_step = time_step
        self.step_counter = 0
        self.current_time = 0.0
        self.params = params

        # Create subsripters for getting observations
        self.scan_sub_list = []
        for i, scan_topic in enumerate(scan_topic_list):
            self.scan_sub_list.append(self.create_subscription(
                LaserScan,
                scan_topic,
                self.scan_callback(i), 10
            ))

        self.odom_sub_list = []
        for i, odom_topic in enumerate(odom_topic_list):
            self.odom_sub_list.append(self.create_subscription(
                Odometry,
                odom_topic,
                self.odom_callback(i), 10
            ))

        # container for holding observations
        self.last_scan_list: List[LaserScan] = [None] * self.num_agents
        self.last_scan_timestamp_list = [self.get_clock().now()] * self.num_agents
        self.last_odom_list: List[Odometry] = [None] * self.num_agents
        self.last_odom_timestamp_list = [self.get_clock().now()] * self.num_agents
        

        # Create publisher for publishing actions
        self.drive_pub_list = []
        for drive_topic in drive_topic_list:
            self.drive_pub_list.append(self.create_publisher(
                AckermannDriveStamped, drive_topic, 10
            ))

        # Create publisher for publishing reset poses to human driver
        self.reset_pub_list = []
        for i, reset_topic in enumerate(reset_topic_list):
            if i == self.ego_idx:
                self.reset_pub_list.append(self.create_publisher(
                    PoseWithCovarianceStamped, reset_topic, 10
                ))
            else:
                self.reset_pub_list.append(self.create_publisher(
                    PoseStamped, reset_topic, 10
                ))
        
        # Create subscriber for getting reset-done message from human driver
        self.reset_done_sub_list = []
        for reset_done_topic in reset_done_topic_list:
            self.reset_done_sub_list.append(self.create_subscription(
                Bool, reset_done_topic, self.reset_done_callback(i), 10
            ))

        # Container for holding reset-done status
        self.reset_done_status_list = [None] * self.num_agents
        self.reset_done_timestamp_list = [self.get_clock().now()] * self.num_agents


    def scan_callback(self, agent_idx: int):
        def scan_callback(scan_msg: LaserScan):
            self.last_scan_list[agent_idx] = scan_msg
            self.last_scan_timestamp_list[agent_idx] = self.get_clock().now()
        return scan_callback
    
    def odom_callback(self, agent_idx: int):
        def odom_callback(odom_msg: Odometry):
            self.last_odom_list[agent_idx] = odom_msg
            self.last_odom_timestamp_list[agent_idx] = self.get_clock().now()
        return odom_callback
    
    def reset_done_callback(self, agent_idx: int):
        def reset_done_callback(reset_done_msg: Bool):
            self.reset_done_status_list[agent_idx] = reset_done_msg
            self.reset_done_timestamp_list[agent_idx] = self.get_clock().now()
        return reset_done_callback


    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2)), control inputs of all agents, first column is desired steering angle, second column is desired velocity

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        self.step_counter += 1
        self.current_time = self.step_counter * self.time_step

        # send action to each agent
        action_sent_timestamp = self.get_clock().now()
        for i in range(self.num_agents):
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.steering_angle = action[i][0]
            drive_msg.drive.speed = action[i][1]
            self.drive_pub_list[i].publish(drive_msg)

        # wait for the next observation from each agent
        # print("step is waiting for the next observation")
        while self.not_all_new_obs(action_sent_timestamp):
            rclpy.spin_once(self, timeout_sec=self.time_step / 100)

        # build the returned values
        obs = self.build_observation(scan_list=self.last_scan_list, odom_list=self.last_odom_list)
        reward = 0
        done = False
        info = {
            'checkpoint_done': np.array([False for _ in range(self.num_agents)]),
        }
        return obs, reward, done, info

    def not_all_new_obs(self, action_sent_timestamp):
        return any((timestamp - action_sent_timestamp).nanoseconds < int(1e9 * self.time_step) for timestamp in self.last_scan_timestamp_list) or \
            any((timestamp - action_sent_timestamp).nanoseconds < int(1e9 * self.time_step) for timestamp in self.last_odom_timestamp_list)

        
    def build_observation(self, scan_list: List[LaserScan], odom_list: List[Odometry]):
        '''
        {
            'ego_idx': 0, 
            'scans': [array([2.59351341, 2.56480895, 2.57242999, ..., 3.91241407, 3.87047525, 3.88441715]), 
                    array([1.27411929, 1.24541482, 1.25303587, ..., 5.71013832, 5.71688463, 5.68082652])], 
            'poses_x': [-5.432158302444511, -3.5936422680106226], 
            'poses_y': [0.457316965026, -0.6331917727595296], 
            'poses_theta': [6.1085530423134475, 0.010375659781753364], 
            'linear_vels_x': [0.004689255326092242, 0.024955937898784204], 
            'linear_vels_y': [0.0, 0.0], 
            'ang_vels_z': [0.0, 0.0], 
            'collisions': array([0., 0.]), 
            'lap_times': array([0.02, 0.02]), 
            'lap_counts': array([0., 0.])
        }
        '''
        obs = {
            'ego_idx': self.ego_idx,
            'scans': [np.array(scan.ranges) for scan in scan_list],
            'poses_x': [odom.pose.pose.position.x for odom in odom_list],
            'poses_y': [odom.pose.pose.position.y for odom in odom_list],
            'poses_theta': [odom.pose.pose.orientation.z for odom in odom_list],
            'linear_vels_x': [np.float64(odom.twist.twist.linear.x) for odom in odom_list],
            'linear_vels_y': [np.float64(odom.twist.twist.linear.y) for odom in odom_list],
            'ang_vels_z': [odom.twist.twist.angular.z for odom in odom_list],
            'collisions': np.zeros(self.num_agents),
            'lap_times': self.current_time * np.ones(self.num_agents),
            'lap_counts': np.zeros(self.num_agents)
        }
        return obs


    def reset(self, poses):
        """
        Reset the gym environment by given poses
        To reset the agents to the poses, 
        1. send a stop command to each agent
        2. send the desired pose to a topic that can be read by human driver
        3. wait for the human driver to reset the car and send a message to the robot driver to start
        4. return the most recent observation from the robot driver

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        self.step_counter = 0 

        # send stop command to each agent
        for i in range(self.num_agents):
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = 0.0
            self.drive_pub_list[i].publish(drive_msg)

        # send the desired pose to the human driver
        desired_pose_sent_timestamp = self.get_clock().now()
        for i in range(self.num_agents):
            if i == self.ego_idx:
                pose_with_covariance_msg = PoseWithCovarianceStamped()
                pose_with_covariance_msg.header.stamp = self.get_clock().now().to_msg()
                pose_with_covariance_msg.pose.pose.position.x = -3.0
                pose_with_covariance_msg.pose.pose.position.y = 0.0
                pose_with_covariance_msg.pose.pose.orientation.z = poses[i][2]
                self.reset_pub_list[i].publish(pose_with_covariance_msg)
            else:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.pose.position.x = poses[i][0]
                pose_msg.pose.position.y = poses[i][1]
                pose_msg.pose.orientation.z = poses[i][2]
                self.reset_pub_list[i].publish(pose_msg)
            time.sleep(0.01) # ROS may limit the frequency of publishing messages
            

        # # wait for the human driver to reset the car and send a message to the robot driver
        # print("reset is waiting for human driver to reset. pose has been sent through topic")
        # while any(timestamp < desired_pose_sent_timestamp for timestamp in self.reset_done_timestamp_list):
        #     rclpy.spin_once(self, timeout_sec=self.time_step / 100)

        # return the most recent observation from the robot driver
        print("reset is waiting for the next observation")
        while self.not_all_new_obs(desired_pose_sent_timestamp):
            rclpy.spin_once(self, timeout_sec=self.time_step / 100)

        obs = self.build_observation(scan_list=self.last_scan_list, odom_list=self.last_odom_list)
        reward = self.time_step
        done = False
        info = {
            'checkpoint_done': np.array([False for _ in range(self.num_agents)]),
        }
        return obs, reward, done, info
