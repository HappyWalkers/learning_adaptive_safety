import numpy as np
import torch
from ray.rllib.policy.policy import Policy

import rclpy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class RLControllor(Node):
    def __init__(self):
        super().__init__('rl_control')

        odom_topic = '/ego_racecar/odom' 
        #odom_topic = '/pf/' 
        scan_topic = '/scan'

        # RL model
        model_pth = './policy'
        self.model = Policy.from_checkpoint(model_pth)

        # sample the lidar scan
        self.n_beams = 54
        assert 1080 // self.n_beams == 1080 / self.n_beams, "n_beams={} is not 1080 divisible".format(self.n_beams)

        # Create subsripters 
        #self.pose_sub = self.create_subscription(
        #    Odometry,
        #    odom_topic,
        #    self.pose_callback, 10
        #)
        self.scan_sub = self.create_subscription(
            LaserScan, 
            scan_topic, 
            self.scan_callback, 10
        )

        # Publishing topics

        self.driver_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

    def scan_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        ranges = ranges[::(1080 // self.n_beams)]

        ranges = torch.tensor(ranges)
        cmds = self.model.compute_single_action(ranges)[0]


        # post process
        low = np.array([-0.4, 0.0])
        high = np.array([0.4, 6.0])
        cmds = low + (cmds + 1.0) * (high - low) / 2.0
        cmds = np.clip(cmds, low, high)
        print(cmds)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(cmds[0])
        drive_msg.drive.speed = float(cmds[1])
        self.driver_pub.publish(drive_msg)
        
        
def main(args=None):
    rclpy.init(args=args)
    rl = RLControllor()
    rclpy.spin(rl)
    
    rl.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
