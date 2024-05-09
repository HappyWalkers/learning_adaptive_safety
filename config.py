# configurations are distributed over many different files and codebases
# this new config file is a trial to centralize all configurations in one place

import pathlib
import numpy as np
import yaml

class Config:
    class CBF:
        frenet_coordinate_diff_oppo_ego_limit_min :float = 0.0
        frenet_coordinate_diff_oppo_ego_limit_max :float = 5.0

        max_brake_lateral = 1.0
        max_brake_longitudinal_func = lambda v: 1 / (1 + v)

        follow_action_optmization_wall_slack_scale: float = 1000
        follow_action_optmization_obstacle_slack_scale: float = 1000

        cbf_params = {
            "acceleration_range": [-9.51, 9.51],
            "beta_range": [-0.22, 0.22],
            "lf": 0.15875,
            "lr": 0.17145,
            "dt": 0.1,
            "track_width": 3.00,
            "wall_margin": 0.5,
            "safe_distance": 0.5,
            # additional params, used only in opt-decay cbf (odcbf)
            "odcbf_gamma_range": [0.0, 1.0],  # range of gamma, for optimal-decay cbf
            "odcbf_gamma_penalty": 1e4,  # penalty for deviation from nominal gamma, for optimal-decay cbf
        }

    class Lattice_Planner:
        dynamic_collision_cost_max: float = 1000
        dynamic_collision_cost_min: float = 1000

        map_collision_cost_for_one_lane: float = 3000

        similarity_cost_scale: float = 0.25

        length_cost_scale: float = 0.5

        follow_optim_cost_scale: float = 0.1

        params = {
            "wb": 0.33,
            "lh_grid_lb": 0.6,
            "lh_grid_ub": 1.2,
            "lh_grid_rows": 3,
            "lat_grid_lb": -1.5,
            "lat_grid_ub": 1.5,
            "lat_grid_cols": 11,
            "weights": 7,
            "score_names": [
                "curvature_cost",
                "get_length_cost",
                "get_similarity_cost",
                "get_follow_optim_cost",
                "get_map_collision",
                "abs_v_cost",
                "collision_cost",
            ],
            # "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "traj_v_span_min": 0.2,
            "traj_v_span_max": 1.0,
            "traj_v_span_num": 5,
            "traj_points": 20,
            "vgain": 10.0,
            "collision_thres": 0.3,
            "planning_frequency": 10,
            "tracker": "advanced_pure_pursuit",
            "tracker_params": {"vgain": 1.0},
        }

    class PPO:
        relative_distance_to_reward_func = lambda rel_distance: np.tanh(rel_distance / 5.0)

        @classmethod
        def reward_at_collision(self):
            return -10 * Config.Env.timeout * Config.Env.planning_freq

        num_steps = 2048

    class Car:
        width = 0.31
        length = 0.58

    class Env:
        reset_pose_dis_min = 2.0
        reset_pose_dis_max = 2.0

        # timeout = 8
        timeout = 1e4

        planning_freq = 10

    class Pure_Pursuit_Planner:
        params = {
            "lookahead_distance": 1.5,
            "max_reacquire": 20.0,
            "wb": 0.33,
            "fixed_speed": None,
            "vgain": 1.0,
            "vgain_std": 0.0,
            "min_speed": 0.0,
            "max_speed": 2.0,
            "max_steering": 1.0,
        }
        v_gain_and_std = [
            {'vgain': 0.7, 'vgain_std': 0.05}, 
        ]
        
        fixed_speed = 1.0