# configurations are distributed over many different files and codebases
# this new config file is a trial to centralize all configurations in one place

import numpy as np

class Config:
    class CBF:
        frenet_coordinate_diff_oppo_ego_limit_min :float = 0.0
        frenet_coordinate_diff_oppo_ego_limit_max :float = 5.0

        max_brake_longitudinal_func = lambda v: 1.0 - (v / 10.0)

        follow_action_optmization_wall_slack_scale: float = 1000
        follow_action_optmization_obstacle_slack_scale: float = 1000

        cbf_params = {
            "acceleration_range": [-9.51, 9.51],
            "beta_range": [-0.22, 0.22],
            "lf": 0.15875,
            "lr": 0.17145,
            "dt": 0.1,
            "track_width": 3.00,
            "wall_margin": 0.1,
            "safe_distance": 0.1 + 0.5,
            # additional params, used only in opt-decay cbf (odcbf)
            "odcbf_gamma_range": [0.0, 1.0],  # range of gamma, for optimal-decay cbf
            "odcbf_gamma_penalty": 1e4,  # penalty for deviation from nominal gamma, for optimal-decay cbf
        }

    class Lattice_Planner:
        dynamic_collision_cost_max: float = 1000
        dynamic_collision_cost_min: float = 1000

        map_collision_cost: float = 3000

        similarity_cost_scale: float = 0.25

        length_cost_scale: float = 0.5

        follow_optim_cost_scale: float = 0.1

    class PPO:
        relative_distance_to_reward_func = lambda rel_distance: np.tanh(rel_distance / 5.0)
        reward_at_collision: float = -1.0

    class Car:
        width = 0.31
        length = 0.58