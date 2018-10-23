import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal - 目标：到达指定的位置 target_pos
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        # 初始化目标距离
        self.target_distance = self.get_distance_from_target()
    
    # 计算智能体和目标的距离
    def get_distance_from_target(self):
        distance_from_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        return distance_from_target

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum() # 计算奖励（reward）
        
        '''
        previous_distance = self.target_distance
        self.target_distance = self.get_distance_from_target()
        term_1 = 100.0/self.target_distance if self.target_distance > 0.01 else 50000
        #term_2 = (previous_distance - self.target_distance)
        #term_3 = -0.4*(abs(self.sim.angular_v)).sum()
        #print("term_1: {}, term_2: {}, term_3: {}".format(term_1, term_2, term_3))
        reward = term_1 # + term_2 + term_3
        '''
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state