"""
pouring_water_env.py

Custom PouringWater environment for robosuite.
Simplified version using standard robosuite objects (no mimicgen dependency).

Task: Pour water from one cup (yellow) to another cup (black).
Success criteria:
    1. Yellow cup is lifted above table
    2. Both cups are aligned in XY plane
    3. Yellow cup is tilted towards black cup
    4. Yellow cup is near black cup
"""

import numpy as np
from collections import OrderedDict
import os

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class PouringWater(ManipulationEnv):
    """
    Custom environment for pouring water task.
    
    Uses two cylindrical cups as objects.
    """
    
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=500,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        base_path=None,
    ):
        # Table settings
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        
        # Reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        
        # Object observation
        self.use_object_obs = use_object_obs
        
        # Placement initializer
        self.placement_initializer = placement_initializer
        
        # Base path for assets
        if base_path is None:
            # Default to the directory containing this file
            base_path = os.path.dirname(os.path.abspath(__file__))
        self.base_path = base_path
        
        # Failure tracking
        self.failure_reasons = []
        
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )
    
    def reward(self, action=None):
        """
        Reward function for the task.
        
        Returns:
            reward: 1.0 if successful, 0.0 otherwise
        """
        reward = 0.
        if self._check_success():
            reward = 1.0
        
        if self.reward_shaping:
            # Add shaping rewards
            yellow_cup_pos = self.sim.data.body_xpos[self.obj_body_id["yellow_cup"]]
            black_cup_pos = self.sim.data.body_xpos[self.obj_body_id["black_cup"]]
            
            # Reward for lifting
            lift_reward = max(0, (yellow_cup_pos[2] - self.table_offset[2] - 0.05) / 0.1)
            
            # Reward for proximity
            dist_xy = np.linalg.norm(yellow_cup_pos[:2] - black_cup_pos[:2])
            proximity_reward = max(0, (0.3 - dist_xy) / 0.3)
            
            reward += 0.1 * (lift_reward + proximity_reward)
        
        return reward
    
    def _load_model(self):
        """
        Loads the arena and objects.
        """
        super()._load_model()
        
        # Adjust robot base pose
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        # Create table arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        
        # Load yellow cup from XML file
        yellow_cup_xml_path = os.path.join(self.base_path, "assets/objects/yellow_cup.xml")
        if not os.path.exists(yellow_cup_xml_path):
            raise FileNotFoundError(f"Yellow cup XML not found: {yellow_cup_xml_path}")
        
        self.yellow_cup = MujocoXMLObject(
            fname=yellow_cup_xml_path,
            name="yellow_cup",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        
        # Load black cup from XML file
        black_cup_xml_path = os.path.join(self.base_path, "assets/objects/black_cup.xml")
        if not os.path.exists(black_cup_xml_path):
            raise FileNotFoundError(f"Black cup XML not found: {black_cup_xml_path}")
        
        self.black_cup = MujocoXMLObject(
            fname=black_cup_xml_path,
            name="black_cup",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        
        self.objects = [self.yellow_cup, self.black_cup]
        
        # Create placement initializer
        self._get_placement_initializer()
        
        # Create manipulation task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )
    
    def _get_placement_initializer(self):
        """
        Set up placement sampler for objects.
        """
        if self.placement_initializer is not None:
            return
        
        # yellow_cup_x_range = (0.0, 0.1)
        # yellow_cup_y_range = (0.0, 0.1)
        # black_cup_x_range = (-0.1, 0.0)
        # black_cup_y_range = (-0.3, -0.2)

        # 固定位置
        # yellow_cup_x_range = (0.0, 0.0)
        # yellow_cup_y_range = (0.15, 0.15)
        # black_cup_x_range = (0.0, 0.0)
        # black_cup_y_range = (-0.15, -0.15)
        
        # 犯错位置
        yellow_cup_x_range = (0.0037, 0.0037)
        yellow_cup_y_range = (0.061, 0.061)
        black_cup_x_range = (-0.0949, -0.0949)
        black_cup_y_range = (-0.2721, -0.2721)


        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        
        # Yellow cup sampler
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="YellowCupSampler",
                mujoco_objects=self.yellow_cup,
                x_range=yellow_cup_x_range,
                y_range=yellow_cup_y_range,
                # rotation=(0.0, 2.0 * np.pi),
                rotation=(-120.0 / 180.0 * np.pi, 120.0 / 180.0 * np.pi),
                # rotation=(-179.0 / 180.0 * np.pi, -179.0 / 180.0 * np.pi),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )
        
        # Black cup sampler
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="BlackCupSampler",
                mujoco_objects=self.black_cup,
                x_range=black_cup_x_range,
                y_range=black_cup_y_range,
                rotation=0.0,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )
    
    def _setup_references(self):
        """
        Set up references to important components.
        """
        super()._setup_references()
        
        # Object body IDs
        self.obj_body_id = dict(
            yellow_cup=self.sim.model.body_name2id(self.yellow_cup.root_body),
            black_cup=self.sim.model.body_name2id(self.black_cup.root_body),
        )
    
    def _reset_internal(self):
        """
        Reset simulation internal configurations.
        """
        super()._reset_internal()
        
        # Reset object positions
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )
    
    def _setup_observables(self):
        """
        Set up observables for the environment.
        """
        observables = super()._setup_observables()
        
        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            
            # World pose in gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return T.pose_inv(T.pose2mat((
                    obs_cache[f"{pf}eef_pos"], 
                    obs_cache[f"{pf}eef_quat"]
                ))) if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
            
            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            actives = [False]
            
            # Add object sensors
            for obj_name in self.obj_body_id:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(
                    obj_name=obj_name, 
                    modality=modality
                )
                sensors += obj_sensors
                names += obj_sensor_names
                actives += [True] * len(obj_sensors)
            
            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )
        
        return observables
    
    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Create sensors for a given object.
        """
        pf = self.robots[0].robot_model.naming_prefix
        
        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])
        
        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], 
                to="xyzw"
            )
        
        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            if any([name not in obs_cache for name in [
                f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"
            ]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((
                obs_cache[f"{obj_name}_pos"], 
                obs_cache[f"{obj_name}_quat"]
            ))
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, 
                obs_cache["world_pose_in_gripper"]
            )
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos
        
        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache.get(f"{obj_name}_to_{pf}eef_quat", np.zeros(4))
        
        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos", 
            f"{obj_name}_quat", 
            f"{obj_name}_to_{pf}eef_pos", 
            f"{obj_name}_to_{pf}eef_quat"
        ]
        
        return sensors, names
    
    def _check_success(self):
        """
        Check if pouring task is successful.
        
        Success criteria:
        1. Yellow cup is lifted above table (> 10cm)
        2. Both cups aligned in XY plane (yaw difference < 10 degrees)
        3. Yellow cup tilted towards black cup (> 10 degrees)
        4. Yellow cup near black cup (< 20cm in XY plane)
        """
        yellow_cup_pos = self.sim.data.body_xpos[self.obj_body_id["yellow_cup"]]
        yellow_cup_quat = self.sim.data.body_xquat[self.obj_body_id["yellow_cup"]]
        black_cup_pos = self.sim.data.body_xpos[self.obj_body_id["black_cup"]]
        black_cup_quat = self.sim.data.body_xquat[self.obj_body_id["black_cup"]]
        
        # 1. Check if yellow cup is lifted
        lifted = yellow_cup_pos[2] > (self.table_offset[2] + 0.1)
        
        # 2. Check XY alignment (yaw difference)
        yellow_cup_mat = T.quat2mat(T.convert_quat(yellow_cup_quat, to="xyzw"))
        black_cup_mat = T.quat2mat(T.convert_quat(black_cup_quat, to="xyzw"))
        
        yellow_euler = T.mat2euler(yellow_cup_mat)
        black_euler = T.mat2euler(black_cup_mat)
        
        yaw_diff = np.abs(yellow_euler[2] - black_euler[2])
        if yaw_diff > np.pi:
            yaw_diff = 2 * np.pi - yaw_diff
        
        xy_aligned = yaw_diff < np.deg2rad(10)
        
        # 3. Check tilt in pouring plane
        diff_xy = black_cup_pos[:2] - yellow_cup_pos[:2]
        dist_xy = np.linalg.norm(diff_xy)
        
        is_plane_tilted = False
        proj_tilt_degree = 0.0
        
        if dist_xy > 1e-4:
            # Plane normal perpendicular to line connecting cups
            plane_normal = np.array([-diff_xy[1], diff_xy[0], 0])
            plane_normal /= np.linalg.norm(plane_normal)
            
            # Yellow cup Z-axis
            yellow_z = yellow_cup_mat[:3, 2]
            
            # Project yellow Z onto plane
            yellow_z_proj = yellow_z - np.dot(yellow_z, plane_normal) * plane_normal
            
            norm_proj = np.linalg.norm(yellow_z_proj)
            if norm_proj > 1e-6:
                yellow_z_proj /= norm_proj
                
                # Angle with world Z-axis
                proj_tilt_angle = np.arccos(np.clip(yellow_z_proj[2], -1.0, 1.0))
                proj_tilt_degree = np.rad2deg(proj_tilt_angle)
                is_plane_tilted = proj_tilt_degree > 10.0
        
        # 4. Check proximity
        is_near = dist_xy < 0.2
        
        success = lifted and xy_aligned and is_plane_tilted and is_near
        
        # Track failure reasons
        self.failure_reasons = []
        if not success:
            if not lifted:
                self.failure_reasons.append(
                    f"Not Lifted (h={yellow_cup_pos[2]:.3f} < {self.table_offset[2] + 0.1:.3f})"
                )
            if not xy_aligned:
                self.failure_reasons.append(
                    f"XY Misaligned (diff={np.rad2deg(yaw_diff):.1f}° > 10°)"
                )
            if not is_plane_tilted:
                self.failure_reasons.append(
                    f"Not Tilted (angle={proj_tilt_degree:.1f}° <= 10°)"
                )
            if not is_near:
                self.failure_reasons.append(
                    f"Not Near (dist={dist_xy:.3f}m > 0.2m)"
                )
        
        return success
