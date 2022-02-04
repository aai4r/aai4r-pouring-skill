import copy

import torch
import math

from utils.torch_jit_utils import *
from utils.utils import *
from torch.nn.functional import normalize
from tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class DemoUR3Pouring(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "x"      # z
        self.up_axis_idx = 0    # 2
        self.dt = 1/30.

        self.use_ik = False

        num_obs = 24 if self.use_ik else 37    # 21 for task space
        num_acts = 8 if self.use_ik else 12   # 8 for task space

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.indices = torch.tensor([0, 1, 2, 3, 4, 5, 8], device=device_id)  # 0~5: ur3 joint, 8: robotiq drive joint

        super().__init__(cfg=self.cfg)

        # set gripper params
        self.grasp_z_offset = 0.135      # (meter)
        self.gripper_stroke = 85 / 1000  # (meter), robotiq 85 gripper stroke: 85 mm -> 0.085 m
        self.angle_stroke_ratio = self.ur3_dof_upper_limits[8] / self.gripper_stroke

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_net_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        print("device:: ", self.device)
        self.ur3_default_dof_pos = to_torch([deg2rad(0.0), deg2rad(-90.0), deg2rad(85.0), deg2rad(0.0), deg2rad(80.0), deg2rad(0.0),
                                             deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.ur3_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur3_dofs]
        # self.ur3_dof_pos = torch.index_select(self.ur3_dof_state[..., 0], 1, self.indices)
        # self.ur3_dof_vel = torch.index_select(self.ur3_dof_state[..., 1], 1, self.indices)
        self.ur3_dof_pos = self.ur3_dof_state[..., 0]
        self.ur3_dof_vel = self.ur3_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.bottle_states = self.root_state_tensor[:, 1]
        self.liquid_states = self.root_state_tensor[:, 2:2 + self.num_water_drops]
        self.cup_states = self.root_state_tensor[:, -1]

        self.contact_net_force = gymtorch.wrap_tensor(contact_net_force_tensor)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur3_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.num_actors = self.root_state_tensor.size()[1]
        self.global_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset(torch.arange(self.num_envs, device=self.device))
        self.refresh_env_tensors()
        self.init_task_path(torch.arange(self.num_envs, device=self.device))

        # expert demo. params.
        self.task_update_buf = torch.zeros_like(self.progress_buf)

        # gripper limit for bottle
        # blim = self.stroke_to_angle(self.bottle_diameter - 0.005)
        # self.ur3_dof_lower_limits[7] = -blim
        # self.ur3_dof_lower_limits[10] = -blim
        #
        # self.ur3_dof_upper_limits[6] = blim
        # self.ur3_dof_upper_limits[8] = blim
        # self.ur3_dof_upper_limits[9] = blim
        # self.ur3_dof_upper_limits[11] = blim

        # TODO, cam setting for debugging with single env.
        cam_pos = gymapi.Vec3(0.9263, 0.4617, 0.5420)
        cam_target = gymapi.Vec3(0.0, -0.3, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_asset_bottle(self):
        self.bottle_height = 0.195
        self.bottle_diameter = 0.065
        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # asset_options.armature = 0.005
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 16
        # asset_options.vhacd_params.max_num_vertices_per_ch = 32

        bottle_asset_file = "urdf/objects/bottle.urdf"
        if "asset" in self.cfg["env"]:
            bottle_asset_file = self.cfg["env"]["asset"].get("assetFileNameBottle", bottle_asset_file)

        bottle_asset = self.gym.load_asset(self.sim, self.asset_root, bottle_asset_file, asset_options)
        return bottle_asset

    def _create_asset_cup(self):
        self.cup_height = 0.074
        asset_options = gymapi.AssetOptions()
        # asset_options.armature = 0.01
        # asset_options.angular_damping = 0.01
        # asset_options.linear_damping = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 250000
        # asset_options.vhacd_params.max_convex_hulls = 128
        # asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.use_mesh_materials = True

        cup_asset_file = "urdf/objects/paper_cup.urdf"
        cup_asset = self.gym.load_asset(self.sim, self.asset_root, cup_asset_file, asset_options)
        return cup_asset

    def create_asset_water_drops(self):
        r = 0.012
        self.expr = [[0, 0], [0, -r], [-r, 0], [0, r], [r, 0]]
        self.num_water_drops = 1

        asset_options = gymapi.AssetOptions()
        asset_options.density = 997
        asset_options.armature = 0.01
        liquid_asset = self.gym.create_sphere(self.sim, 0.015, asset_options)   # radius
        return liquid_asset

    def _create_asset_ur3(self):
        ur3_asset_file = "urdf/ur3_description/robot/ur3_robotiq85_gripper.urdf"
        if "asset" in self.cfg["env"]:
            ur3_asset_file = self.cfg["env"]["asset"].get("assetFileNameUR3", ur3_asset_file)

        # load ur3 asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        ur3_asset = self.gym.load_asset(self.sim, self.asset_root, ur3_asset_file, asset_options)
        return ur3_asset

    def _create_envs(self, num_envs, spacing, num_per_row):
        self.asset_root = "../assets"
        if "asset" in self.cfg["env"]:
            self.asset_root = self.cfg["env"]["asset"].get("assetRoot", self.asset_root)

        self._create_ground_plane()
        ur3_asset = self._create_asset_ur3()
        bottle_asset = self._create_asset_bottle()
        cup_asset = self._create_asset_cup()
        liq_asset = self.create_asset_water_drops()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        ur3_link_dict = self.gym.get_asset_rigid_body_dict(ur3_asset)
        print("ur3 link dictionary: ", ur3_link_dict)

        self.ur3_hand_index = ur3_link_dict["ee_link"]
        self.num_ur3_bodies = self.gym.get_asset_rigid_body_count(ur3_asset)
        self.num_ur3_dofs = self.gym.get_asset_dof_count(ur3_asset)

        print("num ur3 bodies: ", self.num_ur3_bodies)
        print("num ur3 dofs: ", self.num_ur3_dofs)

        # set franka dof properties
        self.ur3_dof_props = self.gym.get_asset_dof_properties(ur3_asset)
        self.ur3_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        # ur3 joints
        self.ur3_dof_props["stiffness"][:6].fill(300.0)
        self.ur3_dof_props["damping"][:6].fill(80.0)
        # robotiq 85 gripper
        self.ur3_dof_props["stiffness"][6:].fill(100000.0)
        self.ur3_dof_props["damping"][6:].fill(100.0)

        self.ur3_dof_lower_limits = self.ur3_dof_props['lower']
        self.ur3_dof_upper_limits = self.ur3_dof_props['upper']

        self.ur3_dof_lower_limits = to_torch(self.ur3_dof_lower_limits, device=self.device)
        self.ur3_dof_upper_limits = to_torch(self.ur3_dof_upper_limits, device=self.device)

        # self.ur3_dof_lower_limits = torch.index_select(self.ur3_dof_lower_limits, 0, self.indices)
        # self.ur3_dof_upper_limits = torch.index_select(self.ur3_dof_upper_limits, 0, self.indices)
        self.ur3_dof_speed_scales = torch.ones_like(self.ur3_dof_lower_limits)

        ur3_start_pose = gymapi.Transform()
        ur3_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ur3_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        bottle_start_pose = gymapi.Transform()
        bottle_start_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx))
        bottle_start_pose.p.x = 0.5
        bottle_start_pose.p.y = 0.0
        bottle_start_pose.p.z = self.bottle_height * 0.05

        liquid_start_pose = bottle_start_pose
        liquid_start_pose.p.z += 0.1

        cup_start_pose = gymapi.Transform()
        cup_start_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx))
        cup_start_pose.p.x = 0.5
        cup_start_pose.p.y = 0.0
        cup_start_pose.p.z = self.cup_height * 0.55

        # compute aggregate size
        num_ur3_bodies = self.gym.get_asset_rigid_body_count(ur3_asset)
        num_ur3_shapes = self.gym.get_asset_rigid_shape_count(ur3_asset)
        num_bottle_bodies = self.gym.get_asset_rigid_body_count(bottle_asset)
        num_bottle_shapes = self.gym.get_asset_rigid_shape_count(bottle_asset)
        num_cup_bodies = self.gym.get_asset_rigid_body_count(cup_asset)
        num_cup_shapes = self.gym.get_asset_rigid_shape_count(cup_asset)
        num_liq_bodies = self.gym.get_asset_rigid_body_count(liq_asset)
        num_liq_shapes = self.gym.get_asset_rigid_shape_count(liq_asset)
        self.max_agg_bodies = num_ur3_bodies + num_bottle_bodies + num_cup_bodies + self.num_water_drops * num_liq_bodies
        self.max_agg_shapes = num_ur3_shapes + num_bottle_shapes + num_cup_shapes + self.num_water_drops * num_liq_shapes

        self.ur3_robots = []
        self.bottles = []
        self.cups = []
        self.default_bottle_states = []
        self.default_cup_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)

            # (1) Create Robot, last number 0: considering self collision
            ur3_actor = self.gym.create_actor(env_ptr, ur3_asset, ur3_start_pose, "ur3", i, 1)
            self.gym.set_actor_dof_properties(env_ptr, ur3_actor, self.ur3_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)

            # (2) Create Bottle
            bottle_actor = self.gym.create_actor(env_ptr, bottle_asset, bottle_start_pose, "bottle", i, 0)
            self.gym.set_rigid_body_color(env_ptr, bottle_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)))
            self.default_bottle_states.append([bottle_start_pose.p.x, bottle_start_pose.p.y, bottle_start_pose.p.z,
                                               bottle_start_pose.r.x, bottle_start_pose.r.y, bottle_start_pose.r.z, bottle_start_pose.r.w,
                                               0, 0, 0, 0, 0, 0])

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)

            # (3) Create liquid
            liq_count = 0
            for j in range(self.num_water_drops):
                liq_actor = self.gym.create_actor(env_ptr, liq_asset, liquid_start_pose, "water_drop{}".format(liq_count), i, 0)
                self.gym.set_rigid_body_color(env_ptr, liq_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, np.random.uniform(0.7, 1)))
                liq_count += 1
                liquid_start_pose.p.z = 0.05 * liq_count

            # (4) Create Cup
            cup_actor = self.gym.create_actor(env_ptr, cup_asset, cup_start_pose, "paper_cup", i, 0)
            self.default_cup_states.append([cup_start_pose.p.x, cup_start_pose.p.y, cup_start_pose.p.z,
                                            cup_start_pose.r.x, cup_start_pose.r.y, cup_start_pose.r.z, cup_start_pose.r.w,
                                            0, 0, 0, 0, 0, 0])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # # (3) Create liquids
            # liq_count = 0
            # while liq_count < self.num_liq_particles:
            #     liquid_pos = copy.deepcopy(bottle_start_pose)
            #     liquid_pos.p.z += self.bottle_height + 0.1 + 0.03 * liq_count
            #     for k in self.expr:
            #         liquid_pos.p.x += k[0]
            #         liquid_pos.p.y += k[1]
            #         liquid_handle = self.gym.create_actor(env_ptr, fluid_asset, liquid_pos, "liquid", i, 0)
            #         color = gymapi.Vec3(0.0, 0.0, 1.0)
            #         self.gym.set_rigid_body_color(env_ptr, liquid_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            #         liq_count += 1

            self.envs.append(env_ptr)
            self.ur3_robots.append(ur3_actor)
            self.bottles.append(bottle_actor)
            self.cups.append(cup_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "tool0")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "robotiq_85_left_finger_tip_link")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "robotiq_85_right_finger_tip_link")
        self.bottle_handle = self.gym.find_actor_rigid_body_handle(env_ptr, bottle_actor, "bottle")
        self.cup_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cup_actor, "paper_cup")
        self.default_bottle_states = to_torch(self.default_bottle_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.default_cup_states = to_torch(self.default_cup_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.init_data()

    def init_data(self):
        # self.lfinger_idxs = []
        # self.rfinger_idxs = []
        # for i in range(self.num_envs):
        #     lfinger_idx = self.gym.find_actor_rigid_body_index(self.envs[i], mirobot_handle, "left_finger", gymapi.DOMAIN_SIM)

        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur3_robots[0], "tool0")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur3_robots[0], "robotiq_85_left_finger_tip_link")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur3_robots[0], "robotiq_85_right_finger_tip_link")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 2     # z-axis
        fwd_offset = 0.02
        ur3_local_grasp_pose = hand_pose_inv * finger_pose
        ur3_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(fwd_offset, grasp_pose_axis))
        self.ur3_local_grasp_pos = to_torch([ur3_local_grasp_pose.p.x, ur3_local_grasp_pose.p.y,
                                             ur3_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.ur3_local_grasp_rot = to_torch([ur3_local_grasp_pose.r.x, ur3_local_grasp_pose.r.y,
                                             ur3_local_grasp_pose.r.z, ur3_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        finger_pose_axis = 1  # y-axis
        _lfinger_pose = gymapi.Transform()
        _lfinger_pose.p += gymapi.Vec3(*get_axis_params(0.0078, finger_pose_axis))
        self.ur3_local_lfinger_pos = to_torch([_lfinger_pose.p.x + fwd_offset, _lfinger_pose.p.y,
                                               _lfinger_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.ur3_local_lfinger_rot = to_torch([_lfinger_pose.r.x, _lfinger_pose.r.y,
                                               _lfinger_pose.r.z, _lfinger_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        _rfinger_pose = gymapi.Transform()
        _rfinger_pose.p += gymapi.Vec3(*get_axis_params(0.0078, finger_pose_axis))
        self.ur3_local_rfinger_pos = to_torch([_rfinger_pose.p.x + fwd_offset, _rfinger_pose.p.y,
                                               _rfinger_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.ur3_local_rfinger_rot = to_torch([_rfinger_pose.r.x, _rfinger_pose.r.y,
                                               _rfinger_pose.r.z, _rfinger_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        bottle_local_grasp_pose = gymapi.Transform()
        bottle_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.005, grasp_pose_axis))
        bottle_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.bottle_local_grasp_pos = to_torch([bottle_local_grasp_pose.p.x, bottle_local_grasp_pose.p.y,
                                               bottle_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.bottle_local_grasp_rot = to_torch([bottle_local_grasp_pose.r.x, bottle_local_grasp_pose.r.y,
                                               bottle_local_grasp_pose.r.z, bottle_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        bottle_local_tip_pose = gymapi.Transform()
        bottle_local_tip_pose.p = gymapi.Vec3(*get_axis_params(self.bottle_height * 0.5 + 0.02, 2))
        bottle_local_tip_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.bottle_local_tip_pos = to_torch([bottle_local_tip_pose.p.x, bottle_local_tip_pose.p.y,
                                              bottle_local_tip_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.bottle_local_tip_rot = to_torch([bottle_local_tip_pose.r.x, bottle_local_tip_pose.r.y,
                                              bottle_local_tip_pose.r.z, bottle_local_tip_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        bottle_local_floor_pose = gymapi.Transform()
        bottle_local_floor_pose.p = gymapi.Vec3(*get_axis_params(-self.bottle_height * 0.5, 2))
        bottle_local_floor_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.bottle_local_floor_pos = to_torch([bottle_local_floor_pose.p.x, bottle_local_floor_pose.p.y,
                                                bottle_local_floor_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.bottle_local_floor_rot = to_torch([bottle_local_floor_pose.r.x, bottle_local_floor_pose.r.y,
                                                bottle_local_floor_pose.r.z, bottle_local_floor_pose.r.w],
                                                device=self.device).repeat((self.num_envs, 1))

        cup_local_tip_pose = gymapi.Transform()
        cup_local_tip_pose.p = gymapi.Vec3(*get_axis_params(self.cup_height * 0.5 + 0.03, 2))
        cup_local_tip_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cup_local_tip_pos = to_torch([cup_local_tip_pose.p.x, cup_local_tip_pose.p.y,
                                           cup_local_tip_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cup_local_tip_rot = to_torch([cup_local_tip_pose.r.x, cup_local_tip_pose.r.y,
                                           cup_local_tip_pose.r.z, cup_local_tip_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.bottle_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cube_left_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        self.bottle_grasp_pos = torch.zeros_like(self.bottle_local_grasp_pos)
        self.bottle_grasp_rot = torch.zeros_like(self.bottle_local_grasp_rot)

        self.bottle_tip_pos = torch.zeros_like(self.bottle_local_grasp_pos)
        self.bottle_tip_rot = torch.zeros_like(self.bottle_local_grasp_rot)

        self.bottle_floor_pos = torch.zeros_like(self.bottle_local_grasp_pos)
        self.bottle_floor_rot = torch.zeros_like(self.bottle_local_grasp_rot)

        self.cup_tip_pos = torch.zeros_like(self.bottle_local_grasp_pos)
        self.cup_tip_rot = torch.zeros_like(self.bottle_local_grasp_rot)

        self.ur3_grasp_pos = torch.zeros_like(self.ur3_local_grasp_pos)
        self.ur3_grasp_rot = torch.zeros_like(self.ur3_local_grasp_rot)
        self.ur3_grasp_rot[..., -1] = 1  # xyzw

        self.ur3_lfinger_pos = torch.zeros_like(self.ur3_local_grasp_pos)
        self.ur3_rfinger_pos = torch.zeros_like(self.ur3_local_grasp_pos)
        self.ur3_lfinger_rot = torch.zeros_like(self.ur3_local_grasp_rot)
        self.ur3_rfinger_rot = torch.zeros_like(self.ur3_local_grasp_rot)

        # jacobians
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur3")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.ur3_hand_index - 1, :]
        self.j_eef = self.j_eef[:, :, :6]  # up to UR3 joints

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ur3_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.bottle_grasp_pos, self.bottle_grasp_rot, self.bottle_pos, self.bottle_rot, self.bottle_tip_pos, self.bottle_floor_pos,
            self.ur3_grasp_pos, self.ur3_grasp_rot, self.cup_pos, self.cup_rot, self.cup_tip_pos, self.liq_pos,
            self.ur3_lfinger_pos, self.ur3_rfinger_pos,
            self.contact_net_force.view(self.num_envs, self.max_agg_bodies, -1)[:, self.lfinger_handle],
            self.contact_net_force.view(self.num_envs, self.max_agg_bodies, -1)[:, self.rfinger_handle],
            self.gripper_forward_axis, self.bottle_up_axis, self.gripper_up_axis, self.cube_left_axis,
            self.num_envs, self.bottle_diameter, self.dist_reward_scale, self.rot_reward_scale, self.open_reward_scale,
            self.action_penalty_scale, self.max_episode_length
        )

    def refresh_env_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def compute_observations(self):
        self.refresh_env_tensors()
        self.sync_gripper()

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.ur3_grasp_rot[:], self.ur3_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.ur3_local_grasp_rot, self.ur3_local_grasp_pos)

        self.ur3_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.ur3_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.ur3_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.ur3_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.ur3_lfinger_rot, self.ur3_lfinger_pos = \
            compute_grasp_transforms(self.ur3_lfinger_rot, self.ur3_lfinger_pos,
                                     self.ur3_local_lfinger_rot, self.ur3_local_lfinger_pos)

        self.ur3_rfinger_rot, self.ur3_rfinger_pos = \
            compute_grasp_transforms(self.ur3_rfinger_rot, self.ur3_rfinger_pos,
                                     self.ur3_local_rfinger_rot, self.ur3_local_rfinger_pos)

        # bottle info
        self.bottle_pos = self.rigid_body_states[:, self.bottle_handle][:, 0:3]
        self.bottle_rot = self.rigid_body_states[:, self.bottle_handle][:, 3:7]

        # cup info.
        # self.cup_pos = self.rigid_body_states[:, self.cup_handle][:, 0:3]
        # self.cup_rot = self.rigid_body_states[:, self.cup_handle][:, 3:7]
        self.cup_pos = self.cup_states[:, 0:3]
        self.cup_rot = self.cup_states[:, 3:7]

        # liquid info., TODO
        self.liq_pos = self.liquid_states[:, 0, 0:3].reshape(self.num_envs, -1)
        self.liq_rot = self.liquid_states[:, 0, 3:7].reshape(self.num_envs, -1)

        self.bottle_grasp_rot[:], self.bottle_grasp_pos[:] = \
            tf_combine(self.bottle_rot, self.bottle_pos, self.bottle_local_grasp_rot, self.bottle_local_grasp_pos)

        self.bottle_tip_rot[:], self.bottle_tip_pos[:] = \
            tf_combine(self.bottle_rot, self.bottle_pos, self.bottle_local_tip_rot, self.bottle_local_tip_pos)

        self.bottle_floor_rot[:], self.bottle_floor_pos[:] = \
            tf_combine(self.bottle_rot, self.bottle_pos, self.bottle_local_floor_rot, self.bottle_local_floor_pos)

        self.cup_tip_rot[:], self.cup_tip_pos[:] = \
            tf_combine(self.cup_rot, self.cup_pos, self.cup_local_tip_rot, self.cup_local_tip_pos)

        dof_pos_scaled = (2.0 * (self.ur3_dof_pos - self.ur3_dof_lower_limits)
                          / (self.ur3_dof_upper_limits - self.ur3_dof_lower_limits) - 1.0)
        # dof_pos_scaled = self.ur3_dof_pos
        dof_pos_scaled = torch.index_select(dof_pos_scaled, 1, self.indices)
        dof_vel = torch.index_select(self.ur3_dof_vel, 1, self.indices)
        dof_pos_vel = torch.cat((dof_pos_scaled, dof_vel), dim=-1)

        to_target_pos = self.bottle_grasp_pos - self.ur3_grasp_pos
        to_target_rot = quat_mul(quat_conjugate(self.bottle_grasp_rot), self.ur3_grasp_rot)
        # to_2nd_target_pos_z = (0.1 - self.cube_pos[:, 2].unsqueeze(-1)).norm()
        # cube_pos_z = self.cube_pos[:, 2].unsqueeze(-1)

        # 7 + 7 + 7 = 21
        # self.obs_buf = torch.cat((dof_pos_scaled, dof_vel * self.dof_vel_scale,
        #                           to_target_pos, to_target_rot), dim=-1)

        to_cup_pos = self.cup_pos - self.bottle_grasp_pos
        # 1 + 7 + 7 + 7 = 22
        # dof_pos_finger = self.angle_to_stroke(self.ur3_dof_pos[:, 8].unsqueeze(-1))
        dof_pos_finger = self.ur3_dof_pos[:, 8].unsqueeze(-1)
        # finger_dist = torch.norm(self.ur3_lfinger_pos - self.ur3_rfinger_pos, p=2, dim=-1).unsqueeze(-1)
        dof_state = dof_pos_finger if self.use_ik else dof_pos_vel
        tip_pos_diff = self.cup_tip_pos - self.bottle_tip_pos
        self.obs_buf = torch.cat((dof_state,
                                  self.ur3_grasp_pos, self.ur3_grasp_rot,
                                  self.bottle_pos, self.bottle_rot,
                                  self.cup_pos, self.liq_pos,
                                  tip_pos_diff), dim=-1)

        # TODO, cam transform
        # cam_tr = self.gym.get_viewer_camera_transform(self.viewer, self.envs[0])
        # print("cam tr: ", cam_tr.p)

        return self.obs_buf

    def reset(self, env_ids):
        self.actions = torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device)
        # self.actions[:, -1] = 1.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.dof_state[:, 0] = torch.zeros_like(self.dof_state[:, 0], dtype=torch.float, device=self.device)  # pos
        self.dof_state[:, 1] = torch.zeros_like(self.dof_state[:, 1], dtype=torch.float, device=self.device)  # vel

        # reset ur3
        pos = tensor_clamp(
            self.ur3_default_dof_pos.unsqueeze(0) + 0.5 * (torch.rand((len(env_ids), self.num_ur3_dofs), device=self.device) - 0.5),
            self.ur3_dof_lower_limits, self.ur3_dof_upper_limits)
        self.ur3_dof_targets[env_ids, :] = pos
        self.ur3_dof_pos[env_ids, :] = pos
        self.ur3_dof_pos[env_ids, 8] = 0.0
        self.ur3_dof_vel[env_ids, :] = torch.zeros_like(self.ur3_dof_vel[env_ids])

        # for gripper sync.
        self.ur3_dof_pos[env_ids, 6] = 1 * self.ur3_dof_pos[env_ids, 8]
        self.ur3_dof_pos[env_ids, 7] = -1. * self.ur3_dof_pos[env_ids, 8]
        self.ur3_dof_pos[env_ids, 9] = 1 * self.ur3_dof_pos[env_ids, 8]
        self.ur3_dof_pos[env_ids, 10] = -1. * self.ur3_dof_pos[env_ids, 8]
        self.ur3_dof_pos[env_ids, 11] = 1 * self.ur3_dof_pos[env_ids, 8]
        # self.ur3_dof_state[:, :, 0] = self.ur3_dof_pos

        # self.ur3_dof_state[:, 0] = torch.ones_like(self.ur3_dof_state[:, 0], dtype=torch.float, device=self.device)

        # reset bottle

        rand_z_angle = torch.rand(len(env_ids)).uniform_(deg2rad(-90.0), deg2rad(90.0))
        quat = []   # z-axis cube orientation randomization
        for gamma in rand_z_angle:
            _q = gymapi.Quat.from_euler_zyx(0, 0, gamma)
            quat.append(torch.FloatTensor([_q.x, _q.y, _q.z, _q.w]))
        quat = torch.stack(quat).to(self.device)

        pick = self.default_bottle_states[env_ids]
        # print("default bottle: ".format(pick[env_ids]))
        pick[:, 3:7] = quat
        xy_scale = to_torch([0.15, 0.45, 0.0,            # position
                             0.0, 0.0, 0.0, 0.0,        # rotation (quat)
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(pick), 1)
        rand_bottle_pos = (torch.rand_like(pick, device=self.device, dtype=torch.float) - 0.5) * xy_scale
        self.bottle_states[env_ids] = pick + rand_bottle_pos

        # reset cup
        place = self.default_cup_states[env_ids]
        place += (torch.rand_like(place, device=self.device, dtype=torch.float) - 0.5) * xy_scale
        place[:, 1] = torch.where(self.bottle_states[env_ids, 1] >= 0,
                                  self.bottle_states[env_ids, 1] - 0.2,
                                  self.bottle_states[env_ids, 1] + 0.2)
        place[:, 3:7] = quat
        self.cup_states[env_ids] = place

        # reset liquid
        init_liq_pose = pick + rand_bottle_pos
        init_liq_pose[:, 2] = init_liq_pose[:, 2] + 0.12
        offset_z = 0.05
        for i in range(self.liquid_states.shape[1]):
            self.liquid_states[env_ids, i] = init_liq_pose
            init_liq_pose[:, 2] = init_liq_pose[:, 2] + offset_z

        # # fluid particle init.
        # for i in range(self.num_envs):
        #     liq_count = 0
        #     z_offset = 0
        #     bottle_pose = gymapi.Transform()
        #     bottle_pose.p = gymapi.Vec3(self.bottle_states[i, 0], self.bottle_states[i, 1], self.bottle_states[i, 2])
        #     bottle_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        #     # for j in range(2, self.root_state_tensor.size(1)):
        #     #     idx = liq_count % len(self.expr)
        #     #     self.root_state_tensor[i, j, :] = to_torch([bottle_pose.p.x + self.expr[idx][0],
        #     #                                                 bottle_pose.p.y + self.expr[idx][1],
        #     #                                                 bottle_pose.p.z + self.bottle_height + 0.1 + 0.03 * z_offset,
        #     #                                                 bottle_pose.r.x, bottle_pose.r.y, bottle_pose.r.z, bottle_pose.r.w,
        #     #                                                 0, 0, 0, 0, 0, 0], device=self.device)
        #     #     liq_count += 1
        #     #     z_offset += 1 if liq_count % len(self.expr) == 0 else 0

        # reset apply
        bottle_liquid_indices = self.global_indices[env_ids, 1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(bottle_liquid_indices),
                                                     len(bottle_liquid_indices))

        # multi_env_ids_int = self.global_indices[env_ids, :1].flatten()
        robot_indices32 = self.global_indices[env_ids, :1].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ur3_dof_targets),
                                                        gymtorch.unwrap_tensor(robot_indices32), len(robot_indices32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(robot_indices32), len(robot_indices32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def sync_gripper_target(self):
        scale = 1.0
        target = self.ur3_dof_targets[:, 8]
        target = tensor_clamp(target, self.ur3_dof_lower_limits[8], self.ur3_dof_upper_limits[8])
        self.ur3_dof_targets[:, 6] = scale * target
        self.ur3_dof_targets[:, 7] = -1. * target
        self.ur3_dof_targets[:, 9] = scale * target
        self.ur3_dof_targets[:, 10] = -1. * target
        self.ur3_dof_targets[:, 11] = 1. * target

    def sync_gripper(self):
        scale = 1.0
        actuator = self.ur3_dof_pos[:, 8]
        actuator = tensor_clamp(actuator, self.ur3_dof_lower_limits[8], self.ur3_dof_upper_limits[8])
        self.ur3_dof_pos[:, 8] = actuator
        self.ur3_dof_pos[:, 6] = scale * actuator
        self.ur3_dof_pos[:, 7] = -1. * actuator
        self.ur3_dof_pos[:, 9] = scale * actuator
        self.ur3_dof_pos[:, 10] = -1. * actuator
        self.ur3_dof_pos[:, 11] = 1 * actuator

        vel = self.ur3_dof_vel[:, 9]
        self.ur3_dof_vel[:, 6] = scale * vel
        self.ur3_dof_vel[:, 7] = -scale * vel
        self.ur3_dof_vel[:, 9] = scale * vel
        self.ur3_dof_vel[:, 10] = -scale * vel
        self.ur3_dof_vel[:, 11] = scale * vel

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def stroke_to_angle(self, m):
        temp = self.ur3_dof_upper_limits[8] - self.angle_stroke_ratio * m
        return tensor_clamp(temp, self.ur3_dof_lower_limits[8], self.ur3_dof_upper_limits[8])

    def angle_to_stroke(self, rad):
        temp = (self.ur3_dof_upper_limits[8] - rad) / self.angle_stroke_ratio
        return tensor_clamp(temp, to_torch(0.0, device=self.device), to_torch(self.gripper_stroke, device=self.device))

    def solve(self, goal_pos, goal_rot, goal_grip, absolute=False):
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.ur3_grasp_rot[:], self.ur3_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.ur3_local_grasp_rot, self.ur3_local_grasp_pos)

        if absolute:
            pos_err = goal_pos - self.ur3_grasp_pos
            orn_err = orientation_error(quat_unit(goal_rot), self.ur3_grasp_rot)    # with quaternion normalize
        else:   # relative
            pos_err = goal_pos
            unit_quat = torch.zeros_like(goal_rot, device=self.device, dtype=torch.float)
            unit_quat[:, -1] = 1.0
            orn_err = orientation_error(quat_unit(goal_rot), unit_quat)    # unit_quat
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        d = 0.05  # damping term
        lmbda = torch.eye(6).to(self.device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6, 1)

        # robotiq gripper sync
        finger_dist = torch.norm(self.ur3_lfinger_pos - self.ur3_rfinger_pos, p=2, dim=-1).unsqueeze(-1)
        u2 = torch.zeros_like(u, device=self.device, dtype=torch.float)
        angle_err = self.stroke_to_angle(goal_grip) - self.stroke_to_angle(finger_dist)

        # PID control to avoid gripper oscillation
        Kp = 2.1
        u2[:, 8-6] = angle_err

        scale = 1.0
        u2[:, 6-6] = scale * u2[:, 8-6]
        u2[:, 7-6] = -scale * u2[:, 8-6]
        u2[:, 9-6] = scale * u2[:, 8-6]
        u2[:, 10-6] = -scale * u2[:, 8-6]
        u2[:, 11-6] = scale * u2[:, 8-6]

        _u = torch.cat((u, u2), dim=1)
        return _u.squeeze(-1)

    def pre_physics_step(self, actions):
        # print("actions: ", actions[61])
        # joint space control
        # self.actions = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float)
        # self.actions[:, :6] = actions[:, :6]
        # grip_act = torch.tanh(actions[:, -1])
        # self.actions[:, 8] = grip_act

        if self.use_ik:
            # TODO, rel. solve test code
            # actions[:, :3] = torch.zeros_like(actions[:, :3])
            # actions[:, 3:7] = torch.zeros_like(actions[:, 3:7])
            # actions[:, 6] = 1.0
            # q = quat_from_euler_xyz(0.0 * torch.ones(self.num_envs, device=self.device, dtype=torch.float),
            #                         0.0 * torch.ones(self.num_envs, device=self.device, dtype=torch.float),
            #                         0.0 * torch.ones(self.num_envs, device=self.device, dtype=torch.float))
            # actions[:, 3:7] = q
            # actions[:, 7] = 0.001

            # task space control
            self.actions = self.solve(goal_pos=actions[:, :3], goal_rot=actions[:, 3:7],
                                      goal_grip=actions[:, 7], absolute=False)
        else:
            self.actions = actions.clone().to(self.device)

        targets = self.ur3_dof_pos + self.ur3_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.ur3_dof_targets = tensor_clamp(targets, self.ur3_dof_lower_limits, self.ur3_dof_upper_limits)

        # gripper on/off
        # grip_act = torch.tanh(actions[:, -1])
        # bottle_grasp_angle = torch.tensor(self.stroke_to_angle(self.bottle_diameter - 0.009), device=self.device, dtype=torch.float)
        # # bottle_grasp_angle = torch.tensor(0.28, device=self.device, dtype=torch.float)
        # gripper_open_angle = self.ur3_dof_lower_limits[8]
        # self.ur3_dof_targets[:, 8] = torch.where(grip_act > 0.0, gripper_open_angle, bottle_grasp_angle)
        # self.ur3_dof_targets[:, 8] = bottle_grasp_angle
        self.sync_gripper_target()

        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ur3_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # compute task update status
        # self.compute_task()

        dof_pos_finger = self.angle_to_stroke(self.ur3_dof_pos[:, 8].unsqueeze(-1))
        done_envs = self.task.update_step_by_checking_arrive(ee_pos=self.ur3_grasp_pos, ee_rot=self.ur3_grasp_rot,
                                                             ee_grip=dof_pos_finger)
        self.reset_buf = torch.where(done_envs > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        self.task_update_buf = torch.where(self.progress_buf == 1,
                                           torch.ones_like(self.progress_buf), torch.zeros_like(self.progress_buf))

        env_ids = self.task_update_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.init_task_path(env_ids)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                # px = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.hand_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # bottle grasp pose
                # px = (self.bottle_grasp_pos[i] + quat_apply(self.bottle_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.bottle_grasp_pos[i] + quat_apply(self.bottle_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.bottle_grasp_pos[i] + quat_apply(self.bottle_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.bottle_grasp_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # bottle tip pose
                # px = (self.bottle_tip_pos[i] + quat_apply(self.bottle_tip_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.bottle_tip_pos[i] + quat_apply(self.bottle_tip_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.bottle_tip_pos[i] + quat_apply(self.bottle_tip_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.bottle_tip_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # bottle floor pose
                # px = (self.bottle_floor_pos[i] + quat_apply(self.bottle_floor_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.bottle_floor_pos[i] + quat_apply(self.bottle_floor_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.bottle_floor_pos[i] + quat_apply(self.bottle_floor_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.bottle_floor_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # cup pose
                # px = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.cup_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # cup tip pose
                # px = (self.cup_tip_pos[i] + quat_apply(self.cup_tip_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.cup_tip_pos[i] + quat_apply(self.cup_tip_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.cup_tip_pos[i] + quat_apply(self.cup_tip_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.cup_tip_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # ur3 grasp pose
                px = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.ur3_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # TODO
                # appr bottle pose for debug
                px = (self.appr_bottle_pos[i] + quat_apply(self.appr_bottle_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.appr_bottle_pos[i] + quat_apply(self.appr_bottle_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.appr_bottle_pos[i] + quat_apply(self.appr_bottle_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.appr_bottle_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # bottle pos init
                px = (self.bottle_pos_init[i] + quat_apply(self.bottle_rot_init[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.bottle_pos_init[i] + quat_apply(self.bottle_rot_init[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.bottle_pos_init[i] + quat_apply(self.bottle_rot_init[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.bottle_pos_init[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # cup pos init
                # px = (self.cup_pos_init[i] + quat_apply(self.cup_rot_init[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.cup_pos_init[i] + quat_apply(self.cup_rot_init[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.cup_pos_init[i] + quat_apply(self.cup_rot_init[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.cup_pos_init[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # direction line
                # p1 = self.bottle_grasp_pos[i].cpu().numpy()
                # p0 = self.ur3_grasp_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]] ,[0.85, 0.85, 0.1])

                # # finger pose
                # px = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.ur3_lfinger_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])
                #
                # px = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.ur3_rfinger_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])
                pass

    def init_task_path(self, env_ids):
        if not hasattr(self, "task"):
            self.task = TaskPathManager(num_env=self.num_envs, num_task_steps=2, device=self.device)

        self.task.reset_task(env_ids=env_ids)
        init_ur3_grasp_pos = to_torch([0.5, 0.0, 0.35], device=self.device).repeat((self.num_envs, 1))
        init_ur3_grasp_rot = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # 1)-1 initial pos variation
        pos_var_meter = 0.03
        pos_var = (torch.rand_like(init_ur3_grasp_pos) - 0.5) * 2.0
        init_ur3_grasp_pos += pos_var * pos_var_meter

        # 1)-2 initial rot variation
        def d2r(deg):
            return deg * (math.pi / 180.0)

        rot_var_deg = 15    # +-
        roll = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg
        pitch = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg
        yaw = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg
        q_var = quat_from_euler_xyz(roll=d2r(roll), pitch=d2r(pitch), yaw=d2r(yaw))
        init_ur3_grasp_rot = quat_mul(init_ur3_grasp_rot, q_var)

        # 1)-3 initial grip variation
        # For 2F-85 gripper, 0x00 --> full open with 85mm, 0xFF --> close
        # Unit: meter ~ [0.0, 0.085]
        init_ur3_grip = to_torch([0.08], device=self.device).repeat((self.num_envs, 1))
        grip_var = (torch.rand_like(init_ur3_grip) - 0.5) * 0.01   # grip. variation range: [0.075, 0.085]
        init_ur3_grip = torch.min(init_ur3_grip + grip_var, torch.tensor(self.gripper_stroke, device=self.device))

        self.task.push_task_pose(env_ids=env_ids,
                                 pos=init_ur3_grasp_pos, rot=init_ur3_grasp_rot, grip=init_ur3_grip)

        # 2) approach bottle
        bottle_pos = self.rigid_body_states[:, self.bottle_handle][:, 0:3]
        bottle_rot = self.rigid_body_states[:, self.bottle_handle][:, 3:7]
        if not hasattr(self, "bottle_pos_init") and not hasattr(self, "bottle_rot_init"):
            self.bottle_pos_init, self.bottle_rot_init = bottle_pos, bottle_rot
        self.bottle_pos_init[env_ids], self.bottle_rot_init[env_ids] = bottle_pos[env_ids], bottle_rot[env_ids]

        cup_pos = self.cup_states[:, 0:3]
        cup_rot = self.cup_states[:, 3:7]
        if not hasattr(self, "cup_pos_init") and not hasattr(self, "cup_rot_init"):
            self.cup_pos_init, self.cup_rot_init = cup_pos, cup_rot
        self.cup_pos_init[env_ids], self.cup_rot_init[env_ids] = cup_pos[env_ids], cup_rot[env_ids]

        vec = bottle_pos - cup_pos
        # dir_z[:, 2] = 0.0  # zero padding on z-axis to make it a planar vectors
        dir_xy = normalize(vec[:, :2], p=2.0, dim=-1)
        appr_bottle_pos = bottle_pos.clone().detach()
        appr_bottle_pos[:, :2] -= (dir_xy * self.bottle_diameter * 2.0)
        appr_bottle_pos[:, 2] *= 1.5

        # roll = torch.tensor([[0.0]] * len(bottle_pos), device=self.device)
        # pitch = torch.tensor([[0.0]] * len(bottle_pos), device=self.device)
        # bottle_rot_mat = quat_to_mat(bottle_rot)
        # rads = torch.bmm(bottle_rot_mat[:, :, 0].view(len(bottle_rot_mat), 1, 3), dir_z.view(len(dir_z), 3, 1)).arccos()
        # yaw = rads.squeeze(1)
        # to_bottle = quat_from_euler_xyz(roll, pitch, yaw).squeeze(1)

        appr_bottle_rot = bottle_rot.clone().detach()
        # appr_bottle_rot = quat_mul(appr_bottle_rot, to_bottle)    # TODO

        # mats = quat_to_mat(bottle_rot)
        # dir_x = mats[:, :, 2].cross(dir_z)
        # dir_y = dir_z.cross(dir_x)
        # appr_bottle_rot = mat_to_quat(torch.stack([dir_x, dir_y, dir_z], dim=-1))

        if not hasattr(self, "appr_bottle_pos") and not hasattr(self, "appr_bottle_rot"):
            self.appr_bottle_pos, self.appr_bottle_rot = appr_bottle_pos, appr_bottle_rot
        self.appr_bottle_pos[env_ids], self.appr_bottle_rot[env_ids] = appr_bottle_pos[env_ids], appr_bottle_rot[env_ids]

        # appr_bottle_rot = init_ur3_grasp_rot.clone()
        appr_bottle_grip = init_ur3_grip.clone()

        self.task.push_task_pose(env_ids=env_ids,
                                 pos=appr_bottle_pos, rot=appr_bottle_rot, grip=appr_bottle_grip)

    def calc_task_error(self):
        pass

    def calc_expert_action(self):
        des_pos, des_rot, des_grip = self.task.get_desired_pose()
        actions = self.solve(goal_pos=des_pos, goal_rot=des_rot, goal_grip=des_grip, absolute=True)
        return actions


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ur3_reward(
    reset_buf, progress_buf, actions,
    bottle_grasp_pos, bottle_grasp_rot, bottle_pos, bottle_rot, bottle_tip_pos, bottle_floor_pos,
    ur3_grasp_pos, ur3_grasp_rot, cup_pos, cup_rot, cup_tip_pos, liq_pos,
    ur3_lfinger_pos, ur3_rfinger_pos,
    lfinger_contact_net_force, rfinger_contact_net_force,
    gripper_forward_axis, bottle_up_axis, gripper_up_axis, cube_left_axis,
    num_envs, bottle_diameter, dist_reward_scale, rot_reward_scale, open_reward_scale,
    action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from fingertip to the cube
    d1 = torch.norm(ur3_grasp_pos - bottle_grasp_pos, p=2, dim=-1)

    axis1 = tf_vector(ur3_grasp_rot, gripper_up_axis)
    axis2 = tf_vector(bottle_grasp_rot, bottle_up_axis)
    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper

    axis3 = tf_vector(ur3_grasp_rot, gripper_forward_axis)[:, :2]
    axis4 = normalize(bottle_grasp_pos - ur3_grasp_pos)[:, :2]
    dot2 = torch.bmm(axis3.view(num_envs, 1, 2), axis4.view(num_envs, 2, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper

    axis5 = tf_vector(bottle_grasp_rot, bottle_up_axis)
    axis6 = bottle_up_axis
    dot3 = torch.bmm(axis5.view(num_envs, 1, 3), axis6.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # check the bottle fallen
    cube_fallen_reward = torch.where((1 - dot3) < 0.8, -1, 0)

    axis7 = tf_vector(cup_rot, bottle_up_axis)
    axis8 = bottle_up_axis
    dot4 = torch.bmm(axis7.view(num_envs, 1, 3), axis8.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

    rot_reward = 0.5 * torch.exp(-10.0 * (1.0 - dot1)) + 0.5 * torch.exp(-10.0 * (1.0 - dot2))

    lfd = torch.norm(ur3_lfinger_pos - bottle_grasp_pos, p=2, dim=-1)
    rfd = torch.norm(ur3_rfinger_pos - bottle_grasp_pos, p=2, dim=-1)
    approach_done = torch.where(d1 <= 0.02, 1.0, 0.0)
    grasp_done = torch.where((lfd < 0.035) & (rfd < 0.035), 1.0, 0.0) * approach_done

    # dist_reward = torch.exp(-5.0 * (0.2 * d1 + 0.4 * lfd + 0.4 * rfd))
    dist_reward = 0.2 * torch.exp(-7.0 * d1) + 0.8 * torch.exp(-7.0 * (lfd + rfd)) * approach_done
    dist_reward = torch.where(approach_done > 0.0,
                              torch.where((lfd < 0.035) & (rfd < 0.035), dist_reward + 7.0,
                                          torch.where((lfd < 0.037) & (rfd < 0.037), dist_reward + 5.0,
                                                      torch.where((lfd < 0.04) & (rfd < 0.04), dist_reward + 3.0,
                                                                  dist_reward + 1.0))),
                              dist_reward)

    # finger reward
    cube_z_axis = tf_vector(bottle_rot, gripper_up_axis)
    _lfinger_vec = ur3_lfinger_pos - bottle_pos
    _rfinger_vec = ur3_rfinger_pos - bottle_pos
    lfinger_dot = torch.bmm(_lfinger_vec.view(num_envs, 1, 3), cube_z_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    rfinger_dot = torch.bmm(_rfinger_vec.view(num_envs, 1, 3), cube_z_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    lfinger_len = torch.norm(_lfinger_vec - lfinger_dot.unsqueeze(-1) * cube_z_axis, p=2, dim=-1)
    rfinger_len = torch.norm(_rfinger_vec - rfinger_dot.unsqueeze(-1) * cube_z_axis, p=2, dim=-1)

    _lfinger_vec_len = _lfinger_vec.norm(p=2, dim=-1)
    _rfinger_vec_len = _rfinger_vec.norm(p=2, dim=-1)
    lfinger_vec = (_lfinger_vec.T / (_lfinger_vec_len + 1e-8)).T
    rfinger_vec = (_rfinger_vec.T / (_rfinger_vec_len + 1e-8)).T

    # cube lifting reward
    lift_reward_scale = 0.3
    des_height = 0.2

    bottle_height = bottle_grasp_pos[:, 2]

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions[:, :6] ** 2, dim=-1)

    # bottle_z = 0.195 * 0.55  # 0.107
    # finger_dist = torch.norm(ur3_lfinger_pos - ur3_rfinger_pos, p=2, dim=-1)
    is_lifted = torch.where((bottle_floor_pos[:, 2] > 0.07), 1.0, 0.0) * grasp_done
    is_grasped = torch.where((approach_done > 0.0) & ((lfd + rfd) <= 0.065), 1.0, 0.0)

    axis_bottle_up = tf_vector(bottle_rot, bottle_up_axis)
    axis_bottle_cup = normalize(cup_pos - bottle_pos)
    # axis_bottle_cup = tf_vector(cup_rot, -bottle_up_axis)
    # bottle_cup_dist_xy = torch.norm(bottle_pos[:, :2] - cup_pos[:, :2], p=2, dim=-1)
    # dot_pouring = torch.bmm(axis_bottle_up.view(num_envs, 1, 3), axis_bottle_cup.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    # pouring_reward = (torch.exp(-5.0 * (1.0 - dot_pouring)) - torch.sigmoid(bottle_cup_dist_xy - 0.2)) * is_lifted

    bottle_cup_tip_dist = torch.norm(bottle_tip_pos - cup_tip_pos, p=2, dim=-1)
    bottle_cup_tip_dist_xy = torch.norm(bottle_tip_pos[:, :2] - cup_tip_pos[:, :2], p=2, dim=-1)
    bottle_cup_tip_dist_z = torch.abs(bottle_tip_pos[:, 2] - cup_tip_pos[:, 2])
    approach_tip = torch.where(bottle_cup_tip_dist < 0.03, 1.0, 0.0)

    liq_cup_dist_xy = torch.norm(liq_pos[:, :2] - cup_pos[:, :2], p=2, dim=-1)
    liq_cup_dist = torch.norm(liq_pos - cup_pos, p=2, dim=-1)
    # bottle_height_rew = torch.where((bottle_pos[:, 2] - bottle_tip_pos[:, 2]) > 0, 1.0, 0.0)
    # bottle_height_rew = torch.min(0.1125 + (bottle_pos[:, 2] - bottle_tip_pos[:, 2]), torch.tensor(0.15))
    bottle_height_rew = 1.0 - torch.max(torch.tanh(20.0 * (bottle_tip_pos[:, 2] - bottle_floor_pos[:, 2])), -torch.tensor(0.5))
    pour_slope_on = torch.where((bottle_tip_pos[:, 2] - bottle_floor_pos[:, 2]) <= 0.0, 1.0, 0.0)

    bottle_slope = torch.min(bottle_floor_pos[:, 2] - bottle_tip_pos[:, 2], torch.zeros_like(bottle_floor_pos[:, 2]))
    pouring_reward = 0.1 * torch.exp(-10.0 * bottle_cup_tip_dist_xy) * is_lifted + \
                     0.2 * torch.exp(-10.0 * bottle_cup_tip_dist_z) * is_lifted + \
                     0.7 * torch.exp(-10.0 * liq_cup_dist) * is_lifted * approach_tip * pour_slope_on \
                     # 0.2 * bottle_height_rew * is_lifted * approach_tip

    # pouring_reward = torch.where(approach_tip > 1.0, pouring_reward + 1.0, pouring_reward)
    pouring_reward_scale = 10.0

    # drop_reward_scale = 10.0
    is_dropped = torch.where((is_lifted > 0.0) & (liq_pos[:, 2] < 0.03), 1.0, 0.0)
    # drop_reward = torch.exp(-5.0 * liq_cup_dist_xy) * is_dropped

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward + \
              pouring_reward_scale * pouring_reward \
              - action_penalty_scale * action_penalty \

    poured_reward_scale = 20.0
    poured_reward = torch.zeros_like(rewards)
    is_poured = (liq_cup_dist_xy < 0.015) & (liq_pos[:, 2] < 0.06)
    poured_reward = torch.where(is_poured, poured_reward + 1.0, poured_reward)
    rewards = torch.max(rewards, poured_reward_scale * poured_reward * is_lifted)

    # check the collisions of both fingers
    # _lfinger_contact_net_force = (lfinger_contact_net_force.T / (lfinger_contact_net_force.norm(p=2, dim=-1) + 1e-8)).T
    # _rfinger_contact_net_force = (rfinger_contact_net_force.T / (rfinger_contact_net_force.norm(p=2, dim=-1) + 1e-8)).T
    _lfinger_contact_net_force = normalize(lfinger_contact_net_force)
    _rfinger_contact_net_force = normalize(rfinger_contact_net_force)
    lf_force_dot = torch.bmm(_lfinger_contact_net_force.view(num_envs, 1, 3), gripper_up_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    rf_force_dot = torch.bmm(_rfinger_contact_net_force.view(num_envs, 1, 3), gripper_up_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

    # rewards = torch.where(lf_force_dot > 0.9, torch.ones_like(rewards) * -1.0, rewards)
    # rewards = torch.where(rf_force_dot > 0.9, torch.ones_like(rewards) * -1.0, rewards)

    # reset_buf = torch.where(lf_force_dot > 0.8, torch.ones_like(reset_buf), reset_buf)
    # reset_buf = torch.where(rf_force_dot > 0.8, torch.ones_like(reset_buf), reset_buf)

    # # bottle / cup fallen penalty
    # rewards = torch.where((bottle_height < 0.07) & (dot3 < 0.5), torch.ones_like(rewards) * -1.0, rewards)
    # rewards = torch.where(dot4 < 0.5, torch.ones_like(rewards) * -1.0, rewards)
    rewards = torch.where(dot4 < 0.5, torch.ones_like(rewards) * -1.0, rewards)  # paper cup fallen reward penalty

    # early stopping
    reset_buf = torch.where((bottle_floor_pos[:, 2] < 0.07) & (dot3 < 0.6), torch.ones_like(reset_buf), reset_buf)   # bottle fallen
    reset_buf = torch.where(dot4 < 0.5, torch.ones_like(reset_buf), reset_buf)  # paper cup fallen
    reset_buf = torch.where(is_poured, torch.ones_like(reset_buf), reset_buf)   # task success
    reset_buf = torch.where(is_dropped > 0.0, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((liq_cup_dist_xy > 0.5) | (bottle_height > des_height + 0.3), torch.ones_like(reset_buf), reset_buf)    # out of range

    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos)

    return global_franka_rot, global_franka_pos
