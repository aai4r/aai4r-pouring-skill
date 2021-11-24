import copy

import torch

from utils.torch_jit_utils import *
from tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


class UR3Pouring(BaseTask):

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
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "x"      # z
        self.up_axis_idx = 0    # 2
        self.dt = 1/60.

        num_obs = 18
        num_acts = 8

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
        self.angle_stroke_ratio = deg2rad(46) / self.gripper_stroke

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
        self.cup_states = self.root_state_tensor[:, 2]

        self.contact_net_force = gymtorch.wrap_tensor(contact_net_force_tensor)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur3_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.num_actors = self.root_state_tensor.size()[1]
        self.global_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset(torch.arange(self.num_envs, device=self.device))

        # gripper limit for bottle
        blim = self.stroke_to_angle(self.bottle_diameter - 0.005)
        self.ur3_dof_lower_limits[7] = -blim
        self.ur3_dof_lower_limits[10] = -blim

        self.ur3_dof_upper_limits[6] = blim
        self.ur3_dof_upper_limits[8] = blim
        self.ur3_dof_upper_limits[9] = blim
        self.ur3_dof_upper_limits[11] = blim

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
        asset_options.vhacd_params.resolution = 30000
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

    def create_asset_fluid_particle(self):
        r = 0.012
        self.expr = [[0, 0], [0, -r], [-r, 0], [0, r], [r, 0]]
        self.num_liq_particles = 5

        asset_options = gymapi.AssetOptions()
        asset_options.density = 997
        liquid_asset = self.gym.create_sphere(self.sim, 0.004, asset_options)
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
        fluid_asset = self.create_asset_fluid_particle()

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
        bottle_start_pose.p.z = self.bottle_height * 0.55

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
        self.max_agg_bodies = num_ur3_bodies + num_bottle_bodies + num_cup_bodies #+ self.num_liq_particles
        self.max_agg_shapes = num_ur3_shapes + num_bottle_shapes + num_cup_shapes #+ self.num_liq_particles

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

            # (1) Create Robot, last number 0: considering self-collision
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

            # (3) Create Cup
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
        _lfinger_pose.p += gymapi.Vec3(*get_axis_params(0.008, finger_pose_axis))
        self.ur3_local_lfinger_pos = to_torch([_lfinger_pose.p.x + fwd_offset, _lfinger_pose.p.y,
                                               _lfinger_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.ur3_local_lfinger_rot = to_torch([_lfinger_pose.r.x, _lfinger_pose.r.y,
                                               _lfinger_pose.r.z, _lfinger_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        _rfinger_pose = gymapi.Transform()
        _rfinger_pose.p += gymapi.Vec3(*get_axis_params(0.008, finger_pose_axis))
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

        self.gripper_forward_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.bottle_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cube_left_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        self.bottle_grasp_pos = torch.zeros_like(self.bottle_local_grasp_pos)
        self.bottle_grasp_rot = torch.zeros_like(self.bottle_local_grasp_rot)

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
            self.bottle_grasp_pos, self.bottle_grasp_rot, self.bottle_pos, self.bottle_rot,
            self.ur3_grasp_pos, self.ur3_grasp_rot, self.cup_pos, self.cup_rot,
            self.ur3_lfinger_pos, self.ur3_rfinger_pos,
            self.contact_net_force.view(self.num_envs, self.max_agg_bodies, -1)[:, self.lfinger_handle],
            self.contact_net_force.view(self.num_envs, self.max_agg_bodies, -1)[:, self.rfinger_handle],
            self.gripper_forward_axis, self.bottle_up_axis, self.gripper_up_axis, self.cube_left_axis,
            self.num_envs, self.bottle_diameter, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
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

        self.bottle_grasp_rot[:], self.bottle_grasp_pos[:] = \
            tf_combine(self.bottle_rot, self.bottle_pos, self.bottle_local_grasp_rot, self.bottle_local_grasp_pos)

        dof_pos_scaled = (2.0 * (self.ur3_dof_pos - self.ur3_dof_lower_limits)
                          / (self.ur3_dof_upper_limits - self.ur3_dof_lower_limits) - 1.0)
        dof_pos_scaled = torch.index_select(dof_pos_scaled, 1, self.indices)
        dof_vel = torch.index_select(self.ur3_dof_vel, 1, self.indices)

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
        self.obs_buf = torch.cat((dof_pos_finger,
                                  self.ur3_grasp_pos, self.ur3_grasp_rot,
                                  self.bottle_pos, self.bottle_rot,
                                  self.cup_pos), dim=-1)

        env_id = 61
        # d1 = torch.norm(self.ur3_grasp_pos - self.bottle_grasp_pos, p=2, dim=-1)
        # appr_done = d1 < 0.005
        # print("d1: {:3f}, appr_done: {}".format(d1[env_id], appr_done[env_id]))
        # print("fin_dist: {}, dof_pos: {}".format(finger_dist[env_id], dof_pos_finger[env_id]))
        # print("actions: ", self.actions[env_id])

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

        robot_indices = self.global_indices[env_ids, :1].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(robot_indices), len(robot_indices))

        # reset bottle
        bottle_liquid_indices = self.global_indices[env_ids, 1:].flatten()

        rand_z_angle = torch.rand(len(env_ids)).uniform_(deg2rad(-90.0), deg2rad(90.0))
        quat = []   # z-axis cube orientation randomization
        for gamma in rand_z_angle:
            _q = gymapi.Quat.from_euler_zyx(0, 0, gamma)
            quat.append(torch.FloatTensor([_q.x, _q.y, _q.z, _q.w]))
        quat = torch.stack(quat).to(self.device)

        pick = self.default_bottle_states[env_ids]
        pick[:, 3:7] = quat
        xy_scale = to_torch([0.2, 0.5, 0.0,            # position
                             0.0, 0.0, 0.0, 0.0,       # rotation (quat)
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

        # apply
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(bottle_liquid_indices),
                                                     len(bottle_liquid_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, :1].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ur3_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def sync_gripper_target(self):
        scale = 1.0
        target = self.ur3_dof_targets[:, 8]
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
        return deg2rad(46) - self.angle_stroke_ratio * m

    def angle_to_stroke(self, rad):
        return (deg2rad(46) - rad) / self.angle_stroke_ratio

    def solve(self, goal_pos, goal_rot, goal_grip, abs=False):
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.ur3_grasp_rot[:], self.ur3_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.ur3_local_grasp_rot, self.ur3_local_grasp_pos)

        if abs:
            # absolute
            pos_err = goal_pos - self.ur3_grasp_pos
            orn_err = orientation_error(quat_unit(goal_rot), self.ur3_grasp_rot)    # with quaternion normalize
        else:
            # relative
            pos_err = goal_pos
            unit_quat = torch.zeros_like(goal_rot, device=self.device, dtype=torch.float)
            unit_quat[:, -1] = 1.0
            orn_err = orientation_error(quat_unit(goal_rot), unit_quat)    # unit_quat
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        d = 0.1  # damping term
        lmbda = torch.eye(6).to(self.device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6, 1)

        # robotiq gripper sync
        u2 = torch.zeros_like(u, device=self.device, dtype=torch.float)
        # u2[:, 8-6] = self.stroke_to_angle(torch.tanh(goal_grip.unsqueeze(-1)))
        # u2[:, 8-6] = torch.tanh(goal_grip.unsqueeze(-1))
        u2[:, 8 - 6] = (self.ur3_dof_pos[:, -1] + goal_grip).unsqueeze(-1)
        # temp = self.stroke_to_angle(self.angle_to_stroke(self.ur3_dof_pos[:, -1]) + goal_grip)
        # u2[:, 8-6] = temp.unsqueeze(-1)
        scale = 1.0

        u2[:, 6-6] = scale * u2[:, 8-6]
        u2[:, 7-6] = -scale * u2[:, 8-6]
        u2[:, 9-6] = scale * u2[:, 8-6]
        u2[:, 10-6] = -scale * u2[:, 8-6]
        u2[:, 11-6] = scale * u2[:, 8-6]

        _u = torch.cat((u, u2), dim=1)
        return _u.squeeze(-1)

    def pre_physics_step(self, actions):
        # self.actions = actions.clone().to(self.device)
        # print("action: ", actions.shape, actions[0, :])

        # joint space control
        # self.actions = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float)
        # self.actions[:, :6] = actions[:, :6]
        # grip_act = torch.tanh(actions[:, -1])
        # self.actions[:, 8] = grip_act

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
                                  goal_grip=actions[:, 7], abs=False)

        targets = self.ur3_dof_targets + self.ur3_dof_speed_scales * self.dt * self.actions * self.action_scale
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

                # # cup pose
                # px = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.cup_pos[i].cpu().numpy()
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

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ur3_reward(
    reset_buf, progress_buf, actions,
    bottle_grasp_pos, bottle_grasp_rot, bottle_pos, bottle_rot,
    ur3_grasp_pos, ur3_grasp_rot, cup_pos, cup_rot,
    ur3_lfinger_pos, ur3_rfinger_pos,
    lfinger_contact_net_force, rfinger_contact_net_force,
    gripper_forward_axis, bottle_up_axis, gripper_up_axis, cube_left_axis,
    num_envs, bottle_diameter, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from fingertip to the cube
    d1 = torch.norm(ur3_grasp_pos - bottle_grasp_pos, p=2, dim=-1)
    # d2 = torch.norm(ur3_lfinger_pos - bottle_grasp_pos, p=2, dim=-1)
    # d3 = torch.norm(ur3_rfinger_pos - bottle_grasp_pos, p=2, dim=-1)
    # dist_reward = torch.exp(-10.0 * d1)

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

    rot_reward = 0.5 * torch.exp(-5.0 * (1.0 - dot1)) + 0.5 * torch.exp(-5.0 * (1.0 - dot2))
    # rot_reward = 1.0 - torch.tanh((10/2) * ((1.0 - dot1) + (1.0 - dot2)))

    d_xy = torch.norm(ur3_grasp_pos[:, :2] - bottle_grasp_pos[:, :2], p=2, dim=-1)
    d_z = torch.abs(ur3_grasp_pos[:, 2] - bottle_grasp_pos[:, 2])

    approach_done = d1 <= 0.01
    lfinger_dist = torch.norm(ur3_lfinger_pos - bottle_grasp_pos, p=2, dim=-1)
    rfinger_dist = torch.norm(ur3_rfinger_pos - bottle_grasp_pos, p=2, dim=-1)

    grasp_reward_scale = 0.02
    grasp_reward = torch.zeros_like(rot_reward)

    lfd = torch.norm(ur3_lfinger_pos - bottle_grasp_pos, p=2, dim=-1)
    rfd = torch.norm(ur3_rfinger_pos - bottle_grasp_pos, p=2, dim=-1)
    d_fd = torch.norm(lfd - rfd, p=2, dim=-1)
    d_z = torch.abs(ur3_grasp_pos[:, 2] - bottle_grasp_pos[:, 2])

    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(d_z <= 0.02,
                                       torch.where(lfd <= 0.05,
                                                   torch.where(rfd <= 0.05,
                                                               around_handle_reward + 1.0, around_handle_reward),
                                                   around_handle_reward),
                                       around_handle_reward)

    # vector approach
    lr_finger_axis = normalize(ur3_lfinger_pos - ur3_rfinger_pos)
    rl_finger_axis = normalize(ur3_rfinger_pos - ur3_lfinger_pos)
    bottle_lfinger_axis = normalize(bottle_grasp_pos - ur3_lfinger_pos)
    bottle_rfinger_axis = normalize(bottle_grasp_pos - ur3_rfinger_pos)

    ldot = torch.bmm(bottle_lfinger_axis.view(num_envs, 1, 3), rl_finger_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    rdot = torch.bmm(bottle_rfinger_axis.view(num_envs, 1, 3), lr_finger_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

    d1_xy = torch.norm(ur3_grasp_pos[:, :2] - bottle_grasp_pos[:, :2], p=2, dim=-1)
    d1_z = torch.abs(ur3_grasp_pos[:, 2] - bottle_grasp_pos[:, 2])
    dist_reward = torch.exp(-5.0 * (0.9 * d1 + 0.05 * lfd + 0.05 * rfd)) #- 0.1 * torch.tanh((10/2) * (lfd + rfd))  # 0.029
    dist_reward = torch.where(d1 <= 0.02, dist_reward * 2, dist_reward)     # approach bonus

    # torch.abs(ur3_grasp_pos[:, 2] - bottle_grasp_pos[:, 2])
    # dist_reward = torch.where(approach_done, dist_reward + 1.0, dist_reward)
    # dist_reward = 1.0 - torch.tanh((10/3) * (d1 + torch.abs(0.029 - lfd) + torch.abs(0.029 - rfd)))
    # dist_reward = 1.0 - torch.tanh((10 / 3) * (d1 + lfd + rfd))
    # dist_reward = torch.where(approach_done, dist_reward * 2, dist_reward)
    # grasp_reward = 0.5 * torch.exp(-5.0 * (1.0 - ldot)) + 0.5 * torch.exp(-5.0 * (1.0 - rdot))

    # finger_dist = torch.norm(ur3_lfinger_pos - ur3_rfinger_pos, p=2, dim=-1)
    # grasp_reward = torch.where(approach_done, -finger_dist, finger_dist)

    # grasp_reward = torch.where(d_z <= 0.02,
    #                            torch.where(ldot > 0.99,  # 0.99 = 8.1 deg
    #                                        torch.where(rdot > 0.99,
    #                                                    (0.029 - lfd) + (0.029 - rfd),
    #                                                    grasp_reward),
    #                                        grasp_reward),
    #                            grasp_reward)

    # grasp_reward = torch.where(d_z <= 0.015,
    #                            torch.where(lfd <= 0.04,
    #                                        torch.where(rfd <= 0.04,
    #                                                    0.5 * torch.exp(-15.0 * torch.abs(0.029 - lfd)) +
    #                                                    0.5 * torch.exp(-15.0 * torch.abs(0.029 - rfd)),
    #                                                    grasp_reward),
    #                                        grasp_reward),
    #                            grasp_reward)

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
    finger_around_dot = torch.bmm(lfinger_vec.view(num_envs, 1, 3), rfinger_vec.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

    # cube lifting reward
    lift_reward_scale = 1.5
    des_height = 0.3

    bottle_lift_pos = bottle_grasp_pos.clone()
    bottle_lift_pos[:, 2] = des_height
    bottle_height = bottle_grasp_pos[:, 2]
    d2 = torch.norm(ur3_grasp_pos - bottle_lift_pos, p=2, dim=-1)
    # lift_reward = torch.where(grasp_done, torch.exp(-20.0 * d2), lift_reward)

    # lift_reward = torch.exp(-15.0 * lift_dist)
    # lift_reward = torch.where(grasp_done, lift_reward + torch.exp(-100.0 * lift_dist), lift_reward)
    # lift_reward = torch.min(cube_pos[:, 2], torch.zeros_like(rot_reward))

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    bottle_z = 0.195 * 0.55  # 0.107
    lift_reward = torch.zeros_like(rot_reward)
    lift_reward = torch.where(bottle_height > 0.3, lift_reward + 1.0,
                              torch.where(bottle_height > 0.2, lift_reward + 0.8,
                                          torch.where(bottle_height > 0.15, lift_reward + 0.6,
                                                      torch.where(bottle_height > 0.2, lift_reward + 0.4, lift_reward))))

    is_lifted = torch.where(approach_done & (bottle_height > 0.3), 1.0, 0.0)

    align_reward_scale = 2.0
    # align_reward = (1.0 - torch.tanh(10.0 * torch.norm(bottle_pos[:, :2] - cup_pos[:, :2], p=2, dim=-1))) * is_lifted
    align_reward = torch.exp(-5.0 * torch.norm(bottle_pos[:, :2] - cup_pos[:, :2], p=2, dim=-1)) * is_lifted

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward + \
              lift_reward_scale * is_lifted \
              # + align_reward_scale * align_reward
              # - action_penalty_scale * action_penalty
              # grasp_reward_scale * grasp_reward
              # + lift_reward_scale * is_lifted \
              #+ align_reward_scale * align_reward \


              # lift_reward_scale * is_lifted + align_reward_scale * align_reward
              # lift_reward_scale * lift_reward + align_reward_scale * align_reward + \


    # + align_reward_scale * align_reward
    # grasp_reward_scale * grasp_reward + around_handle_reward_scale * around_handle_reward + \

    # rewards = torch.where(lift_reward > 0.3, rewards + 1.0,
    #                       torch.where(lift_reward > 0.2, rewards + 0.8,
    #                                   torch.where(lift_reward > 0.15, rewards + 0.6,
    #                                               torch.where(lift_reward > 0.2, rewards + 0.4, rewards))))

    # rewards = torch.where(lift_reward > 0.3,
    #                       rewards + 2.0 * torch.exp(-10.0 * torch.norm(bottle_pos[:, :2] - cup_pos[:, :2])),
    #                       rewards)

    # grasp = torch.norm(finger_dist - (cube_size - 0.002), p=2, dim=-1)
    # grasp_reward = torch.exp(-10.0 * grasp)
    # grasp_reward_scale = 10.0
    # rewards = torch.where(approach_done,
    #                       rewards * 1.5 + grasp_reward_scale * grasp_reward,
    #                       rewards + finger_dist_reward_scale * finger_dist_reward)

    # rewards = torch.where(lift_dist < 0.01, rewards * 3.0, rewards)
    # rewards = torch.where(bottle_pos[:, 2] >= des_height, rewards * 2.0, rewards)

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

    # # bottle fallen penalty
    # rewards = torch.where((bottle_height < 0.3) & (dot3 < 0.85), torch.ones_like(rewards) * -1.0, rewards)
    # rewards = torch.where(dot4 < 0.85, torch.ones_like(rewards) * -1.0, rewards)

    reset_buf = torch.where((bottle_height < 0.3) & (dot3 < 0.85), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(dot4 < 0.85, torch.ones_like(reset_buf), reset_buf)
    # reset_buf = torch.where(lift_dist < 0.01, torch.ones_like(reset_buf), reset_buf)
    # reset_buf = torch.where(bottle_pos[:, 2] >= des_height, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos)

    return global_franka_rot, global_franka_pos
