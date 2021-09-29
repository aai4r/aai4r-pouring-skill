# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import torch

from utils.torch_jit_utils import *
from tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class MirobotCube(BaseTask):

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

        num_obs = 23
        num_acts = 8

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_net_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.mirobot_default_dof_pos = to_torch([deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), 0.017, 0.017], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.mirobot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.mirobot_dof_pos = self.mirobot_dof_state[..., 0]
        self.mirobot_dof_vel = self.mirobot_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.cube_states = self.root_state_tensor[:, 1]

        self.contact_net_force = gymtorch.wrap_tensor(contact_net_force_tensor)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        mirobot_asset_file = "urdf/mirobot_description/mirobot.urdf"
        cube_asset_file = "urdf/objects/cube.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            mirobot_asset_file = self.cfg["env"]["asset"].get("assetFileNameMirobot", mirobot_asset_file)
            cube_asset_file = self.cfg["env"]["asset"].get("assetFileNameCube", cube_asset_file)

        # load mirobot asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        mirobot_asset = self.gym.load_asset(self.sim, asset_root, mirobot_asset_file, asset_options)

        # load cube asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.density = 40
        cube_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, asset_options)

        mirobot_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        mirobot_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(mirobot_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(mirobot_asset)

        print("num mirobot bodies: ", self.num_franka_bodies)
        print("num mirobot dofs: ", self.num_franka_dofs)

        # set franka dof properties
        mirobot_dof_props = self.gym.get_asset_dof_properties(mirobot_asset)
        self.mirobot_dof_lower_limits = []
        self.mirobot_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            mirobot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                mirobot_dof_props['stiffness'][i] = mirobot_dof_stiffness[i]
                mirobot_dof_props['damping'][i] = mirobot_dof_damping[i]
            else:
                mirobot_dof_props['stiffness'][i] = 7000.0
                mirobot_dof_props['damping'][i] = 50.0

            self.mirobot_dof_lower_limits.append(mirobot_dof_props['lower'][i])
            self.mirobot_dof_upper_limits.append(mirobot_dof_props['upper'][i])

        self.mirobot_dof_lower_limits = to_torch(self.mirobot_dof_lower_limits, device=self.device)
        self.mirobot_dof_upper_limits = to_torch(self.mirobot_dof_upper_limits, device=self.device)
        self.mirobot_dof_speed_scales = torch.ones_like(self.mirobot_dof_lower_limits)
        self.mirobot_dof_speed_scales[[6, 7]] = 0.1
        mirobot_dof_props['effort'][6] = 200
        mirobot_dof_props['effort'][7] = 200

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        cube_start_pose = gymapi.Transform()
        cube_start_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx))

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(mirobot_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(mirobot_asset)
        num_cube_bodies = self.gym.get_asset_rigid_body_count(cube_asset)
        num_cube_shapes = self.gym.get_asset_rigid_shape_count(cube_asset)
        self.max_agg_bodies = num_franka_bodies + num_cube_bodies
        self.max_agg_shapes = num_franka_shapes + num_cube_shapes

        self.mirobots = []
        self.cubes = []
        self.default_cube_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)

            mirobot_actor = self.gym.create_actor(env_ptr, mirobot_asset, franka_start_pose, "mirobot", i, 1)
            self.gym.set_actor_dof_properties(env_ptr, mirobot_actor, mirobot_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)

            # cube_pose = cube_start_pose
            # cube_pose.p.x = self.start_position_noise * (np.random.rand() - 0.5)
            # dz = 0.025
            # dy = np.random.rand() - 0.5
            # cube_pose.p.y = self.start_position_noise * dy
            # cube_pose.p.z = dz
            cube_actor = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", i, 2)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)

            self.default_cube_states.append([cube_start_pose.p.x + 0.23, cube_start_pose.p.y, cube_start_pose.p.z + 0.02,
                                             cube_start_pose.r.x, cube_start_pose.r.y, cube_start_pose.r.z, cube_start_pose.r.w,
                                             0, 0, 0, 0, 0, 0])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.mirobots.append(mirobot_actor)
            self.cubes.append(cube_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, mirobot_actor, "Link6")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, mirobot_actor, "left_finger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, mirobot_actor, "right_finger")
        self.cube_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cube_actor, "cube")
        self.default_cube_states = to_torch(self.default_cube_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.init_data()

    def init_data(self):
        # self.lfinger_idxs = []
        # self.rfinger_idxs = []
        # for i in range(self.num_envs):
        #     lfinger_idx = self.gym.find_actor_rigid_body_index(self.envs[i], mirobot_handle, "left_finger", gymapi.DOMAIN_SIM)


        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.mirobots[0], "Link6")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.mirobots[0], "left_finger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.mirobots[0], "right_finger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 2     # z-axis
        mirobot_local_grasp_pose = hand_pose_inv * finger_pose
        mirobot_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(-0.025, grasp_pose_axis))
        self.mirobot_local_grasp_pos = to_torch([mirobot_local_grasp_pose.p.x, mirobot_local_grasp_pose.p.y,
                                                mirobot_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.mirobot_local_grasp_rot = to_torch([mirobot_local_grasp_pose.r.x, mirobot_local_grasp_pose.r.y,
                                                mirobot_local_grasp_pose.r.z, mirobot_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        _lfinger_pose = gymapi.Transform()
        _lfinger_pose.p += gymapi.Vec3(*get_axis_params(0.025, grasp_pose_axis))
        self.mirobot_local_lfinger_pos = to_torch([_lfinger_pose.p.x, _lfinger_pose.p.y,
                                                   _lfinger_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.mirobot_local_lfinger_rot = to_torch([_lfinger_pose.r.x, _lfinger_pose.r.y,
                                                   _lfinger_pose.r.z, _lfinger_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        _rfinger_pose = gymapi.Transform()
        _rfinger_pose.p += gymapi.Vec3(*get_axis_params(0.025, grasp_pose_axis))
        self.mirobot_local_rfinger_pos = to_torch([_rfinger_pose.p.x, _rfinger_pose.p.y,
                                                   _rfinger_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.mirobot_local_rfinger_rot = to_torch([_rfinger_pose.r.x, _rfinger_pose.r.y,
                                                   _rfinger_pose.r.z, _rfinger_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        cube_local_grasp_pose = gymapi.Transform()
        cube_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.005, grasp_pose_axis))
        cube_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cube_local_grasp_pos = to_torch([cube_local_grasp_pose.p.x, cube_local_grasp_pose.p.y,
                                              cube_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cube_local_grasp_rot = to_torch([cube_local_grasp_pose.r.x, cube_local_grasp_pose.r.y,
                                              cube_local_grasp_pose.r.z, cube_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cube_down_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_right_axis = to_torch([0, -1, 0], device=self.device).repeat((self.num_envs, 1))
        self.cube_left_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        self.cube_grasp_pos = torch.zeros_like(self.cube_local_grasp_pos)
        self.cube_grasp_rot = torch.zeros_like(self.cube_local_grasp_rot)

        self.mirobot_grasp_pos = torch.zeros_like(self.mirobot_local_grasp_pos)
        self.mirobot_grasp_rot = torch.zeros_like(self.mirobot_local_grasp_rot)
        self.mirobot_grasp_rot[..., -1] = 1  # xyzw

        self.mirobot_lfinger_pos = torch.zeros_like(self.mirobot_local_grasp_pos)
        self.mirobot_rfinger_pos = torch.zeros_like(self.mirobot_local_grasp_pos)
        self.mirobot_lfinger_rot = torch.zeros_like(self.mirobot_local_grasp_rot)
        self.mirobot_rfinger_rot = torch.zeros_like(self.mirobot_local_grasp_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_mirobot_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.cube_grasp_pos, self.cube_grasp_rot, self.cube_pos, self.cube_rot,
            self.mirobot_grasp_pos, self.mirobot_grasp_rot,
            self.mirobot_lfinger_pos, self.mirobot_rfinger_pos,
            self.contact_net_force.view(self.num_envs, self.max_agg_bodies, -1)[:, self.lfinger_handle],
            self.contact_net_force.view(self.num_envs, self.max_agg_bodies, -1)[:, self.rfinger_handle],
            self.gripper_forward_axis, self.cube_down_axis, self.gripper_right_axis, self.cube_left_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.mirobot_grasp_rot[:], self.mirobot_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.mirobot_local_grasp_rot, self.mirobot_local_grasp_pos)

        self.mirobot_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.mirobot_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.mirobot_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.mirobot_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.mirobot_lfinger_rot, self.mirobot_lfinger_pos = \
            compute_grasp_transforms(self.mirobot_lfinger_rot, self.mirobot_lfinger_pos,
                                     self.mirobot_local_lfinger_rot, self.mirobot_local_lfinger_pos)

        self.mirobot_rfinger_rot, self.mirobot_rfinger_pos = \
            compute_grasp_transforms(self.mirobot_rfinger_rot, self.mirobot_rfinger_pos,
                                     self.mirobot_local_rfinger_rot, self.mirobot_local_rfinger_pos)

        # cube info
        self.cube_pos = self.rigid_body_states[:, self.cube_handle][:, 0:3]
        self.cube_rot = self.rigid_body_states[:, self.cube_handle][:, 3:7]

        self.cube_grasp_rot[:], self.cube_grasp_pos[:] = \
            tf_combine(self.cube_rot, self.cube_pos, self.cube_local_grasp_rot, self.cube_local_grasp_pos)

        dof_pos_scaled = (2.0 * (self.mirobot_dof_pos - self.mirobot_dof_lower_limits)
                          / (self.mirobot_dof_upper_limits - self.mirobot_dof_lower_limits) - 1.0)
        to_target_pos = self.cube_grasp_pos - self.mirobot_grasp_pos
        to_target_rot = quat_mul(quat_conjugate(self.cube_grasp_rot), self.mirobot_grasp_rot)
        # to_2nd_target_pos_z = (0.1 - self.cube_pos[:, 2].unsqueeze(-1)).norm()
        # cube_pos_z = self.cube_pos[:, 2].unsqueeze(-1)
        self.obs_buf = torch.cat((dof_pos_scaled, self.mirobot_dof_vel * self.dof_vel_scale,
                                  to_target_pos, to_target_rot), dim=-1)

        return self.obs_buf

    def reset(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # reset mirobot
        pos = tensor_clamp(
            self.mirobot_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.mirobot_dof_lower_limits, self.mirobot_dof_upper_limits)
        self.mirobot_dof_pos[env_ids, :] = pos
        self.mirobot_dof_vel[env_ids, :] = torch.zeros_like(self.mirobot_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # reset cube
        cube_indices = self.global_indices[env_ids, 1].flatten()

        rand_z_angle = torch.rand(len(env_ids)).uniform_(deg2rad(-90.0), deg2rad(90.0))
        quat = []   # z-axis cube orientation randomization
        for gamma in rand_z_angle:
            _q = gymapi.Quat.from_euler_zyx(0, 0, gamma)
            quat.append(torch.FloatTensor([_q.x, _q.y, _q.z, _q.w]))
        quat = torch.stack(quat).to(self.device)

        pick = self.default_cube_states[env_ids]
        pick[:, 3:7] = quat
        xy_scale = to_torch([0.12, 0.2, 0.0,            # position
                             0.0, 0.0, 0.0, 0.0,        # rotation
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(pick), 1)
        rand_cube_pos = (torch.rand_like(pick, device=self.device, dtype=torch.float) - 0.5) * xy_scale
        self.cube_states[env_ids] = pick + rand_cube_pos
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(cube_indices), len(cube_indices))

        # apply
        multi_env_ids_int32 = self.global_indices[env_ids, :1].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # act_grip_mean = actions[:, 6:8].mean(dim=-1)
        # actions[:, 6], actions[:, 7] = act_grip_mean, act_grip_mean
        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.mirobot_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.mirobot_dof_lower_limits, self.mirobot_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

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

                # cube grasp pose
                px = (self.cube_grasp_pos[i] + quat_apply(self.cube_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cube_grasp_pos[i] + quat_apply(self.cube_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cube_grasp_pos[i] + quat_apply(self.cube_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cube_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # mirobot grasp pose
                px = (self.mirobot_grasp_pos[i] + quat_apply(self.mirobot_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.mirobot_grasp_pos[i] + quat_apply(self.mirobot_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.mirobot_grasp_pos[i] + quat_apply(self.mirobot_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.mirobot_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # finger pose
                # px = (self.mirobot_lfinger_pos[i] + quat_apply(self.mirobot_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.mirobot_lfinger_pos[i] + quat_apply(self.mirobot_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.mirobot_lfinger_pos[i] + quat_apply(self.mirobot_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.mirobot_lfinger_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])
                #
                # px = (self.mirobot_rfinger_pos[i] + quat_apply(self.mirobot_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.mirobot_rfinger_pos[i] + quat_apply(self.mirobot_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.mirobot_rfinger_pos[i] + quat_apply(self.mirobot_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.mirobot_rfinger_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_mirobot_reward(
    reset_buf, progress_buf, actions,
    cube_grasp_pos, cube_grasp_rot, cube_pos, cube_rot,
    mirobot_grasp_pos, mirobot_grasp_rot,
    mirobot_lfinger_pos, mirobot_rfinger_pos,
    lfinger_contact_net_force, rfinger_contact_net_force,
    gripper_forward_axis, cube_down_axis, gripper_right_axis, cube_left_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from fingertip to the cube
    d = torch.norm(mirobot_grasp_pos - cube_grasp_pos, p=2, dim=-1)
    dist_reward = torch.exp(-10.0 * d)

    axis1 = tf_vector(mirobot_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(cube_grasp_rot, cube_down_axis)
    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper

    axis3 = tf_vector(mirobot_grasp_rot, gripper_right_axis)
    axis4 = tf_vector(cube_grasp_rot, cube_left_axis)
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper

    axis5 = tf_vector(cube_grasp_rot, gripper_forward_axis)
    axis6 = gripper_forward_axis
    dot3 = torch.bmm(axis5.view(num_envs, 1, 3), axis6.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # check the cube fallen
    cube_fallen_reward = torch.where((1 - dot3) < 0.8, -1, 0)
    rot_reward = 0.5 * torch.exp(-5.0 * (1.0 - dot1)) + 0.5 * torch.exp(-5.0 * (1.0 - dot2)) - cube_fallen_reward

    approach_done = (d <= 0.01) & ((1 - dot1) <= 0.2) & ((1 - dot2) <= 0.2)

    finger_dist_reward = torch.zeros_like(rot_reward)
    finger_dist = torch.norm(mirobot_lfinger_pos - mirobot_rfinger_pos, p=2, dim=-1)
    cube_size = 0.02
    finger_dist_reward = torch.where(finger_dist > cube_size, finger_dist_reward + 0.1, finger_dist_reward)

    grasp_reward_scale = 1.0
    grasp_reward = torch.zeros_like(rot_reward)
    grasp_reward = torch.where(approach_done,
                               grasp_reward + 1.0 - 100.0 * torch.max(finger_dist - cube_size, torch.zeros_like(finger_dist) - 0.005),
                               finger_dist_reward)

    grasp_done = approach_done & (finger_dist <= cube_size)

    # finger reward
    cube_z_axis = tf_vector(cube_rot, gripper_forward_axis)
    _lfinger_vec = mirobot_lfinger_pos - cube_pos
    _rfinger_vec = mirobot_rfinger_pos - cube_pos
    lfinger_dot = torch.bmm(_lfinger_vec.view(num_envs, 1, 3), cube_z_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    rfinger_dot = torch.bmm(_rfinger_vec.view(num_envs, 1, 3), cube_z_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    lfinger_len = torch.norm(_lfinger_vec - lfinger_dot.unsqueeze(-1) * cube_z_axis, p=2, dim=-1)
    rfinger_len = torch.norm(_rfinger_vec - rfinger_dot.unsqueeze(-1) * cube_z_axis, p=2, dim=-1)

    _lfinger_vec_len = _lfinger_vec.norm(p=2, dim=-1)
    _rfinger_vec_len = _rfinger_vec.norm(p=2, dim=-1)
    lfinger_vec = (_lfinger_vec.T / (_lfinger_vec_len + 1e-8)).T
    rfinger_vec = (_rfinger_vec.T / (_rfinger_vec_len + 1e-8)).T
    finger_around_dot = torch.bmm(lfinger_vec.view(num_envs, 1, 3), rfinger_vec.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

    finger_around_reward_scale = 5.0
    margin = 0.6
    lfinger_around_reward = torch.zeros_like(rot_reward)
    lfinger_around_reward = torch.where((lfinger_dot < cube_size * 2) & (lfinger_dot > cube_size * 0.5),
                                       torch.where(lfinger_len < cube_size * 0.5, lfinger_around_reward - 1.0,
                                                   lfinger_around_reward),
                                       lfinger_around_reward)
    rfinger_around_reward = torch.zeros_like(rot_reward)
    rfinger_around_reward = torch.where((rfinger_dot < cube_size * 2) & (rfinger_dot > cube_size * 0.5),
                                        torch.where(rfinger_len < cube_size * 0.5, rfinger_around_reward - 1.0,
                                                    rfinger_around_reward),
                                        rfinger_around_reward)
    finger_around_reward = 0.5 * lfinger_around_reward + 0.5 * rfinger_around_reward
    # finger_around_reward = torch.where(lfinger_len < cube_size * 0.5, finger_around_reward - 1.0, finger_around_reward)
    # finger_around_reward = torch.where(finger_around_dot < -0.0, -1.0 * finger_around_dot, finger_around_reward)
    # finger_around_reward = torch.where(finger_around_dot < -0.98, finger_around_reward * 2.0, finger_around_reward)
    # finger_around_reward = torch.where(finger_around_dot < -0.99,
    #                                    torch.where(_lfinger_vec_len < cube_size * margin,
    #                                                torch.where(_rfinger_vec_len < cube_size * margin,
    #                                                            finger_around_reward +
    #                                                            2.0 * cube_size * margin - (_lfinger_vec_len + _rfinger_vec_len),
    #                                                            finger_around_reward + (cube_size * margin - _lfinger_vec_len)),
    #                                                finger_around_reward + 1.0),
    #                                    finger_around_reward)

    # cube lifting reward
    lift_reward_scale = 5.0
    des_height = 0.1

    lift_reward = torch.zeros_like(rot_reward)
    cube_lift_pos = cube_grasp_pos.clone()
    cube_lift_pos[:, 2] = des_height
    d2 = torch.norm(mirobot_grasp_pos - cube_lift_pos, p=2, dim=-1)
    lift_reward = torch.where(grasp_done, torch.exp(-20.0 * d2), lift_reward)
    # lift_dist = torch.norm(des_height - cube_pos[:, 2], p=2, dim=-1)
    # lift_reward = torch.exp(-100.0 * lift_dist)
    # lift_reward = torch.where(grasp_done, lift_reward + torch.exp(-100.0 * lift_dist), lift_reward)
    # lift_reward = torch.min(cube_pos[:, 2], torch.zeros_like(rot_reward))

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward + \
              grasp_reward_scale * grasp_reward + \
              lift_reward_scale * lift_reward - action_penalty * action_penalty_scale

    # grasp = torch.norm(finger_dist - (cube_size - 0.002), p=2, dim=-1)
    # grasp_reward = torch.exp(-10.0 * grasp)
    # grasp_reward_scale = 10.0
    # rewards = torch.where(approach_done,
    #                       rewards * 1.5 + grasp_reward_scale * grasp_reward,
    #                       rewards + finger_dist_reward_scale * finger_dist_reward)

    # rewards = torch.where(lift_dist < 0.01, rewards * 3.0, rewards)
    rewards = torch.where(cube_pos[:, 2] >= des_height, rewards * 2.0, rewards)

    # check the collisions of both fingers
    _lfinger_contact_net_force = (lfinger_contact_net_force.T / (lfinger_contact_net_force.norm(p=2, dim=-1) + 1e-8)).T
    _rfinger_contact_net_force = (rfinger_contact_net_force.T / (rfinger_contact_net_force.norm(p=2, dim=-1) + 1e-8)).T
    lf_force_dot = torch.bmm(_lfinger_contact_net_force.view(num_envs, 1, 3), gripper_forward_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    rf_force_dot = torch.bmm(_rfinger_contact_net_force.view(num_envs, 1, 3), gripper_forward_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

    # rewards = torch.where(lf_force_dot > 0.9, rewards - 1.0, rewards)
    # rewards = torch.where(rf_force_dot > 0.9, rewards - 1.0, rewards)
    #
    # reset_buf = torch.where(lf_force_dot > 0.9, torch.ones_like(reset_buf), reset_buf)
    # reset_buf = torch.where(rf_force_dot > 0.9, torch.ones_like(reset_buf), reset_buf)

    reset_buf = torch.where(dot3 < 0.8, torch.ones_like(reset_buf), reset_buf)
    # reset_buf = torch.where(lift_dist < 0.01, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(cube_pos[:, 2] >= des_height, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, mirobot_local_grasp_rot, mirobot_local_grasp_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, mirobot_local_grasp_rot, mirobot_local_grasp_pos)

    return global_franka_rot, global_franka_pos
