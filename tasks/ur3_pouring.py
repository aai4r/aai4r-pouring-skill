from utils.torch_jit_utils import *
from tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

import torch
import torch.nn.init as init
import os
import cv2
# from dataset_manager import DatasetManager
# from model.model import VisuomotorPolicy
import time
from utils import *


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


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

        self.dt = 1 / 60

        num_obs = 24
        num_acts = 8

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=cfg)
        self.asset_root = "./assets"
        np.random.seed(42)

        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.max_episode_length = 1200
        self.img_stack = 4
        self.rgb_num_ch = 3
        self.angle_stroke_ratio = deg2rad(46) / 85
        self.debug_view = False

        # gripper params
        self.grasp_z_offset = 0.135   # (m)
        self.gripper_stroke = 60      # (mm)

        # fluid modeling params
        r = 0.012
        self.expr = [[0, 0], [0, -r], [-r, 0], [0, r], [r, 0]]
        self.num_liq_particles = 140

        # self.envs = []
        # self.hand_idxs = []
        # self.cup_idxs = []
        # self.bottle_idxs = []
        # self.default_liq_states = []
        # self.ur3_handles = []
        # self.init_pos_list = []
        # self.init_rot_list = []
        # self.init_cup_pos_list = []
        # self.init_cup_rot_list = []
        # self.cameras = []
        # self.cam_handles = []
        # self.cam_tensors = []
        # self.img_tensors = torch.zeros(self.num_envs, cfg.camera_height, cfg.camera_width, self.rgb_num_ch * self.img_stack, device=self.device, dtype=torch.uint8)
        # self.task_step = torch.zeros(self.num_envs, 1, 4, device=self.device, dtype=torch.long)
        # self.task_name = "ur3_pouring"
        #
        # self.init_task()
        #
        # self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # self.action = torch.zeros(self.num_envs, self.num_dofs, 1, device=self.device, dtype=torch.float)
        # self.grip_action = torch.ones(self.num_envs, device=self.device, dtype=torch.uint8)
        #
        # self.gather_dataset = cfg.gather_sim_dataset
        # path = os.path.join("dataset", self.task_name)
        # num_frames = cfg.num_frames    # This will be modified considering num_envs
        # # self.dataset = DatasetManager(num_envs=self.num_envs, num_frames=num_frames, path=path,
        # #                               max_sub_frames=cfg.max_sub_frames,
        # #                               img_shape=(cfg.camera_height, cfg.camera_width, self.rgb_num_ch * self.img_stack),
        # #                               action_shape=(self.num_dofs,), state_shape=(38,),
        # #                               dynamic_partition=cfg.dynamic_partition_storage)
        #
        # self.joint_pos = []
        # self.joint_vel = []
        # self.grip_pos = []
        # self.grip_rot = []
        # self.grip_act = []
        # self.cube_pos = []
        # self.cube_rot = []
        # self.cup_pos = []
        # self.cup_rot = []
        # self.frame_count = 0

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        pass

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_table(self):
        # create table asset
        self.table_dims = gymapi.Vec3(0.45, 0.9, 0.001)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, asset_options)

        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(self.table_dims.x * 0.6, 0.0, self.table_dims.z * 0.5)
        return table_asset

    def create_cup(self):
        # load paper cup
        self.cup_height = 0.074
        asset_options = gymapi.AssetOptions()
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 30000
        asset_options.vhacd_params.max_convex_hulls = 8
        asset_options.vhacd_params.max_num_vertices_per_ch = 16
        paper_cup_asset_file = "urdf/target_objects/urdf/paper_cup.urdf"
        cup_asset = self.gym.load_asset(self.sim, self.asset_root, paper_cup_asset_file, asset_options)
        return cup_asset

    def create_cube(self):
        # create box asset
        self.box_size = 0.02
        asset_options = gymapi.AssetOptions()
        cube_asset_file = "urdf/target_objects/urdf/cube.urdf"
        # box_asset = self.gym.create_box(self.sim, self.box_size, self.box_size, self.box_size, asset_options)
        box_asset = self.gym.load_asset(self.sim, self.asset_root, cube_asset_file, asset_options)
        return box_asset

    def create_bottle(self):
        self.bottle_height = 0.195
        self.bottle_radius = 0.065 / 2
        asset_options = gymapi.AssetOptions()
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 30000
        asset_options.vhacd_params.max_convex_hulls = 8
        asset_options.vhacd_params.max_num_vertices_per_ch = 16
        bottle_asset_file = "urdf/target_objects/urdf/bottle.urdf"
        bottle_asset = self.gym.load_asset(self.sim, self.asset_root, bottle_asset_file, asset_options)
        return bottle_asset

    def create_fluid_particle(self):
        asset_options = gymapi.AssetOptions()
        asset_options.density = 997
        liquid_asset = self.gym.create_sphere(self.sim, 0.004, asset_options)
        return liquid_asset

    def create_robot(self):
        # load mirobot asset
        ur3_asset_file = "urdf/ur3_description/robot/ur3_robotiq85_gripper.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        # asset_options.collapse_fixed_joints = True
        ur3_asset = self.gym.load_asset(self.sim, self.asset_root, ur3_asset_file, asset_options)

        # configure mirobot dofs
        self.mirobot_dof_props = self.gym.get_asset_dof_properties(ur3_asset)
        mirobot_lower_limits = self.mirobot_dof_props["lower"]
        mirobot_upper_limits = self.mirobot_dof_props["upper"]
        mirobot_ranges = mirobot_upper_limits - mirobot_lower_limits
        mirobot_mids = 0.0 * (mirobot_upper_limits + mirobot_lower_limits)

        # use position drive for all dofs
        self.mirobot_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.mirobot_dof_props["stiffness"][:6].fill(400.0)
        self.mirobot_dof_props["damping"][:6].fill(40.0)
        # grippers
        self.mirobot_dof_props["stiffness"][6:].fill(1000.0)
        self.mirobot_dof_props["damping"][6:].fill(40.0)

        # default dof states and position targets
        robot_dofs_names = self.gym.get_asset_dof_names(ur3_asset)
        print("dof names: ", robot_dofs_names)
        num_dofs = self.gym.get_asset_dof_count(ur3_asset)
        print("num dof: ", num_dofs)
        self.default_dof_pos = np.zeros(num_dofs, dtype=np.float32)
        self.default_dof_pos[:7] = np.array([deg2rad(0.0), deg2rad(-90.0), deg2rad(85.0),
                                             deg2rad(0.0), deg2rad(80.0), deg2rad(0.0), mirobot_upper_limits[6]])  # with gripper
        # grippers open
        # self.default_dof_pos[6] = mirobot_upper_limits[6]

        self.default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        # get link index of panda hand, which we will use as end effector
        mirobot_link_dict = self.gym.get_asset_rigid_body_dict(ur3_asset)
        print("dict: ", mirobot_link_dict)
        self.ur_hand_index = mirobot_link_dict["tool0"]

        ur3_init_pose = gymapi.Transform()
        ur3_init_pose.p = gymapi.Vec3(0, 0, 0)
        return ur3_asset, ur3_init_pose

    def generate_random_joint_angles(self, rand_deg=3.0):
        return torch.cat((
            init.uniform_(torch.FloatTensor(self.num_envs, 1), a=deg2rad(-rand_deg), b=deg2rad(rand_deg)),                  # shoulder pan joint
            init.uniform_(torch.FloatTensor(self.num_envs, 1), a=deg2rad(-90.0 - rand_deg), b=deg2rad(-90.0 + rand_deg)),   # shoulder lift joint
            init.uniform_(torch.FloatTensor(self.num_envs, 1), a=deg2rad(85.0 - rand_deg), b=deg2rad(85.0 + rand_deg)),     # elbow joint
            init.uniform_(torch.FloatTensor(self.num_envs, 1), a=deg2rad(0.0 - rand_deg), b=deg2rad(0.0 + rand_deg)),       # wrist 1
            init.uniform_(torch.FloatTensor(self.num_envs, 1), a=deg2rad(80.0 - rand_deg), b=deg2rad(80.0 + rand_deg)),     # wrist 2
            init.uniform_(torch.FloatTensor(self.num_envs, 1), a=deg2rad(-rand_deg), b=deg2rad(rand_deg)),                  # wrist 3
            torch.zeros(self.num_envs, 1),  # 6
            torch.zeros(self.num_envs, 1),  # 7
            torch.zeros(self.num_envs, 1),  # 8
            # init.uniform_(torch.FloatTensor(self.num_envs, 1), a=deg2rad(0.0), b=0.80285),                          # robotiq85 left knuckle
            torch.zeros(self.num_envs, 1),  # 9
            torch.zeros(self.num_envs, 1),  # 10
            torch.zeros(self.num_envs, 1),  # 11
        ), dim=-1)

    def generate_random_cube_pos(self):
        box_pose = gymapi.Transform()
        box_pose.p.x = self.table_pose.p.x + np.random.uniform(-0.05, 0.07)
        box_pose.p.y = self.table_pose.p.y + np.random.uniform(-0.03, 0.03)
        box_pose.p.z = self.table_dims.z + 0.5 * self.box_size
        box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
        return box_pose

    def generate_random_cup_pos(self):
        cup_pose = gymapi.Transform()
        cup_pose.p.x = self.table_pose.p.x + np.random.uniform(-0.015, 0.015)
        cup_pose.p.y = self.table_pose.p.y + np.random.uniform(-0.015, 0.015)
        cup_pose.p.z = self.table_dims.z + 0.5 * self.cup_height + 0.01
        cup_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
        return cup_pose

    def generate_random_bottle_pos(self):
        bottle_pos = gymapi.Transform()
        bottle_pos.p.x = self.table_pose.p.x + 0.15
        bottle_pos.p.y = self.table_pose.p.y + np.where(np.random.rand() > 0.5, 0.27, -0.27)
        bottle_pos.p.z = self.table_dims.z + 0.5 * self.bottle_height + 0.01
        bottle_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
        return bottle_pos

    def set_ur3_body_color(self, env, robot_handle):
        self.gym.set_rigid_body_color(env, robot_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 0.7, 0.7))     # base_link
        self.gym.set_rigid_body_color(env, robot_handle, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 0.7, 0.7))     # shoulder_link
        self.gym.set_rigid_body_color(env, robot_handle, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 0.7, 0.7))     # upper_arm_link
        self.gym.set_rigid_body_color(env, robot_handle, 3, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 0.7, 0.7))     # forearm_link
        self.gym.set_rigid_body_color(env, robot_handle, 4, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 0.7, 0.7))     # wrist_1_link
        self.gym.set_rigid_body_color(env, robot_handle, 5, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 0.7, 0.7))     # wrist_2_link
        self.gym.set_rigid_body_color(env, robot_handle, 6, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 0.7, 0.7))     # wrist_3_link
        self.gym.set_rigid_body_color(env, robot_handle, 7, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 0.7, 0.7))     # tool0

    def obs_stack(self):
        pass

    def init_task(self):
        table_asset = self.create_table()
        bottle_asset = self.create_bottle()
        cup_asset = self.create_cup()
        fluid_asset = self.create_fluid_particle()
        ur3_asset, ur3_pose = self.create_robot()
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)

            # (0) add robot
            ur3_handle = self.gym.create_actor(env, ur3_asset, ur3_pose, "ur3", i, 2)
            self.ur3_handles.append(ur3_handle)
            self.set_ur3_body_color(env, ur3_handle)

            # set dof properties
            self.gym.set_actor_dof_properties(env, ur3_handle, self.mirobot_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, ur3_handle, self.default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, ur3_handle, self.default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, ur3_handle, "tool0")
            hand_pos = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([hand_pos.p.x + self.grasp_z_offset, hand_pos.p.y, hand_pos.p.z])
            self.init_rot_list.append([hand_pos.r.x, hand_pos.r.y, hand_pos.r.z, hand_pos.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, ur3_handle, "tool0", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # (1) add table
            table_handle = self.gym.create_actor(env, table_asset, self.table_pose, "table", i, 0)

            # (2) add paper cup
            cup_pos = self.generate_random_cup_pos()
            cup_handle = self.gym.create_actor(env, cup_asset, cup_pos, "cup", i, 0)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, cup_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            # cup_texture = self.gym.create_texture_from_buffer(self.sim, 10, 10, np.array([1, 1, 1, 1]).astype(np.uint8))
            # self.gym.set_rigid_body_texture(env, cup_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cup_texture)

            # get initial cup pose
            self.init_cup_pos_list.append([cup_pos.p.x, cup_pos.p.y, cup_pos.p.z])
            self.init_cup_rot_list.append([cup_pos.r.x, cup_pos.r.y, cup_pos.r.z, cup_pos.r.w])

            # get global index of box in rigid body state tensor
            cup_idx = self.gym.get_actor_rigid_body_index(env, cup_handle, 0, gymapi.DOMAIN_SIM)
            self.cup_idxs.append(cup_idx)

            # (3) add bottle
            bottle_pos = self.generate_random_bottle_pos()
            bottle_handle = self.gym.create_actor(env, bottle_asset, bottle_pos, "bottle", i, 0)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, bottle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # get global index of box in rigid body state tensor
            bottle_idx = self.gym.get_actor_rigid_body_index(env, bottle_handle, 0, gymapi.DOMAIN_SIM)
            self.bottle_idxs.append(bottle_idx)

            # (4) add liquids
            liq_count = 0
            while liq_count < self.num_liq_particles:
                liquid_pos = bottle_pos
                liquid_pos.p.z += self.bottle_height + 0.1 + 0.03 * liq_count
                for k in self.expr:
                    liquid_pos.p.x += k[0]
                    liquid_pos.p.y += k[1]
                    liquid_handle = self.gym.create_actor(env, fluid_asset, liquid_pos, "liquid", i, 0)
                    color = gymapi.Vec3(0.0, 0.0, 1.0)
                    self.gym.set_rigid_body_color(env, liquid_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                    liq_count += 1

            # add camera
            camera_handle = self.gym.create_camera_sensor(env, self.camera_props)
            self.cameras.append(camera_handle)
            self.gym.set_camera_location(camera_handle, env, gymapi.Vec3(0.5, 0.1, 0.25), ur3_pose.p)

            # camera tensor
            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_COLOR)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            self.cam_tensors.append(torch_cam_tensor)

        # point camera at middle env
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        self.gym.prepare_sim(self.sim)

        # initial hand position and orientation tensors
        self.init_pos = torch.Tensor(self.init_pos_list).view(self.num_envs, 3).to(self.device)
        self.init_rot = torch.Tensor(self.init_rot_list).view(self.num_envs, 4).to(self.device)

        # hand orientation for grasping
        self.down_q = torch.stack(self.num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(self.device).view((self.num_envs, 4))

        # initial cup position and orientation tensors
        self.init_cup_pos = torch.Tensor(self.init_cup_pos_list).view(self.num_envs, 3).to(self.device)
        self.init_cup_rot = torch.Tensor(self.init_cup_rot_list).view(self.num_envs, 4).to(self.device)

        # downard axis
        self.down_dir = torch.Tensor([0, 0, -1]).to(self.device).view(1, 3)

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur3")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to ur hand
        self.j_eef = jacobian[:, self.ur_hand_index - 1, :, :6]

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = dof_states[:, 0].view(self.num_envs, 12, 1)  # with robotiq85 gripper UR3(6DoF) + Robotiq(6DoF)
        self.dof_vel = dof_states[:, 1].view(self.num_envs, 12, 1)
        self.dof_pos_list = []

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # [num_envs * num_dofs, 2]
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # actor_root_state_tensor = actor_root_state_tensor[:self.num_envs * 7, :]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.num_actors = self.root_state_tensor.size()[1]
        self.global_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.mirobot_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # print("root state: ", self.root_state_tensor.shape)
        # print("dof state: ", self.dof_state.shape)
        # print("rigid body state: ", self.rigid_body_states.shape)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

    def reset(self, env_ids):
        print("reset!! ", env_ids)
        robot_indices = self.global_indices[env_ids, :1].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.mirobot_dof_targets),
                                                        gymtorch.unwrap_tensor(robot_indices), len(robot_indices))

        # joint random position
        self.dof_state = torch.zeros_like(self.dof_state, dtype=torch.float, device=self.device)
        init_pos = self.generate_random_joint_angles()
        self.dof_state[:, 0] = init_pos.view(-1)   # pos, [num_envs * num_dofs, 1]
        self.dof_state[:, 1] = torch.zeros_like(self.dof_state[:, 1], dtype=torch.float, device=self.device)    # vel
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
        # cup, bottle & liquid reset
        cup_bottle_liq_indices = self.global_indices[env_ids, 2:].flatten()
        for i in range(self.num_envs):
            cup_pose = self.generate_random_cup_pos()
            self.root_state_tensor[i, 2, :] = to_torch([cup_pose.p.x, cup_pose.p.y, cup_pose.p.z,
                                                        cup_pose.r.x, cup_pose.r.y, cup_pose.r.z, cup_pose.r.w,
                                                        0, 0, 0, 0, 0, 0], device=self.device)
            bottle_pos = self.generate_random_bottle_pos()
            self.root_state_tensor[i, 3, :] = to_torch([bottle_pos.p.x, bottle_pos.p.y, bottle_pos.p.z,
                                                        bottle_pos.r.x, bottle_pos.r.y, bottle_pos.r.z, bottle_pos.r.w,
                                                        0, 0, 0, 0, 0, 0], device=self.device)
            liq_count = 0
            z_offset = 0
            for j in range(4, self.root_state_tensor.size(1)):
                idx = liq_count % len(self.expr)
                self.root_state_tensor[i, j, :] = to_torch([bottle_pos.p.x + self.expr[idx][0],
                                                            bottle_pos.p.y + self.expr[idx][1],
                                                            bottle_pos.p.z + self.bottle_height + 0.1 + 0.03 * z_offset,
                                                            bottle_pos.r.x, bottle_pos.r.y, bottle_pos.r.z, bottle_pos.r.w,
                                                            0, 0, 0, 0, 0, 0], device=self.device)
                liq_count += 1
                z_offset += 1 if liq_count % len(self.expr) == 0 else 0

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(cup_bottle_liq_indices), len(cup_bottle_liq_indices))
        # image tensor stack reset
        for i in range(len(env_ids)):
            self.image_stacking_reset(i)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.task_step[env_ids] = 0

    def reset_task_points(self, env_ids):
        """
        retrieve position & orientation of each object and set the via-points list
        Ingredients: pos & rot of hand, cube and cup
        (1) init pos & gripper_open
        (2) approach target
        (3) grasp ready
        (4) gripper_close
        (5) lifting
        (6) approach_goal
        (7) pouring
        (8) turning back
        (9) approach origin
        (10) putting down
        (11) grasp ready
        (12) gripper open
        (13) init pos
        """
        bottle_pos, bottle_rot = self.rb_states[self.bottle_idxs, :3], self.rb_states[self.bottle_idxs, 3:7]
        cup_pos, cup_rot = self.rb_states[self.cup_idxs, :3], self.rb_states[self.cup_idxs, 3:7]

        # init pos & gripper open
        init_pos = self.init_pos.clone()
        init_rot = self.init_rot.clone()
        init_grip = torch.Tensor([[0.0]] * self.num_envs).to(self.device)
        init_err = torch.Tensor([[0.02, 0.02]] * self.num_envs).to(self.device)   # position error, orientation error

        # approach target
        dir_z = bottle_pos - cup_pos
        dir_z[:, 2] = 0.0  # zero padding to z-axis
        dir_z = dir_z / dir_z.norm(dim=-1).unsqueeze(-1)  # normalize

        appr_target_pos = bottle_pos.clone()
        appr_target_pos[:, 2] *= 1.2
        appr_target_pos[:, :2] -= dir_z[:, :2] * self.bottle_radius * 4.0

        mats = quat_to_mat(bottle_rot)
        # dir_x = dir_z.cross(mats[:, :, 2])    # gripper flip (reverse)
        # dir_y = dir_z.cross(dir_x)
        dir_x = mats[:, :, 2].cross(dir_z)
        dir_y = dir_z.cross(dir_x)
        appr_target_rot = mat_to_quat(torch.stack([dir_x, dir_y, dir_z], dim=-1))
        self.appr_target_rot = appr_target_rot  # for debugging
        appr_target_grip = torch.Tensor([[0.0]] * self.num_envs).to(self.device)
        appr_target_err = torch.Tensor([[0.03, 0.03]] * self.num_envs).to(self.device)

        # grasp_ready-1
        grasp_ready_pos = bottle_pos.clone()
        grasp_ready_pos[:, 2] *= 1.1
        grasp_ready_pos[:, :2] -= dir_z[:, :2] * self.bottle_radius * 1.4
        grasp_ready_rot = appr_target_rot.clone()
        grasp_ready_grip = torch.Tensor([[0.0]] * self.num_envs).to(self.device)
        grasp_ready_err = torch.Tensor([[0.02, 0.02]] * self.num_envs).to(self.device)

        # grasp_ready-2
        grasp_ready_pos_2 = bottle_pos.clone()
        grasp_ready_pos_2[:, 2] *= 1.1
        grasp_ready_pos_2[:, :2] -= dir_z[:, :2] * self.bottle_radius * -0.2
        grasp_ready_rot_2 = grasp_ready_rot.clone()
        grasp_ready_2_grip = torch.Tensor([[0.0]] * self.num_envs).to(self.device)
        grasp_ready_2_err = torch.Tensor([[0.01, 0.01]] * self.num_envs).to(self.device)

        # gripper_close
        gripper_close_pos = grasp_ready_pos_2.clone()
        gripper_close_rot = grasp_ready_rot_2.clone()
        gripper_close_grip = torch.Tensor([[1.0]] * self.num_envs).to(self.device)
        gripper_close_err = torch.Tensor([[0.02, 0.02]] * self.num_envs).to(self.device)

        # lifting
        lifting_pos = gripper_close_pos.clone()
        lifting_pos[:, 2] = self.bottle_height * 1.4
        lifting_rot = gripper_close_rot.clone()
        lifting_grip = torch.Tensor([[1.0]] * self.num_envs).to(self.device)
        lifting_err = torch.Tensor([[0.06, 0.07]] * self.num_envs).to(self.device)

        # approach_goal
        roll = torch.FloatTensor([[deg2rad(0.0)]] * self.num_envs).to(self.device)
        pitch = torch.FloatTensor([[deg2rad(0.0)]] * self.num_envs).to(self.device)
        yaw = torch.where((bottle_pos[:, 1] > cup_pos[:, 1]), deg2rad(-90.0), deg2rad(90.0)).view(self.num_envs, -1).to(self.device)
        q = quat_from_euler_xyz(roll, pitch, yaw).squeeze(1)
        offset_dir = quat_rotate(q, dir_z)

        pour_offset = 0.08
        appr_goal_pos = cup_pos.clone()
        appr_goal_pos[:, :2] = appr_goal_pos[:, :2] + offset_dir[:, :2] * pour_offset
        appr_goal_pos[:, 2] = self.cup_height * 3.0
        appr_goal_rot = lifting_rot.clone()
        appr_goal_grip = torch.Tensor([[1.0]] * self.num_envs).to(self.device)
        appr_goal_err = torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device)

        # pouring
        pouring_pos = appr_goal_pos.clone()
        roll = torch.FloatTensor([[deg2rad(0.0)]] * self.num_envs).to(self.device)
        pitch = torch.FloatTensor([[deg2rad(0.0)]] * self.num_envs).to(self.device)
        yaw = torch.where((bottle_pos[:, 1] > cup_pos[:, 1]), deg2rad(-135.0), deg2rad(135.0)).view(self.num_envs, -1).to(self.device)
        q = quat_from_euler_xyz(roll, pitch, yaw)
        pouring_rot = quat_mul(appr_goal_rot, q.squeeze(1))
        pouring_grip = torch.Tensor([[1.0]] * self.num_envs).to(self.device)
        pouring_err = torch.Tensor([[0.015, 0.01]] * self.num_envs).to(self.device)

        # turning back
        turning_back_pos = pouring_pos.clone()
        roll = torch.FloatTensor([[deg2rad(0.0)]] * self.num_envs).to(self.device)
        pitch = torch.FloatTensor([[deg2rad(0.0)]] * self.num_envs).to(self.device)
        yaw = torch.where((bottle_pos[:, 1] > cup_pos[:, 1]), deg2rad(135.0), deg2rad(-135.0)).view(self.num_envs, -1).to(self.device)
        q = quat_from_euler_xyz(roll, pitch, yaw)
        turning_back_rot = quat_mul(pouring_rot, q.squeeze(1))
        turning_back_grip = torch.Tensor([[1.0]] * self.num_envs).to(self.device)
        turning_back_err = torch.Tensor([[0.02, 0.02]] * self.num_envs).to(self.device)

        # appr_origin
        appr_origin_pos = bottle_pos.clone()
        appr_origin_pos[:, 2] = turning_back_pos[:, 2]
        appr_origin_rot = turning_back_rot.clone()
        appr_origin_grip = torch.Tensor([[1.0]] * self.num_envs).to(self.device)
        appr_origin_err = torch.Tensor([[0.02, 0.02]] * self.num_envs).to(self.device)

        # putting down
        put_down_pos = bottle_pos.clone()
        put_down_pos[:, 2] += 0.02
        put_down_rot = appr_origin_rot.clone()
        put_down_grip = torch.Tensor([[1.0]] * self.num_envs).to(self.device)
        put_down_err = torch.Tensor([[0.025, 0.02]] * self.num_envs).to(self.device)

        # gripper_open
        gripper_open_pos = put_down_pos.clone()
        gripper_open_rot = put_down_rot.clone()
        gripper_open_grip = torch.Tensor([[0.0]] * self.num_envs).to(self.device)
        gripper_open_err = torch.Tensor([[0.007, 0.007]] * self.num_envs).to(self.device)

        # release
        release_pos = grasp_ready_pos.clone()
        release_rot = grasp_ready_rot.clone()
        release_grip = torch.Tensor([[0.0]] * self.num_envs).to(self.device)
        release_err = torch.Tensor([[0.01, 0.01]] * self.num_envs).to(self.device)

        pos_list = torch.stack([init_pos[env_ids], appr_target_pos[env_ids],
                                grasp_ready_pos[env_ids], grasp_ready_pos_2[env_ids], gripper_close_pos[env_ids],
                                lifting_pos[env_ids], appr_goal_pos[env_ids], pouring_pos[env_ids],
                                turning_back_pos[env_ids], appr_origin_pos[env_ids], put_down_pos[env_ids],
                                gripper_open_pos[env_ids], release_pos[env_ids], init_pos[env_ids]], dim=0).transpose(1, 0).to(self.device)

        rot_list = torch.stack([init_rot[env_ids], appr_target_rot[env_ids],
                                grasp_ready_rot[env_ids], grasp_ready_rot_2[env_ids], gripper_close_rot[env_ids],
                                lifting_rot[env_ids], appr_goal_rot[env_ids], pouring_rot[env_ids],
                                turning_back_rot[env_ids], appr_origin_rot[env_ids], put_down_rot[env_ids],
                                gripper_open_rot[env_ids], release_rot[env_ids], init_rot[env_ids]], dim=0).transpose(1, 0).to(self.device)

        grip_list = torch.stack([init_grip[env_ids], appr_target_grip[env_ids],
                                 grasp_ready_grip[env_ids], grasp_ready_2_grip[env_ids], gripper_close_grip[env_ids],
                                 lifting_grip[env_ids], appr_goal_grip[env_ids], pouring_grip[env_ids],
                                 turning_back_grip[env_ids], appr_origin_grip[env_ids], put_down_grip[env_ids],
                                 gripper_open_grip[env_ids], release_grip[env_ids], init_grip[env_ids]], dim=0).transpose(1, 0).to(self.device)

        err_list = torch.stack([init_err[env_ids], appr_target_err[env_ids],
                                grasp_ready_err[env_ids], grasp_ready_2_err[env_ids], gripper_close_err[env_ids],
                                lifting_err[env_ids], appr_goal_err[env_ids], pouring_err[env_ids],
                                turning_back_err[env_ids], appr_origin_err[env_ids], put_down_err[env_ids],
                                gripper_open_err[env_ids], release_err[env_ids], init_err[env_ids]], dim=0).transpose(1, 0).to(self.device)

        if not hasattr(self, 'task_pos'):
            num_task_steps = pos_list.shape[1]
            self.task_pos = torch.zeros(self.num_envs, num_task_steps, 3, device=self.device)
            self.task_rot = torch.zeros(self.num_envs, num_task_steps, 4, device=self.device)
            self.task_grip = torch.zeros(self.num_envs, num_task_steps, 1, device=self.device)
            self.task_err = torch.zeros(self.num_envs, num_task_steps, 2, device=self.device)

        self.task_pos[env_ids] = pos_list
        self.task_rot[env_ids] = rot_list
        self.task_grip[env_ids] = grip_list
        self.task_err[env_ids] = err_list

    def viz_camera_debug_view(self, env_idx):
        env_idx = min(max(0, env_idx), self.num_envs-1)
        # retrieve camera image
        if self.camera_debug_view:
            self.gym.start_access_image_tensors(self.sim)  # transfer img_tensor from GPU to memory
            img = self.img_tensors[env_idx]
            cimg = []
            ch = 3
            for j in range(self.img_stack):
                s = img[:, :, ch * j:ch * (j + 1)].cpu().numpy()
                s = cv2.cvtColor(s, cv2.COLOR_BGRA2RGB)
                cimg.append(s)
            cv2.imshow("Image", cv2.hconcat(tuple([x for x in cimg])))
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                exit()
            self.gym.end_access_image_tensors(self.sim)

    def image_stacking(self):
        ch = self.rgb_num_ch
        for i in range(len(self.cam_tensors)):
            # shift existing frames
            temp = self.img_tensors[i, :, :, :ch * (self.img_stack - 1)].clone()  # (env, height, width, ch * stack)
            self.img_tensors[i, :, :, ch:] = temp

            # insert new image to the last channel
            new_img_tensor = self.cam_tensors[i].clone()[:, :, :3]
            self.img_tensors[i, :, :, :ch] = new_img_tensor

    def image_stacking_reset(self, env_idx):
        ch = self.rgb_num_ch
        self.gym.start_access_image_tensors(self.sim)  # transfer img_tensor from GPU to memory
        new_img_tensor = self.cam_tensors[env_idx].clone()
        for i in range(self.img_stack):
            self.img_tensors[env_idx, :, :, ch * i:ch * (i+1)] = new_img_tensor[:, :, :3]
        self.gym.end_access_image_tensors(self.sim)

    def gather_sim_dataset(self):
        if not self.gather_dataset:
            return

        self.joint_pos.append(self.dof_pos.clone())
        self.joint_vel.append(self.dof_vel.clone())
        self.grip_pos.append(self.rb_states[self.hand_idxs, :3].clone())
        self.grip_rot.append(self.rb_states[self.hand_idxs, 3:7].clone())
        self.grip_act.append(self.grip_action.clone())
        self.cube_pos.append(self.rb_states[self.box_idxs, :3].clone())
        self.cube_rot.append(self.rb_states[self.box_idxs, 3:7].clone())
        self.cup_pos.append(self.rb_states[self.cup_idxs, :3].clone())
        self.cup_rot.append(self.rb_states[self.cup_idxs, 3:7].clone())

        # insert the dataset, there is 3 frame shift between obs and state
        if self.frame_count < 4:    # to align the frame shift
            return
        self.gym.start_access_image_tensors(self.sim)  # transfer img_tensor from GPU to memory
        joint_pos = self.joint_pos.pop(0)
        joint_vel = self.joint_vel.pop(0)
        grip_pos = self.grip_pos.pop(0)
        grip_rot = self.grip_rot.pop(0)
        grip_act = self.grip_act.pop(0)
        cube_pos = self.cube_pos.pop(0)
        cube_rot = self.cube_rot.pop(0)
        cup_pos = self.cup_pos.pop(0)
        cup_rot = self.cup_rot.pop(0)
        for i in range(len(self.cam_tensors)):
            img_tensor = self.img_tensors[i].clone()
            action_tensor = self.action[i].squeeze(-1).clone()
            state_tensor = torch.cat((joint_pos.squeeze(-1)[i], joint_vel.squeeze(-1)[i],
                                      grip_pos[i], grip_rot[i], grip_act[i].unsqueeze(-1),
                                      cube_pos[i], cube_rot[i], cup_pos[i], cup_rot[i]))
            self.dataset.insert_tensor(env_idx=i, cam_tensor=img_tensor,
                                       action_tensor=action_tensor, state_tensor=state_tensor)
        self.gym.end_access_image_tensors(self.sim)

        # save
        if self.dataset.is_tensor_full():
            self.dataset.save()

    def get_robotiq_gripper_target_pos(self, grip_acts):
        gripper_pos_target = torch.zeros_like(self.dof_pos[:, 6:, :])
        gripper_pos_target[:, 8 - 6, 0] = grip_acts
        gripper_pos_target[:, 6 - 6, 0] = 1 * gripper_pos_target[:, 8 - 6, 0]
        gripper_pos_target[:, 7 - 6, 0] = -1 * gripper_pos_target[:, 8 - 6, 0]
        gripper_pos_target[:, 9 - 6, 0] = 1 * gripper_pos_target[:, 8 - 6, 0]
        gripper_pos_target[:, 10 - 6, 0] = -1 * gripper_pos_target[:, 8 - 6, 0]
        gripper_pos_target[:, 11 - 6, 0] = 1 * gripper_pos_target[:, 8 - 6, 0]
        return gripper_pos_target

    def sync_gripper(self):
        self.dof_pos[:, 6, 0] = 1 * self.dof_pos[:, 8, 0]
        self.dof_pos[:, 7, 0] = -1 * self.dof_pos[:, 8, 0]
        self.dof_pos[:, 9, 0] = 1 * self.dof_pos[:, 8, 0]
        self.dof_pos[:, 10, 0] = -1 * self.dof_pos[:, 8, 0]
        self.dof_pos[:, 11, 0] = 1 * self.dof_pos[:, 8, 0]

        self.dof_state[:, 0] = self.dof_pos.view(-1)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def stroke_to_angle(self, mm):
        return deg2rad(46) - self.angle_stroke_ratio * mm

    def angle_to_stroke(self, rad):
        return (deg2rad(46) - rad) / self.angle_stroke_ratio

    def gripper_z_offset(self, hand_pos, hand_rot):  # mirobot gripper is offset in x-axis
        pz = quat_apply(hand_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1))
        hand_pos[:] += (self.grasp_z_offset * pz)
        return hand_pos

    def solve(self, goal_pos, goal_rot, goal_grip):
        hand_pos, hand_rot = self.rb_states[self.hand_idxs, :3], self.rb_states[self.hand_idxs, 3:7]
        hand_pos = self.gripper_z_offset(hand_pos=hand_pos, hand_rot=hand_rot)

        des_stroke = self.gripper_stroke
        gripper_stroke = self.angle_to_stroke(self.dof_pos[:, 8, 0]).to(self.device)
        dist = torch.where(goal_grip.squeeze(-1).to(self.device),
                           (gripper_stroke - des_stroke).abs(),
                           torch.ones_like(gripper_stroke).to(self.device) * -1)
        grip_rst = torch.where(dist < 0.4, 1, 0)

        # compute position and orientation error
        pos_err = goal_pos - hand_pos
        orn_err = orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        d = 0.15    # damping term
        lmbda = torch.eye(6).to(self.device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6, 1)
        self.action = u.clone()

        # update position targets
        return self.dof_pos[:, :6, :] + u, pos_err.norm(dim=-1), orn_err.norm(dim=-1), grip_rst

    def pouring(self):
        goal_pos = torch.gather(self.task_pos, 1, self.task_step[:, :, :3]).squeeze(1)
        goal_rot = torch.gather(self.task_rot, 1, self.task_step).squeeze(1)
        goal_grip = torch.gather(self.task_grip, 1, self.task_step[:, :, :1]).squeeze(1).type(torch.ByteTensor)
        goal_err = torch.gather(self.task_err, 1, self.task_step[:, :, :2]).squeeze(1)
        self.grip_action = goal_grip.squeeze(-1).to(self.device)

        actuator_pos_target, e_pos_err, e_rot_err, grip_rst = self.solve(goal_pos=goal_pos, goal_rot=goal_rot, goal_grip=goal_grip)

        arrive = torch.where((e_pos_err < goal_err[:, 0]) & (e_rot_err < goal_err[:, 1]) & (grip_rst > 0), 1, 0)
        self.task_step = self.task_step + arrive.repeat(4, 1).transpose(0, 1).unsqueeze(1)
        self.task_step = torch.min(self.task_step, torch.ones_like(self.task_step) * self.task_pos.size()[1] - 1)

        # always open the gripper above a certain height, dropping the box and restarting from the beginning
        des_rad = self.stroke_to_angle(mm=self.gripper_stroke)
        grip_acts = torch.where(goal_grip.squeeze(-1).to(self.device), torch.Tensor([des_rad] * self.num_envs).to(self.device),
                                torch.Tensor([0.] * self.num_envs).to(self.device))

        gripper_pos_target = self.get_robotiq_gripper_target_pos(grip_acts)
        pos_target = torch.cat((actuator_pos_target, gripper_pos_target), dim=1).contiguous()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_target))

        # debug viz
        if self.viewer and self.debug_view:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            hand_pos, hand_rot = self.rb_states[self.hand_idxs, :3], self.rb_states[self.hand_idxs, 3:7]
            for i in range(self.num_envs):
                px = (hand_pos[i] + quat_apply(self.appr_target_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (hand_pos[i] + quat_apply(self.appr_target_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (hand_pos[i] + quat_apply(self.appr_target_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = hand_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    def check_task_complete(self, episodic=True):
        # TODO
        # define task complete by checking the cube is inside the cup
        bottle_pos = self.rb_states[self.bottle_idxs, :3]
        cup_pos = self.rb_states[self.cup_idxs, :3]
        to_xy = bottle_pos[:, :2] - cup_pos[:, :2]
        dist_xy = torch.norm(to_xy, dim=-1)
        # height_z = torch.norm(, dim=-1)

        cup_bottom_radius = 0.025
        # cube_half = 0.5 * self.box_size
        # thres_xy = cup_bottom_radius - cube_half
        # thres_z = self.box_size + 0.02

        # episode progress
        self.progress_buf += 1
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where((dist_xy < thres_xy) & (bottle_pos[:, 2] < thres_z), torch.ones_like(self.reset_buf), self.reset_buf)

    def run(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # here, image changes!! after fetch_results!

            # check and reset
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                self.reset(env_ids)

            # refresh tensors
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.sync_gripper()     # implementation of robotiq85 mimic continuous joints

            # task
            if len(env_ids) > 0:
                self.reset_task_points(env_ids)
            self.pouring()  # calc action

            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            # check whether the task is complete
            self.check_task_complete()

            if self.gather_dataset:
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    print("Current Frame Count: {}".format(self.frame_count * self.num_envs))

                if self.dataset.storage_achievement():
                    print(":::: Message :::: Storage is finished...")
                    break

            # store image and action
            self.image_stacking()
            self.viz_camera_debug_view(env_idx=0)
            self.gather_sim_dataset()

        # cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def load_model(self, num=-1):   # default is the last model
        path = os.path.join("model", self.task_name)
        file_list = os.listdir(path)
        file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # dummy model
        self.model = VisuomotorPolicy()
        load_path = os.path.join(path, file_list[num])
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Model load success!: ", file_list[num])

    def inference(self):
        img_tensors = (self.img_tensors.permute(0, 3, 1, 2)[:, :, :, :] / 255.0)
        dof_pos = self.dof_pos_list.pop(0)
        out = self.model(x=img_tensors, joints=dof_pos[:, :6].squeeze(-1))

        u = torch.zeros(self.num_envs, 8, 1, device=self.device)
        u[:, :6] = out[:, :6].unsqueeze(-1)  # motor velocities (action)

        # print("u: ", rad2deg(u[0]), " dof: ", rad2deg(self.dof_pos[0]))
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:  # ESC key
        #     cv2.destroyAllWindows()

        goal_grip = out[:, 6]
        grip_acts = torch.where(goal_grip.unsqueeze(-1) > 0.5,
                                torch.Tensor([[0., 0.]] * self.num_envs).to(self.device),
                                torch.Tensor([[0.0175, 0.0175]] * self.num_envs).to(self.device))
        pos_target = self.dof_pos + u #* self.dt
        pos_target[:, 6:8] = grip_acts.unsqueeze(-1)
        pos_target = torch.clamp(pos_target, min=-1, max=1)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_target))

    def model_test(self, model_idx=-1):
        self.load_model(num=model_idx)
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # dof append
            self.dof_pos_list.append(self.dof_pos)

            # check and reset
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                self.reset(env_ids)

            # refresh tensors
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            # check whether the task is complete
            self.check_task_complete()

            # model inference
            self.image_stacking()
            self.viz_camera_debug_view(env_idx=0)

            if self.frame_count > 1:
                self.inference()

            self.frame_count += 1

        # cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)