import copy
import random
import cv2
import os

from utils.torch_jit_utils import *
from utils.utils import *
from torch.nn.functional import normalize
from tasks.base.base_task import BaseTask, AttrDict
from isaacgym import gymtorch
from isaacgym import gymapi

# vr interface
from vr import triad_openvr


def _uniform(low, high, size=1):
    return np.random.uniform(low=low, high=high, size=size)


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
        self.rand_init_pos_scale = self.cfg["env"]["rand_init_pos_scale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.debug_cam = self.cfg["expert"]["debug_cam"]

        self.up_axis = "x"      # z
        self.up_axis_idx = 0    # 2
        self.dt = 1/30.

        self.use_ik = False
        self.action_noise = self.cfg["env"]["action_noise"]
        self.action_noise_scale = self.cfg["env"]["action_noise_scale"]

        """ Camera Sensor setting """
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = self.cfg["env"]["cam_width"]
        self.camera_props.height = self.cfg["env"]["cam_height"]
        self.camera_props.horizontal_fov = 69  # degree, Default: 90, RealSense D435 FoV = H69 / V42
        self.camera_props.enable_tensors = True

        self.img_obs = self.cfg["expert"]["img_obs"]

        if self.img_obs:
            num_obs = (self.camera_props.height, self.camera_props.width, 3)
            num_states = 47
        else:
            num_obs = (47, )
            num_states = 47     # TODO

        num_acts = 8 if self.use_ik else 7   # 8 for task space ==> pos(3), ori(4), grip(1)

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = num_acts

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.indices = torch.tensor([0, 1, 2, 3, 4, 5, 8], device=device_id)  # 0~5: ur3 joint, 8: robotiq drive joint

        super().__init__(cfg=self.cfg)

        if self.interaction_mode or self.teleoperation_mode:
            """ VR interface setting """
            self.vr = triad_openvr.triad_openvr()
            self.vr.print_discovered_objects()
            self.vr_ref_rot = euler_to_mat3d(roll=deg2rad(-90.0), pitch=deg2rad(0.0), yaw=deg2rad(179.9))

        # set gripper params
        self.grasp_z_offset = 0.135      # (meter)
        self.gripper_stroke = 85 / 1000  # (meter), robotiq 85 gripper stroke: 85 mm -> 0.085 m
        self.max_grip_rad = torch.tensor(0.80285, device=self.device)
        self.angle_stroke_ratio = self.max_grip_rad / self.gripper_stroke

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

        # object order.
        # [0: robot, 1: bottle, 2: water drop, 3: cup, 4: ]
        self.ur3_default_dof_pos = to_torch([deg2rad(0.0), deg2rad(-110.0), deg2rad(100.0), deg2rad(0.0), deg2rad(80.0), deg2rad(0.0),
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
        self.cup_states = self.root_state_tensor[:, 3]

        self.contact_net_force = gymtorch.wrap_tensor(contact_net_force_tensor)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur3_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.num_actors = self.root_state_tensor.size()[1]
        self.global_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset(torch.arange(self.num_envs, device=self.device))
        self.refresh_env_tensors()

        self.task_status = AttrDict(
            approach_bottle=torch.zeros(self.num_envs, 1, device=self.device),
            grasping=torch.zeros(self.num_envs, 1, device=self.device),
            lifting=torch.zeros(self.num_envs, 1, device=self.device),
            pouring=torch.zeros(self.num_envs, 1, device=self.device),
            task_success=torch.zeros(self.num_envs, 1, device=self.device),
            bottle_fallen=torch.zeros(self.num_envs, 1, device=self.device),
            grasp_stability=torch.zeros(self.num_envs, 1, device=self.device),
        )
        self.set_task_viapoints(torch.arange(self.num_envs, device=self.device))

        # task_rl demo. params.
        self.task_update_buf = torch.zeros_like(self.progress_buf)

        # set gripper limit to fit the bottle's diameter
        blim = self.stroke_to_angle(self.bottle_diameter - 0.002)
        self.ur3_dof_lower_limits[7] = -blim
        self.ur3_dof_lower_limits[10] = -blim

        self.ur3_dof_upper_limits[6] = blim
        self.ur3_dof_upper_limits[8] = blim
        self.ur3_dof_upper_limits[9] = blim
        self.ur3_dof_upper_limits[11] = blim

        """ Camera Viewer setting """
        cam_pos_third_person = gymapi.Vec3(0.9263, 0., 0.5420)   # gymapi.Vec3(0.9263, 0.4617, 0.5420)
        cam_target_third_person = gymapi.Vec3(0.0, 0.0, 0.0)        # gymapi.Vec3(0.0, -0.3, 0.0)

        # Vec3(-0.114076, 0.199471, 0.953120)
        # Quat(0.673537, 0.682974, 0.201253, 0.198472)
        cam_pos_first_person = gymapi.Vec3(-0.114076, 0.0, 0.953120)
        cam_target_first_person = gymapi.Vec3(0.5, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos_first_person, cam_target_first_person)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M, "mouse_tr")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "set_ood")

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

    def _create_background_plane(self):
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        # asset_options.use_mesh_materials = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 15000
        asset_options.vhacd_params.max_convex_hulls = 128
        asset_options.vhacd_params.max_num_vertices_per_ch = 64

        # asset_file_path = "urdf/objects/background_plane.urdf"
        # bg_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file_path, asset_options)
        bg_asset = self.gym.create_box(self.sim, 0.001, 3.2, 1.0, asset_options)
        return bg_asset

    def _create_floor_plane(self):
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        floor_asset = self.gym.create_box(self.sim, 3.2, 3.2, 0.001, asset_options)
        return floor_asset

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

    def _create_asset_gourd_bottle(self):
        self.bottle_height = 0.2
        self.bottle_diameter = 0.037 * 2
        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # asset_options.armature = 0.005
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 2000000
        # asset_options.vhacd_params.max_convex_hulls = 128
        # asset_options.vhacd_params.max_num_vertices_per_ch = 32

        bottle_asset_file = "urdf/objects/gourd_bottle.urdf"
        if "asset" in self.cfg["env"]:
            bottle_asset_file = self.cfg["env"]["asset"].get("assetFileNameBottle", bottle_asset_file)

        bottle_asset = self.gym.load_asset(self.sim, self.asset_root, bottle_asset_file, asset_options)
        return bottle_asset

    def _create_asset_cup(self):
        asset_options = gymapi.AssetOptions()
        # asset_options.armature = 0.01
        # asset_options.angular_damping = 0.01
        # asset_options.linear_damping = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 250000
        # asset_options.vhacd_params.max_convex_hulls = 128
        # asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.use_mesh_materials = True

        # asset_options.fix_base_link = True
        # asset_options.disable_gravity = True

        target_cup_asset_name = "paper_cup_broad.urdf"      # paper_cup.urdf, paper_cup_broad.urdf
        cup_asset_file = "urdf/objects/" + target_cup_asset_name
        cup_asset = self.gym.load_asset(self.sim, self.asset_root, cup_asset_file, asset_options)

        cup_prop_dict = {
            "paper_cup.urdf": AttrDict(height=0.073, inner_radius=0.0328),
            "paper_cup_broad.urdf": AttrDict(height=0.073, inner_radius=0.066)
        }

        self.cup_height = cup_prop_dict[target_cup_asset_name].height
        self.cup_inner_radius = cup_prop_dict[target_cup_asset_name].inner_radius
        return cup_asset

    def create_asset_water_drops(self):
        r = 0.012
        self.expr = [[0, 0], [0, -r], [-r, 0], [0, r], [r, 0]]
        self.num_water_drops = 1
        self.water_drop_radius = 0.015

        asset_options = gymapi.AssetOptions()
        asset_options.density = 997
        asset_options.armature = 0.005
        liquid_asset = self.gym.create_sphere(self.sim, self.water_drop_radius, asset_options)   # radius
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
        bg_asset = self._create_background_plane()
        floor_asset = self._create_floor_plane()
        texture_file_path = os.path.join(self.asset_root, "textures", "PerlinNoiseTexture.png")
        self.perlin_texture_handle = self.gym.create_texture_from_file(self.sim, texture_file_path)
        texture = np.random.randint(low=0, high=255, size=(128, 128, 4), dtype=np.uint8)
        texture[:, :, -1] = 255
        self.random_texture_handle = self.gym.create_texture_from_buffer(self.sim, 128, 128, texture)

        ur3_asset = self._create_asset_ur3()
        bottle_asset = self._create_asset_bottle()
        # bottle_asset = self._create_asset_gourd_bottle()
        cup_asset = self._create_asset_cup()
        liq_asset = self.create_asset_water_drops()
        self.water_in_boundary_xy = self.cup_inner_radius - self.water_drop_radius
        self.water_in_boundary_z = self.cup_height * 0.8 - self.water_drop_radius

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
        self.ur3_dof_props["stiffness"][6:].fill(1000.0)
        self.ur3_dof_props["damping"][6:].fill(100.0)

        self.ur3_dof_lower_limits = self.ur3_dof_props['lower']
        self.ur3_dof_upper_limits = self.ur3_dof_props['upper']

        self.ur3_dof_lower_limits = to_torch(self.ur3_dof_lower_limits, device=self.device)
        self.ur3_dof_upper_limits = to_torch(self.ur3_dof_upper_limits, device=self.device)

        # self.ur3_dof_lower_limits = torch.index_select(self.ur3_dof_lower_limits, 0, self.indices)
        # self.ur3_dof_upper_limits = torch.index_select(self.ur3_dof_upper_limits, 0, self.indices)
        self.ur3_dof_speed_scales = torch.ones_like(self.ur3_dof_lower_limits)

        l_bg_pose = gymapi.Transform()
        l_bg_pose.p = gymapi.Vec3(-0.8, 0.0, 0.5)
        l_bg_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        r_bg_pose = gymapi.Transform()
        r_bg_pose.p = gymapi.Vec3(-0.8, -0.6, 0.6)
        r_bg_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        floor_bg_pose = gymapi.Transform()
        floor_bg_pose.p = gymapi.Vec3(-0.0, 0.0, 0.0)
        floor_bg_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        ur3_start_pose = gymapi.Transform()
        ur3_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ur3_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        bottle_start_pose = gymapi.Transform()
        bottle_start_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx))
        bottle_start_pose.p.x = 0.55
        bottle_start_pose.p.y = 0.0
        bottle_start_pose.p.z = self.bottle_height * 0.05

        liquid_start_pose = bottle_start_pose
        liquid_start_pose.p.z += 0.1

        cup_start_pose = gymapi.Transform()
        cup_start_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx))
        cup_start_pose.p.x = 0.5
        cup_start_pose.p.y = 0.0
        cup_start_pose.p.z = self.cup_height * 0.55

        third_person_top_pos_stare = [[0.78, 0.0, 0.5], [0.6, 0.0, 0.6]]
        third_person_view_pos_stare = [[0.9, 0.0, 0.6], [0.28, 0.0, 0.1]]
        first_person_view_pos_stare = [[-0.068866, -0.014887, 0.777630], [0.5, 0.0, 0.0]]
        self.default_cam_pos = third_person_view_pos_stare[0]
        self.default_cam_stare = third_person_view_pos_stare[1]

        # compute aggregate size
        num_bg_bodies = self.gym.get_asset_rigid_body_count(bg_asset)
        num_bg_shapes = self.gym.get_asset_rigid_shape_count(bg_asset)
        num_floor_bodies = self.gym.get_asset_rigid_body_count(floor_asset)
        num_floor_shapes = self.gym.get_asset_rigid_shape_count(floor_asset)

        num_ur3_bodies = self.gym.get_asset_rigid_body_count(ur3_asset)
        num_ur3_shapes = self.gym.get_asset_rigid_shape_count(ur3_asset)
        num_bottle_bodies = self.gym.get_asset_rigid_body_count(bottle_asset)
        num_bottle_shapes = self.gym.get_asset_rigid_shape_count(bottle_asset)
        num_cup_bodies = self.gym.get_asset_rigid_body_count(cup_asset)
        num_cup_shapes = self.gym.get_asset_rigid_shape_count(cup_asset)
        num_liq_bodies = self.gym.get_asset_rigid_body_count(liq_asset)
        num_liq_shapes = self.gym.get_asset_rigid_shape_count(liq_asset)

        self.max_agg_bodies = num_bg_bodies + num_floor_bodies + num_ur3_bodies + num_bottle_bodies + num_cup_bodies + \
                              self.num_water_drops * num_liq_bodies
        self.max_agg_shapes = num_bg_shapes + num_floor_shapes + num_ur3_shapes + num_bottle_shapes + num_cup_shapes + \
                              self.num_water_drops * num_liq_shapes

        self.l_bgs = []
        self.r_bgs = []
        self.floor_bgs = []
        self.ur3_robots = []
        self.bottles = []
        self.cups = []
        self.default_bottle_states = []
        self.default_cup_states = []
        self.camera_handles = []
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
            self.gym.set_rigid_body_texture(env_ptr, bottle_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.perlin_texture_handle)

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
            self.gym.set_rigid_body_texture(env_ptr, cup_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.perlin_texture_handle)

            # (5) Create Left Background
            l_bg_actor = self.gym.create_actor(env_ptr, bg_asset, l_bg_pose, "background_left", i, 0)
            # self.gym.reset_actor_materials(env_ptr, bg_actor, gymapi.MESH_VISUAL_AND_COLLISION)
            self.gym.set_rigid_body_texture(env_ptr, l_bg_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.random_texture_handle)
            self.gym.set_rigid_body_color(env_ptr, l_bg_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(_uniform(0.2, 1), _uniform(0.2, 1), _uniform(0.2, 1)))

            # (6) Create Floor Background
            floor_actor = self.gym.create_actor(env_ptr, floor_asset, floor_bg_pose, "floor_background", i, 0)
            # self.gym.reset_actor_materials(env_ptr, floor_actor, gymapi.MESH_VISUAL_AND_COLLISION)
            self.gym.set_rigid_body_texture(env_ptr, floor_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                            self.random_texture_handle)
            self.gym.set_rigid_body_color(env_ptr, floor_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(_uniform(0.2, 1), _uniform(0.2, 1), _uniform(0.2, 1)))

            # # (6) Create Left Background
            # r_bg_actor = self.gym.create_actor(env_ptr, bg_asset, r_bg_pose, "background_right", i, 0)
            # # self.gym.reset_actor_materials(env_ptr, bg_actor, gymapi.MESH_VISUAL_AND_COLLISION)
            # self.gym.set_rigid_body_texture(env_ptr, r_bg_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, texture_handle)
            # self.gym.set_rigid_body_color(env_ptr, r_bg_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
            #                               gymapi.Vec3(_uniform(0.2, 1), _uniform(0.2, 1), _uniform(0.2, 1)))

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

            # (3) or (4) Create Camera sensors
            camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
            cx, cy, cz = self.default_cam_pos
            sx, sy, sz = self.default_cam_stare
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(cx, cy, cz), gymapi.Vec3(sx, sy, sz))

            self.camera_handles.append(camera_handle)
            self.envs.append(env_ptr)
            self.ur3_robots.append(ur3_actor)
            self.bottles.append(bottle_actor)
            self.cups.append(cup_actor)
            self.l_bgs.append(l_bg_actor)
            # self.r_bgs.append(r_bg_actor)
            self.floor_bgs.append(floor_actor)

        self.robot_base_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "base_link")
        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "tool0")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "robotiq_85_left_finger_tip_link")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "robotiq_85_right_finger_tip_link")
        self.bottle_handle = self.gym.find_actor_rigid_body_handle(env_ptr, bottle_actor, "bottle")
        self.cup_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cup_actor, "paper_cup_broad")
        self.default_bottle_states = to_torch(self.default_bottle_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.default_cup_states = to_torch(self.default_cup_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.init_data()

    def init_data(self):
        # self.lfinger_idxs = []
        # self.rfinger_idxs = []
        # for i in range(self.num_envs):
        #     lfinger_idx = self.gym.find_actor_rigid_body_index(self.envs[i], mirobot_handle, "left_finger", gymapi.DOMAIN_SIM)
        ref_pose = gymapi.Transform()
        ref_pose.p = gymapi.Vec3(0.0, 0.0, 0.005)
        ref_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.ref_pos = to_torch([ref_pose.p.x, ref_pose.p.y, ref_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.ref_rot = to_torch([ref_pose.r.x, ref_pose.r.y, ref_pose.r.z, ref_pose.r.w], device=self.device).repeat((self.num_envs, 1))

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
            self.reset_buf, self.progress_buf, actions,
            self.bottle_grasp_pos, self.bottle_grasp_rot, self.bottle_pos, self.bottle_rot, self.bottle_tip_pos, self.bottle_floor_pos,
            self.ur3_grasp_pos, self.ur3_grasp_rot, self.cup_pos, self.cup_rot, self.cup_tip_pos, self.liq_pos,
            self.ur3_lfinger_pos, self.ur3_rfinger_pos,
            self.contact_net_force.view(self.num_envs, self.max_agg_bodies, -1)[:, self.lfinger_handle],
            self.contact_net_force.view(self.num_envs, self.max_agg_bodies, -1)[:, self.rfinger_handle],
            self.gripper_forward_axis, self.bottle_up_axis, self.gripper_up_axis, self.cube_left_axis,
            self.num_envs, self.water_in_boundary_xy, self.dist_reward_scale, self.rot_reward_scale, self.open_reward_scale,
            self.action_penalty_scale, self.max_episode_length
        )

    def refresh_env_tensors(self):
        self.gym.end_access_image_tensors(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

    def render_camera(self, to_numpy=False, color_order='rgb'):
        img = None
        for i in range(len(self.envs)):
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i],
                                                                 self.camera_handles[i], gymapi.IMAGE_COLOR)
            img = gymtorch.wrap_tensor(camera_tensor)[:, :, :3]

        if to_numpy:
            img = img.cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if color_order == 'bgr' else img
        return (img / 255.0).astype(np.float32)

    def _render_camera(self, to_numpy=False, color_order='rgb'):    # for multi env
        img = torch.zeros(self.num_envs, self.camera_props.width, self.camera_props.height, 3, device=self.device)
        for i in range(len(self.envs)):
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i],
                                                                 self.camera_handles[i], gymapi.IMAGE_COLOR)
            img[i] = gymtorch.wrap_tensor(camera_tensor)[:, :, :3]

        if to_numpy:
            img = img.cpu().numpy()
            for i in range(len(img)):
                img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR) if color_order == 'bgr' else img[i]
        return (img / 255.0).astype(np.float32)

    def compute_observations(self):
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
        self.cup_pos = self.rigid_body_states[:, self.cup_handle][:, 0:3]
        self.cup_rot = self.rigid_body_states[:, self.cup_handle][:, 3:7]
        # self.cup_pos = self.cup_states[:, 0:3]
        # self.cup_rot = self.cup_states[:, 3:7]

        # TODO
        # 1.(self: isaacgym._bindings.linux - x86_64.gym_37.Gym, arg0: isaacgym._bindings.linux - x86_64.gym_37.Env, arg1: int, arg2: int) ->
        # numpy.ndarray[isaacgym._bindings.linux - x86_64.gym_37.RigidBodyState]
        # [Error][carb.gym.plugin] Function GymGetActorRigidBodyStates cannot be used with the GPU pipeline after simulation starts.
        # Please use the tensor API if possible.See docs / programming / tensors.html for more info.

        # state = self.gym.get_actor_rigid_body_states(self.envs[0], self.cups[0], gymapi.STATE_NONE)
        # print("state: ", state)
        # temp = gymtorch.unwrap_tensor(self.cup_states)
        # self.gym.set_actor_rigid_body_states(self.envs[0], self.cups[0],
        #                                      gymtorch.unwrap_tensor(self.cup_states), gymapi.STATE_ALL)

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
        # dof_pos = self.ur3_dof_pos
        dof_pos = torch.index_select(dof_pos_scaled, 1, self.indices)
        # dof_pos[:, -1] = self.ur3_dof_pos[:, -1]  # useless code line...
        dof_vel = torch.index_select(self.ur3_dof_vel, 1, self.indices)
        dof_pos_vel = torch.cat((dof_pos, dof_vel[:, :-1]), dim=-1)     # except for gripper joint speed

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
        # dof_state = dof_pos_finger if self.use_ik else dof_pos
        tip_pos_diff = self.cup_tip_pos - self.bottle_tip_pos

        if self.img_obs:
            self.states_buf = torch.cat((dof_pos_vel,                               # 13, [0, 12]
                                        self.ur3_grasp_pos, self.ur3_grasp_rot,     # 7, [13, 19]
                                        self.cup_tip_pos - self.bottle_tip_pos,     # 3, [20, 22]
                                        self.bottle_pos, self.bottle_rot,           # 7, [23, 29]
                                        self.cup_pos, self.cup_rot,                 # 7, [30, 36]
                                        self.liq_pos, to_target_pos,                # 6, [37, 42]
                                        to_target_rot                               # 4, [43, 46]
                                         ), dim=-1)

            """ Camera Sensor Visualization """
            for i in range(len(self.envs)):
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i],
                                                                     self.camera_handles[i], gymapi.IMAGE_COLOR)
                torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)[:, :, :3]
                self.obs_buf[i, :] = torch_camera_tensor
                if self.debug_cam:
                    bgr_cam = cv2.cvtColor(torch_camera_tensor.cpu().numpy(), cv2.COLOR_RGB2BGR)
                    cv2.imshow("camera sensor", bgr_cam)
                    k = cv2.waitKey(0)
                    if k == 27:     # ESC
                        exit()
        else:
            self.obs_buf = torch.cat((dof_pos_vel,                              # 13, [0, 12]
                                     self.ur3_grasp_pos, self.ur3_grasp_rot,    # 7, [13, 19]
                                     self.cup_tip_pos - self.bottle_tip_pos,    # 3, [20, 22]
                                     self.bottle_pos, self.bottle_rot,          # 7, [23, 29]
                                     self.cup_pos, self.cup_rot,                # 7, [30, 36]
                                     self.liq_pos, to_target_pos,               # 6, [37, 42]
                                     to_target_rot                              # 4, [43, 46]
                                     ), dim=-1)

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
            self.ur3_default_dof_pos.unsqueeze(0) + self.rand_init_pos_scale * (torch.rand((len(env_ids), self.num_ur3_dofs), device=self.device) - 0.5),
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
        xy_scale = to_torch([0.13, 0.4, 0.0,            # position, 0.13, 0.4, 0.0,
                             0.0, 0.0, 0.0, 0.0,        # rotation (quat)
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(pick), 1)

        # both side randomization: 0.5, right_side only: 1.0
        both_size_rand = True
        const = 0.5 if both_size_rand else 1.0
        rand_bottle_pos = (torch.rand_like(pick) - const) * xy_scale

        self.bottle_states[env_ids] = pick + rand_bottle_pos

        for e_id in env_ids:
            self.gym.set_rigid_body_color(self.envs[e_id], self.bottles[e_id], 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(_uniform(1, 1), _uniform(0.0, 0.5), _uniform(0.0, 0.5)))
                                          # gymapi.Vec3(_uniform(0, 0.1), _uniform(0.0, 1.0), _uniform(0.0, 1.0)))


        # reset cup
        place = self.default_cup_states[env_ids]
        place += (torch.rand_like(place) - 0.5) * xy_scale
        place[:, 1] = torch.where(self.bottle_states[env_ids, 1] >= 0,
                                  self.bottle_states[env_ids, 1] - 0.25 + (torch.rand(1, device=self.device) - 0.5) * 0.2,
                                  self.bottle_states[env_ids, 1] + 0.25 + (torch.rand(1, device=self.device) - 0.5) * 0.2)
        place[:, 3:7] = quat
        self.cup_states[env_ids] = place

        for e_id in env_ids:
            self.gym.set_rigid_body_color(self.envs[e_id], self.cups[e_id], 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(_uniform(0.0, 0.5), _uniform(1, 1), _uniform(0.0, 0.5)))

        # reset liquid
        init_liq_pose = pick + rand_bottle_pos
        init_liq_pose[:, 2] = init_liq_pose[:, 2] + 0.22
        offset_z = 0.05
        for i in range(self.liquid_states.shape[1]):
            self.liquid_states[env_ids, i] = init_liq_pose
            init_liq_pose[:, 2] = init_liq_pose[:, 2] + offset_z

        # background color
        for e_id in env_ids:
            self.gym.set_rigid_body_color(self.envs[e_id], self.l_bgs[e_id], 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(_uniform(0.7, 1), _uniform(0.7, 1), _uniform(0.7, 1)))
            # self.gym.set_rigid_body_color(self.envs[e_id], self.r_bgs[e_id], 0, gymapi.MESH_VISUAL_AND_COLLISION,
            #                               gymapi.Vec3(_uniform(0.7, 1), _uniform(0.7, 1), _uniform(0.7, 1)))
            self.gym.set_rigid_body_color(self.envs[e_id], self.floor_bgs[e_id], 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(_uniform(0.7, 1), _uniform(0.7, 1), _uniform(0.7, 1)))

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

        for i in range(len(self.envs)):
            if i in env_ids:
                _cp = self.default_cam_pos
                _cs = self.default_cam_stare
                rand_pos = gymapi.Vec3(_cp[0] + _uniform(low=-0.05, high=0.05, size=1),     # [-0.08, 0.08]
                                       _cp[1] + _uniform(low=-0.05, high=0.05, size=1),       # [-0.1, 0.1]
                                       _cp[2] + _uniform(low=-0.05, high=0.05, size=1))     # [-0.05, 0.05]
                rand_stare = gymapi.Vec3(_cs[0] + _uniform(low=-0.05, high=0.05, size=1),   # [-0.05, 0.05]
                                         _cs[1] + _uniform(low=-0.05, high=0.05, size=1),   # [-0.05, 0.05]
                                         _cs[2] + _uniform(low=-0.05, high=0.05, size=1))   # [-0.05, 0.05]
                self.gym.set_camera_location(self.camera_handles[i], self.envs[i], rand_pos, rand_stare)

        if self.img_obs:
            # reset camera sensor pose
            for i in range(len(self.envs)):
                if i in env_ids:
                    _cp = self.default_cam_pos
                    _cs = self.default_cam_stare
                    rand_pos = gymapi.Vec3(_cp[0] + _uniform(low=-0.05, high=0.05, size=1),
                                           _cp[1] + _uniform(low=-0.05, high=0.05, size=1),
                                           _cp[2] + _uniform(low=-0.05, high=0.05, size=1))
                    rand_stare = gymapi.Vec3(_cs[0] + _uniform(low=-0.05, high=0.05, size=1),
                                             _cs[1] + _uniform(low=-0.05, high=0.05, size=1),
                                             _cs[2] + _uniform(low=-0.05, high=0.05, size=1))
                    self.gym.set_camera_location(self.camera_handles[i], self.envs[i], rand_pos, rand_stare)

        # light params, affecting all envs
        l_color = gymapi.Vec3(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1))
        l_ambient = gymapi.Vec3(random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0))
        l_direction = gymapi.Vec3(random.uniform(0., 1), random.uniform(0., 1), random.uniform(0., 1))
        self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))
        self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))
        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))

        # apply
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
        temp = self.max_grip_rad - self.angle_stroke_ratio * m
        return tensor_clamp(temp, self.ur3_dof_lower_limits[8], self.ur3_dof_upper_limits[8])

    def angle_to_stroke(self, rad):
        temp = (self.max_grip_rad - rad) / self.angle_stroke_ratio
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

        # u2[:, 8-6] = angle_err
        #
        # scale = 1.0
        # u2[:, 6-6] = scale * u2[:, 8-6]
        # u2[:, 7-6] = -scale * u2[:, 8-6]
        # u2[:, 9-6] = scale * u2[:, 8-6]
        # u2[:, 10-6] = -scale * u2[:, 8-6]
        # u2[:, 11-6] = scale * u2[:, 8-6]

        _u = torch.cat((u, angle_err.unsqueeze(-1)), dim=1)
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
            if len(actions.shape) < 2:
                actions = actions.unsqueeze(0)
            _actions = actions[:, :7].clone().to(self.device)
            if self.action_noise:
                _actions += (torch.rand_like(_actions) - 0.5) * self.action_noise_scale   # add joint action noise
            grip_act = _actions[:, -1].unsqueeze(-1).repeat(1, 5) * torch.tensor([-1., 1., 1., -1., 1.], device=self.device)
            self.actions = torch.cat((_actions, grip_act), dim=-1)

        # pause
        def get_img_with_text(text=''):
            img = np.zeros((128, 512, 3), np.uint8)

            # Write some Text
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (256-64, 64-8)
            fontScale = 1
            fontColor = (255, 255, 255)
            thickness = 1
            lineType = 2

            cv2.putText(img, text,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            return img

        if self.interaction_mode:
            if self.pause:
                img = get_img_with_text('Pause')
                cv2.imshow('pause', img)
                cv2.waitKey(1)
                self.actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device)
                self.ur3_dof_vel = torch.zeros_like(self.ur3_dof_vel)
                self.progress_buf -= 1
            else:
                img = get_img_with_text('Resume')
                cv2.imshow('pause', img)
                cv2.waitKey(1)

        if self.teleoperation_mode:
            self.teleoperation()

        targets = self.ur3_dof_pos + self.ur3_dof_speed_scales * self.dt * self.actions * self.action_scale
        # targets1 = self.ur3_dof_pos[:, :6] + self.ur3_dof_speed_scales * (1/32) * self.actions[:, :6] * self.action_scale
        # targets2 = self.ur3_dof_pos[:, 6:] + self.ur3_dof_speed_scales * (1/28) * self.actions[:, 6:] * self.action_scale
        # targets = torch.cat(targets1, targets2, dim=-1)
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

        if self.interaction_mode:
            self.interaction()

        self.key_event_handling()

    def key_event_handling(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "mouse_tr" and evt.value > 0:
                tr = self.gym.get_viewer_camera_transform(self.viewer, self.envs[0])
                print("p: ", tr.p)
                print("r: ", tr.r)
            if evt.action == "set_ood" and evt.value > 0:
                if self.debug_viz:
                    self.debug_viz = False
                    self.gym.clear_lines(self.viewer)
                else:
                    self.debug_viz = True

    def interaction(self):
        if self.viewer:
            # print("Interaction Mode is Running...")
            if self.gym.query_viewer_has_closed(self.viewer):
                exit()

            # vr controller
            pose = self.vr.devices["controller_1"].get_pose_euler()
            vel = self.vr.devices["controller_1"].get_velocity()
            d = self.vr.devices["controller_1"].get_controller_inputs()

            env_ids = (self.reset_buf == 0).nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                # to remove the button pressed noise
                btn_menu = d["menu_button"]
                self.btn_pause_que.append(btn_menu)
                in_seq = 4
                if len(self.btn_pause_que) > in_seq:
                    self.btn_pause_que.pop(0)

                if len(self.btn_pause_que) >= in_seq:
                    if self.btn_pause_que.count(True) >= len(self.btn_pause_que):
                        self.pause = False if self.pause else True
                        print("btn pressed...", self.btn_pause_que, self.pause)
                        self.btn_pause_que = [False for _ in range(len(self.btn_pause_que))]

                if d["trackpad_pressed"]:
                    self.reset_buf = torch.ones_like(self.reset_buf)

                if d["trigger"]:
                    if vel:
                        scale = 0.025
                        self.cup_states[0, 0] = torch.clamp(self.cup_states[0, 0] + scale * -vel[2], min=0.45, max=0.6)
                        self.cup_states[0, 1] = torch.clamp(self.cup_states[0, 1] + scale * -vel[0], min=-0.28, max=0.28)
                        self.cup_states[0, 2] = torch.clamp(self.cup_states[0, 2] + scale * vel[1], min=0.04, max=0.05)

                        _indices = self.global_indices[env_ids, 3].flatten()
                        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                                     gymtorch.unwrap_tensor(_indices),
                                                                     len(_indices))

            # # check mouse event
            # for evt in self.gym.query_viewer_action_events(self.viewer):
            #     if evt.action == "mouse_left":
            #         pos = self.gym.get_viewer_mouse_position(self.viewer)
            #         window_size = self.gym.get_viewer_size(self.viewer)
            #         u = (pos.x * window_size.x - window_size.x / 2) / window_size.x
            #         v = (pos.y * window_size.y - window_size.y / 2) / window_size.x
            #         # xcoord = pos.x - 0.5
            #         # ycoord = pos.y - 0.5
            #         print("Left mouse was clicked at x: {:.3f},  y: {:.3f}".format(pos.x, pos.y))
            #         print(f"Mouse coords: {u}, {v}")
            #
            #         # camera pose
            #         _cam_pose = self.gym.get_viewer_camera_transform(self.viewer, self.envs[0])
            #         _cam_look = _cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))
            #
            #         cam_pos = np.array([_cam_pose.p.x, _cam_pose.p.y, _cam_pose.p.z])
            #         cam_look = np.array([_cam_look.x, _cam_look.y, _cam_look.z])
            #         print("cam pos: {},  cam_fwd: {}".format(cam_pos, cam_look))
            #
            #         projection_matrix = self.gym.get_camera_proj_matrix(self.sim, self.envs[0], self.camera_handles[0])
            #         view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[0], self.camera_handles[0]))
            #         print("proj_matrix: \n{}, \nview_matrix: \n{}".format(projection_matrix, view_matrix))
            #
            #         fu = 2 / projection_matrix[0, 0]
            #         fv = 2 / projection_matrix[1, 1]
            #         d = 1.0
            #         p = np.array([fu * u, fv * v, d, 1.0])
            #         print("inv proj: {}".format(p * np.linalg.inv(view_matrix)))
            #         print("cup pos: {}".format(self.cup_pos[0, :]))
            #         print("======================================")

    def teleoperation(self):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer): exit()
            goal = torch.zeros(1, 8, device=self.device)
            goal[:, -2:] = 1.0
            d = self.vr.devices["controller_1"].get_controller_inputs()
            if d['trigger']:
                pv = np.array([v for v in self.vr.devices["controller_1"].get_velocity()]) * 1.0
                av = np.array([v for v in self.vr.devices["controller_1"].get_angular_velocity()]) * 1.0  # incremental
                # av = np.array([v for v in vr.devices["controller_1"].get_pose_quaternion()]) * 1.0        # absolute

                pv = torch.matmul(self.vr_ref_rot, torch.tensor(pv).unsqueeze(0).T)
                av = torch.tensor(av).unsqueeze(0)
                _q = mat_to_quat(self.vr_ref_rot.unsqueeze(0))
                av = quat_apply(_q, av)

                goal[:, :3] = pv.T
                _quat = quat_from_euler_xyz(roll=av[0, 0], pitch=av[0, 1], yaw=av[0, 2])
                goal[:, 3:7] = _quat

                # trackpad button transition check and gripper manipulation
                self.trk_btn_trans.append(0) if d["trackpad_pressed"] else self.trk_btn_trans.append(1)
                if len(self.trk_btn_trans) > 2: self.trk_btn_trans.pop(0)
                if len(self.trk_btn_trans) > 1:
                    a, b = self.trk_btn_trans
                    if (b - a) < 0:
                        self.trk_btn_toggle = 0 if self.trk_btn_toggle else 1
                        print("track pad button pushed, ", self.trk_btn_toggle)

                goal[:, 7] = self.trk_btn_toggle
                _actions = self.solve(goal_pos=goal[:, :3], goal_rot=goal[:, 3:7], goal_grip=goal[:, 7], absolute=False)

                actions = torch.zeros_like(self.actions)
                actions[:, :6] = _actions[:, :6]
                gripper_actuation = _actions[:, -1]
                actions[:, 6] = actions[:, 8] = actions[:, 9] = actions[:, 11] = gripper_actuation
                actions[:, 7] = actions[:, 11] = -1 * gripper_actuation
                self.actions = actions

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.refresh_env_tensors()

        self.compute_observations()
        self.compute_reward(self.actions)

        # compute task update status
        # self.compute_task()

        t = self.gym.get_sim_time(self.sim)
        dof_pos_finger = self.angle_to_stroke(self.ur3_dof_pos[:, 8].unsqueeze(-1))
        done_envs = self.tpm.update_step_by_checking_arrive(ee_pos=self.ur3_grasp_pos, ee_rot=self.ur3_grasp_rot,
                                                            ee_grip=dof_pos_finger,
                                                            dof_pos=torch.index_select(self.ur3_dof_pos, 1, self.indices),
                                                            sim_time=t)

        self.reset_buf = torch.where(done_envs > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        self.task_update_buf = torch.where(self.progress_buf == 10,     # for stability
                                           torch.ones_like(self.progress_buf), torch.zeros_like(self.progress_buf))

        _env_ids = self.task_update_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(_env_ids) > 0:
            self.set_task_viapoints(_env_ids)

        self.task_evaluation()

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.ref_pos[i] + quat_apply(self.ref_rot[i], to_torch([1, 0, 0], device=self.device) * 0.4)).cpu().numpy()
                py = (self.ref_pos[i] + quat_apply(self.ref_rot[i], to_torch([0, 1, 0], device=self.device) * 0.4)).cpu().numpy()
                pz = (self.ref_pos[i] + quat_apply(self.ref_rot[i], to_torch([0, 0, 1], device=self.device) * 0.4)).cpu().numpy()

                p0 = self.ref_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # px = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.hand_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # bottle grasp pose
                px = (self.bottle_grasp_pos[i] + quat_apply(self.bottle_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.bottle_grasp_pos[i] + quat_apply(self.bottle_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.bottle_grasp_pos[i] + quat_apply(self.bottle_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.bottle_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # bottle tip pose
                px = (self.bottle_tip_pos[i] + quat_apply(self.bottle_tip_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.bottle_tip_pos[i] + quat_apply(self.bottle_tip_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.bottle_tip_pos[i] + quat_apply(self.bottle_tip_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.bottle_tip_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # bottle floor pose
                # px = (self.bottle_floor_pos[i] + quat_apply(self.bottle_floor_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.bottle_floor_pos[i] + quat_apply(self.bottle_floor_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.bottle_floor_pos[i] + quat_apply(self.bottle_floor_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.bottle_floor_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # cup pose
                px = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cup_pos[i] + quat_apply(self.cup_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cup_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # cup tip pose
                px = (self.cup_tip_pos[i] + quat_apply(self.cup_tip_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cup_tip_pos[i] + quat_apply(self.cup_tip_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cup_tip_pos[i] + quat_apply(self.cup_tip_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cup_tip_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # ur3 grasp pose
                px = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.ur3_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # TODO
                # # appr bottle pose for debug
                # px = (self.appr_bottle_pos[i] + quat_apply(self.appr_bottle_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.appr_bottle_pos[i] + quat_apply(self.appr_bottle_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.appr_bottle_pos[i] + quat_apply(self.appr_bottle_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.appr_bottle_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
                #
                # # bottle pos init
                # px = (self.bottle_pos_init[i] + quat_apply(self.bottle_rot_init[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.bottle_pos_init[i] + quat_apply(self.bottle_rot_init[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.bottle_pos_init[i] + quat_apply(self.bottle_rot_init[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.bottle_pos_init[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

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

    def task_evaluation(self):
        """
        Target:
            1) approach bottle (O)
            2) grasping (O)
            3) lifting (O)
            4) approach cup, grasping bottle
            5) pouring (O)
            6) task success (O)

        Failure:
            1) bottle fall (O)
            2) cup fall
            3) unstable grasping (O)
        """

        # Target: 1) approach bottle
        d1 = torch.norm(self.ur3_grasp_pos - self.bottle_grasp_pos, p=2, dim=-1)
        self.task_status.approach_bottle = torch.where(d1 < 0.1, 1.0, self.task_status.approach_bottle.double())

        # Target: 2) grasping
        finger_dist = torch.norm(self.ur3_lfinger_pos - self.ur3_rfinger_pos, p=2, dim=-1).unsqueeze(-1)
        grasp_cond = (d1 < 0.05) & (finger_dist < self.bottle_diameter + 0.005)
        self.task_status.grasping = torch.where(grasp_cond, 1.0, self.task_status.grasping.double())

        # Target: 3) lifting
        self.task_status.lifting = torch.where((self.task_status.grasping == 1.0) & (self.bottle_floor_pos[:, -1] > 0.08),
                                                1.0, self.task_status.lifting.double())

        # Target: 5) pouring
        r = self.bottle_tip_pos[:, -1] - self.bottle_floor_pos[:, -1]
        self.task_status.pouring = torch.where((self.task_status.lifting == 1.0) & (r < 0.0),
                                                1.0, self.task_status.pouring.double())

        # Target: 6) task success
        # Target: 6) task success
        # is_cup_fallen = dot4 < 0.5
        # is_bottle_fallen = (bottle_floor_pos[:, 2] < 0.02) & (dot3 < 0.8)
        # is_pouring_finish = (bottle_pos[:, 2] > 0.09 + 0.074 * 0.5) & (liq_pos[:, 2] < 0.03)
        liq_cup_dist_xy = torch.norm(self.liq_pos[:, :2] - self.cup_pos[:, :2], p=2, dim=-1)
        is_poured = (liq_cup_dist_xy < self.water_in_boundary_xy) & (self.liq_pos[:, 2] < self.water_in_boundary_z)
        self.task_status.task_success = torch.where(is_poured, 1.0, self.task_status.task_success.double())

        # Failure: 1) bottle fallen
        axis1 = tf_vector(self.bottle_grasp_rot, self.bottle_up_axis)
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), self.bottle_up_axis.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        # is_bottle_fallen = (self.bottle_floor_pos[:, 2] < 0.02) & (dot1 < 0.7)
        is_bottle_fallen = torch.logical_not(grasp_cond) & (dot1 < 0.7)
        self.task_status.bottle_fallen = torch.where(is_bottle_fallen, 1.0, self.task_status.bottle_fallen.double())

        # Failure: 2) grasping stability
        ee_up_axis = tf_vector(self.ur3_grasp_rot, self.gripper_up_axis)
        bottle_up_axis = tf_vector(self.bottle_grasp_rot, self.bottle_up_axis)
        cos_loss = torch.bmm(ee_up_axis.view(self.num_envs, 1, 3), bottle_up_axis.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        self.task_status.grasp_stability = torch.where(self.task_status.grasping == True,
                                                       cos_loss, self.task_status.grasp_stability)

        # if not hasattr(self.extras, "pouring_task_eval"):
        #     self.extras = []
        # else:
        _task_status = copy.deepcopy(self.task_status)
        for k, v, in _task_status.items():
            _task_status[k] = v.cpu().numpy()
        self.extras = _task_status

        # print("approach bottle: {}, grasping: {}, lifting: {}, pouring: {}, task success: {}, bottle fallen: {}, grasp cos loss: {}"
        #       .format(self.task_status.approach_bottle.item(),
        #               self.task_status.grasping.item(),
        #               self.task_status.lifting.item(),
        #               self.task_status.pouring.item(),
        #               self.task_status.task_success.item(),
        #               self.task_status.bottle_fallen.item(),
        #               self.task_status.grasp_stability.item()))

    def set_task_viapoints(self, env_ids):
        # task progress reset, TODO, should be in reset() function?
        for k, v in self.task_status.items():
            v[env_ids] = 0.0

        if not hasattr(self, "task_pose_list"):
            self.tpl = TaskPoseList(task_name="pouring", num_envs=self.num_envs, device=self.device)

        """ 
            1)-1 initial pos variation 
        """
        init_ur3_hand_pos = self.tpl.gen_pos_variation(pivot_pos=[0.5, 0.0, 0.35], pos_var_meter=0.02)
        init_ur3_hand_pos = to_torch([0.5, 0.0, 0.32], device=self.device).repeat((self.num_envs, 1))
        pos_var_meter = 0.02
        pos_var = (torch.rand_like(init_ur3_hand_pos) - 0.5) * 2.0
        init_ur3_hand_pos += pos_var * pos_var_meter

        """ 
            1)-2 initial rot variation 
        """
        init_ur3_hand_rot = self.tpl.gen_rot_variation(pivot_quat=[0.0, 0.0, 0.0, 1.0], rot_var_deg=15)
        init_ur3_hand_rot = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        rot_var_deg = 15    # +-
        roll = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg
        pitch = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg
        yaw = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * rot_var_deg
        q_var = quat_from_euler_xyz(roll=deg2rad(roll), pitch=deg2rad(pitch), yaw=deg2rad(yaw))
        init_ur3_hand_rot = quat_mul(init_ur3_hand_rot, q_var)

        """
            1)-3 initial grip variation
            For 2F-85 gripper, 0x00 --> full open with 85mm, 0xFF --> close
            Unit: meter ~ [0.0, 0.085]
        """
        init_ur3_grip = self.tpl.gen_grip_variation(pivot_grip=[0.08], grip_var_meter=0.01)
        init_ur3_grip = to_torch([0.08], device=self.device).repeat((self.num_envs, 1))     # meter
        grip_var = (torch.rand_like(init_ur3_grip) - 0.5) * 0.01   # grip. variation range: [0.075, 0.085]
        init_ur3_grip = torch.min(init_ur3_grip + grip_var, torch.tensor(self.gripper_stroke, device=self.device))
        self.tpl.append_pose(pos=init_ur3_hand_pos, rot=init_ur3_hand_rot, grip=init_ur3_grip)

        """
            2) approach above the bottle
        """
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

        robot_base_pos = self.rigid_body_states[:, self.robot_base_handle][:, 0:3]
        vx = bottle_pos - robot_base_pos
        vx[:, 2] = 0.0
        vx = normalize(vx, p=2.0, dim=-1)
        appr_bottle_pos = bottle_pos.clone().detach() - (vx * self.bottle_diameter * 1.7)
        appr_bottle_pos[:, 2] = self.bottle_height + 0.05

        q_z90 = torch.tensor([[0.0, 0.0, 0.707, 0.707]] * self.num_envs, device=self.device)
        vy = quat_apply(q_z90, vx)
        vz = vx.cross(vy)

        mat = torch.stack([vx, vy, vz], dim=-1)
        appr_bottle_rot = mat_to_quat(mat)

        if not hasattr(self, "appr_bottle_pos") and not hasattr(self, "appr_bottle_rot"):
            self.appr_bottle_pos, self.appr_bottle_rot = appr_bottle_pos, appr_bottle_rot
        self.appr_bottle_pos[env_ids], self.appr_bottle_rot[env_ids] = appr_bottle_pos[env_ids], appr_bottle_rot[env_ids]
        appr_bottle_grip = to_torch([0.085], device=self.device).repeat((self.num_envs, 1))  # full open
        self.tpl.append_pose(pos=appr_bottle_pos, rot=appr_bottle_rot, grip=appr_bottle_grip,
                             err=ViaPointProperty(pos=3.e-1, rot=2.e-2, grip=1.e-2))

        """
            3) grasp ready
        """
        grasp_ready_pos = bottle_pos.clone().detach()
        grasp_ready_rot = appr_bottle_rot.clone().detach()
        grasp_ready_grip = appr_bottle_grip.clone().detach()
        self.tpl.append_pose(pos=grasp_ready_pos, rot=grasp_ready_rot, grip=grasp_ready_grip,
                             err=ViaPointProperty(pos=3.e-2, rot=3.e-2, grip=3.e-3))

        """
            4) grasp
        """
        grasp_pos = grasp_ready_pos.clone().detach()
        grasp_rot = grasp_ready_rot.clone().detach()
        grasp_grip = to_torch([self.bottle_diameter - 0.002], device=self.device).repeat((self.num_envs, 1))
        self.tpl.append_pose(pos=grasp_pos, rot=grasp_rot, grip=grasp_grip,
                             err=ViaPointProperty(pos=3.e-2, rot=3.e-2, grip=1.e-3))

        """
            5) lift
        """
        lift_pos = grasp_pos.clone().detach()
        lift_pos[:, 2] += 0.2
        lift_rot = grasp_rot.clone().detach()
        lift_grip = grasp_grip.clone().detach()
        self.tpl.append_pose(pos=lift_pos, rot=lift_rot, grip=lift_grip,
                             err=ViaPointProperty(pos=1.e-1, rot=1.e-1, grip=1.e-3))

        """
            6) approach cup
        """
        a = bottle_pos - cup_pos
        b = torch.tensor([[0.0, 1.0, 0.0]] * self.num_envs, device=self.device)
        nom = torch.bmm(a.view(len(a), 1, 3), b.view(len(b), 3, 1)).squeeze(-1)
        denom = torch.bmm(b.view(len(b), 1, 3), b.view(len(b), 3, 1)).squeeze(-1)
        proj = (nom / denom) * b
        proj = proj / proj.norm(dim=-1).unsqueeze(-1)
        appr_cup_pos = cup_pos + proj * 0.12
        appr_cup_pos[:, 2] = self.cup_height + 0.068
        appr_cup_rot = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        appr_cup_grip = lift_grip.clone().detach()
        self.tpl.append_pose(pos=appr_cup_pos, rot=appr_cup_rot, grip=appr_cup_grip,
                             err=ViaPointProperty(pos=1.e-1, rot=1.e-1, grip=1.e-3))

        """
            7) pouring
        """
        pour_cup_pos = appr_cup_pos.clone().detach()
        pour_cup_rot = appr_cup_rot.clone().detach()

        roll = deg2rad(torch.tensor(110, device=self.device).repeat(self.num_envs))
        direction = torch.where(bottle_pos[:, 1] > cup_pos[:, 1], torch.ones_like(roll), -1.0 * torch.ones_like(roll))
        pour_rot = quat_from_euler_xyz(roll=roll * direction, pitch=torch.zeros_like(roll), yaw=torch.zeros_like(roll))
        pour_cup_rot = quat_mul(pour_cup_rot, pour_rot)
        pour_cup_grip = appr_cup_grip.clone().detach()
        self.tpl.append_pose(pos=pour_cup_pos, rot=pour_cup_rot, grip=pour_cup_grip, err=ViaPointProperty(wait=2.))

        """
            8) put up bottle
        """
        put_up_bottle_pos = pour_cup_pos.clone().detach()
        put_up_bottle_rot = appr_cup_rot.clone().detach()
        put_up_bottle_grip = pour_cup_grip.clone().detach()
        self.tpl.append_pose(pos=put_up_bottle_pos, rot=put_up_bottle_rot, grip=put_up_bottle_grip)

        # """
        #     9) return bottle
        # """
        #
        # return_bottle_pos = lift_pos.clone().detach()
        # return_bottle_rot = lift_rot.clone().detach()
        # return_bottle_grip = lift_grip.clone().detach()
        # self.task_pose_list.append_pose(pos=return_bottle_pos, rot=return_bottle_rot, grip=return_bottle_grip)

        """
            Last) push poses to the task path manager
        """
        if not hasattr(self, "tpm"):
            num_task_steps = self.tpl.length()
            print("num_task_steps: ", num_task_steps)
            self.tpm = TaskPathManager(num_env=self.num_envs, num_task_steps=num_task_steps, device=self.device)
            self.tpm.set_init_joint_pos(joint=torch.index_select(self.ur3_default_dof_pos, 0, self.indices).clone())
        self.tpm.reset_task(env_ids=env_ids)

        for i in range(self.tpl.length()):
            p, r, g, e = self.tpl.pose_pop(index=0)
            self.tpm.push_pose(env_ids=env_ids, pos=p, rot=r, grip=g, err=e)

    def calc_task_error(self):
        pass

    def calc_expert_action(self):
        des_pos, des_rot, des_grip, _ = self.tpm.get_desired_pose()
        actions = self.solve(goal_pos=des_pos, goal_rot=des_rot, goal_grip=des_grip, absolute=True)

        # desired joint action for initial position
        curr_joint = torch.index_select(self.ur3_dof_pos, 1, self.indices)
        des_joint = self.tpm.get_init_des_joint()

        # follow the joint space trajectory when initial step
        actions[:, :6] = torch.where(self.tpm._step[:, 0].repeat((1, 2))[:, :6] == 0,
                                     des_joint[:, :6] - curr_joint[:, :6],
                                     actions[:, :6])

        return torch.clamp(actions, min=-1.0, max=1.0)

    """
        Dataset normalization implementation on tensors for each line by line
        # a, b = -1, 1    # range [a, b]
        # # normalize does not preserve the data shape
        # numer = torch.sub(actions, actions.min(-1).values.unsqueeze(-1))
        # denom = actions.max(-1).values.unsqueeze(-1) - actions.min(-1).values.unsqueeze(-1)
        # actions = (b - a) * (numer / denom) + a

        # normalizing does preserve the data shape
        # numer = actions - actions.mean(-1).unsqueeze(-1)
        # sig_in = actions.var(-1).unsqueeze(-1)
        # sig_out = (b - a) / 2
        # u_out = (b + a) / 2
        # actions = (numer / sig_in) #* sig_out + u_out
    """


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
    num_envs, water_in_boundary_xy, dist_reward_scale, rot_reward_scale, open_reward_scale,
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
    # dist_reward = torch.where(approach_done > 0.0,
    #                           torch.where((lfd < 0.035) & (rfd < 0.035), dist_reward + 7.0,
    #                                       torch.where((lfd < 0.037) & (rfd < 0.037), dist_reward + 5.0,
    #                                                   torch.where((lfd < 0.04) & (rfd < 0.04), dist_reward + 3.0,
    #                                                               dist_reward + 1.0))),
    #                           dist_reward)

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

    # dist_reward = torch.exp(-5.0 * d1)  # between bottle and ur3 grasp
    approach_done = d1 < 0.02
    approach_reward = 1.5 * torch.where(approach_done, torch.exp(-5.0 * d1), 0.0 * torch.exp(-5.0 * d1))
    # grasping_reward = torch.where((approach_reward > 0.0) & (), 1.0, 0.0)
    lift_reward = 3.0 * torch.where((bottle_floor_pos[:, 2] > 0.05) & approach_done, 1.0, 0.0)
    # bottle_lean_rew = torch.where(bottle_floor_pos[:, 2] < 0.04, torch.exp(-7.0 * (1 - dot3)), torch.ones_like(dist_reward))
    up_rot_reward = 5.0 * torch.exp(-3.0 * (1.0 - dot1))


    # tip_dist_reward =
    rewards = approach_reward + lift_reward + up_rot_reward  #- action_penalty_scale * action_penalty

    poured_reward = torch.zeros_like(rewards)
    poured_reward_scale = 10.0
    is_poured = (liq_cup_dist_xy < water_in_boundary_xy) & (liq_pos[:, 2] < 0.04)  # 0.015, 0.04
    poured_reward = torch.where(is_poured, poured_reward + 1.0, poured_reward)
    rewards += poured_reward_scale * poured_reward
    # rewards = poured_reward_scale * poured_reward

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
    is_cup_fallen = dot4 < 0.5
    is_bottle_fallen = (dot3 < 0.5) & (d1 > 0.06)
    # is_bottle_fallen = (bottle_floor_pos[:, 2] < 0.02) & (dot3 < 0.6)
    # is_pouring_finish = (bottle_pos[:, 2] > 0.09 + 0.074 * 0.5) & (liq_pos[:, 2] < 0.03)
    rewards = torch.where(is_cup_fallen, torch.ones_like(rewards) * -1.0, rewards)  # paper cup fallen reward penalty
    rewards = torch.where(is_bottle_fallen, torch.ones_like(rewards) * -1.0, rewards)  # bottle fallen reward penalty

    # early stopping
    reset_buf = torch.where(is_bottle_fallen, torch.ones_like(reset_buf), reset_buf)    # bottle fallen
    reset_buf = torch.where(is_cup_fallen, torch.ones_like(reset_buf), reset_buf)       # paper cup fallen
    # reset_buf = torch.where(is_pouring_finish, torch.ones_like(reset_buf), reset_buf)   # pouring task anyway
    reset_buf = torch.where(is_poured, torch.ones_like(reset_buf), reset_buf)   # task success
    # reset_buf = torch.where(is_dropped > 0.0, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((liq_cup_dist_xy > 0.5) | (bottle_height > des_height + 0.3), torch.ones_like(reset_buf), reset_buf)    # out of range

    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos)

    return global_franka_rot, global_franka_pos
