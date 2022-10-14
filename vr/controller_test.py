import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi, gymtorch
from math import sqrt
import torch
from utils.torch_jit_utils import *
from utils.utils import *

import triad_openvr
import time
import sys


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos)

    return global_franka_rot, global_franka_pos


def controller_test():
    vr = triad_openvr.triad_openvr()
    vr.print_discovered_objects()

    if len(sys.argv) == 1:
        interval = 1 / 250
    elif len(sys.argv) == 2:
        interval = 1 / float(sys.argv[1])
    else:
        print("Invalid number of arguments")
        interval = False

    if interval:
        while True:
            start = time.time()
            pv, av = [], []
            for each in zip(vr.devices["controller_1"].get_velocity(), vr.devices["controller_1"].get_angular_velocity()):
                # [x, y, z, yaw, pitch, roll]
                pv.append(each[0]), av.append(each[1])
            print(pv + av)

            d = vr.devices["controller_1"].get_controller_inputs()
            if d['trigger']:
                print("trigger is pushed")

            sleep_time = interval - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


class SimVR:
    def __init__(self):
        # initialize VR device
        self.vr = triad_openvr.triad_openvr()
        self.vr.print_discovered_objects()
        self.rot = mat3d(roll=deg2rad(0), pitch=deg2rad(179.9), yaw=deg2rad(0))

        self.trk_btn_trans = []
        self.trk_btn_toggle = 1

        # initialize the isaac gym simulation
        self.gym = gymapi.acquire_gym()

        # parse arguments
        self.args = gymutil.parse_arguments(
            description="Projectiles Example: Press SPACE to fire a projectile. Press R to reset the simulation.",
            custom_parameters=[
                {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"}])

        dev_num = 1 if torch.cuda.device_count() else 0
        # self.args.sim_device = "cuda:{}".format(dev_num)
        self.args.sim_device = "cpu"
        self.args.compute_device_id = dev_num
        self.args.graphics_device_id = dev_num

        self.device = self.args.sim_device

        # configure sim
        sim_params = gymapi.SimParams()
        if self.args.physics_engine == gymapi.SIM_FLEX:
            sim_params.flex.shape_collision_margin = 0.05
            sim_params.flex.num_inner_iterations = 6
        elif self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = self.args.num_threads = 4
            sim_params.physx.use_gpu = self.args.use_gpu

        sim_params.use_gpu_pipeline = False
        if self.args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id,
                                       self.args.physics_engine, sim_params)

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.loop_on = True
        self.asset_root = "../assets"

        self.init_env()

        self.obj_handles = []
        # self._create_cube()
        self._create_ur3()

    def init_env(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)

        # subscribe to input events. This allows input to be used to interact
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "exit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

        # viewer camera setting
        cam_pos = gymapi.Vec3(3.58, 1.58, 0.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # set up the grid of environments
        self.num_envs = self.args.num_envs
        num_per_row = int(sqrt(self.num_envs))
        spacing = 2.0

        # only consider one environment
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, lower, upper, num_per_row)

    def _create_cube(self):
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 100.
        cube_asset_options.disable_gravity = True
        cube_asset_options.linear_damping = 20  # damping is important for stabilizing the movement
        cube_asset_options.angular_damping = 20

        cube_asset = self.gym.create_box(self.sim, 0.5, 0.5, 0.5, cube_asset_options)

        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(0.0, 0.25, 0.0)
        init_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cube_handle = self.gym.create_actor(self.env, cube_asset, init_pose, "cube", -1, 0)
        self.obj_handles.append(self.cube_handle)

        c = 0.5 + 0.5 * np.random.random(3)
        self.gym.set_rigid_body_color(self.env, self.cube_handle, 0,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))

        # save initial state for reset
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.cube_initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

    def _create_ur3(self):
        ur3_asset_file = "urdf/ur3_description/robot/ur3_robotiq85_gripper.urdf"

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
        ur3_link_dict = self.gym.get_asset_rigid_body_dict(ur3_asset)
        print("ur3 link dictionary: ", ur3_link_dict)

        self.ur3_default_dof_pos = to_torch(
            [deg2rad(0.0), deg2rad(-110.0), deg2rad(100.0), deg2rad(0.0), deg2rad(80.0), deg2rad(0.0),
             deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)], device=self.device)

        # ur3 dof properties
        ur3_dof_props = self.gym.get_asset_dof_properties(ur3_asset)
        ur3_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        # ur3 joints
        ur3_dof_props["stiffness"][:6].fill(300.0)
        ur3_dof_props["damping"][:6].fill(80.0)

        # robotiq 85 gripper
        ur3_dof_props["stiffness"][6:].fill(1000.0)
        ur3_dof_props["damping"][6:].fill(100.0)

        # ur3 joint limits
        self.ur3_dof_lower_limits = ur3_dof_props['lower']
        self.ur3_dof_upper_limits = ur3_dof_props['upper']

        self.ur3_dof_lower_limits = to_torch(self.ur3_dof_lower_limits, device=self.device)
        self.ur3_dof_upper_limits = to_torch(self.ur3_dof_upper_limits, device=self.device)

        ur3_start_pose = gymapi.Transform()
        ur3_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ur3_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # correct the ur3 base
        rot = quat_mul(to_torch([ur3_start_pose.r.x, ur3_start_pose.r.y,
                                 ur3_start_pose.r.z, ur3_start_pose.r.w], device=self.device),
                       quat_from_euler_xyz(torch.tensor(deg2rad(-90), device=self.device),
                                           torch.tensor(deg2rad(0), device=self.device),
                                           torch.tensor(deg2rad(0), device=self.device)))
        ur3_start_pose.r = gymapi.Quat(rot[0], rot[1], rot[2], rot[3])
        self.ur3_handle = self.gym.create_actor(self.env, ur3_asset, ur3_start_pose, "ur3", 0, 1)
        self.gym.set_actor_dof_properties(self.env, self.ur3_handle, ur3_dof_props)

        # init tensors
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur3_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float,
                                           device=self.device)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).to(self.device)
        self.ur3_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dofs]
        self.ur3_dof_pos = self.ur3_dof_state[..., 0]
        self.ur3_dof_vel = self.ur3_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13).to(self.device)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.num_actors = self.root_state_tensor.size()[1]
        self.global_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env, self.ur3_handle, "tool0")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(self.env, self.ur3_handle, "robotiq_85_left_finger_tip_link")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(self.env, self.ur3_handle, "robotiq_85_right_finger_tip_link")

        hand_pose = self.gym.get_rigid_transform(self.env, self.hand_handle)
        lfinger_pose = self.gym.get_rigid_transform(self.env, self.lfinger_handle)
        rfinger_pose = self.gym.get_rigid_transform(self.env, self.rfinger_handle)

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

        self.ur3_grasp_pos = torch.zeros_like(self.ur3_local_grasp_pos)
        self.ur3_grasp_rot = torch.zeros_like(self.ur3_local_grasp_rot)
        self.ur3_grasp_rot[..., -1] = 1
        self.ur3_lfinger_pos = torch.zeros_like(self.ur3_local_grasp_pos)
        self.ur3_rfinger_pos = torch.zeros_like(self.ur3_local_grasp_pos)

        # set gripper params
        self.grasp_z_offset = 0.135  # (meter)
        self.gripper_stroke = 85 / 1000  # (meter), robotiq 85 gripper stroke: 85 mm -> 0.085 m
        self.max_grip_rad = torch.tensor(0.80285, device=self.device)
        self.angle_stroke_ratio = self.max_grip_rad / self.gripper_stroke

        # jacobians
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur3")
        jacobian = gymtorch.wrap_tensor(_jacobian).to(self.device)

        # jacobian entries corresponding to franka hand
        self.ur3_hand_index = ur3_link_dict["ee_link"]
        self.j_eef = jacobian[:, self.ur3_hand_index - 1, :]
        self.j_eef = self.j_eef[:, :, :6]  # up to UR3 joints

    def reset(self):
        if hasattr(self, "cube_handle"):
            self.gym.set_sim_rigid_body_states(self.sim, self.cube_initial_state, gymapi.STATE_ALL)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

        if hasattr(self, "ur3_handle"):
            print("ur3 reset!")
            id = 0
            pos = tensor_clamp(
                self.ur3_default_dof_pos.unsqueeze(0) + 0.1 * (torch.rand((1, self.num_dofs), device=self.device) - 0.5),
                self.ur3_dof_lower_limits, self.ur3_dof_upper_limits)
            self.ur3_dof_targets = pos      # targets
            self.ur3_dof_pos[id, :] = pos
            self.ur3_dof_pos[id, 8] = 0.0
            self.ur3_dof_vel[id, :] = torch.zeros_like(self.ur3_dof_vel)

            # for gripper sync.
            self.ur3_dof_pos[id, 6] = 1 * self.ur3_dof_pos[id, 8]
            self.ur3_dof_pos[id, 7] = -1. * self.ur3_dof_pos[id, 8]
            self.ur3_dof_pos[id, 9] = 1 * self.ur3_dof_pos[id, 8]
            self.ur3_dof_pos[id, 10] = -1. * self.ur3_dof_pos[id, 8]
            self.ur3_dof_pos[id, 11] = 1 * self.ur3_dof_pos[id, 8]

            robot_indices32 = self.global_indices[:1].flatten()
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.ur3_dof_targets),
                                                            gymtorch.unwrap_tensor(robot_indices32),
                                                            len(robot_indices32))
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(robot_indices32), len(robot_indices32))

    def event_handler(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):

            if evt.action == "exit" and evt.value > 0:
                self.loop_on = False

            if evt.action == "reset" and evt.value > 0:
                self.reset()

                # current viewer camera transformation
                tr = self.gym.get_viewer_camera_transform(self.viewer, self.env)
                print("p: ", tr.p)
                print("r: ", tr.r)

    def vr_handler_for_cube(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        state = self.gym.get_actor_rigid_body_states(self.env, self.cube_handle, gymapi.STATE_NONE)
        d = self.vr.devices["controller_1"].get_controller_inputs()
        if d['trigger']:
            pv = np.array([v for v in self.vr.devices["controller_1"].get_velocity()]) * 10.0
            av = np.array([v for v in self.vr.devices["controller_1"].get_angular_velocity()]) * 1.0

            state['vel']['linear'].fill((pv[0], pv[1], pv[2]))
            state['vel']['angular'].fill((av[0], av[1], av[2]))
            print("trigger is pushed, ", pv)

        self.gym.set_actor_rigid_body_states(self.env, self.cube_handle, state, gymapi.STATE_ALL)
        self.draw_coord(pos=np.asarray(state['pose']['p'][0].tolist()),
                        rot=np.asarray(state['pose']['r'][0].tolist()))

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
            # orn_err = orientation_error(quat_unit(goal_rot), self.ur3_grasp_rot)  # unit_quat

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

        _u = torch.cat((u, angle_err.unsqueeze(-1)), dim=1)
        return _u.squeeze(-1)

    def compute_obs(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

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

    def vr_handler_for_ur3(self):
        self.compute_obs()
        goal = torch.zeros(1, 8, device=self.device)
        goal[:, -2:] = 1.0

        state = self.gym.get_actor_rigid_body_states(self.env, self.ur3_handle, gymapi.STATE_NONE)
        d = self.vr.devices["controller_1"].get_controller_inputs()
        if d['trigger']:
            pv = np.array([v for v in self.vr.devices["controller_1"].get_velocity()]) * 1.0
            av = np.array([v for v in self.vr.devices["controller_1"].get_angular_velocity()]) * 1.0    # incremental
            # av = np.array([v for v in vr.devices["controller_1"].get_pose_quaternion()]) * 1.0        # absolute

            pv = torch.matmul(self.rot, torch.tensor(pv).unsqueeze(0).T)
            av = torch.tensor(av).unsqueeze(0)
            _q = mat_to_quat(self.rot.unsqueeze(0))
            av = quat_apply(_q, av)

            goal[:, :3] = pv.T
            _quat = quat_from_euler_xyz(roll=av[0, 0], pitch=av[0, 1], yaw=av[0, 2])
            goal[:, 3:7] = _quat
            # print("trigger is pushed, ", pv, av)

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
        dt = 1 / 30
        action_scale = 10.0
        actions = torch.zeros_like(self.ur3_dof_pos)
        actions[:, :6] = _actions[:, :6]
        drv = _actions[:, -1]
        actions[:, 6] = actions[:, 8] = actions[:, 9] = actions[:, 11] = drv
        actions[:, 7] = actions[:, 11] = -1 * drv

        targets = self.ur3_dof_pos + dt * actions * action_scale
        self.ur3_dof_targets = tensor_clamp(targets, self.ur3_dof_lower_limits, self.ur3_dof_upper_limits)
        # print("dof targets: ", self.ur3_dof_targets)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ur3_dof_targets))

        # self.gym.set_actor_rigid_body_states(self.env, self.ur3_handle, state, gymapi.STATE_ALL)

        self.draw_coord(pos=[np.asarray(state['pose']['p'][0].tolist()), self.ur3_grasp_pos[0].numpy()],
                        rot=[np.asarray(state['pose']['r'][0].tolist()), self.ur3_grasp_rot[0].numpy()])

    def draw_coord(self, pos, rot, scale=0.2):     # args type: numpy arrays
        self.gym.clear_lines(self.viewer)
        for p, r in zip(pos, rot):
            pos = torch.tensor(p, device=self.device, dtype=torch.float32)
            rot = torch.tensor(r, device=self.device, dtype=torch.float32)
            px = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device, dtype=torch.float32) * scale)).cpu().numpy()
            py = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device, dtype=torch.float32) * scale)).cpu().numpy()
            pz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device, dtype=torch.float32) * scale)).cpu().numpy()

            p0 = pos.cpu().numpy()
            self.gym.add_lines(self.viewer, self.env, 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
            self.gym.add_lines(self.viewer, self.env, 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
            self.gym.add_lines(self.viewer, self.env, 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    def run(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            if not self.loop_on: break
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # processing
            self.event_handler()
            if hasattr(self, "cube_handle"): self.vr_handler_for_cube()
            if hasattr(self, "ur3_handle"): self.vr_handler_for_ur3()

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def vr_manipulation_test():
    sv = SimVR()
    sv.run()


def mat3d(roll, pitch, yaw):    # radian input
    rx = torch.tensor([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = torch.tensor([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = torch.tensor([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return torch.matmul(rx, torch.matmul(ry, rz))


def rotation_matrix_test():
    v = torch.tensor([[0.0, 0.0, 1.0]]).T
    rot = mat3d(roll=deg2rad(0), pitch=deg2rad(179.5), yaw=deg2rad(0))
    result = torch.matmul(rot.float(), v)
    print("input: {}".format(v))
    print("rot \n", rot)
    print("result: {}".format(result))

    q = mat_to_quat(rot.unsqueeze(0))
    print("mat to quat: ", q)


if __name__ == "__main__":
    # controller_test()
    vr_manipulation_test()
    # rotation_matrix_test()
