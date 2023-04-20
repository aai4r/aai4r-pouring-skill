"""
WARNING::
This code is incomplete!
Should be fixed later.....
"""

from isaacgym import gymapi, gymtorch

from tasks.base.base_task import BaseTask
from vr_teleop import triad_openvr
from utils.utilities import *
import cv2


class VRInteraction(BaseTask):
    def __init__(self, cfg):
        super(VRInteraction, self).__init__(cfg)

        self.interaction_mode = cfg["env"]["interaction_mode"]
        self.pause = False
        self.btn_pause_que = []

        # VR Teleoperation
        self.teleoperation_mode = cfg["env"]["teleoperation_mode"]
        self.trk_btn_trans = []
        self.trk_btn_toggle = 1

        if self.interaction_mode:
            self.gym.subscribe_viewer_mouse_event(self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_left")

        if self.interaction_mode or self.teleoperation_mode:
            """ VR interface setting """
            self.vr = triad_openvr.triad_openvr()
            self.vr.print_discovered_objects()
            self.vr_ref_rot = euler_to_mat3d(roll=deg2rad(-90.0), pitch=deg2rad(0.0), yaw=deg2rad(179.9))

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M, "mouse_tr")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "set_ood")

    # pause
    def get_img_with_text(self, text=''):
        img = np.zeros((128, 512, 3), np.uint8)

        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (256 - 64, 64 - 8)
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

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions=actions)

        if self.interaction_mode:
            if self.pause:
                img = self.get_img_with_text('Pause')
                cv2.imshow('pause', img)
                cv2.waitKey(1)
                self.actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device)
                self.ur3_dof_vel = torch.zeros_like(self.ur3_dof_vel)
                self.progress_buf -= 1
            else:
                img = self.get_img_with_text('Resume')
                cv2.imshow('pause', img)
                cv2.waitKey(1)

        # TODO, between self.actions and targets
        if self.teleoperation_mode:
            self.teleoperation()

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

            # vr_teleop controller
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
                # av = np.array([v for v in vr_teleop.devices["controller_1"].get_pose_quaternion()]) * 1.0        # absolute

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
