import random

from vr_teleop.tasks.lib_modules import *
from vr_teleop.tasks.base import VRWrapper
from vr_teleop.tasks.rollout_manager import RolloutManager, RolloutManagerExpand, RobotState, RobotState2


class RealUR3(BaseRTDE, UR3ControlMode):
    def __init__(self, task_name):
        self.init_vr()
        BaseRTDE.__init__(self, HOST="192.168.0.75")
        UR3ControlMode.__init__(self, init_mode="forward")

        self.cam = RealSense()
        self.timer = CustomTimer(duration_sec=1.0)
        self.rand_control_mode = True

        self.rollout = RolloutManagerExpand(task_name=task_name)
        self.collect_demo = True

    def init_vr(self):
        self.vr = VRWrapper(device="cpu", rot_d=(-90.0, 0.0, -90.0))

    def random_change_control_mode(self, move_j=False):
        idx = self.cmodes.index(self.CONTROL_MODE)
        to = random.randrange(0, len(self.cmodes))
        self.CONTROL_MODE = self.cmodes[to]
        print("CONTROL_MODE: {} --> {}".format(self.cmodes[idx], self.CONTROL_MODE))
        if move_j:
            self.speed_stop()
            self.move_j(self.iposes)

    def shift_control_mode(self, move_j=False):
        super().shift_control_mode()
        if move_j:
            self.speed_stop()
            self.move_j(self.iposes)

    def control_mode_to(self, cont_to, move_j):
        if cont_to not in self.cmodes:
            raise IndexError("cont_to should be one of the {}".format(self.cmodes))
        idx = self.cmodes_d[cont_to]
        self.CONTROL_MODE = self.cmodes[idx]
        if move_j:
            self.speed_stop()
            self.move_j(self.iposes)

    def get_state(self):
        tcp_pos, tcp_aa = self.get_actual_tcp_pos_ori()
        target_diff = np.array([0.5196, -0.1044, 0.088]) - np.array(tcp_pos)
        state = RobotState2(joint=self.get_actual_q(),
                            ee_pos=tcp_pos,
                            ee_quat=self.quat_from_tcp_axis_angle(tcp_aa),
                            target_diff=target_diff.tolist(),
                            # gripper_one_hot=self.grip_one_hot_state(),    # for RobotState
                            gripper_pos=[self.gripper.gripper_to_mm_normalize()],
                            control_mode_one_hot=self.cont_mode_one_hot_state())
        return state

    def record_frame(self, observation, state, action_pos, action_quat, action_grip, action_mode, done, extra=None):
        """
        :param observation
        :param state:
        :param action_pos:      Relative positional diff. (vel), list type
        :param action_quat:     Rotation as a quaternion (real-last), list type
        :param action_grip:     Gripper ON/OFF one-hot vector, list type
        :param done:            Scalar value of 0 or 1
        :return:
        """
        info = str({"gripper": self.grip_on, "control_mode": self.CONTROL_MODE})
        action = action_pos + action_quat + action_grip + action_mode
        self.rollout.append(image=observation, state=state, action=action, done=done, info=info, extra=extra)

    def play_demo(self):
        # go to initial state in joint space
        self.speed_stop()
        init_obs, init_state, _, _, _ = self.rollout.get(0)
        if hasattr(self, 'cam') and init_obs is not None:
            visualize(np.zeros(init_obs.shape), init_obs)
        self.move_j(init_state.joint)

        # loop for playing demo
        for idx in range(1, self.rollout.len()):
            start_t = self.init_period()
            obs, state, action, done, info = self.rollout.get(index=idx)
            if hasattr(self, 'cam') and obs is not None:
                visualize(np.zeros(obs.shape), obs)

            act_pos, act_quat, grip = action[:3], action[3:7], action[7:8]
            if len(action) >= 9:
                cmode = action[8:9]
                if cmode[0] != 0.0:
                    self.control_mode_to(cont_to="forward" if cmode[0] > 0 else "downward", move_j=True)
            else:
                if self.cont_mode_to_str(state.control_mode_one_hot) != self.CONTROL_MODE:
                    self.shift_control_mode(move_j=True)

            if len(grip) == 2:  # gripper one-hot state
                self.move_grip_on_off(self.grip_onehot_to_bool(grip))
            elif len(grip) == 1:    # cont. control
                # self.gripper.rq_move_mm_norm(grip[0])
                self.gripper.grasping_by_hold(grip[0])
            else:
                raise NotImplementedError

            actual_tcp_pos, actual_tcp_ori = self.get_actual_tcp_pos_ori()
            des_pos = np.array(actual_tcp_pos) + np.array(act_pos)
            des_rot = self.goal_axis_angle_from_act_quat(act_quat=act_quat, actual_tcp_aa=actual_tcp_ori)

            goal_pose = self.goal_pose(des_pos=des_pos, des_rot=des_rot)
            goal_j = self.get_inverse_kinematics(tcp_pose=goal_pose)
            diff_j = (np.array(goal_j) - np.array(self.get_actual_q())) * 1.0
            self.speed_j(list(diff_j), self.acc, self.dt)
            self.wait_period(start_t)
        self.speed_stop()

    def robot_reset(self):
        self.speed_stop()
        if self.rand_control_mode:
            self.random_change_control_mode()
            print("cont mode one hot: ", self.get_state().control_mode_one_hot)
        _pose = self.add_noise_angle(inputs=self.iposes)
        self.move_j(_pose.tolist())
        self.gripper.rq_move_mm_norm(1.)
        self.grip_on = False

    def run_vr_teleop(self):
        print("Run VR teleoperation mode")
        self.robot_reset()
        try:
            for _ in range(10):
                depth, color = self.cam.get_np_images()
                visualize(depth, color)

            self.move_j(self.iposes)
            while True:
                start_t = self.init_period()

                # get velocity command from VR
                cont_status = self.vr.get_controller_status()
                if cont_status["btn_reset_timeout"]:
                    print("discard current demo...")
                    # self.rollout.save_to_file()
                    self.rollout.reset()
                    self.robot_reset()
                    continue

                if cont_status["btn_reset_pose"]:
                    depth, color = self.cam.get_np_images()
                    self.record_frame(observation=copy.deepcopy(color),
                                      state=self.get_state(),
                                      action_pos=[0., 0., 0.],
                                      action_quat=[0., 0., 0., 1.],
                                      action_grip=[1.0],
                                      action_mode=[0.0],
                                      done=1,
                                      extra=[0., 0., 0., 1.])   # source rotation of VR controller...
                    # self.play_demo()
                    self.rollout.show_rollout_summary()
                    self.rollout.save_to_file()
                    self.rollout.reset()

                    self.robot_reset()
                    continue

                diff_j = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                action_mode = [0.0]
                if cont_status["btn_trigger"]:
                    depth, color = self.cam.get_np_images()
                    visualize(depth, color)
                    state = self.get_state()

                    if cont_status["trk_up"]:
                        # self.shift_control_mode(move_j=True)
                        self.control_mode_to(cont_to="forward", move_j=True)
                        action_mode[0] = 1.0
                    elif cont_status["trk_down"]:
                        self.control_mode_to(cont_to="downward", move_j=True)
                        action_mode[0] = -1.0

                    if cont_status["btn_grip"]:
                        self.grip_toggle()

                    gripper_action = -1.0 if self.grip_on else 1.0
                    self.gripper.grasping_by_hold(step=gripper_action)
                    # gripper_action_norm = self.gripper.get_gripper_action(normalize=True)

                    vr_curr_pos_vel, vr_curr_quat = cont_status["lin_vel"], cont_status["pose_quat"]
                    act_pos = list(vr_curr_pos_vel)

                    actual_tcp_pos, actual_tcp_ori_aa = self.get_actual_tcp_pos_ori()

                    actual_tcp_ori_quat = tr.axis_angle_to_quaternion(torch.tensor(actual_tcp_ori_aa))
                    actual_tcp_ori_quat = quaternion_real_last(actual_tcp_ori_quat)
                    conj = quat_conjugate(actual_tcp_ori_quat)
                    act_quat = quat_mul(torch.tensor(vr_curr_quat), conj)

                    if self.collect_demo: self.record_frame(observation=copy.deepcopy(color),
                                                            state=state,
                                                            action_pos=act_pos,
                                                            action_quat=act_quat.tolist(),
                                                            action_grip=[gripper_action],
                                                            action_mode=action_mode,
                                                            done=0,
                                                            extra=vr_curr_quat.tolist())

                    d_pos = np.array(actual_tcp_pos) + np.array(act_pos)
                    d_rot = self.goal_axis_angle_from_act_quat(act_quat=act_quat, actual_tcp_aa=actual_tcp_ori_aa)
                    goal_pose = self.goal_pose(des_pos=d_pos, des_rot=d_rot)    # limit handling

                    goal_j = self.get_inverse_kinematics(tcp_pose=goal_pose)
                    diff_j = (np.array(goal_j) - np.array(self.get_actual_q())) * 1.0

                self.speed_j(list(diff_j), self.acc, self.dt)
                self.wait_period(start_t)
        except ValueError:
            print("Value Error... ")
        finally:
            print("end of control... ")
            self.cam.stop_stream()
            if not hasattr(self, "rtde_c"): return
            self.speed_stop()
            print("speed stop.. ")
            self.stop_script()
            print("script stop.. ")

    def replay_mode(self, batch_idx, rollout_idx):
        self.move_j(self.iposes)
        print("[Reset] current CONT MODE: ", self.CONTROL_MODE, self.rpy_base)
        while True:
            cont_status = self.vr.get_controller_status()
            if cont_status["btn_reset_pose"]:
                print("reset & replay")
                self.rollout.load_from_file(batch_idx=batch_idx, rollout_idx=rollout_idx)
                self.rollout.show_rollout_summary()
                self.play_demo()


if __name__ == "__main__":
    tasks = ["pouring_skill_img", "pick_and_place_img"]
    # tasks2 = ["pouring_constraint", "pick_and_place_constraint"]
    u = RealUR3(task_name="pouring_skill_img")
    u.run_vr_teleop()
    # u.replay_mode(batch_idx=1, rollout_idx=3)
