import time
from vr_teleop.tasks.lib_modules import *
from vr_teleop.tasks.base import VRWrapper
from vr_teleop.tasks.rollout_manager import RolloutManager, RolloutManagerExpand, RobotState, RobotState2


def vr_test():
    # vr = VRWrapper(device="cpu", rot_d=(-89.9, 0.0, 89.9))
    vr = VRWrapper(device="cpu", rot_d=(0.0, 0.0, 0.0))

    def gripper_test():
        HOST = "192.168.0.75"
        rtde_c = rtde_control.RTDEControlInterface(HOST)
        gripper = RobotiqGripperExpand(rtde_c, HOST)
        return gripper
    gripper = gripper_test()
    grip = False

    start = time.time()
    while True:
        cont_status = vr.get_controller_status()
        if cont_status["btn_trigger"]:
            curr_time = time.time()
            if (curr_time - start) > 1:
                print("btn_trigger on!")
                pq = cont_status["pose_quat"]
                print("pq: ", pq)
                start = curr_time

        if cont_status["btn_reset_timeout"]:
            print("btn_reset timeout")
        if cont_status["btn_reset_pose"]:
            print("btn_reset_mode")
        if cont_status["btn_grip"]:
            print("btn_grip is pressed..")
            grip = not grip
        gripper_action = -1.0 if grip else 1.0
        gripper.grasping_by_hold(step=gripper_action)

        # if cont_status["btn_gripper"]:
        #     print("btn_gripper")


def camera_test():
    rollout = RolloutManagerExpand(task_name="pouring_skill_img")
    cam = RealSenseMulti()
    idx1 = cam.index_from_key('rear')
    idx2 = cam.index_from_key('front')
    cam.display_info(idx1)

    class Check_dt:
        def __init__(self):
            self.start = time.time()

        def print_dt(self):
            current = time.time()
            print("dt: {}, Hz: {}".format(current - self.start, 1.0 / (current - self.start)))
            self.start = current

    try:
        dt = Check_dt()
        while True:
        # for i in range(10):
            depth, color = cam.get_np_images(idx1)
            depth2, color2 = cam.get_np_images(idx2)
            st = RobotState2()
            st.gen_random_data(n_joint=6, n_cont_mode=2)
            # rollout.append(observation=color, state=st,
            #                action=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], done=[0], info=[""])
            if cam.visualize([depth, depth2], [color, color2]) == 27:
                break
            # dt.print_dt()
        # rollout.save_to_file()
    finally:
        cam.stop_stream_all()


def camera_load_test(batch_idx, rollout_idx):
    cam = RealSense()
    rollout = RolloutManagerExpand(task_name="pouring_skill_img")
    rollout.load_from_file(batch_idx=batch_idx, rollout_idx=rollout_idx)
    for i in range(rollout.len()):
        obs, state, action, done, info = rollout.get(i)
        print("obs ", obs.shape, obs.dtype)
        _depth = np.zeros(obs.shape[:2])
        visualize(_depth, obs)
        print(i, cv2.waitKey(0))


def gripper_test():
    pass


if __name__ == "__main__":
    # vr_test()
    gripper_test()
    # camera_test()
    # camera_load_test(batch_idx=1, rollout_idx=0)
