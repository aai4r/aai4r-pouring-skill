import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt
import torch
from utils.torch_jit_utils import *

import triad_openvr
import time
import sys

vr = triad_openvr.triad_openvr()
vr.print_discovered_objects()


def controller_test():
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
        # initialize the isaac gym simulation
        self.gym = gymapi.acquire_gym()

        # parse arguments
        self.args = gymutil.parse_arguments(
            description="Projectiles Example: Press SPACE to fire a projectile. Press R to reset the simulation.",
            custom_parameters=[
                {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"}])

        dev_num = 1 if torch.cuda.device_count() else 0
        self.args.sim_device = "cuda:{}".format(dev_num)
        self.args.compute_device_id = dev_num
        self.args.graphics_device_id = dev_num

        # configure sim
        sim_params = gymapi.SimParams()
        if self.args.physics_engine == gymapi.SIM_FLEX:
            sim_params.flex.shape_collision_margin = 0.05
            sim_params.flex.num_inner_iterations = 6
        elif self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu

        sim_params.use_gpu_pipeline = False
        if self.args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id,
                                       self.args.physics_engine, sim_params)

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.loop_on = True

        self.init_env()
        self.create_cube()

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
        num_envs = self.args.num_envs
        num_per_row = int(sqrt(num_envs))
        spacing = 2.0

        # only consider one environment
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, lower, upper, num_per_row)

    def create_cube(self):
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
        c = 0.5 + 0.5 * np.random.random(3)
        self.gym.set_rigid_body_color(self.env, self.cube_handle, 0,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))

        # save initial state for reset
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.cube_initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

    def event_handler(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):

            if evt.action == "exit" and evt.value > 0:
                self.loop_on = False

            if evt.action == "reset" and evt.value > 0:
                self.gym.set_sim_rigid_body_states(self.sim, self.cube_initial_state, gymapi.STATE_ALL)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                state = self.gym.get_actor_rigid_body_states(self.env, self.cube_handle, gymapi.STATE_NONE)
                print("current cube state: ", state['pose']['p'][0])
                print("init cube state", self.cube_initial_state['pose']['p'][0])

                # current viewer camera transformation
                tr = self.gym.get_viewer_camera_transform(self.viewer, self.env)
                print("p: ", tr.p)
                print("r: ", tr.r)

    def vr_handler(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        state = self.gym.get_actor_rigid_body_states(self.env, self.cube_handle, gymapi.STATE_NONE)
        d = vr.devices["controller_1"].get_controller_inputs()
        if d['trigger']:
            pv = np.array([v for v in vr.devices["controller_1"].get_velocity()]) * 10.0
            av = np.array([v for v in vr.devices["controller_1"].get_angular_velocity()]) * 1.0

            _p = state['pose']['p'][0]
            state['vel']['linear'].fill((pv[0], pv[1], pv[2]))
            state['vel']['angular'].fill((av[0], av[1], av[2]))
            print("trigger is pushed, ", pv)

        self.gym.set_actor_rigid_body_states(self.env, self.cube_handle, state, gymapi.STATE_ALL)
        self.draw_coord(pos=np.asarray(state['pose']['p'][0].tolist()),
                        rot=np.asarray(state['pose']['r'][0].tolist()))

    def draw_coord(self, pos, rot):     # as numpy arrays
        self.gym.clear_lines(self.viewer)
        pos = torch.tensor(pos, device=self.args.sim_device, dtype=torch.float32)
        rot = torch.tensor(rot, device=self.args.sim_device, dtype=torch.float32)
        px = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.args.sim_device, dtype=torch.float32))).cpu().numpy()
        py = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.args.sim_device, dtype=torch.float32))).cpu().numpy()
        pz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.args.sim_device, dtype=torch.float32))).cpu().numpy()

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
            self.vr_handler()

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


if __name__ == "__main__":
    # controller_test()
    vr_manipulation_test()
