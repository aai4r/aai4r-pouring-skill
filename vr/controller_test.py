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


def vr_manipulation_test():
    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Projectiles Example: Press SPACE to fire a projectile. Press R to reset the simulation.",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"}])

    dev_num = 1 if torch.cuda.device_count() else 0
    args.sim_device = "cuda:{}".format(dev_num)
    args.compute_device_id = dev_num
    args.graphics_device_id = dev_num

    # configure sim
    sim_params = gymapi.SimParams()
    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.shape_collision_margin = 0.05
        sim_params.flex.num_inner_iterations = 6
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # create viewer. Not optional in this example
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # subscribe to input events. This allows input to be used to interact
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "exit")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

    # set up the grid of environments
    num_envs = args.num_envs
    num_per_row = int(sqrt(num_envs))
    spacing = 2.0

    # only consider one environment
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower, upper, num_per_row)

    cube_asset_options = gymapi.AssetOptions()
    cube_asset_options.density = 100.
    cube_asset_options.disable_gravity = True
    cube_asset_options.linear_damping = 20      # damping is important for stabilizing the movement
    cube_asset_options.angular_damping = 20

    cube_asset = gym.create_box(sim, 0.5, 0.5, 0.5, cube_asset_options)

    init_pose = gymapi.Transform()
    init_pose.p = gymapi.Vec3(0.0, 0.0, 1.5)
    init_pose.r = gymapi.Quat(0, 0, 0, 1)
    cube_handle = gym.create_actor(env, cube_asset, init_pose, "cube", -1, 0)
    c = 0.5 + 0.5 * np.random.random(3)
    gym.set_rigid_body_color(env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))

    # save initial state for reset
    gym.refresh_rigid_body_state_tensor(sim)
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    # viewer camera setting
    # cam_pos = gymapi.Vec3(0.0, 1.58, 3.58)
    cam_pos = gymapi.Vec3(5.616872, 1.241800, 1.494509)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # get state of cube
        state = gym.get_actor_rigid_body_states(env, cube_handle, gymapi.STATE_NONE)

        gym.refresh_rigid_body_state_tensor(sim)
        d = vr.devices["controller_1"].get_controller_inputs()
        if d['trigger']:
            pv = np.array([v for v in vr.devices["controller_1"].get_velocity()]) * 10.0
            av = np.array([v for v in vr.devices["controller_1"].get_angular_velocity()]) * 1.0

            _p = state['pose']['p'][0]
            state['vel']['linear'].fill((pv[0], pv[1], pv[2]))
            state['vel']['angular'].fill((av[0], av[1], av[2]))
            print("trigger is pushed, ", pv)

        gym.set_actor_rigid_body_states(env, cube_handle, state, gymapi.STATE_ALL)

        # draw orientation line
        gym.clear_lines(viewer)
        pos = torch.tensor(np.asarray(state['pose']['p'][0].tolist()), device=args.sim_device, dtype=torch.float32)
        rot = torch.tensor(np.asarray(state['pose']['r'][0].tolist()), device=args.sim_device, dtype=torch.float32)
        px = (pos + quat_apply(rot, to_torch([1, 0, 0], device=args.sim_device, dtype=torch.float32))).cpu().numpy()
        py = (pos + quat_apply(rot, to_torch([0, 1, 0], device=args.sim_device, dtype=torch.float32))).cpu().numpy()
        pz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=args.sim_device, dtype=torch.float32))).cpu().numpy()

        p0 = pos.cpu().numpy()
        gym.add_lines(viewer, env, 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
        gym.add_lines(viewer, env, 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
        gym.add_lines(viewer, env, 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

        # Get input actions from the viewer and handle them appropriately
        for evt in gym.query_viewer_action_events(viewer):

            if evt.action == "exit" and evt.value > 0:
                quit()

            if evt.action == "reset" and evt.value > 0:
                gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

                # current viewer camera transformation
                tr = gym.get_viewer_camera_transform(viewer, env)
                print("p: ", tr.p)
                print("r: ", tr.r)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    # controller_test()
    vr_manipulation_test()
