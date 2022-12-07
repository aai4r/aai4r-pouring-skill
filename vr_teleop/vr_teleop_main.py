from math import sqrt

from objects.base import *
from objects.cube import Cube
from objects.ur3_robotiq85 import IsaacUR3
from spirl.utils.general_utils import AttrDict
from utils.torch_jit_utils import *

import triad_openvr

import socket
from threading import Event, Thread


class SimVR:
    def __init__(self, cfg):
        # initialize VR device
        self.cfg = cfg
        self.vr = triad_openvr.triad_openvr() if self.cfg.vr_on else None
        if self.cfg.vr_on:
            self.vr.print_discovered_objects()
        self.rot = euler_to_mat3d(roll=deg2rad(0.0), pitch=deg2rad(90.0), yaw=deg2rad(0.0))

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

        _isaac = IsaacElement(gym=self.gym, viewer=self.viewer, sim=self.sim, env=self.env, num_envs=self.num_envs,
                              device=self.device, asset_root=self.asset_root)
        _vr = VRElement(vr=self.vr, rot=self.rot)
        self.obj = self.cfg.target_obj(isaac_elem=_isaac, vr_elem=_vr)

        if self.cfg.socket_open:
            self.threads = []
            self.events = []
            self.open_socket_server()

    def init_env(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)

        # subscribe to input events. This allows input to be used to interact
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "exit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

        # viewer camera setting
        # cam_pos = gymapi.Vec3(3.58, 1.58, 0.0)  # third person view
        # cam_pos = gymapi.Vec3(-0.202553, 0.890771, -0.211403)  # tele.op. view
        cam_pos = gymapi.Vec3(0.223259, 0.694804, 0.573643)
        cam_target = gymapi.Vec3(-0.2, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # set up the grid of environments
        self.num_envs = self.args.num_envs
        num_per_row = int(sqrt(self.num_envs))
        spacing = 2.0

        # only consider one environment
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, lower, upper, num_per_row)

    def open_socket_server(self):
        HOST = "127.0.0.1"
        PORT = 9999

        # create socket and listen
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen()

        print("socket server start!")
        event = Event()
        t = Thread(target=self._threaded, args=(event, ))
        self.events.append(event)
        self.threads.append(t)
        t.start()
        # t.join()

    def _threaded(self, event):
        print("Socket server starts!")
        client_socket, addr = self.server_socket.accept()
        print("Connected by {}:{} ".format(addr[0], addr[1]))
        start = time.time()
        while True:
            if event.is_set():
                print("Kill Thread...")
                self.server_socket.close()
                break

            try:
                data = client_socket.recv(1024)

                if (time.time() - start) > 1:
                    if not data: raise ConnectionError
                    print("Received: ", data.decode())
                    start = time.time()
                s_data = self.obj.get_status()
                client_socket.send(s_data.encode())
            except ConnectionResetError as e:
                print("Disconnected from {}:{}".format(addr[0], addr[1]))
                break
            except ConnectionError as e:
                try:
                    client_socket, addr = self.server_socket.accept()
                except:
                    break

    def event_handler(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):

            if evt.action == "exit" and evt.value > 0:
                self.loop_on = False

            if evt.action == "reset" and evt.value > 0:
                self.obj.reset()

                # current viewer camera transformation
                tr = self.gym.get_viewer_camera_transform(self.viewer, self.env)
                print("p: ", tr.p)
                print("r: ", tr.r)

    def run(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            if not self.loop_on: break
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # processing
            self.event_handler()
            self.obj.move()

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        self.kill_all_threads()

    def kill_all_threads(self):
        self.server_socket.shutdown(socket.SHUT_RDWR)
        [e.set() for e in self.events]


if __name__ == "__main__":
    target_list = [Cube, IsaacUR3]
    cfg = AttrDict(vr_on=True, socket_open=True, target_obj=IsaacUR3)
    sv = SimVR(cfg=cfg)
    sv.run()

