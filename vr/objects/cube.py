from .base import *


class Cube(BaseObject):
    def __init__(self, isaac_elem, vr_elem):
        self.vr = vr_elem.vr
        self.rot = vr_elem.rot
        super().__init__(isaac_elem)

    def _create(self):
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

        # set viewer perspective setting
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(3.58, 1.58, 0.0), gymapi.Vec3(0.0, 0.0, 0.0))

    def reset(self):
        self.gym.set_sim_rigid_body_states(self.sim, self.cube_initial_state, gymapi.STATE_ALL)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def vr_handler(self):
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
        self.draw_coord(pos=[np.asarray(state['pose']['p'][0].tolist())],
                        rot=[np.asarray(state['pose']['r'][0].tolist())])