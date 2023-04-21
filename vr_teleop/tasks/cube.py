from .base import *


class Cube(BaseObject):
    def __init__(self, isaac_elem, vr_elem):
        self.vr = vr_elem
        super().__init__(isaac_elem)

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13).to(self.device)
        self.cube_pos = self.rigid_body_states[:, self.cube_handle][:, 0:3]
        self.cube_rot = self.rigid_body_states[:, self.cube_handle][:, 3:7]

        self.count = 0
        self.vr_q = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # initialize to proper q

    def _create(self):
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 100.
        cube_asset_options.disable_gravity = True
        cube_asset_options.linear_damping = 20  # damping is important for stabilizing the movement
        cube_asset_options.angular_damping = 20

        cube_asset = self.gym.create_box(self.sim, 0.5, 0.5, 0.5, cube_asset_options)

        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(0.0, 0.0, 0.6)
        init_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cube_handle = self.gym.create_actor(self.env, cube_asset, init_pose, "cube", -1, 0)

        c = 0.5 + 0.5 * np.random.random(3)
        self.gym.set_rigid_body_color(self.env, self.cube_handle, 0,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))

        # save initial state for reset
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.cube_initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

        # set viewer perspective setting
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(-3.58, 0.0, 1.0), gymapi.Vec3(0.0, 0.0, 0.0))

    def reset(self):
        self.gym.set_sim_rigid_body_states(self.sim, self.cube_initial_state, gymapi.STATE_ALL)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def get_count(self, target):
        self.count += 1
        if self.count % target == 0:
            self.count = 0
            return True
        return False

    def get_vr_status(self):
        if self.vr is None: return self.vr
        cont_status = self.vr.get_controller_status()
        return cont_status

    def vr_handler(self, state):
        cont_status = self.vr.get_controller_status()
        if cont_status["btn_trigger"]:
            lv = cont_status["lin_vel"] * 10.0
            av = cont_status["ang_vel"] * 1.0
            state['vel']['linear'].fill((lv[0], lv[1], lv[2]))
            state['vel']['angular'].fill((av[0], av[1], av[2]))     # relative

            # orientation (absolute)
            self.cube_pos = self.rigid_body_states[:, self.cube_handle][:, 0:3]
            self.cube_rot = self.rigid_body_states[:, self.cube_handle][:, 3:7]
            self.vr_q = torch.tensor(cont_status["pose_quat"], device=self.device)
        curr = self.cube_rot

        s = 10.0
        dq = orientation_error(desired=self.vr_q.unsqueeze(0), current=curr).squeeze(0) * s
        if self.get_count(50):
            print(" -------------------- ")
            print("des_q: ", self.vr_q)
            print("curr_q: ", curr)
            print("dq: ", dq)
        # self.cube_rot = quat_mul(curr, dq)
        state['vel']['angular'].fill((dq[0], dq[1], dq[2]))

    def move(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        state = self.gym.get_actor_rigid_body_states(self.env, self.cube_handle, gymapi.STATE_NONE)
        if self.vr is not None:
            self.vr_handler(state=state)

        self.gym.set_actor_rigid_body_states(self.env, self.cube_handle, state, gymapi.STATE_ALL)
        # print("asdasdasd")
        # print(np.asarray(state['pose']['p'][0].tolist()))
        # print(np.asarray(state['pose']['r'][0].tolist()))
        self.draw_coord(pos=[np.asarray(state['pose']['p'][0].tolist()), np.array([0.0, 0.0, 0.0])],
                        rot=[np.asarray(state['pose']['r'][0].tolist()), np.array([0.0, 0.0, 0.0, 1.0])], scale=1.0)