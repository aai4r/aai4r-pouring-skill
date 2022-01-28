import torch
from utils import TaskPathManager


def function_test():
    device = "cuda:0"
    num_env = 10
    num_task_steps = 5
    task = TaskPathManager(num_env=num_env, num_task_steps=num_task_steps, device=device)
    task.print_task_status()

    pos = torch.ones(num_env, 3, device=device)
    rot = torch.ones(num_env, 4, device=device)
    grip = torch.ones(num_env, 1, device=device)
    for i in range(5):
        task.push_task_pose(env_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pos=pos, rot=rot, grip=grip)
        pos += 1
        rot += 1
        grip += 1
    task.print_task_status()
    pos, rot, grip = task.get_desired_pose()
    print("get pos: \n{}, rot: \n{}, grip: \n{}".format(pos, rot, grip))
    print("shapes, pos: {}, rot: {}, grip: {}".format(pos.shape, rot.shape, grip.shape))

    task.update_step_by_checking_arrive(torch.rand_like(pos), torch.rand_like(rot), torch.rand_like(grip))


if __name__ == '__main__':
    function_test()
