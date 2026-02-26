from bruce_gym.envs import *
from bruce_gym.utils import get_args, task_registry

import torch

from bruce_gym.scripts.gazebo.simple_env import SimpleTask

speed_x = 0.1
speed_y = 0.0
speed_w = 0.0


def log_step(load_run,step, ckpt):
    log_file_path = "./log_file_"+load_run+".log"
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{ckpt}-{step}\n")
        
def play(args):
    global speed_x, speed_y, speed_w
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = 'plane'

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 1.0
    env_cfg.init_state.reset_ratio = 0.8

    env_gazebo = SimpleTask(env_cfg)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env_gazebo, name=args.task, args=args, train_cfg=train_cfg)

    policy = ppo_runner.get_inference_policy(device=env_gazebo.device)

    step = 0
    obs_gazebo, _ = env_gazebo.get_observations(torch.zeros(10, dtype=torch.float, device=env_gazebo.device, requires_grad=False), [speed_x, speed_y, speed_w])


    while True:
        actions_gazebo = policy(obs_gazebo.detach())
        delay = 0.1
        obs_gazebo, slip = env_gazebo.step(actions_gazebo[:,:10].detach(), [speed_x, speed_y, speed_w])
        step += 1

        if slip:
            log_step(args.load_run, step, args.checkpoint)
            step = 0
            print("env_gazebo.phase", env_gazebo.phase.item())
            break

if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_CRITIC = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
        
    args = get_args()
    play(args)
