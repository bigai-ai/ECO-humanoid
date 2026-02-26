# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# Copyright (c) 2026 ECO Authors. All rights reserved.


from bruce_gym.envs import *
from bruce_gym.utils import get_args, task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
# python humanoid/scripts/train.py --task=bruce_ssrl  --headless --run_name ppo_critic_same
# python humanoid/scripts/train.py --task=booster_ssrl  --run_name no_ratio_1batch --sim_device=cuda:0 --rl_device=cuda:0
# export CUDA_VISIBLE_DEVICES=0
# seed=0
# nohup python -u humanoid/scripts/train.py --task=bruce_ppolag  --headless --run_name eco_0.1vel_6000cost_env4096_lr1e2_plane_seed${seed} --sim_device=cuda:0 --seed ${seed} --rl_device=cuda:0 > eco_0.1vel_6000cost_env4096_plane_seed${seed}.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=5
# nohup python -u humanoid/scripts/train.py --task=bruce_p3o  --headless  --run_name p3o_0.1vel_6000cost_env4096_plane_seed175 --sim_device=cuda:0 --rl_device=cuda:0 --seed 173 > p3o_0.1vel_6000cost_env4096_plane_seed173.log 2>&1 &
# nohup python -u humanoid/scripts/train.py --task=bruce_p3o  --headless  --run_name p3o_0.1vel_6000cost_env4096_plane_seed125 --sim_device=cuda:0 --rl_device=cuda:0 --seed 124 > p3o_0.1vel_6000cost_env4096_plane_seed124.log 2>&1 &

# nohup python -u humanoid/scripts/train.py --task=bruce_p3o  --headless  --run_name p3o_0.1vel_6000cost_env4096_plane_seed125 --sim_device=cuda:0 --rl_device=cuda:0 --seed 125 > p3o_0.1vel_6000cost_env4096_plane_seed125.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=7

#  nohup python -u humanoid/scripts/train.py --task=bruce_ppo  --headless --run_name ppo_0.03_0.1vel_6000cost_env4096_plane_seed182 --sim_device=cuda:0 --rl_device=cuda:0 --seed 182 > ppo0.03_0.1vel_6000cost_env4096_plane_seed182.log 2>&1 &
#  nohup python -u humanoid/scripts/train.py --task=bruce_ppo  --headless --run_name ppo_0.08_0.1vel_6000cost_env4096_plane_seed180 --sim_device=cuda:0 --rl_device=cuda:0 --seed 180 > ppo0.08_0.1vel_6000cost_env4096_plane_seed180.log 2>&1 &
#  nohup python -u humanoid/scripts/train.py --task=bruce_ppo  --headless --run_name ppo_0.025_0.1vel_6000cost_env4096_plane_seed181 --sim_device=cuda:0 --rl_device=cuda:0 --seed 181 > ppo0.025_0.1vel_6000cost_env4096_plane_seed181.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python -u humanoid/scripts/train.py --task=bruce_ipo  --headless --run_name ipo_0.1vel_6000cost_env4096_plane_seed126_kappa10 --sim_device=cuda:0 --rl_device=cuda:0 --seed 126 > ipo_0.1vel_6000cost_env4096_plane_seed126_kappa10.log 2>&1 &
# nohup python -u humanoid/scripts/train.py --task=bruce_ipo  --headless --run_name ipo_0.1vel_6000cost_env4096_plane_seed127_kappa10 --sim_device=cuda:0 --rl_device=cuda:0 --seed 127 > ipo_0.1vel_6000cost_env4096_plane_seed127_kappa10.log 2>&1 &
# nohup python -u humanoid/scripts/train.py --task=bruce_ipo  --headless --run_name ipo_0.1vel_6000cost_env4096_plane_seed128_kappa10 --sim_device=cuda:0 --rl_device=cuda:0 --seed 128 > ipo_0.1vel_6000cost_env4096_plane_seed128_kappa10.log 2>&1 &
# nohup python -u humanoid/scripts/train.py --task=bruce_ipo  --headless --run_name ipo_0.1vel_6000cost_env4096_plane_seed129_kappa10 --sim_device=cuda:0 --rl_device=cuda:0 --seed 129 > ipo_0.1vel_6000cost_env4096_plane_seed129_kappa10.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=5
# nohup python -u humanoid/scripts/train.py --task=bruce_crpo  --headless --run_name crpo_0.1vel_6000cost_env4096_plane_seed126 --sim_device=cuda:0 --rl_device=cuda:0 --seed 126 > crpo_0.1vel_6000cost_env4096_plane_seed126.log 2>&1 &
# nohup python -u humanoid/scripts/train.py --task=bruce_crpo  --headless --run_name crpo_0.1vel_6000cost_env4096_plane_seed127 --sim_device=cuda:0 --rl_device=cuda:0 --seed 127 > crpo_0.1vel_6000cost_env4096_plane_seed127.log 2>&1 &
# nohup python -u humanoid/scripts/train.py --task=bruce_crpo  --headless --run_name crpo_0.1vel_6000cost_env4096_plane_seed128 --sim_device=cuda:0 --rl_device=cuda:0 --seed 128 > crpo_0.1vel_6000cost_env4096_plane_seed128.log 2>&1 &




# python humanoid/scripts/train.py --task=kuavo_ppolag  --headless

# python humanoid/scripts/train.py --task=humanoid_ppo --run_name test_1 --headless

if __name__ == '__main__':
    args = get_args()
    train(args)
