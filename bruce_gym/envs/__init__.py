# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# Copyright (c) 2026 ECO Authors. All rights reserved.

from .base.legged_robot import LeggedRobot
from .custom.brucewalk_env import BruceWalkFreeEnv
from .custom.brucewalk_config import BruceWalkCfg, BruceWalkCfgPPOLag
from .custom.brucewalk_config_ipo import BruceWalkIPOCfg, BruceWalkCfgIPO
from .custom.brucewalk_config_p3o import BruceWalkP3OCfg, BruceWalkCfgP3O
from .custom.brucewalk_config_crpo import BruceWalkCRPOCfg, BruceWalkCfgCRPO
from .custom.brucewalk_config_ppo import BruceWalkShapingCfg, BruceWalkShapingCfgPPO

from bruce_gym.utils.task_registry import task_registry


task_registry.register("bruce_ppolag", BruceWalkFreeEnv, BruceWalkCfg(), BruceWalkCfgPPOLag())
task_registry.register("bruce_ipo", BruceWalkFreeEnv, BruceWalkIPOCfg(), BruceWalkCfgIPO())
task_registry.register("bruce_p3o", BruceWalkFreeEnv, BruceWalkP3OCfg(), BruceWalkCfgP3O())
task_registry.register("bruce_crpo", BruceWalkFreeEnv, BruceWalkCRPOCfg(), BruceWalkCfgCRPO())
task_registry.register("bruce_ppo", BruceWalkFreeEnv, BruceWalkShapingCfg(), BruceWalkShapingCfgPPO())
