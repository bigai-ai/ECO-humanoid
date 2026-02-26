# Copyright 2023 OmniSafe Team. All Rights Reserved.
# Copyright 2026 ECO Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Lagrange."""

from __future__ import annotations

import torch


class P3OLagrange:
    def __init__(
        self,
        cost_index: int = 1,
        **args,
    ) -> None:
        
        cost_limit_key = f"cost_limit{cost_index}"
        lagrangian_multiplier_init_key = f"lagrangian_multiplier_init{cost_index}"
        self.kappa = args.get("kappa", 0.01)
        self.penalty_max = args.get("penalty_max", 1.0)
        self.cost_limit: float = args.get(cost_limit_key, 0.0)
        lagrangian_multiplier_init: float = args.get(
            lagrangian_multiplier_init_key, 0.0
        )

        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )

        self.previous_cost = None  # Cache previous episode cost.


    def update_lagrange_multiplier(self, Jc: float) -> None:
        r"""Update Lagrange multiplier (lambda).

        We update the Lagrange multiplier by minimizing the penalty loss, which is defined as:

        .. math::

            \lambda ^{'} = \lambda + \eta \cdot (J_C - J_C^*)

        where :math:`\lambda` is the Lagrange multiplier, :math:`\eta` is the learning rate,
        :math:`J_C` is the mean episode cost, and :math:`J_C^*` is the cost limit.

        Args:
            Jc (float): mean episode cost.
        """


        self.lagrangian_multiplier = torch.nn.Parameter(
             torch.as_tensor(Jc - self.cost_limit),
            requires_grad=True,
        )
