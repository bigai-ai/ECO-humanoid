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


class Lagrange:
    def __init__(
        self,
        cost_index: int = 1,
        **args,
    ) -> None:
        
        cost_limit_key = f"cost_limit{cost_index}"
        lambda_lr_key = f"lambda_lr{cost_index}"
        lagrangian_multiplier_init_key = f"lagrangian_multiplier_init{cost_index}"
        lambda_optimizer_key = f"lambda_optimizer{cost_index}"
        
        self.cost_limit: float = args.get(cost_limit_key, 0.0)
        lambda_lr: float = args.get(lambda_lr_key, 0.0)
        lagrangian_multiplier_init: float = args.get(
            lagrangian_multiplier_init_key, 0.0
        )
        lambda_optimizer = args.get(lambda_optimizer_key, "Adam")

        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()

        assert hasattr(torch.optim, lambda_optimizer), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(torch.optim, lambda_optimizer)
        
        # Use a dedicated optimizer for the multiplier.
        self.lambda_optimizer: torch.optim.Optimizer = torch_opt(
            [self.lagrangian_multiplier],
            lr=lambda_lr,
        )
        
        self.previous_cost = None  # Cache previous episode cost.

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        """Penalty loss for Lagrange multiplier.

        .. note::
            ``mean_ep_cost`` is obtained from ``self.logger.get_stats('EpCosts')[0]``, which is
            already averaged across MPI processes.

        Args:
            mean_ep_cost (float): mean episode cost.

        Returns:
            Penalty loss for Lagrange multiplier.
        """
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

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
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0.0,
        )  # enforce: lambda in [0, inf]
        
