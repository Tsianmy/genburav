import torch
import numpy as np
from typing import Dict, Any, List, Optional

class GaussSplatSched:
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            param_group_name: str,
            decay_t: float,
            param_group_field: str = 'lr',
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_t: int = 0,
            t_in_epochs: bool = True,
            epoch_len = None,
            **kwargs,
    ) -> None:
        if t_in_epochs:
            assert epoch_len is not None
            decay_t *= epoch_len
            t_in_epochs = False
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self.param_group_name = param_group_name
        self._initial_param_group_field = f"initial_{param_group_field}"
        self.base_values = []
        for i, group in enumerate(self.optimizer.param_groups):
            if group['name'] != self.param_group_name:
                continue
            if param_group_field not in group:
                raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
            group.setdefault(self._initial_param_group_field, group[param_group_field])
            self.base_values.append(group[param_group_field])
        self.metric = None  # any point to having this for all?
        self.decay_t = decay_t
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_t = cycle_t
        self.t_in_epochs = t_in_epochs
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def _get_lr(self, t: int) -> List[float]:
        if t < 0:
            return 0.0
        if self.cycle_t > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.cycle_mul + (1 - self.cycle_mul) * np.sin(
                0.5 * np.pi * np.clip(t / self.cycle_t, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(t / self.decay_t, 0, 1)
        log_lerps = [np.exp(np.log(v) * (1 - t) + np.log(self.lr_min) * t) for v in self.base_values]
        return [delay_rate * v for v in log_lerps]

    def _get_values(self, t: int, on_epoch: bool = True) -> Optional[List[float]]:
        proceed = (on_epoch and self.t_in_epochs) or (not on_epoch and not self.t_in_epochs)
        if not proceed:
            return None
        return self._get_lr(t)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self._get_values(num_updates, on_epoch=False)
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        param_groups = []
        for group in self.optimizer.param_groups:
            if group['name'] == self.param_group_name:
                param_groups.append(group)
        if not isinstance(values, (list, tuple)):
            values = [values] * len(param_groups)
        for param_group, value in zip(param_groups, values):
            if 'lr_scale' in param_group:
                param_group[self.param_group_field] = value * param_group['lr_scale']
            else:
                param_group[self.param_group_field] = value
