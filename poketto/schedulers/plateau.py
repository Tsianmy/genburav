import torch
from timm.scheduler import PlateauLRScheduler
from timm.scheduler.scheduler import Scheduler

class PlateauLR(PlateauLRScheduler):
    def __init__(
        self,
        optimizer,
        decay_rate=0.1,
        patience_t=10,
        verbose=True,
        threshold=0.0001,
        cooldown_t=0,
        warmup_it=0,
        warmup_lr_init=0,
        lr_min=0,
        mode='max',
        noise_range_t=None,
        noise_type='normal',
        noise_pct=0.67,
        noise_std=1,
        noise_seed=None,
        initialize=True,
        t_in_epochs=True,
        epoch_len=None
    ):
        if t_in_epochs:
            assert epoch_len is not None
            patience_t *= epoch_len
            cooldown_t *= epoch_len
            t_in_epochs = False

        Scheduler.__init__(
            self,
            optimizer,
            'lr',
            noise_range_t=noise_range_t,
            noise_type=noise_type,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=patience_t,
            factor=decay_rate,
            verbose=verbose,
            threshold=threshold,
            cooldown=cooldown_t,
            mode=mode,
            min_lr=lr_min
        )

        self.warmup_t = warmup_it
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        self.restore_lr = None

        self.metric = float('inf') if mode == 'min' else float('-inf')
        self.need_metric = True

    def step_update(self, t, metric=None):
        if t <= self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
            super().update_groups(lrs)
        else:
            if self.restore_lr is not None:
                # restore actual LR from before our last noise perturbation before stepping base
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.restore_lr[i]
                self.restore_lr = None

            self.lr_scheduler.step(self.metric, t)  # step the base scheduler

            if self._is_apply_noise(t):
                self._apply_noise(t)