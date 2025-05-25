from timm.scheduler import CosineLRScheduler

class CosineLR(CosineLRScheduler):
    def __init__(
        self,
        optimizer,
        t_initial: int,
        lr_min: float = 0,
        cycle_mul: float = 1,
        cycle_decay: float = 1,
        cycle_limit: int = 1,
        warmup_it=0,
        warmup_lr_init=0,
        warmup_prefix=False,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1,
        noise_seed=42,
        k_decay=1,
        initialize=True,
        epoch_len=None
    ):
        if t_in_epochs:
            assert epoch_len is not None
            t_initial *= epoch_len
            t_in_epochs = False
        super().__init__(
            optimizer,
            t_initial,
            lr_min,
            cycle_mul,
            cycle_decay,
            cycle_limit,
            warmup_it,
            warmup_lr_init,
            warmup_prefix,
            t_in_epochs,
            noise_range_t,
            noise_pct,
            noise_std,
            noise_seed,
            k_decay,
            initialize
        )
