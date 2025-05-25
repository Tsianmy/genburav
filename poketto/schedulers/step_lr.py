from timm.scheduler import StepLRScheduler

class StepLR(StepLRScheduler):
    def __init__(
        self,
        optimizer,
        decay_t,
        decay_rate=1,
        warmup_it=0,
        warmup_lr_init=0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1,
        noise_seed=42,
        initialize=True,
        epoch_len=None
    ):
        if t_in_epochs:
            assert epoch_len is not None
            decay_t *= epoch_len
            t_in_epochs = False
        super().__init__(
            optimizer,
            decay_t,
            decay_rate,
            warmup_it,
            warmup_lr_init,
            t_in_epochs,
            noise_range_t,
            noise_pct,
            noise_std,
            noise_seed,
            initialize
        )
