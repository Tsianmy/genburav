from poketto.utils import glogger

class BaseVisualizer:
    def __init__(self, save_dir, use_tensorboard=False, tb_log_metrics=None):
        self.save_dir = save_dir
        self.mode = 'eval'
        self.tb_writer = None
        if use_tensorboard:
            glogger.info(f'[{self.__class__.__name__}] use Tensorboard')
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.save_dir)
                layout = {'losses': {'train': ['Multiline', ['train/loss*']]}}
                if tb_log_metrics is not None:
                    layout['metrics'] = {
                        k: ['Multiline', [f'^(train|eval)/{k}']] for k in tb_log_metrics}
                self.tb_writer.add_custom_scalars(layout)
            except:
                glogger.warning(f'[{self.__class__.__name__}] Tensorboard not available')
    
    def set_mode(self, name='eval'):
        assert name in ('train', 'eval', 'infer')
        self.mode = name

    def fetch(self, data, *args, **kwargs):
        pass
    
    def add_data(self, data, *args, **kwargs):
        pass
    
    def save(self, data, *args, **kwargs):
        pass