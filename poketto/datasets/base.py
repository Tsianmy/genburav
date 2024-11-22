import numpy as np
from torch.utils.data import Dataset
from .transforms import Compose

class BaseDataset(Dataset):
    def __init__(
        self,
        data_root='',
        transforms=None,
        test_mode=False,
        max_refetch=100
    ):
        super().__init__()
        self.data_root = data_root
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list = []

        self.transforms = Compose(transforms)

        self.data_list = self.load_data_list()
    
    def load_data_list(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! ')
    
    def prepare_data(self, idx):
        data = self.raw_data(idx)
        return self.transforms(data)
    
    def raw_data(self, idx):
        raise NotImplementedError
    
    def _rand_another(self) -> int:
        """Get random index.
        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def __len__(self):
        return len(self.data_list)