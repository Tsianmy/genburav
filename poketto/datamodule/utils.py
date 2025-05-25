from typing import Union, List, Tuple
from torch.utils.data import default_collate

def get_collate_fn(collate_keys: Union[List[str], Tuple[str]]):
    def collate_fn(batch):
        result = {}
        for key in collate_keys:
            result[key] = default_collate([elem[key] for elem in batch])
        rest_keys = batch[0].keys() - collate_keys
        for key in rest_keys:
            result[key] = [elem[key] for elem in batch]
        return result
    
    return collate_fn