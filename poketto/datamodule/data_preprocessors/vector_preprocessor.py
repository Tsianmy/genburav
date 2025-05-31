from .base_preprocessor import BaseDataPreprocessor

class VecDataPreprocessor(BaseDataPreprocessor):
    def __call__(self, data: dict, **kwargs):
        data['gt'] = self.to_cuda(data['gt'].float())
        return data