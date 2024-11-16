from typing import Sequence, Callable

class Compose:
    """Compose multiple transforms sequentially.
    """

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if not callable(transform):
                raise TypeError(f'transform {transform} is not callable')
            self.transforms.append(transform)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string