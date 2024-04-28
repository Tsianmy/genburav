from poketto import build

class Compose:
    """Compose multiple transforms sequentially.
    """

    def __init__(self, transforms):
        self.transforms = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                self.transforms.append(build.build_transform(transform))

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