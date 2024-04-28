from torchvision.transforms import functional as F

class ToTensor:
    def __call__(self, data):
        img = data['img']
        data['img'] = F.to_tensor(img)
        return data
    
    def __repr__(self):
        return self.__class__.__name__ + '()'