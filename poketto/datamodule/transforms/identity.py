class Identity:
    def __call__(self, data):
        return data
    
    def __repr__(self):
        return self.__class__.__name__ + '()'