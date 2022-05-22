class MyNormalizer:
    def __init__(self):
        pass

    def fit(self, array):
        self.min = array.min()
        self.max = array.max()

    def transform_single(self, val):
        if val < self.min or val > self.max:
            raise ValueError("Out of bound value")

        z = val - self.min
        return z / (self.max - self.min)