class AlwaysZero:
    def fit(self, *args, **kwargs):
        return self
    def predict(self, X):
        import numpy as np
        return np.zeros(len(X), dtype=int)
