import numpy as np
from abc import ABC, abstractmethod

class ParametrizationSampler(ABC):
    @abstractmethod
    def sample(self, n):
        raise(NotImplementedError)

class UniformIntSampler(ParametrizationSampler):
    def __init__(self, low, high):
        self._low = low
        self._high = high
        
    def sample(self, n):
        return np.random.randint(low = self._low, high= self._high, size=(n, ))

class UniformSampler(ParametrizationSampler):
    def __init__(self, low, high):
        self._low = low
        self._high = high

    def sample(self, n):
        return np.random.sample(n)*(self._high - self._low) + self._low

class SinusoidNaturalParametrizationSampler(ParametrizationSampler):
    def sample(self, n):
        return map(tuple, np.random.sample(size = (n,3)))