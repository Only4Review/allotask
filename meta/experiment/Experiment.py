from abc import ABC, abstractmethod

class Experiment(ABC):

    @abstractmethod
    def read_config(self, config):
        pass

    @abstractmethod
    def setup_logs(self):
        pass

    @abstractmethod
    def run(self):
        pass
