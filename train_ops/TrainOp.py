from abc import ABC, abstractmethod

class TrainOP(ABC):
    """
    This class implements the meta-training procedure for a meta-model.
    """

    @abstractmethod
    def train(model, data, train_config):
        pass
    
    """
    @abstractmethod
    def throw_hook(self) -> TrainHook:
        pass
    """

    
class TrainHook(ABC):
    """
    A TrainHook is a method that is called when a certain event occurs during training.
    It represents a stateful interface for the training OP to interface with other modules.
    It is instantiated with all the objects necessary to execute the operation.
    It has no functionality in of itself other than serving as an external playground for other methods
    of other classes.
    """

    @abstractmethod
    def resolve(self):
        "Calls the method it was instantiated with."
        pass