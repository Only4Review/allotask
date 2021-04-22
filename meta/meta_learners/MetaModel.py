from torch import nn
from abc import ABC, abstractmethod


class MetaModel(nn.Module, ABC):
    """
    This class stores an estimator for a task.
    It has a method to switch this estimator - task_forward, but does not store multiple estimators simultaneously.
    It also reads a training operation to extract meta-information from a meta-dataset through the meta_fit method.
    The usage of the class should be:
        meta_fit -> task_forward -> forward
    """

    @abstractmethod
    def task_forward(self, task_information):
        '''
        Input task data and return and store an estimator (a torch Module). 
        The forward method performs inderence with this estimator.
        '''
        pass

    @abstractmethod
    def meta_fit(self, train_op, meta_data):
        """
        Stores in self an estimator factory. A call to task_forward iterates the factory.
        """
        pass


class BlackBoxMM(MetaModel):
    """
    Implements h, g, f as in project ducumentation.
    task_forward = f.__init__(g.forward(h.forward(task_data)))
    """
    pass