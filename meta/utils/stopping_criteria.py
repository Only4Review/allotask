from abc import abstractmethod, ABC

class StoppingCriterion(ABC):
    @abstractmethod
    def __call__(self, lr_rates):
        return False


class StopByAnnealing(StoppingCriterion):
    #input: a list of learning rates used in each epoch
    #output: True if training must stop, False otherwise
            
    """Description: checks if we have more than k lr annealings in lr_rates
        to determine whether to stop the training or not.
    """
    def __init__(self, after_k_annealing_steps = 3):
        self.max_no_of_annealing_steps = after_k_annealing_steps

    def __call__(self, lr_rates):
        annealing_steps=0
        for i in range(len(lr_rates)-1):
            if lr_rates[i] != lr_rates[i+1]:
                annealing_steps += 1

        if annealing_steps > self.max_no_of_annealing_steps:
            return True
        else:
            return False