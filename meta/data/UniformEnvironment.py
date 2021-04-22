from typing import Callable
import numpy as np
from abc import abstractmethod, ABC


class Environment(ABC):
    @abstractmethod
    def current_state(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

class UniformEnvironment(Environment):
    def __init__(self, state_value: Callable, budget: int, initial_state = None):
        # states are pairs of $K_out$, $K_in$. We may consider adding a time state (i.e. number of grad steps).
        # This is a function. It takes as arguments a state and returns a value obtained from training in that state.
        self.state_value = state_value

        self._initial_state = initial_state
        if initial_state is None:
            initial_state = np.random.randint(1, 2 * np.floor(np.sqrt(budget)), 2)
        self.initial_state = np.concatenate((initial_state, np.array([self.state_value(initial_state)])))
        self._current_state = self.initial_state

        self.budget = budget
        self.value_history = []
        self.value_history.append(self.state_value(initial_state))

    @property
    def current_state(self):
        return self._current_state

    def next_state(self, state, action):
        ground_state = state[:-1] + action
        return np.concatenate((ground_state, np.array([self.state_value(ground_state)])))

    def reset(self):
        self.__init__(self.state_value, self.budget, self._initial_state)

    def step(self, action):
        next_state = self.next_state(self.current_state, action)
        self._current_state = next_state

        next_value = next_state[-1]
        reward = np.log(next_value/self.value_history[-1])
        self.value_history.append(next_value)
        if next_state[0] * next_state[1] >= self.budget:
            done = 1
        else:
            done = 0
        return next_state, reward, done