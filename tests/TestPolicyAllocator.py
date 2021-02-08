import unittest
import numpy as np
import torch
from data_allocator.PolicyBasedAllocator import UniformPolicyAllocator
from data_allocator.UniformAllocationPolicy import UniformAllocationPolicy
from data.UniformEnvironment import Environment

class MockENV(Environment):
    def __init__(self):
        self._current_state = np.array([0,0])
        self.state_value = lambda x: np.random.randint(0,10)

    @property
    def current_state(self):
        return self._current_state

    def reset(self):
        pass

    def step(self, action):
        self._current_state = np.random.randint(0,10,2) + action
        return self.current_state, np.random.randint(0,10), 1


class TestPollicyAllocator(unittest.TestCase):
    def setUp(self):
        self.env = MockENV()
        self.policy = UniformAllocationPolicy(2,2,2,2)
        self.eval_env = MockENV()
        self.optim = torch.optim.Adam(self.policy.parameters(), lr = 0.1)
        self.allocator = UniformPolicyAllocator(self.policy, self.env, self.eval_env, self.optim, 'cpu')

    def test_end_to_end(self):
        _, __ = self.allocator.policy_gradient_optimization(1)

    def test_policy_params_updating(self):
        before = [param.data.detach().clone() for param in self.policy.parameters()]
        _, __ = self.allocator.policy_gradient_optimization(3, batch_size=1, evaluate = False)
        after = [param.data.detach().clone() for param in self.policy.parameters()]
        for index in range(len(before)):
            self.assertFalse(torch.allclose(before[index], after[index]))

    def test_discount_rewards(self):
        rewards = [0,0,0,1]
        expected = [1/8, 1/4, 1/2, 1]
        actual = self.allocator.discount_rewards(rewards, 0.5)
        self.assertEqual(expected, actual)

    # TODO
    def test_evaluate_policy(self):
        pass

    # TODO
    def test_GPU_computation(self):
        pass



