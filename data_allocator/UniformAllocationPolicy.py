import torch
from torch import nn


class UniformAllocationPolicy(nn.Module):
    """
    A thin wrapper around a nn which maps states to actions.
    """

    def __init__(self, state_space_dimension=2, task_budget_increment_size = 100, no_tasks_increment = 10, hidden_dim=64, dropout = 0.6, bias = True):
        super(UniformAllocationPolicy, self).__init__()

        self.action_space_size = task_budget_increment_size * no_tasks_increment

        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(state_space_dimension, self.hidden_dim, bias=bias)
        self.linear2 = nn.Linear(self.hidden_dim, task_budget_increment_size, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.linear3 = nn.Linear(state_space_dimension, self.hidden_dim, bias=bias)
        self.linear4 = nn.Linear(self.hidden_dim, no_tasks_increment, bias=bias)

        # Episode policy and reward history
        self._policy_history = torch.Tensor()
        self.register_buffer('policy_history', self._policy_history)
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        # The assumption is that an action is built out of two independent increments. We are factorizing the
        # distribution over the action space this way for simplicity.

        h1 = self.linear1(x)
        h1 = self.dropout(h1)
        h1 = self.relu(h1)
        h1 = self.linear2(h1)
        out_data_increment = nn.Softmax(dim=-1)(h1)

        h2 = self.linear3(x)
        h2 = self.dropout(h2)
        h2 = self.relu(h2)
        h2 = self.linear4(h2)
        out_task_increment = nn.Softmax(dim=-1)(h2)

        return out_data_increment, out_task_increment