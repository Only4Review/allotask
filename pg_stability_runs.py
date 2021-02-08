import torch.optim as optim
from data_allocator.PolicyBasedAllocator import UniformPolicyAllocator
from data_allocator.UniformAllocationPolicy import UniformAllocationPolicy
from data.UniformEnvironment import UniformEnvironment
import numpy as np
from utils.setup_logging import setup_logging
import CONSTANTS as see
import torch


def main(no_episodes, budget, lr, gamma, hidden_dim, output_dim, dropout, start_point, trials, device):

    see.logs.write('================================\n')
    see.logs.write('Learning rate: {}\n'.format(lr))
    see.logs.write('Gamma: {}\n'.format(gamma))
    see.logs.write('Hidden dim: {}\n'.format(hidden_dim))
    see.logs.write('Dropout: {}\n'.format(dropout))

    performance = []

    for trial in range(trials):
        see.logs.write('\n Trial {}: \n'.format(trial))

        env = UniformEnvironment(state_value, budget, start_point)
        eval_env = UniformEnvironment(state_value, budget, np.array([1, 1]))
        policy = UniformAllocationPolicy(3, output_dim, output_dim, hidden_dim, dropout)
        policy.to(device)
        policy.train()
        allocator = UniformPolicyAllocator(policy, env, eval_env, optim.Adam(policy.parameters(), lr=lr), device=device,
                                           gamma=gamma, logging=True)

        deterministic, max_state_value = allocator.policy_gradient_optimization(no_episodes)
        performance.append(max_state_value)

    see.logs.write('Policy trained {} times with {} average performance and {} max performance'.format(
        trials,
        sum(performance)/trials,
        max(performance)
    ))


def state_value(pair):
    x = pair[0]
    y = pair[1]
    return x*y/(x+3*y)


def grid_search_max(some_function, grid_size, start_point):
    best = -np.inf
    for i in range(start_point[0], grid_size):
        for j in range(start_point[1], grid_size):
            if i * j < grid_size:
                curr = some_function((i, j))
                if curr >= best:
                    best = curr
                    argmax = (i, j)
    see.logs.write('\nState value maximized at {} with value {}.\n'.format(argmax, best))


if __name__ == '__main__':

    setup_logging('PG', 'sinusoid', '')

    BUDGET = 2000
    N_EPISODES = 5000
    START = np.array([1,1])
    OUTPUT_DIM = 2
    TRIALS = 3
    LR = 0.001
    GAMMA = 0.99

    see.logs.write('Budget is {}'.format(BUDGET))
    grid_search_max(state_value, BUDGET, start_point=np.array([1,1]))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config_index = 0
    for hidden_dim in [8, 16]:
        for dropout in [0.5, 0.8]:
            see.logs.write('\n\n CONFIG {}: \n'.format(config_index))
            main(N_EPISODES, BUDGET, LR, GAMMA, hidden_dim, OUTPUT_DIM, dropout, START, TRIALS, device)
            config_index += 1
