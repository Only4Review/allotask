import torch
import numpy as np
from torch.distributions import Categorical
import CONSTANTS as see
from tqdm import tqdm
from collections import Counter
from typing import Tuple
from data.UniformEnvironment import Environment


class UniformPolicyAllocator:
    def __init__(
            self,
            policy: torch.nn.Module,
            environment: Environment,
            eval_env: Environment,
            optimizer: torch.optim.Optimizer,
            device: str = 'cpu',
            gamma: float  = 0.99,
            logging: bool = False
    ) -> None:
        self.policy= policy
        self.env = environment
        self.eval_env = eval_env
        self.optimizer= optimizer
        self.gamma= gamma
        self.device= device
        self.logging= logging

    def select_action(self, state, evaluate = False):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = state.to(self.device)
        data_incr, task_incr = self.policy(state)

        # Something very subtle is going on here. When turning to categorical, we are effectively reassigning the
        # the outputs [0, 1, 2 ...] to the increment distribution. The network never knows otherwise. It is only through
        # the reward feedback that this assignment is made meaningful.
        data_incr_dist = Categorical(data_incr)
        task_incr_dist = Categorical(task_incr)

        # Exploration / Exploitation
        action = (data_incr_dist.sample(), task_incr_dist.sample())
        if not evaluate:
            # Add log probability of our chosen action to our history
            log_prob_action = (data_incr_dist.log_prob(action[0]) + task_incr_dist.log_prob(action[1]))
            if self.policy.policy_history.dim != 0:
                self.policy.policy_history = torch.cat([self.policy.policy_history, log_prob_action.view(1,-1)])
            else:
                self.policy.policy_history = log_prob_action
        return np.array([action[0].item(), action[1].item()])

    def update_state(self, environment, evaluate = False):
        current_state = environment.current_state
        action = self.select_action(current_state, evaluate)
        state, reward, done = environment.step(action)
        if not evaluate:
            self.policy.reward_episode.append(reward)
        return done, action

    def update_policy(self):
        loss = self.compute_loss()
        self.update_weights(loss)
        self.save_and_reset_episode_history(loss)

    def save_and_reset_episode_history(self, loss):
        self.policy.loss_history.append(loss.item())
        self.policy.reward_history.append(np.sum(self.policy.reward_episode))
        self.policy.policy_history = self.policy.policy_history.new_empty(0)
        self.policy.reward_episode = []

    def update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self):
        rewards = self.discount_rewards(self.policy.reward_episode, self.gamma)
        # Scale rewards
        rewards = torch.tensor(rewards).float().to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        # Calculate loss
        loss = (-1) * torch.sum(torch.mul(self.policy.policy_history.flatten(), rewards), -1)
        loss.to(self.device)
        return loss

    def discount_rewards(self, undiscounted_rewards, gamma):
        R = 0
        rewards = []
        # Discount future rewards back to the present using gamma
        for r in undiscounted_rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        return rewards

    def evaluate_policy(self, iterations, episode_time = 300):
        self.policy.eval()
        performance = []
        episode_paths = []
        for i in range(iterations):
            actions = []
            self.eval_env.reset()
            for j in range(episode_time):
                done, action = self.update_state(self.eval_env, evaluate=True)
                if done:
                    break
                # This happens after break, since we exceeded the budget already.
                actions.append(action)
            performance.append(self.eval_env.state_value(self.eval_env.current_state))
            episode_paths.append(np.array(actions))
        self.policy.train()

        # TODO: This is a hack. I don't like this way of evaluating, need to switch.
        deterministic = 1
        for item in episode_paths:
            if np.array_equal(item, episode_paths[0]):
                pass
            else:
                deterministic = 0
                break

        list_of_actions = []
        for item in episode_paths:
            for ac in item:
                list_of_actions.append((ac[0], ac[1]))

        counts = Counter(list_of_actions)
        return max(performance), min(performance), deterministic, counts

    def check_distribution_deterministic(self, action_logits, threshold = 0.99):
        if action_logits.max() > threshold:
            return 1
        else:
            return 0

    def policy_gradient_optimization(
            self,
            episodes: int,
            episode_time = 150,
            batch_size = 50,
            iter_capped = True,
            evaluate = True,
            evaluate_every = 100,
            evaluate_for = 10
    ) -> Tuple[int, float]:
        """
        :param episodes: int, number of episodes
        :param episode_time: int, maximum length of each episode
        :param batch_size: int, PG update batch size
        :param iter_capped: bool, whether to stop the episode when the episode is done vs when episode time runs out
        :param evaluate: bool, whether to evaluate
        :param evaluate_every: int, number of episodes between policy evaluations, used for stopping, should be an integer multiple of batch_size
        :param evaluate_for: int, number of times to MC estimate the policy behaviour
        :return:
            tuple(int, float), whether at evaluation stage all policy trajectories coincide, and the max state value reached by the policy
        """

        # Always cast before switching device. The other way round gives backprop error.
        running_loss = torch.tensor(0).type(torch.FloatTensor).to(self.device)

        for episode in tqdm(range(1, episodes+1)):
            self.unroll_policy(episode_time, iter_capped, self.env)
            # self.update_policy()
            loss = self.compute_loss()
            self.save_and_reset_episode_history(loss)
            running_loss += loss

            if episode % batch_size == 0:
                self.update_weights(running_loss / batch_size)
                running_loss = torch.tensor(0).type(torch.FloatTensor).to(self.device)

            if episode % evaluate_every == 0 and evaluate:
                max_perf, min_perf, deterministic, action_counts = self.evaluate_policy(evaluate_for)
                if deterministic:
                    break

        max_perf, min_perf, deterministic, action_counts = self.evaluate_policy(evaluate_for)
        if self.logging:
            see.logs.write('Policy performance is {}, with worst performance {}\n'.format(max_perf, min_perf))
            see.logs.write('Policy is deterministic: {}\n'.format(deterministic))
            see.logs.write('Action spectrum is {}\n'.format(action_counts))
        return int(deterministic), max_perf

    def unroll_policy(self, episode_time, iter_capped, environment):
        environment.reset()  # Reset environment and record the starting state
        for time in range(episode_time):
            done, _ = self.update_state(environment)
            if done and not iter_capped:
                break