import numpy as np
import random
from enum import Enum
from collections import namedtuple
from scipy.linalg import solve

Step = namedtuple('Step', ['state', 'action', 'reward', 'next_state'], verbose=True)
Model = namedtuple('Model', ['kernel', 'reward', 'terminal_state'], verbose=True)

class RandomWalk:
    def __init__(self, _Ns):

        self.Na = 2
        self.Ns = _Ns + 2  # We add two terminal states

        self.kernel = np.zeros((self.Ns, self.Na, self.Ns))
        self.reward = np.zeros((self.Ns, self.Na))
        self.terminal_state = [0, self.Ns - 1]

        move = [-1, +1]
        for s in range(self.Ns):
            for a in range(self.Na):

                    if s in self.terminal_state:
                        self.kernel[s, a, s] = 1
                    else:
                        self.kernel[s, a, s+move[a]] = 1

        # reaching right terminal state-> reward
        self.reward[self.Ns-2, 1] = 1

    def start(self):
        return self.Ns//2

    def get_random_policy(self):
        return np.ones((self.Ns, self.Na))/self.Na

    def get_model(self):
        return Model(kernel=self.kernel, reward=self.reward, terminal_state=self.terminal_state)


class Sampler:
    def __init__(self, model):
        self.kernel = model.kernel
        self.reward = model.reward
        self.terminal_state = model.terminal_state

        assert len(self.terminal_state) > 0  # force terminal state

        [self.Ns, self.Na] = self.reward.shape

    def get_next(self, state, policy):
        action = np.random.choice(self.Na, 1, p=policy[state])[0]
        reward = self.reward[state, action]
        next_state = np.random.choice(self.Ns, 1, p=self.kernel[state, action])[0]

        return Step(state=state, action=action, reward=reward, next_state=next_state)

    def get_episode(self, policy, start=None):

        if start is None:
            start = random.randint(0, self.Ns-1)

        state = start
        episode = []
        while state not in self.terminal_state:
            step = self.get_next(state, policy=policy)
            episode.append(step)
            state = step.next_state

        return episode

    def get_batch(self, size, policy, start=None):
        return [self.get_episode(policy=policy, start=start) for _ in range(size)]


class DP:
    def __init__(self, model):
        self.kernel = model.kernel
        self.reward = model.reward

        [self.Ns, self.Na] = self.reward.shape

    def __get_policy_model(self, policy):

        # Normalize the kernel by the policy
        buf_kernel = np.transpose(self.kernel, axes=[0, 2, 1])
        pi_kernel = np.zeros((self.Ns, self.Ns))
        for i, row_kernel in enumerate(buf_kernel):
            pi_kernel[i] = np.sum(row_kernel * policy[i], axis=1)

        # Normalize the reward by the policy
        pi_reward = np.sum(self.reward * policy, axis=1)

        return pi_kernel, pi_reward

    def value_function_linear_solve(self, gamma, policy):

        # Compute P_pi / R_pi
        pi_kernel, pi_reward = self.__get_policy_model(policy)

        # Solve the Bellman linear system
        A = np.identity(self.Ns) - gamma * pi_kernel
        b = pi_reward
        v = solve(A,b)
        #v = np.linalg.inv(A).dot(b)  # TODO: use solver

        return v

    def value_function_fix_point(self, gamma, policy, epsilon=1e-4, v0=None):

        # Compute P_pi / R_pi
        pi_kernel, pi_reward = self.__get_policy_model(policy)

        # Initialize v values
        if v0 is None:
            v0 = np.zeros(self.Ns)
        v = v0
        v_old = v + 10 * epsilon

        # Compute Bellman fix point
        while np.max(np.abs(v - v_old)) > epsilon:
            v, v_old = pi_reward + gamma*pi_kernel.dot(v), v

        return v

    def q_function(self, gamma, policy, epsilon=1e-4, v0=None):

        # Compute V values
        v = self.value_function_fix_point(gamma, policy, epsilon=epsilon, v0=v0)

        # Compute Q
        Q = np.copy(self.reward)
        for i, q in enumerate(Q):
            q += gamma*self.kernel[i].dot(v)

        return Q


if __name__ == "__main__":

    Ns = 5

    random_walk = RandomWalk(Ns)
    model = random_walk.get_model()

    sampler = Sampler(model)
    batch = sampler.get_batch(5, policy=random_walk.get_random_policy(), start=random_walk.start())

    for episode in batch:
        print(episode)
        print("")

    dp_solver = DP(model)

    v1 = dp_solver.value_function_fix_point(gamma=0.99, policy=random_walk.get_random_policy())
    v2 = dp_solver.value_function_linear_solve(gamma=0.99, policy=random_walk.get_random_policy())
    q1 = dp_solver.q_function(gamma=0.99, policy=random_walk.get_random_policy())

    print(v1)
    print(v2)
    print(q1)