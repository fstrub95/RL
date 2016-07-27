import numpy as np
import random

from collections import namedtuple
from scipy.linalg import solve
from scipy import signal

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


class ModelSampler:
    def __init__(self, model):
        self.kernel = model.kernel
        self.reward = model.reward
        self.terminal_state = model.terminal_state

        assert len(self.terminal_state) > 0  # force terminal state

        [self.Ns, self.Na] = self.reward.shape

    def get_space(self):
        return self.Ns, self.Na

    def get_next(self, state, action):

        reward = self.reward[state, action]
        next_state = np.random.choice(self.Ns, 1, p=self.kernel[state, action])[0]

        return Step(state=state, action=action, reward=reward, next_state=next_state)

    def __get_episode(self, state0, policy, episode, extra_step=False):

        state = state0
        while state not in self.terminal_state :
            # pick action regarding policy
            action = np.random.choice(self.Na, 1, p=policy[state])[0]
            step = self.get_next(state=state, action=action)

            episode.append(step)
            state = step.next_state

        if extra_step:
            action = np.random.choice(self.Na, 1, p=policy[state])[0]
            step = self.get_next(state=state, action=action)
            episode.append(step)

        return episode

    def get_v_episode(self, policy, state0=None):

        # pick a random state/action
        if state0 is None:
            state0 = random.randint(0, self.Ns - 1)

        # compute the episode by following the policy
        return self.__get_episode(policy=policy, state0=state0, episode=[])

    def get_v_batch(self, size, policy, state0=None):
        return [self.get_v_episode(policy=policy, state0=state0) for _ in range(size)]

    def get_q_episode(self, policy, state0=None, action0=None, extra_step=False):

        # pick a random state/action
        if state0 is None:
            state0 = random.randint(0, self.Ns-1)
        if action0 is None:
            action0 = random.randint(0, self.Na-1)

        # evaluate the first state/action
        step = self.get_next(state=state0, action=action0)

        # compute the episode by following the policy
        return self.__get_episode(state0=step.next_state, policy=policy, episode=[step], extra_step=extra_step)

    def get_q_batch(self, size, policy, state0=None, action0=None):
        return [self.get_q_episode(policy=policy, state0=state0, action0=action0) for _ in range(size)]



class Solver:

    def get_greedy_policy(self, Q):
        policy = np.zeros((self.Ns, self.Na))
        greedy_indices = np.argmax(Q, axis=1)

        for i, state_policy in zip(greedy_indices, policy):
            state_policy[i] = 1
        return policy

    def get_epsilon_greedy_policy(self, Q, epsilon=0.1):

        policy = np.zeros((self.Ns, self.Na))
        policy.fill(epsilon/self.Na)

        greedy_indices = np.argmax(Q, axis=1)

        for i, state_policy in zip(greedy_indices, policy):
            state_policy[i] += (1-epsilon)
        return policy


class DP(Solver):
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
        v = solve(A, b)

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

    def value_to_q_function(self, gamma, v):
        Q = np.copy(self.reward)
        for i, q in enumerate(Q):
            q += gamma*self.kernel[i].dot(v)
        return Q

    def q_function(self, gamma, policy, epsilon=1e-4, v0=None):

        # Compute V values
        v = self.value_function_fix_point(gamma, policy, epsilon=epsilon, v0=v0)

        # Compute Q
        Q = self.value_to_q_function(gamma, v)

        return Q, v

    def policy_iteration(self, gamma, epsilon=1e-4, v0=None):

        # Initialization
        Q = np.random.uniform(size=(self.Ns, self.Na))
        v = v0
        policy0 = self.get_greedy_policy(Q)
        policy, old_policy = policy0, policy0

        # Policy Iteration
        while True:
            Q, v = self.q_function(gamma, policy, epsilon=epsilon, v0=v)
            policy, old_policy = self.get_greedy_policy(Q), policy

            if np.array_equal(policy, old_policy):
                break

        return policy, Q, v

    def value_iteration(self, gamma, epsilon=1e-4, v0=None):

        # Initialization
        if v0 is None:
            v0 = np.zeros(self.Ns)
        v = v0
        v_old = v + 10 * epsilon

        # Bellman optimal
        while np.max(np.abs(v - v_old)) > epsilon:
            v, v_old = np.max(self.reward + gamma * self.kernel.dot(v), axis=1), v

        Q = self.value_to_q_function(gamma, v)
        policy = self.get_greedy_policy(Q)

        return policy, Q, v



class MC(Solver):
    def __init__(self, sampler):
        self.sampler = sampler
        self.Ns, self.Na = sampler.get_space()



    # every-visit MC
    def value_function(self, gamma, policy, no_samples=1000, state0=None, v0=None):

        if v0 is None:
            v0 = np.zeros(self.Ns)
        v = v0
        state_counter = np.zeros(self.Ns)

        for _ in range(no_samples):

            episode = self.sampler.get_v_episode(policy=policy, state0=state0)
            cum_returns = 0

            for step in reversed(episode):

                #compute the cumulative return
                cum_returns = step.reward + gamma*cum_returns

                #iterative Mean to compute expected rewards
                state_counter[step.state] += 1
                n = state_counter[step.state]
                v[step.state] = (n-1) / n * v[step.state] + cum_returns/n

        return v


    def value_function_grad(self, gamma, policy, no_samples=1000, state0=None, v0=None, alpha=0):

        if v0 is None:
            v0 = np.zeros(self.Ns)
        v = v0
        state_counter = np.zeros(self.Ns)

        for _ in range(no_samples):

            episode = self.sampler.get_v_episode(policy=policy, state0=state0)
            cum_returns = 0

            for step in reversed(episode):
                # compute the cumulative return
                cum_returns = step.reward + gamma * cum_returns

                #compute the number of states
                state_counter[step.state] += 1
                n = state_counter[step.state]

                # either use fix/dynamic gradient step (learning rate=lrt)
                if alpha > 0: lrt = alpha
                else : lrt = 1/n

                #update v
                v[step.state] +=  lrt * ( cum_returns - v[step.state] )

        return v



    def __update_q_function(self, episode, gamma, q, state_counter):

        cum_returns = 0

        for step in reversed(episode):
            # compute the cumulative return
            cum_returns = step.reward + gamma * cum_returns

            # iterative Mean to compute expected rewards
            state_counter[step.state, step.action] += 1
            n = state_counter[step.state, step.action]
            q[step.state, step.action] = (n - 1) / n * q[step.state, step.action] + cum_returns / n

        return q


    # every-visit MC
    def q_function(self, gamma, policy, no_samples=1000, q0=None, state0=None, action0=None):

        # Intialization
        if q0 is None:
            q0 = np.zeros((self.Ns, self.Na))
        q = q0
        state_counter = np.zeros((self.Ns, self.Na))

        #MC carlo evaluation
        for _ in range(no_samples):
            episode = self.sampler.get_q_episode(policy=policy, state0=state0, action0=action0)
            q = self.__update_q_function(episode, gamma, q, state_counter)

        return q


    def q_control_on_policy(self, gamma, policy, no_samples=1000, epsilon=0.1, q0=None):

        if q0 is None:
            q0 = np.zeros((self.Ns, self.Na))
        q = q0
        state_counter = np.zeros((self.Ns, self.Na))

        #MC carlo control
        for _ in range(no_samples):
            episode = self.sampler.get_q_episode(policy=policy)
            q = self.__update_q_function(episode, gamma, q, state_counter)
            policy = self.get_epsilon_greedy_policy(Q=q, epsilon=epsilon)

        # Compute a final greedy policy
        policy = self.get_greedy_policy(q)

        return policy, q




class TD_Backward(Solver):
    def __init__(self, sampler, _lambda=0):
        self.sampler = sampler
        self._lambda = _lambda
        self.Ns, self.Na = sampler.get_space()

    def value_function(self, gamma, policy, no_samples=1000, state0=None, v0=None, alpha=0.1):
        if v0 is None:
            v0 = np.zeros(self.Ns)
        v = v0


        for _ in range(no_samples):

            episode = self.sampler.get_v_episode(policy=policy, state0=state0)
            etrace = np.zeros(self.Ns)

            # Approximate the value function
            for step in episode:

                #compute the error for one state
                td_error = step.reward + gamma*v[step.next_state] - v[step.state]
                etrace[step.state] += 1

                #update v and the eligibility trace for all the states
                v += alpha*td_error*etrace
                etrace *= gamma*self._lambda

        return v


    def sarsa(self, gamma, policy, no_samples=1000, alpha=0.1, q0=None, state0=None, action0=None):

        if q0 is None:
            q0 = np.zeros((self.Ns, self.Na))
        q = q0


        for _ in range(no_samples):

            episode = self.sampler.get_q_episode(policy=policy, state0=None, action0=None, extra_step=True)
            etrace = np.zeros((self.Ns, self.Na))

            # Approximate the value function
            for step, next_step in zip(episode[:-1], episode[1:]):
                # compute the error for one state
                td_error = step.reward + gamma * q[next_step.state, next_step.action] - q[step.state, step.action]
                etrace[step.state, step.action] += 1

                # update v and the eligibility trace for all the states
                q += alpha * td_error * etrace
                etrace *= gamma * self._lambda

            policy = self.get_epsilon_greedy_policy(q)

        policy = self.get_greedy_policy(q)

        return policy, q


    def q_learning(self, gamma, policy, no_samples=1000, alpha=0.1, q0=None, state0=None, action0=None):
        if q0 is None:
            q0 = np.zeros((self.Ns, self.Na))
        q = q0

        for _ in range(no_samples):

            episode = self.sampler.get_q_episode(policy=policy, state0=state0, action0=action0, extra_step=True)
            etrace = np.zeros((self.Ns, self.Na))

            # Approximate the value function
            for step, next_step in zip(episode[:-1], episode[1:]):

                #get next greedy action (Q-Learning specific)
                a_greedy = q[next_step.state].argmax()

                # compute the error for one state
                td_error = step.reward + gamma * q[next_step.state, a_greedy] - q[step.state, step.action]
                etrace[step.state, step.action] += 1

                # update v and the eligibility trace for all the states
                q += alpha * td_error * etrace

                if a_greedy == next_step.action:
                    etrace *= gamma * self._lambda
                else:
                    etrace.fill(0)

            policy = self.get_epsilon_greedy_policy(q)

        policy = self.get_greedy_policy(q)

        return policy, q


class TD_Forward(Solver):
    def __init__(self, sampler, _lambda=0):
        self.sampler = sampler
        self._lambda = _lambda
        self.Ns, self.Na = sampler.get_space()

    def __compute_gain(self, episode, gamma, v):

        gain = np.zeros(len(episode))


        # compute the gain with a T step horizon
        gamma_prod = 1
        cum_reward = 0
        for i, step in enumerate(episode):
            gamma_prod *= gamma
            cum_reward += gamma_prod*step.reward

            gain[i] = cum_reward + gamma*gamma_prod*v[step.next_state]

        # compute the gain with a geometrical mean
        gain_prev = 1
        lambda_norm = 1-self._lambda
        for i, g in reversed(list(enumerate(gain))):
            gain_prev = g + self._lambda*gain_prev
            gain[i] = lambda_norm * gain_prev

        return gain

    def value_function(self, gamma, policy, no_samples=1000, state0=None, v0=None, alpha=0.1):

        if v0 is None:
            v0 = np.zeros(self.Ns)
        v = v0

        for _ in range(no_samples):

            episode = self.sampler.get_v_episode(policy=policy, state0=state0)
            gain = self.__compute_gain(episode=episode, gamma=gamma, v=v)

            # Approximate the value function
            for step, cum_reward in zip(episode, gain):
                v[step.state] += alpha*( cum_reward - v[step.state] )
        return v




class tabular_basis:
    def __init__(self, _Ns, _Na):
        self.Ns = _Ns
        self.Na = _Na

    def size(self):
        return self.Ns*self.Na

    def eval(self, state, action):
        phi = np.zeros((self.Ns, self.Na))
        phi[state, action] = 1
        return phi.reshape(-1)

class gaussian_basis:
    def __init__(self, _Ns, _Na, no_gaussian, sigma):
        self.Ns = _Ns
        self.Na = _Na
        self.mean = np.array([ i / (no_gaussian + 1) for i in range(_Ns) ])
        self.sigma = sigma

    def gaussian(self, x, mean, std):
        return np.exp(-np.power(x - mean, 2.) / (2 * np.power(std, 2.)))

    def size(self):
        return self.Na * len(self.mean)

    def eval(self, state, action):
        phi = np.zeros((self.Na, len(self.mean)))
        phi[action] = self.gaussian(state, self.mean, std=self.sigma)
        return phi.reshape(-1)


class soft_max_policy_1D:

    def __init__(self, basis):
        self.Ns = basis.Ns
        self.Na = basis.Na
        self.basis = basis
        self.w = np.random.uniform(-0.5,0.5,size=basis.size())

    def __linear_eval(self, state):
        return [np.dot(self.w, self.basis.eval(state, action)) for action in range(self.Na)]

    def __softmax(self, x):
        e = np.exp(x)
        dist = e / np.sum(e)
        return dist

    def eval(self, state, action=None):
        linear_comb = self.__linear_eval(state)
        soft_max = self.__softmax(linear_comb)

        if action:
            return soft_max[action]
        else:
            return soft_max

    def log_grad(self, state, action):

        phi = self.basis.eval(state, action)

        E_phi = np.zeros(self.basis.size())

        pi = self.eval(state)

        for a in range(self.Na):
            E_phi += (self.basis.eval(state, a)*pi[a])


        return phi -  E_phi

    def update(self, state, action, alpha, reward):
        self.w += alpha * self.log_grad(state, action) * reward


class REINFORCE(Solver):
    def __init__(self, sampler, policy):
        self.sampler = sampler
        self.Ns, self.Na = sampler.get_space()
        self.policy = policy

    def __policy2tabular(self, approx_policy):

        tabular_policy = np.zeros((self.Ns, self.Na))
        for state in range(self.Ns):
            tabular_policy[state] = approx_policy.eval(state=state, action=None)
        return tabular_policy

    def compute_pi(self, gamma, no_samples = 1000, alpha = 0.1, state0=None, action0=None, v0=None):

        tabular_policy = self.__policy2tabular(self.policy)

        for _ in range(no_samples):

            episode = self.sampler.get_v_episode(policy=tabular_policy,state0=state0)

            cum_returns = 0
            for step in reversed(episode):

                # compute the cumulative return
                cum_returns = step.reward + gamma * cum_returns
                self.policy.w += alpha*self.policy.log_grad(step.state, step.action)*cum_returns

            tabular_policy = self.__policy2tabular(self.policy)

        return tabular_policy





if __name__ == "__main__":

    Ns = 5

    random_walk = RandomWalk(Ns)
    model = random_walk.get_model()

    sampler = ModelSampler(model)
    batch = sampler.get_v_batch(5, policy=random_walk.get_random_policy(), state0=random_walk.start())

    for episode in batch:
        print(episode)
        print("")

    dp_solver = DP(model)

    #v1 = dp_solver.value_function_fix_point(gamma=0.99, policy=random_walk.get_random_policy())
    #v2 = dp_solver.value_function_linear_solve(gamma=0.99, policy=random_walk.get_random_policy())
    #q1 = dp_solver.q_function(gamma=0.99, policy=random_walk.get_random_policy())

    #print(v1)
    #print(v2)
    #print(q1)

    #pi3, q3, v3 = dp_solver.policy_iteration(gamma=0.99)
    #print(pi3)
    #print(q3)
    #print(v3)

    pi4, q4, v4 = dp_solver.value_iteration(gamma=0.99)
    print(pi4)
    #print(q4)
    #print(v4)

    #mc_solver = MC(sampler)
    #vv1 = mc_solver.value_function(gamma=0.99, no_samples=1000, state0=random_walk.start(), policy=random_walk.get_random_policy())
    #vv1 = mc_solver.value_function_grad(gamma=0.99, no_samples=1000, state0=random_walk.start(), policy=random_walk.get_random_policy(), alpha=0.01)
    #qq1 = mc_solver.q_function(gamma=0.99, no_samples=1000, policy=random_walk.get_random_policy())

    #print(vv1)
    #print(qq1)
    #print(pi)


    #ppi3, qq3 = mc_solver.q_control_on_policy(gamma=0.99, policy=random_walk.get_random_policy(), no_samples=1000)

    #print(qq3)
    #print(ppi3)


    print("yo")

    reinforce = REINFORCE(sampler, soft_max_policy_1D(tabular_basis(Ns+2, 2)))
    q = reinforce.compute_pi(gamma=0.99, no_samples=1000, state0=random_walk.start(), alpha=0.2)
    print(q)


    #reinforce = REINFORCE(sampler, soft_max_policy_1D(gaussian_basis(Ns + 2, 2, 3, 0.1)))
    #q = reinforce.compute_pi(gamma=0.99, no_samples=1000, state0=random_walk.start(), alpha=0.05)
    #print(q)

    #td_solver = TD_Forward(sampler, _lambda=0)
    #v4 = td_solver.value_function(gamma=0.99, state0=random_walk.start(), policy=random_walk.get_random_policy(), alpha=0.05)
    #print(v4)

    #td_f_solver = TD_Forward(sampler, _lambda=0.8)
    #v5 = td_f_solver.value_function(gamma=0.99, state0=random_walk.start(), policy=random_walk.get_random_policy(),
    #                                alpha=0.01, no_samples=1000)

    #print(v5)

    #td_b_solver = TD_Backward(sampler, _lambda=0.9)
    #v6 = td_b_solver.value_function(gamma=0.99, state0=random_walk.start(), policy=random_walk.get_random_policy(),
    #                                alpha=0.1, no_samples=1000)

    #ppi6, qq6 = td_b_solver.sarsa(gamma=0.99, state0=random_walk.start(), policy=random_walk.get_random_policy(),
    #                                alpha=0.1, no_samples=1000)

    #print(v6)
    #print(ppi6)
    #print(qq6)

    #ppi7, qq7 = td_b_solver.q_learning(gamma=0.99, state0=random_walk.start(), policy=random_walk.get_random_policy(),
    #                              alpha=0.1, no_samples=1000)

    # print(v6)
    #print(ppi7)
    #print(qq7)

