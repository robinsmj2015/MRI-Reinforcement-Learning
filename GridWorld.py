import numpy as np
from prettytable import PrettyTable
from typing import List

class GridWorld:
    def __init__(self, nrows: int, ncols: int, gamma: float, pos_reward: int, neg_reward: int, reward_type: str, policy: str, terminal_states: List[str]) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.rows = np.arange(nrows)
        self.cols = np.arange(ncols)
        self.gamma = gamma
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward
        self.terminal_states = terminal_states
        self.values = {}
        for i in self.rows:
            for j in self.cols:
                self.values[str(i) + "_" + str(j)] = 0

        self.policy = policy
        self.reward_type = reward_type
        self.table = None

    def update(self, iteration: int) -> None:
        values_copy = self.values.copy()
        for i in self.rows:
            for j in self.cols:
                current_state = str(i) + "_" + str(j)
                if current_state in self.terminal_states:
                    continue
                u = str(i - 1) + "_" + str(j) if i > 0 else current_state
                d = str(i + 1) + "_" + str(j) if i < self.nrows - 1 else current_state
                l = str(i) + "_" + str(j - 1) if j > 0 else current_state
                r = str(i) + "_" + str(j + 1) if j < self.ncols - 1 else current_state

                if self.policy == "RAND":
                    discounted_sum = 0
                    for s_prime in [u, d, l, r]:
                        discounted_sum += self.values[s_prime] * self.gamma

                    if self.reward_type == "JUST_NEG":
                        # every step is negative reward
                        discounted_sum += self.neg_reward * 4

                    elif self.reward_type == "BOTH":
                        # positive reward when moving closer to terminal state 0
                        # negative reward when moving away from terminal state 0
                        if i == int(self.terminal_states[0][0]) or j == int(self.terminal_states[0][2]):
                            discounted_sum += self.neg_reward * 3
                            discounted_sum += self.pos_reward

                    else:
                        assert False, "Specify other reward type!"
                    discounted_sum /= 4

                elif self.policy == "GREEDY":
                    if self.reward_type == "JUST_NEG":
                        discounted_sum = max([self.values[s_prime] * self.gamma for s_prime in [u, d, l, r]]) - 1

                    elif self.reward_type == "BOTH":
                        qs = []
                        for s_prime in [u, d, l, r]:
                            r = -1
                            if abs(int(current_state[0]) - int(self.terminal_states[0][0])) > abs(int(s_prime[0]) - int(self.terminal_states[0][0])):
                                r = 1
                            if abs(int(current_state[-1]) - int(self.terminal_states[0][-1])) > abs(int(s_prime[-1]) - int(self.terminal_states[0][-1])):
                                r = 1
                            qs.append(self.values[s_prime] * self.gamma + r)
                        discounted_sum = max(qs)

                values_copy[current_state] = discounted_sum


        self.values = values_copy
        print(f"\nIteration {iteration} complete!\n")

    def get_row(self, i: int) -> List[str]:
        return [f'{self.values[str(i) + "_" + str(j)]:.2f}' for j in self.cols]

    def display_table(self, iteration: int) -> None:
        self.table = PrettyTable()
        self.table.field_names = ["After", "iteration", "number:", str(iteration)]
        for i in self.rows:
            self.table.add_row(self.get_row(i))
        print(self.table)


gw = GridWorld(nrows=4, ncols=4, gamma=0.99, pos_reward=1, neg_reward=-1, reward_type="BOTH", policy="GREEDY", terminal_states=["0_0"])
num_iterations = 100
display_period = 10
gw.display_table(0)
for num in range(1, num_iterations + 1):
    gw.update(num)
    if num % display_period == 0:
        gw.display_table(num)
