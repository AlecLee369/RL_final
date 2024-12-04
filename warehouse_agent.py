import numpy as np
from itertools import product

class warehouse_agent:
    def __init__(self):
        self.environment = None
        # Action = ['Up' 0, 'Right' 1, 'Down' 2, 'Left' 3], clockwise
        self.A = np.array([0, 1, 2, 3])
        # Transition function
        self.P = [0.7, 0.12, 0.12, 0.06]
        self.reward = None
        self.terminal_states = None
        self.policies = None
        self.q_values = None

    def generate_reward(self, rows, columns):
        H = 3 * (rows + 1)
        W = 3 * columns + 2
        reward = -np.ones((H, W))
        reward[:, np.arange(1, W-3, 3)] = -9
        reward[:, np.arange(3, W, 3)] = -9
        reward[np.arange(0, H, 4)] = -1
        self.reward = reward

        num_aisles = rows * columns
        # number of aisles, number of states, number of actions
        self.q_values = np.random.rand(num_aisles, H*W, 4)
        terminal_states_x = np.arange(2, H, 4)
        terminal_states_y = np.arange(2, W, 3)  
        self.terminal_states = list(product(terminal_states_x, terminal_states_y))


if __name__ == '__main__':
    agent = warehouse_agent()
    agent.generate_reward(2, 3)
    print(agent.reward)

