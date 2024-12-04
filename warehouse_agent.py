import numpy as np
from itertools import product

class warehouse_agent:
    def __init__(self, rows, columns):
        # Action = ['Up' 0, 'Right' 1, 'Down' 2, 'Left' 3], clockwise
        self.A = np.array([0, 1, 2, 3])
        # Transition function
        self.P = [0.7, 0.12, 0.12, 0.06]
        self.reward = None
        self.terminal_states = None
        self.policies = None
        self.q_values = None
        self.num_aisles = None 
        self.H = None  # environment height
        self.W = None  # environment width

    # def generate_environment(self, rows, columns)
        # initial gridworld
        H = 3 * (rows + 1)
        W = 3 * columns + 2
        reward = -np.ones((H, W))
        reward[:, np.arange(1, W-3, 3)] = -9
        reward[:, np.arange(3, W, 3)] = -9
        reward[np.arange(0, H, 4)] = -1
        self.reward = reward
        self.num_aisles = rows * columns
        # number of aisles, number of states, number of actions
        self.q_values = np.random.rand(self.num_aisles, H*W, 4)
        terminal_states_x = np.arange(2, H, 4)
        terminal_states_y = np.arange(2, W, 3)  
        self.terminal_states = np.array(list(product(terminal_states_x, terminal_states_y)))
        self.H = H
        self.W = W

    # generate item that need to be delivered, sort terminal states the agent need to visit based on their distance to [0, 0] where the agent start,
    # return sorted terminal states and their sequency
    def generate_items(self, num_items):
        if num_items > self.num_aisles:
            print(f'bad number items, make it <= {self.num_aisles}')
            return None
        random_numbers = np.random.choice(np.arange(0, self.num_aisles, 1), size=num_items, replace=False)
        items_terminal_states = self.terminal_states[random_numbers]
        sequence = np.argsort(np.linalg.norm(items_terminal_states, axis=1))
        return items_terminal_states[sequence], np.sort(random_numbers)
    
if __name__ == '__main__':
    agent = warehouse_agent(2, 3)
    agent.generate_items(4)

