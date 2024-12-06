# Episodic semi-gradient n-step Sarsa(\lambda)
import numpy as np
import torch
from warehouse_agent import warehouse_agent
DTYPE = torch.float32

class QNetwork(torch.nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(6, 30)
        self.fc2 = torch.nn.Linear(30, 50)
        self.fc3 = torch.nn.Linear(50, 30)
        self.fc4 = torch.nn.Linear(30, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class SemiGradNStepSarsa(warehouse_agent):

    def __init__(self, rows, columns):
        super().__init__(rows, columns)
        self.len_action = len(self.A)
        self.q_net = QNetwork()
        self.num_episodes = 10000

    def generate_ini_state(self, terminate_x, terminate_y):
        while True:
            x = int(np.random.uniform(0, self.H))
            y = int(np.random.uniform(0, self.W))
            if x != terminate_x or y != terminate_y:
                return x, y

    def estimate_q_value(self, x, y, a):
        action_one_hot = torch.zeros(self.len_action).detach()
        action_one_hot[int(a)] = 1.0
        state = torch.tensor([x, y]).detach()
        input_tensor = torch.cat([state, action_one_hot], dim=0)
        q_value = self.q_net(input_tensor)
        return q_value

    def estimate_q_values(self, x, y):
        q_values = torch.zeros(self.len_action)
        for i in range(self.len_action):
            # action_one_hot = torch.zeros(self.len_action).detach()
            # action_one_hot[i] = 1.0
            # state = torch.tensor([x, y]).detach()
            # input_tensor = torch.cat([state, action_one_hot], dim=0)
            # q_value = self.q_net(input_tensor)
            q_value = self.estimate_q_value(x, y, i)
            q_values[i] = q_value
        return q_values


    def apply_transition_function(self, intend_action):
        c = np.random.uniform(0, 1)
        if c < self.P[0]:  # move intended direction
            return intend_action
        elif c < self.P[1]:  # turn Left
            # if intend_action == 0:
            #     real_action = 3
            # else:
            #     real_action = intend_action - 1
            return self.A[intend_action - 1]
        elif c < self.P[2]:  # turn Right
            # if intend_action == 3:
            #     real_action = 0
            # else:
            #     real_action = intend_action + 1
            return self.A[(intend_action + 1) % self.len_action]
        else:
            return None

    def generateAction_epsGreedy(self, x, y, eps):
        q_values = self.estimate_q_values(x, y)
        # print(q_values)
        max_q_values = torch.max(q_values)
        best_action_positions = (q_values == max_q_values).nonzero(as_tuple=True)[0]
        num_best_actions = best_action_positions.shape[0]
        c = np.random.uniform(0, 1)
        total_p = 0
        intend_action = 0
        for a in range(self.len_action):
            pi_a = (eps / self.len_action)
            term = (1 - eps) / num_best_actions
            if a in best_action_positions:
                total_p += term + pi_a
            else:
                total_p += pi_a
            if c <= total_p:
                intend_action = a
                break
        # transition function
        real_action = self.apply_transition_function(intend_action)
        return intend_action, real_action


    def computeNextPosition(self, row: int, column: int, action) -> tuple[int, int]:
        if action is None:  # no action, stay on the same state
            return row, column
        new_row = row
        new_column = column
        if action == 0:
            new_row = row - 1
        elif action == 1:
            new_column = column + 1
        elif action == 2:
            new_row = row + 1
        elif action == 3:
            new_column = column - 1
        # move outside of map
        if new_row < 0 or new_row > 4 or new_column < 0 or new_column > 4:
            return row, column
        # # hit wall
        # elif REWARD[new_row][new_column] == 'Wall':
        #     return row, column
        # regular move or stay same place
        else:  # action == 'None'
            return new_row, new_column
    def computeG(self, history, gamma, i):
        G = 0
        for tt in range(i):
            R_t = tt * 4 + 3
            G += (gamma ** tt) * history[R_t]
        return G

    def train(self, n: int, alpha: float, gamma: float, epsilon: float):
        '''

        :param n: n step
        :param alpha: learning rate
        :param gamma:
        :param epsilon: for epsilon-greedy policy
        :return:
        '''
        for terminate_x, terminate_y in self.terminal_states:  # multiple terminate states
            temp_reward = self.reward[terminate_x, terminate_y]
            self.reward[terminate_x, terminate_y] = 20  # set reward
            for episode in range(self.num_episodes):
                print(episode)
                stored_n_step = []  # len should equal 4n, [x_0, y_0, A_0, R_1,   x_1, y_1, A_1, R_2,  ...]
                x, y = self.generate_ini_state(terminate_x, terminate_y)
                stored_n_step.append(torch.tensor(x, dtype=DTYPE).item())
                stored_n_step.append(torch.tensor(y, dtype=DTYPE).item())
                intend_action, real_action = self.generateAction_epsGreedy(x, y, epsilon)
                stored_n_step.append(intend_action)
                T = np.inf
                t = 0
                tau = None
                while True:
                    print(t)
                    if t < T:
                        x, y = self.computeNextPosition(x, y, real_action)
                        stored_n_step.append(torch.tensor(self.reward[x, y], dtype=DTYPE).item())
                        stored_n_step.append(torch.tensor(x, dtype=DTYPE).item())
                        stored_n_step.append(torch.tensor(y, dtype=DTYPE).item())
                        # stored_n_step.append(self.reward[x, y])
                        # stored_n_step.append(x)
                        # stored_n_step.append(y)
                    if x == terminate_x and y == terminate_y:
                        T = t + 1
                    else:
                        intend_action, real_action = self.generateAction_epsGreedy(x, y, epsilon)
                        # stored_n_step.append(torch.tensor(intend_action, dtype=DTYPE))
                        stored_n_step.append(intend_action)
                    tau = t - n + 1
                    if tau >= 0:
                        stored_n_step = stored_n_step[-15:]
                        i = np.minimum(tau + n, T)
                        G = self.computeG(stored_n_step, gamma, int(i - tau))
                        if tau + n < T:
                            G += self.estimate_q_value(stored_n_step[-3], stored_n_step[-2], stored_n_step[-1])

                        action_one_hot = torch.zeros(self.len_action, dtype=DTYPE)
                        print(stored_n_step)
                        action_one_hot[int(stored_n_step[2])] = 1.0
                        state = torch.tensor([stored_n_step[0], stored_n_step[1]], dtype=DTYPE)
                        input_tensor = torch.cat([state, action_one_hot], dim=0)
                        q_value_tau_step = self.q_net(input_tensor)
                        q_value_tau_step.backward()

                        error = G - q_value_tau_step
                        # Manually update weights based on the TD error and learning rate
                        with torch.no_grad():
                            for param in self.q_net.parameters():
                                param += alpha * error * param.grad
                        self.q_net.zero_grad()
                    if tau == T - 1:
                        break
                    t += 1
            self.reward[terminate_x, terminate_y] = temp_reward
        return

if __name__ == '__main__':
    agent = SemiGradNStepSarsa(1, 1)
    alpha = 1e-4
    gamma = 0.925
    epsilon = 0.05
    n = 3
    agent.train(n, alpha, gamma, epsilon)
    print()
