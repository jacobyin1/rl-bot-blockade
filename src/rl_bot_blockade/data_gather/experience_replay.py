import collections
import torch
import numpy as np


class ExperienceReplay:

    def __init__(self, buffer_size, environment):
        self.data = collections.deque(maxlen=buffer_size)
        self.env = environment

    def gather(self, qmodel, episodes=200):
        for episode in range(episodes):
            is_alive, observation = self.env.step(qmodel)
            self.data.extend(observation)
            while is_alive:
                is_alive, observation = self.env.step(qmodel)
                self.data.extend(observation)
            self.env.reset()

    def sample(self, batch_size=50, device='cuda'):
        idx = torch.randint(high=len(self.data), size=(batch_size,))
        batch = [self.data[i] for i in idx]
        s1, a, o, r, s2 = zip(*batch)
        s1_t = torch.stack(s1).type(torch.float32).to(device=device)
        a_t = torch.stack(a).type(torch.float32).to(device=device)
        o_t = torch.stack(o).type(torch.float32).to(device=device)
        r_t = torch.tensor(r).type(torch.float32).to(device=device)
        s2_t = torch.stack(s2).type(torch.float32).to(device=device)
        return s1_t, a_t, o_t, r_t, s2_t

    def get_episode(self, qmodel):
        obs = []
        is_alive, observation = self.env.step(qmodel)
        obs.extend(observation)
        while is_alive:
            is_alive, observation = self.env.step(qmodel)
            obs.extend(observation)
        self.env.reset()
        s1, _, _, _, _ = zip(*obs)
        indices = np.arange(len(s1) - 4, step=2)
        boards = np.zeros((indices.shape[0], self.env.board_length, self.env.board_length), dtype=np.uint8)
        for i, i_value in enumerate(indices):
            boards[i] = self.env.prettify_state(s1[i_value]).numpy()
        return boards[:boards.shape[0]]

