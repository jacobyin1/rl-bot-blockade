import time

from torch import optim

from data_gather import environment, experience_replay, simulator
import matplotlib.pyplot as plt
from model import qmodel


# f(0) = 1, f(15) = 0.05, f(30) = 0.001
def e_func(n):
    # 10 ** (-0.001 * n)
    return 0.05


env = environment.Environment(20, (15, 15), (5, 5))
replay = experience_replay.ExperienceReplay(20000, env)
qmodel = qmodel.QModel(e_func, [418, 418, 418, 418], device='cuda')
qmodel.load_model()

optimizer = optim.Adam(qmodel.parameters(), lr=0.0001)

for epoch in range(10000):
    replay.gather(qmodel, episodes=100)

    # if epoch % 10 == 0:
    #     sim_states = replay.get_episode(qmodel)
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     anim = simulator.simulate(fig, ax, sim_states)
    #     plt.show()

    if epoch % 10 == 0:
        qmodel.save_model()
        print(epoch)

    for n_train in range(20):
        data = replay.sample(batch_size=50, device='cuda')
        qmodel.train(data, optimizer)
    qmodel.update_target()

fig, ax = plt.subplots(figsize=(10, 10))
qmodel.graph_loss(ax)
plt.show()
qmodel.save_model()





