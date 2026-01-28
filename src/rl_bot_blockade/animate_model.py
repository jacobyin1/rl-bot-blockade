import matplotlib.pyplot as plt

from data_gather import environment, experience_replay, simulator
from model import qmodel

env = environment.Environment(20, (15, 15), (5, 5))
replay = experience_replay.ExperienceReplay(10000, env)
qmodel = qmodel.QModel(lambda x: 0, [418, 418, 418, 418])
qmodel.load_model()

sim_states = replay.get_episode(qmodel)
fig, ax = plt.subplots(figsize=(10, 10))
anim = simulator.simulate(fig, ax, sim_states)
plt.show()
anim.save('animation.mp4')

