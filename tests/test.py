import matplotlib.pyplot as plt
from data_gather import environment, experience_replay, simulator
from model import qmodel
import torch

env = environment.Environment(20, (15, 15), (5, 5))
replay = experience_replay.ExperienceReplay(1000, env)
qmodel = qmodel.QModel(lambda x: 0, [418, 418, 418, 418])
qmodel.load_model()
replay.gather(qmodel, 1)
a, b, c, d, e = replay.sample(batch_size=1)
# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
t = (a, b, c, d, e)
print(qmodel.loss(t, gamma=0.99))
