from torch_ant import AntTorch
import numpy as np

env = AntTorch()
# qp = env.reset()
a = np.zeros(env.action_size)

for i in range(10):
    qp = env.step(a)
