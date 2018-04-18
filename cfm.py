import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_trainer.trainer import Trainer
from torch_trainer.callbacks import auc_callback
from torch.optim import Adam
from torch.optim import SGD

from scipy.special import expit

from nuclear_embedding import NuclearEmbedding


class CFM(nn.Module):
    def __init__(self, n, k, threshold=1.0):
        super().__init__()
        self.intx = NuclearEmbedding(n, k * 10, threshold=threshold)
        self.bias = nn.Embedding(n, 1)
        self.c = Parameter(torch.randn(1))
        self.k = k

    def forward(self, i, j, y):
        bi = self.bias(i).squeeze()
        bj = self.bias(j).squeeze()
        mij = self.intx(i, j)
        logodds = self.c + bi + bj + mij
        return logodds

    def loss(self, pred, i, j, y):
        llh = F.binary_cross_entropy_with_logits(pred, y)
        return llh


k = 10
n_dim = 4
n_usr = 5000
n_itm = 5000
n_rat = int(1e5)

# Generate fake data
np.random.seed(42)
b_usr = np.random.randn(n_usr)
b_itm = np.random.randn(n_itm)
v_usr = np.random.randn(n_usr, n_dim)
v_itm = np.random.randn(n_itm, n_dim)
i_usr = np.random.randint(0, n_usr, size=n_rat)
i_itm = np.random.randint(0, n_itm, size=n_rat)
y = expit(b_usr[i_usr] + b_itm[i_itm] +
          np.sum(v_usr[i_usr, :] * v_itm[i_itm, :], axis=1))
y = np.rint(y)
print(b_usr)
print(v_itm)
print(y)


model = CFM(n_usr + n_itm, k, threshold=1e5)
# optim = Adam(model.parameters(), lr=1e-4)
optim = SGD(model.parameters(), lr=1e-2)

callbacks = {'auc': auc_callback}
trainer = Trainer(model, optim, batchsize=512, window=32,
                  callbacks=callbacks, seed=42)
for e in range(1000):
    trainer.fit(i_usr, n_usr + i_itm, y)
    print(e)
    print(model.intx.u.weight.data[0])
    print(model.intx.vt.weight.data[0])
