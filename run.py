import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.parameter import Parameter
from torch_trainer.trainer import Trainer
from torch_trainer.callbacks import auc_callback
from torch.autograd import Variable
from scipy.special import expit


def t(x):
    return x.transpose(0, 1)


def frobenius(m):
    return (m ** 2.0).sum()


class RCFM(nn.Module):
    def __init__(self, n_usr, n_itm, k, l0=1.0, l1=1.0):
        super().__init__()
        self.busr = nn.Embedding(n_usr, 1)
        self.bitm = nn.Embedding(n_itm, 1)
        self.usr = nn.Embedding(n_usr, k)
        self.itm = nn.Embedding(n_itm, k)
        self.c = Parameter(torch.randn(1))
        self.k = k
        self.l0 = l0
        self.l1 = l1

    def forward(self, i, j, y):
        bi = self.busr(i).squeeze()
        bj = self.bitm(j).squeeze()
        vi = self.usr(i)
        vj = self.itm(j)
        logodds = self.c + bi + bj + (vi * vj).sum(dim=1)
        return logodds

    def approx_trace(self):
        # Use Hutchinson's trick
        # http://blog.shakirm.com/2015/09/
        # machine-learning-trick-of-the-day-3-hutchinsons-trick/
        z = Variable(torch.randn(self.k))
        u = self.usr.weight
        v = self.itm.weight
        zu = z[None, :] @ t(u)
        vz = v @ z[:, None]
        trace = zu @ vz
        return torch.abs(trace)

    def approx_orth(self, n_samples=1024):
        u = self.usr.weight
        un = u.size()[0]
        v = self.itm.weight
        vn = v.size()[0]
        xu = torch.LongTensor(n_samples).random_(0, un)
        xv = torch.LongTensor(n_samples).random_(0, vn)
        iu = Variable((xu[None, :] == xu[:, None]).float())
        iv = Variable((xv[None, :] == xv[:, None]).float())
        us = u[xu]
        vs = u[xu]
        loss = (frobenius(iu - us @ t(us)) +
                frobenius(iv - vs @ t(vs)) +
                frobenius(us @ t(vs)))
        return loss

    def bias_l2(self):
        lbu = (self.busr.weight ** 2.0).sum()
        lbi = (self.bitm.weight ** 2.0).sum()
        return lbu + lbi

    def likelihood(self, pred, y):
        return F.binary_cross_entropy_with_logits(pred, y)

    def loss(self, pred, i, j, y):
        # at = sum(self.approx_trace() for _ in range(int(n)))
        at = self.approx_orth()
        bl = self.bias_l2()
        lh = self.likelihood(pred, y)
        return lh + self.l1 * bl + self.l0 * at


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


model = RCFM(n_usr, n_itm, k)
optim = Adam(model.parameters(), lr=5e-2)

callbacks = {'auc': auc_callback}
trainer = Trainer(model, optim, batchsize=512, window=32,
                  callbacks=callbacks, seed=42)
for e in range(1000):
    trainer.fit(i_usr, i_itm, y)
    print(e)
    print(model.usr.weight.data[0])
    print(model.usr.weight.data[1])
    print(model.usr.weight.data[0] /
          model.usr.weight.data[1])
