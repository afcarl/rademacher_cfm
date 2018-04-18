import torch

from torch import nn


class NuclearEmbedding(nn.Module):
    def __init__(self, n, max_rank=100, threshold=1.0, min_rank=2):
        super().__init__()
        self.u = nn.Embedding(n, max_rank)
        self.s = nn.Parameter(torch.ones(max_rank).uniform_(0, 1))
        self.vt = nn.Embedding(n, max_rank)
        self.threshold = threshold
        self.min_rank = min_rank
        # Initialize close orthogonal
        for _ in range(100):
            self._update_u_v(truncate=False)

    def linear(self, x):
        # Right-matrix multiplication, where we take
        # care not to fully materialize the whole matrix
        u, s, vt = self.u.weight.data, self.s.data, self.vt.weight.data
        v = vt.t()
        vx = torch.matmul(v, x)
        svx = torch.matmul(torch.diag(s), vx)
        usvx = torch.matmul(u, svx)
        return usvx

    def adjoint(self, x):
        # Left-matrix multiplication, where we take
        # care not to fully materialize the whole matrix
        u, s, vt = self.u.weight.data, self.s.data, self.vt.weight.data
        utx = torch.matmul(u.t(), x)
        sutx = torch.matmul(torch.diag(s), utx)
        vtsutx = torch.matmul(vt, sutx)
        return vtsutx

    def _update_u_v(self, truncate=False):
        # Our big interaction matrix is M = u . s . v,
        # but we don't materialize it directly to keep mem low.
        # But note: these values are not orthogonalized,
        # because we just took a small step in SGD
        u, s, vt = self.u.weight.data, self.s.data, self.vt.weight.data

        # Do a single iteration of qr decomp;
        # since we recompute this for every mini-batch
        # of data, this is done frequently and our eigenvectors
        # change slowly, so we don't repeat the following lines
        # more than once per batch
        X_U = self.adjoint(u)
        V, _ = torch.qr(X_U)
        X_V = self.linear(V)
        U, R = torch.qr(X_V)
        S = torch.abs(torch.diag(R))

        # Replace old values with orthogonal ones
        u[...] = U[...]
        vt[...] = V.t()[...]
        s[...] = S[...]

        # Compute index where we cross the threshold
        # but just up to the singular value threshold
        if truncate:
            idxs = torch.nonzero(torch.cumsum(S, 0) > self.threshold)
            if len(idxs) > 0:
                rank = max(int(idxs[0][0]), self.min_rank)
                u[:, rank:] = 0.0
                vt[:, rank:] = 0.0
                s[rank:] = 0.0
                print(f"rank: {rank}")

    def forward(self, i, j):
        self._update_u_v()
        u = self.u(i)
        s = self.s
        vt = self.vt(j)
        r = torch.sum(u * s[None, :] * vt, dim=1)
        return r
