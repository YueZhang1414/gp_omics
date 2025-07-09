import os
import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp
from pyro.contrib.gp.kernels import Kernel
from pyro.infer import TraceMeanField_ELBO
from pyro.optim import Adam
import pandas as pd


# ---------------- read data---------------- #

def load_feature_table(path):
    df = pd.read_csv(path, header=None)
    print(df.shape) 

    X = df.iloc[:, 1:101].values.astype(float)
    y = df.iloc[:, 101].values.astype(float)

    print(df.iloc[:, 101].head(10))


    print(X.shape)
    print(y.shape)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_adjacency_matrices(root_path):
    matrices = []
    for i in range(1, 101):
        folder = os.path.join(root_path, f"Sample_{i:03d}")
        file_path = os.path.join(folder, "weights.csv")
        mat = np.loadtxt(file_path, delimiter=",")
        matrices.append(torch.tensor(mat, dtype=torch.float32))
    return matrices


def load_node_features(root_path):
    features = []
    for i in range(1, 101):
        file_path = os.path.join(root_path, f"Sample_{i:03d}.csv")
        vals = []
        with open(file_path, "r") as f:
            lines = f.readlines()[2:102]
            for line in lines:
                token = line.strip().split(",")[0]
                try:
                    vals.append(float(token))
                except:
                    vals.append(0.0)
        features.append(torch.tensor(vals, dtype=torch.float32))
    return features


# ---------------- Graph Kernel + Hybrid Kernel ---------------- #

class PropagationKernel:
    def __init__(self, t_max=5, sigma=1.0):
        self.t_max = t_max
        self.sigma = sigma

    def compute_graph_embedding(self, A, node_features):
        F_ = node_features.clone()
        for _ in range(self.t_max):
            D_inv = 1.0 / (A.sum(dim=1) + 1e-8)
            F_ = D_inv.unsqueeze(1) * torch.matmul(A, F_.unsqueeze(1))
            F_ = F_.squeeze()
        return F_

    def graph_similarity(self, g1, g2):
        dist_sq = torch.norm(g1 - g2, p=2) ** 2
        return torch.exp(-dist_sq / (2.0 * self.sigma ** 2))


class HybridKernel(Kernel):
    def __init__(self, X, As, Fs, theta, lam):
        super().__init__(input_dim=X.shape[1])
        self.X = X
        self.As = As
        self.Fs = Fs
        self.theta = theta
        self.lam = lam

    def forward(self, x1, x2, diag=False):
        n1, n2 = x1.shape[0], x2.shape[0]
        K_mat = torch.zeros(n1, n2)

        pk = PropagationKernel(t_max=5, sigma=self.theta[1].item())

        for i in range(n1):
            for j in range(n2):
                idx1 = (self.X == x1[i]).all(dim=1).nonzero().item()
                idx2 = (self.X == x2[j]).all(dim=1).nonzero().item()

                k_genes = torch.exp(-torch.sum((x1[i] - x2[j]) ** 2) / self.theta[0])
                g1 = pk.compute_graph_embedding(self.As[idx1], self.Fs[idx1])
                g2 = pk.compute_graph_embedding(self.As[idx2], self.Fs[idx2])
                k_graph = pk.graph_similarity(g1, g2)

                K_mat[i, j] = k_genes + self.lam * k_graph

        if diag:
            return torch.diag(K_mat)
        return K_mat


# ---------------- Main: Sparse GP ---------------- #

# load data
X_all, y_all = load_feature_table("/Users/zhangyue/Library/Mobile Documents/com~apple~CloudDocs/ZhangY/mock_data/feature_table_alldata.csv")
As = load_adjacency_matrices("/Users/zhangyue/Library/Mobile Documents/com~apple~CloudDocs/ZhangY/mock_data/adj_matrix")
Fs = load_node_features("/Users/zhangyue/Library/Mobile Documents/com~apple~CloudDocs/ZhangY/mock_data/node_feature")
print(X_all.shape)
print(y_all.unsqueeze(1).shape)


theta = torch.tensor([0.1, 0.1])
lam = 1.0

kernel = HybridKernel(X_all, As, Fs, theta, lam)
Xu = X_all[::10]


#model = gp.models.SparseGPRegression(X_all, y_all.float().unsqueeze(1), kernel, Xu)
model = gp.models.SparseGPRegression(X_all, y_all.unsqueeze(0), kernel, Xu)


optimizer = Adam({"lr": 0.1})

# ------------------ train ------------------

num_steps = 100
for i in range(num_steps):
    optimizer.zero_grad()
    loss = -model.model.log_prob(model.y)
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print(f"Iter {i+1} Loss: {loss.item():.4f}")

print("Training Done.")

# ------------------ test ------------------

with torch.no_grad():
    f_loc, f_var = model(X_all, full_cov=False, noiseless=False)
    pred_prob = torch.sigmoid(f_loc.squeeze())
    pred_label = (pred_prob > 0.5).float()
    acc = (pred_label == y_all).float().mean()
    print(f"Train Accuracy (interpreted as classification): {acc.item() * 100:.2f}%")