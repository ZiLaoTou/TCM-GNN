import math
import torch
def one_hot(site, site_num, device):
    sub = site.shape[0]
    D = torch.zeros((sub, site_num))
    for i in range(sub):
        D[i, site[i].long()] = 1
    D = D.to(device)
    return D

def centering(X, device):
    sub = X.shape[0]
    H = torch.eye(sub) - (1 / sub) * torch.ones((sub, sub))
    H = H.to(device)
    return torch.mm(X, H)

def rbf(X, sigma=None):
    GX = torch.mm(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / sigma / sigma
    KX = torch.exp(KX)
    return KX

def HSIC(data, site, site_num, device, kernel="linear"):
    sub = data.shape[0]
    D = one_hot(site, site_num, device)
    if kernel == "linear":
        K = torch.mm(data, data.T)
    elif kernel == "rbf":
        K = rbf(data)
    L = torch.mm(D, D.T)
    loss = torch.trace(torch.mm(centering(K, device), centering(L, device)))/((sub-1)**2)
    return loss