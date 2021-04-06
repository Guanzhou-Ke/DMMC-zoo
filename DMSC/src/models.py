import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.nn import init
import math
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from munkres import Munkres

from data import EYaleBDataset


class ConvAE(nn.Module):
    def __init__(self, kernel_size, n_hidden, strides, paddings, reg_constant1:int = 1.0, re_constant2:int = 1.0, 
                 batch_size:int = 100, reg:int = None, denoise:bool = False, model_path:str = None, restore_path:str = None, 
                 logs_path:str = './logs', num_modalities:int = 2):
        super().__init__()
        self._encoders = nn.ModuleList()
        self._decoders = nn.ModuleList()
        
        self.bs = batch_size
        
        self.reg_constant1 = reg_constant1
        self.re_constant2 = re_constant2
        
        
        for i in range(num_modalities):
            self._encoders.append(self._make_encoder(kernel_size[0], n_hidden[0], strides[0], paddings[0]))
            self._decoders.append(self._make_decoder(kernel_size[1], n_hidden[1], strides[1], paddings[1]))
            
        self.apply(self.weights_init('xavier'))
        
        Coef = 1.0e-4*torch.ones(self.bs, self.bs)
        Coef -= torch.diag(torch.diag(Coef))
        self.Coef = nn.Parameter(Coef)
        
        self.flatten = nn.Flatten()
        self.mseloss = nn.MSELoss()

    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        return init_fun
    
    def _make_encoder(self, kernel_size, n_hidden, strides, paddings):
        modules = []
        for i in range(len(n_hidden)):
            if i == 0:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=n_hidden[0], kernel_size=kernel_size[0], 
                                         stride=strides[0], padding=paddings[0]),
                        nn.ReLU(True))
                    )
            else:
                modules.append(nn.Sequential(
                        nn.Conv2d(in_channels=n_hidden[i-1], out_channels=n_hidden[i], kernel_size=kernel_size[i], 
                                         stride=strides[i], padding=paddings[i]),
                        nn.ReLU(True))
                    )
        return nn.Sequential(*modules)
    
    def _make_decoder(self, kernel_size, n_hidden, strides, paddings):
        modules = []
        for i in range(len(n_hidden)):
            if i == 0:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=30, out_channels=n_hidden[0], kernel_size=kernel_size[0], 
                                         stride=strides[0], padding=paddings[0]),
                        nn.ReLU(True))
                    )
            else:
                modules.append(nn.Sequential(
                        nn.ConvTranspose2d(in_channels=n_hidden[i-1], out_channels=n_hidden[i], kernel_size=kernel_size[i], 
                                         stride=strides[i], padding=paddings[i]),
                        nn.ReLU(True))
                              )
        return nn.Sequential(*modules)
    
    def forward(self, xs):
        zs = []
        latents = []
        recons = []
        for i, x in enumerate(xs):
            z = self._encoders[i](x)
            latent_shape = z.shape
            z = self.flatten(z)
            zs.append(z)
            z_c = torch.matmul(self.Coef, z)
            latents.append(z_c.view(latent_shape))
        for i, latent in enumerate(latents):
            recons.append(self._decoders[i](latent))
            
        return recons, zs
    
    def get_loss(self, xs):
        recons, zs = self(xs)
        reg_loss = torch.sum(torch.pow(self.Coef, 2.0))
        recon_loss = 0.6*self.mseloss(recons[0], xs[0])
        self_express_loss = 0.3*self.mseloss(zs[0], torch.matmul(self.Coef, zs[0]))
        for i in range(len(xs), 1):
            recon_loss += 0.1*self.mseloss(recons[i], xs[i])
            self_express_loss += 0.05*self.mseloss(zs[i], torch.matmul(self.Coef, zs[i]))
        loss = recon_loss + self.reg_constant1*reg_loss + self.re_constant2*self_express_loss
        return loss
    
def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    # calc error rate.
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)

    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2   

def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp

def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs,0)
    for i in range(N):
        Cabs[:,i]= Cabs[:,i] / (Cabs[ind[0,i],i] + 1e-6)
    Cksym = Cabs + Cabs.T;
    return Cksym

def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
#     U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]), ncv=alpha*r)
    U, S, _ = svds(C,r)
    U = U[:,::-1]    
    S = np.sqrt(S[::-1])
    S = np.diag(S)    
    U = U.dot(S)    
    U = normalize(U, norm='l2', axis = 1)       
    Z = U.dot(U.T)
    Z = Z * (Z>0)    
    L = np.abs(Z ** alpha)
    L = L/L.max()   
    L = 0.5 * (L + L.T)    
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    nmi = normalized_mutual_info_score(gt_s[:], c_x[:])
    ari = adjusted_rand_score(gt_s[:], c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate ,nmi, ari

def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C,axis=0)         
    W = np.diag(1.0/W)
    L = W.dot(C)    
    return L

def train_model(model, data_loader, batch_size, learning_rate, num_class):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alpha = max(0.4 - (num_class-1)/10 * 0.1, 0.1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    max_step =1000#500 + num_class*25# 100+num_class*20
    display_step = 10
    lr = 1.0e-3
    # fine-tune network
    epoch = 0
    label = None
    print("Training begin")
    while epoch < max_step:
        epoch = epoch + 1
        train_loss = 0.
        
        for data in data_loader:
            Xs, y = data[0], data[1].to(device)
            Xs = [d.to(device) for d in data[0]]
            if epoch <= 1:
                label = y.detach().cpu().numpy()
            loss = model.get_loss(Xs)
            loss.backward()
            train_loss += loss.detach().cpu().item()
            optimizer.step()
        Coef = model.Coef.detach().cpu().numpy()
        

        if epoch % display_step == 0:
            print(f"epoch: {epoch} loss: {train_loss/batch_size:.5f}" )
            Coef = thrC(Coef,alpha)
            print(np.diag(Coef))
            y_x, _ = post_proC(Coef, label.max())
            missrate_x,nmi,ari = err_rate(label, y_x)
            acc = 1 - missrate_x
            print(f"accuracy: {acc:.4f} NMI: {nmi:.4f} ARI: {ari:.4f}")


    print("%d subjects:" % num_class)    
    print("ACC: %.4f%%" % (acc*100))
    print("NMI: %.4f%%" % (nmi*100))
    print("ARI: %.4f%%" % (ari*100))

    return acc, Coef,nmi,ari


if __name__ == '__main__':
    num_class = 38
    batch_size = 2424
    reg1 = 1.0
    reg2 = 1.0 * 10 ** (num_class / 10.0 - 3.0)
    lr = 1.0e-3
    
    kernel_size = [
        [5, 3, 3],
        [3, 3, 5, 4]
    ]
    n_hidden = [
        [10, 20, 30],
        [30, 20, 10, 1]
    ]
    strides = [
        [2, 1, 2],
        [2, 1, 2, 1]
    ]
    paddings = [
        [1, 1, 0],
        [0, 1, 1, 1]
    ]

    datapath = '../datasets/EYB_fc.mat'

    eyb_dataset = EYaleBDataset(datapath, transform=ToTensor())
    train_loader = DataLoader(eyb_dataset, batch_size=batch_size, shuffle=True)
    
    model = ConvAE(kernel_size, n_hidden, strides, paddings, num_modalities=5, batch_size=batch_size, 
                   reg_constant1=reg1, re_constant2=reg2)

    train_model(model, data_loader=train_loader, batch_size=batch_size, learning_rate=lr, num_class=num_class)