import torch
from torch import nn
import pytorch_lightning as pl

from objectives import CCALoss


class DCCAE(pl.LightningModule):
    
    def __init__(self, 
                 encoder1_layers, encoder2_layers,
                 decoder1_layers, decoder2_layers,
                 outdim,
                 lam=1,
                 learning_rate=1e-3,
                 reg_par=1e-4,
                 use_all_singular_values=False,
                 device='cuda'):
        super().__init__()
        # According to the original paper, it used fully-connectted nerual network
        # to implement the auto-encoder structure. But you could modifiy this 
        # setting, e.g. CNN-based auto-encoder.
        self.encoder1 = MLPNet(encoder1_layers)
        self.decoder1 = MLPNet(decoder1_layers)
        
        self.encoder2 = MLPNet(encoder2_layers)
        self.decoder2 = MLPNet(decoder2_layers)
        
        self.outdim = outdim
        self.learning_rate = learning_rate
        self.lam = lam
        self.reg_par = reg_par
        self.use_all_singular_values = use_all_singular_values
        
        self.cca_loss = CCALoss(self.outdim,
                                device=device,
                                use_all_singular_values=self.use_all_singular_values)
        
        self.recon_loss = nn.MSELoss()
        
    def forward(self, x1, x2):
        H1 = self.encoder1(x1)
        H2 = self.encoder2(x2)
        
        rx1 = self.decoder1(H1)
        rx2 = self.decoder2(H2)
        return H1, H2, rx1, rx2
    
    def training_step(self, batch, batch_idx):
        x1, x2, target = batch
        H1, H2, rx1, rx2 = self(x1, x2)
        
        corr_loss = self.cca_loss(H1, H2)
        recon_loss = self.lam * (self.recon_loss(rx1, x1) + self.recon_loss(rx2, x2) )
        
        self.log('train_loss', (corr_loss + recon_loss))
        
        return corr_loss + recon_loss
    
    def validation_step(self, batch, batch_idx):
        x1, x2, target = batch
        H1, H2, rx1, rx2 = self(x1, x2)
        
        corr_loss = self.cca_loss(H1, H2)
        recon_loss = self.lam * (self.recon_loss(rx1, x1) + self.recon_loss(rx2, x2) )
        
        self.log('valid_loss', (corr_loss + recon_loss))
        
        return corr_loss + recon_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.reg_par)


class DCCA(pl.LightningModule):
    """Deep CCA implementation via pytroch lightning.
    """
    
    def __init__(self, 
                 view1_layers, 
                 view2_layers,
                 outdim,
                 learning_rate=1e-3,
                 reg_par=1e-5,
                 use_all_singular_values=False,
                 device='cuda'):
        super().__init__()
        self.view1_model = MLPNet(view1_layers)
        self.view2_model = MLPNet(view2_layers)
        self.outdim = outdim
        self.use_all_singular_values = use_all_singular_values
        self.learning_rate = learning_rate
        self.reg_par = reg_par
        self.criterion = CCALoss(self.outdim,
                                 device=device,
                                 use_all_singular_values=self.use_all_singular_values)
        
    def forward(self, x1, x2):
        
        H1 = self.view1_model(x1)
        H2 = self.view2_model(x2)
        return H1, H2
    
    def training_step(self, batch, batch_idx):
        x1, x2, target = batch
        H1, H2 = self(x1, x2)
        loss = self.criterion(H1, H2)
        
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x1, x2, target = batch
        H1, H2 = self(x1, x2)
        loss = self.criterion(H1, H2)
        
        self.log('valid_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.reg_par)
        return optimizer
    
    
class MLPNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x