import unittest

import torch
from torch import nn
import pytorch_lightning as pl

from models import DCCA, DCCAE, MLPNet


class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Prepare some paramerters.
        """
        # Common parameters.
        self.batch_size = 16
        self.indim, self.outdim = 784, 10
        self.test_input = torch.randn(self.batch_size, self.indim)
        self.device = 'cpu'

        # For DCCA/MLPNet
        self.view1_layers = [self.indim, 16, self.outdim]
        self.view2_layers = [self.indim, 16, self.outdim]
        
        # For DCCAE
        self.encoder1_layers = [self.indim, 16, self.outdim]
        self.encoder2_layers = [self.indim, 16, self.outdim]
        
        self.decoder1_layers = [self.outdim, 16, self.indim]
        self.decoder2_layers = [self.outdim, 16, self.indim]
        
    
    def test_mlpnet(self):
        model = MLPNet(self.view1_layers)
        self.assertIsInstance(model, nn.Module)
        output = model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.outdim))
        
    def test_dcca(self):
        model = DCCA(self.view1_layers, self.view1_layers, 
                     outdim=self.outdim, device=self.device)
        self.assertIsInstance(model, pl.LightningModule)
        H1, H2 = model(self.test_input, self.test_input)
        self.assertEqual(H1.shape, (self.batch_size, self.outdim))
        self.assertEqual(H2.shape, (self.batch_size, self.outdim))
        
    def test_dccae(self):
        model = DCCAE(self.encoder1_layers, self.encoder2_layers, 
                      self.decoder1_layers, self.decoder2_layers,
                      outdim=self.outdim,
                      device=self.device)
        self.assertIsInstance(model, pl.LightningModule)
        H1, H2, rx1, rx2 = model(self.test_input, self.test_input)
        self.assertEqual(H1.shape, (self.batch_size, self.outdim))
        self.assertEqual(H2.shape, (self.batch_size, self.outdim))
        self.assertEqual(rx1.shape, (self.batch_size, self.indim))
        self.assertEqual(rx2.shape, (self.batch_size, self.indim))