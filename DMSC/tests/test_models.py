import unittest

import torch
from torch import nn

from models import ConvAE


class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Prepare some paramerters.
        """
        # Common parameters.
        self.num_class = 38
        self.batch_size = 2424
        self.reg1 = 1.0
        self.reg2 = 1.0 * 10 ** (self.num_class / 10.0 - 3.0)
        self.lr = 1.0e-3

        self.kernel_size = [
            [5, 3, 3],
            [3, 3, 5, 4]
        ]
        self.n_hidden = [
            [10, 20, 30],
            [30, 20, 10, 1]
        ]
        self.strides = [
            [2, 1, 2],
            [2, 1, 2, 1]
        ]
        self.paddings = [
            [1, 1, 0],
            [0, 1, 1, 1]
        ]
        
        self.test_input = [torch.rand(self.batch_size, 1, 32, 32) for i in range(5)]
        
    
    def test_model(self):
        model = ConvAE(self.kernel_size, self.n_hidden, self.strides, self.paddings, num_modalities=5, batch_size=self.batch_size, 
                   reg_constant1=self.reg1, re_constant2=self.reg2)
        self.assertIsInstance(model, nn.Module)
        output = model(self.test_input)
        self.assertEqual(len(output), 2)
        self.assertEqual(len(output[0]), 5)
