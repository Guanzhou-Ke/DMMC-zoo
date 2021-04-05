import os
import unittest

from data import NoisyMnistDataset
from utils import load_noisy_mnist


class TestNoisyMnist(unittest.TestCase):
    
    def setUp(self):
        self.DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
        
    def test_load_data(self):
        train_set, valid_set, test_set = load_noisy_mnist(self.DATA_ROOT)
        self.assertIsInstance(train_set, NoisyMnistDataset)
        self.assertIsInstance(valid_set, NoisyMnistDataset)
        self.assertIsInstance(test_set, NoisyMnistDataset)
        self.assertEqual(len(train_set), 5*1e4)
        self.assertEqual(len(valid_set), 1*1e4)
        self.assertEqual(len(test_set), 1*1e4)