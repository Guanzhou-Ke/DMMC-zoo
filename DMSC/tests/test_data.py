import os
import unittest

from torchvision.transforms import ToTensor

from data import EYaleBDataset


class TestEYaleBDataset(unittest.TestCase):
    
    def setUp(self):
        self.DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'EYB_fc.mat')
        
    def test_load_data(self):
        eyb_dataset = EYaleBDataset(self.DATA_ROOT, transform=ToTensor())
        self.assertIsInstance(eyb_dataset, EYaleBDataset)
        self.assertEqual(len(eyb_dataset), 2424)
        self.assertEqual(len(eyb_dataset[0][0]), 5)
        self.assertEqual(eyb_dataset[0][0][0].shape, (1, 32, 32))