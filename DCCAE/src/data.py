from torch.utils.data import Dataset

class NoisyMnistDataset(Dataset):
    """Custom a noisy mnist dataset.
    """
    
    def __init__(self, data):
        """
        data: must be like [x1 for train, x2 for train, and training label]
        """
        self.view1 = data[0]
        self.view2 = data[1]
        self.target = data[2]
        assert (len(self.view1) - len(self.view2) + len(self.target)) == len(self.view1)
    
    def __len__(self):
        return len(self.view1)
    
    def __getitem__(self, index):
        target = self.target[index]
        view1 = self.view1[index]
        view2 = self.view2[index]
        return view1, view2, target