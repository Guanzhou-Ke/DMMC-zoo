import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset


class EYaleBDataset(Dataset):
    def __init__(self, root:str = './data/EYB.mat', transform=None):
        """
        0: face, 1: left eye, 2: nose, 3: mouth, 4:right eye
        """
        raw_data = sio.loadmat(root)
        self.num_modalities = int(raw_data['num_modalities'])
        self.data = []
        self.labels = np.array(raw_data['Label']).reshape(-1, 1)
        self.transform = transform
        for modality in range(self.num_modalities):
            I = []
            img = raw_data[f'modality_{modality}']
            for i in range(img.shape[1]):
                temp = np.reshape(img[:, i], (32,32))
                I.append(temp)
            self.data.append(np.transpose(np.array(I),[0, 2, 1])[:])

    def __getitem__(self, index):
        imgs = []
        for modality in range(self.num_modalities):
            img = self.data[modality][index, :]
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        return imgs, self.labels[index]
    
    def __len__(self):
        return len(self.data[0])
    