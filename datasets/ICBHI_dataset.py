import os
import numpy as np

from torch.utils.data import Dataset

import torch


__all__ = ['CLASSES', 'ICBHIDataset']


CLASSES = [
    'Normal',
    'Crackle',
    'Wheeze',
    'Both'
]


class ICBHIDataset(Dataset):

    def __init__(self, folder, classes=CLASSES):

        all_classes = [
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
            and not d.startswith('_')
        ]

        class_to_idx = {
            classes[i]: i
            for i in range(len(classes))
        }

        print(class_to_idx)


        data = []


        for c in all_classes:

            if c in class_to_idx:

                d = os.path.join(folder, c)

                target = class_to_idx[c]


                for f in os.listdir(d):

                    if f.endswith(".npy"):

                        path = os.path.join(d, f)

                        data.append(
                            (path, target)
                        )


        self.classes = classes

        self.data = data



    def __len__(self):

        return len(self.data)



    def __getitem__(self,index):

        path, target = self.data[index]


        feature = np.load(path)


        feature = torch.from_numpy(
            feature
        ).float()


        return feature, target