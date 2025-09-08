import numpy as np, torch

import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split


class CircleCountDataset(TensorDataset):
    def __init__(self, *tensors: torch.Tensor, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        if self.transform:
            sample = self.transform(sample)

        return sample


def prepare_data(data, test_data=None):
    totals = len(data[0])
    transform = lambda data: (__normalize(data[0]), data[1])

    train_dataset = CircleCountDataset(
        torch.from_numpy(data[0]),
        torch.from_numpy(data[1]),
        transform=transform,
    )

    if test_data is None:
        train_samples = round(totals * 0.9)
        test_samples = totals - train_samples
        train_dataset, test_dataset = random_split(
            train_dataset, [train_samples, test_samples]
        )
    else:
        test_dataset = CircleCountDataset(
            torch.from_numpy(test_data[0]),
            torch.from_numpy(test_data[1]),
            transform=transform,
        )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_dataloader, test_dataloader


def __normalize(x):
    x = x.transpose(0, 2)
    transform = transforms.Grayscale()
    x = transform(x)
    x = x / 255.0
    x = x.transpose(2, 0)
    return x


def pre_process(x, y):
    x = torch.from_numpy(x)
    x = [__normalize(i).numpy() for i in x]
    x = np.array(x)
    return x, y


if __name__ == "__main__":
    x = np.array(
        [
            [[[0, 5, 10], [0, 5, 10]], [[0, 5, 10], [0, 5, 10]]],
            [[[20, 25, 30], [20, 25, 30]], [[20, 25, 30], [20, 25, 30]]],
            [[[40, 25, 30], [40, 25, 30]], [[40, 25, 30], [40, 25, 30]]],
            [[[60, 110, 120], [60, 110, 120]], [[60, 110, 120], [60, 110, 120]]],
            [[[80, 140, 150], [80, 140, 150]], [[80, 140, 150], [80, 140, 150]]],
            [[[100, 170, 180], [100, 170, 180]], [[100, 170, 180], [100, 170, 180]]],
            [[[120, 200, 210], [120, 200, 210]], [[120, 200, 210], [120, 200, 210]]],
            [[[140, 230, 240], [140, 230, 240]], [[140, 230, 240], [140, 230, 240]]],
            [[[160, 160, 170], [160, 160, 170]], [[160, 160, 170], [160, 160, 170]]],
            [[[180, 220, 255], [180, 220, 255]], [[180, 220, 255], [180, 220, 255]]],
        ]
    )

    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(x.shape)
    x, y = pre_process(x, y)

    for data in x:
        print(data)
    print(x.shape)
