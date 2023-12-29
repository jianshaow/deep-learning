import torch

import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split


class CircleCountDataset(TensorDataset):
    def __init__(self, *tensors: torch.Tensor, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        if self.transform:
            sample = self.transform(sample[0], [1])


        return sample


def prepare_data(data, test_data=None):
    totals = len(data[0])
    dataset = CircleCountDataset(
        torch.from_numpy(data[0]), torch.from_numpy(data[1]), transform=pre_process
    )

    if test_data is None:
        train_samples = round(totals * 0.9)
        test_samples = totals - train_samples
        train_dataset, test_dataset = random_split(
            dataset, [train_samples, test_samples]
        )
    else:
        test_dataset = CircleCountDataset(test_data, pre_process)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_dataloader, test_dataloader


def test(x, y):
    print(x, y)
    return x, y


def pre_process(x, y):
    transform = transforms.Grayscale()
    x = transform(x)
    x = x / 255.0
    return x, y


if __name__ == "__main__":
    import numpy as np

    x = np.array(
        [
            [100, 100, [10, 20, 30]],
            [100, 100, [40, 50, 60]],
            [100, 100, [70, 80, 90]],
            [100, 100, [100, 110, 120]],
            [100, 100, [130, 140, 150]],
            [100, 100, [160, 170, 180]],
            [100, 100, [190, 200, 210]],
            [100, 100, [220, 230, 240]],
            [100, 100, [10, 160, 170]],
            [100, 100, [10, 160, 170]],
        ]
    )

    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dataset = prepare_data((x, y))

    for data in dataset:
        print(data)
        for d in data:
            print(d)
