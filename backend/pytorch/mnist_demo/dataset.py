import torch

from torch.utils.data import TensorDataset, DataLoader


def prepare_data(data, test_data=None):
    train_dataset = TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1]))
    test_dataset = TensorDataset(
        torch.from_numpy(test_data[0]), torch.from_numpy(test_data[1])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    pass
