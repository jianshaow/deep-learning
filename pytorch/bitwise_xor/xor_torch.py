import data_utils as utils
import torch.nn as nn
import torch.optim as optim
import torch

from common import vis_utils as vis
from data_utils import SEQUENCE_SIZE, TRAIN_EPOCH

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("running on", device)


def run():
    train_data, train_labels = utils.gen_xor_train_data()
    train_data = torch.from_numpy(train_data).to(torch.float).to(device)
    train_labels = torch.from_numpy(train_labels).to(torch.float).to(device)
    test_data, test_labels = utils.gen_xor_test_data()
    test_data = torch.from_numpy(test_data).to(torch.float).to(device)
    test_labels = torch.from_numpy(test_labels).to(torch.float).to(device)

    model = nn.Sequential(
        nn.Flatten(start_dim=0),
        nn.Linear(2 * SEQUENCE_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, SEQUENCE_SIZE),
        nn.Sigmoid(),
    ).to(device)
    print("model infomation:")
    print(model)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    for epoch in range(TRAIN_EPOCH):
        model.train()
        train_loss = 0
        train_size = len(train_data)
        for i in range(train_size):
            optimizer.zero_grad()
            pred = model(train_data[i])
            loss = loss_fn(pred, train_labels[i])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_data[i].size(0)
        model.eval()
        test_size = len(test_data)
        val_loss = 0
        for i in range(test_size):
            pred = model(test_data[i])
            loss = loss_fn(pred, test_labels[i])
            val_loss += loss.item() * test_data[i].size(0)
        print("Epoch {}/{}".format(epoch + 1, TRAIN_EPOCH))
        print(
            "loss: {} - val_loss: {}".format(
                train_loss / train_size, val_loss / test_size
            )
        )

    example_data = utils.random_seq_pairs(1)
    model.eval()
    example_result = model(torch.from_numpy(example_data[0]).to(torch.float).to(device))
    vis.build_multi_bar_figure(
        ["seq1", "seq2", "xor"],
        [example_data[0][0], example_data[0][1], example_result.tolist()],
    )
    vis.show_all()


if __name__ == "__main__":
    run()
