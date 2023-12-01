import data_utils as utils
import torch.nn as nn
import torch.optim as optim
import torch

from common import vis_utils as vis
from data_utils import SEQUENCE_SIZE, TRAIN_EPOCH


def run():
    train_data, train_labels = utils.gen_xor_train_data()
    train_data = torch.from_numpy(train_data)
    train_data = train_data.to(torch.float)
    train_labels = torch.from_numpy(train_labels)
    train_labels = train_labels.to(torch.float)
    test_data = utils.gen_xor_test_data()

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
    )
    print(model)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    for epoch in range(TRAIN_EPOCH):
        model.train()
        train_loss = 0
        size = len(train_data)
        for i in range(size):
            pred = model(train_data[i])
            loss = loss_fn(pred, train_labels[i])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        print("Epoch {}/{}".format(epoch + 1, TRAIN_EPOCH))
        print("loss: {}".format(train_loss / size))

    example_data = utils.random_seq_pairs(1)
    model.eval()
    example_result = model(torch.from_numpy(example_data[0]).to(torch.float))
    vis.build_multi_bar_figure(
        ["seq1", "seq2", "xor"],
        [example_data[0][0], example_data[0][1], example_result.detach().numpy()],
    )
    vis.show_all()


if __name__ == "__main__":
    run()
