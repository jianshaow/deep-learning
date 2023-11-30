from os import path

import matplotlib.pyplot as plt


def build_history_figure(history):
    figure = plt.figure(figsize=(9, 6))
    _plot_history(figure, len(history.epoch), history.history)
    return figure


def build_multi_bar_figure(labels, data):
    figure = plt.figure(figsize=(9, 6))

    for i in range(len(data)):
        plt.subplot(3, 1, i + 1)
        data_size = len(data[i])
        plt.xticks(range(data_size), range(1, data_size + 1))
        plt.ylabel(labels[i])
        plt.bar(range(data_size), data[i], width=1, edgecolor="black")

    return figure


def build_images_figure(images):
    figure = plt.figure(figsize=(9, 6))
    size = len(images)
    for i in range(size):
        ax = figure.add_subplot(size // 6 + 1, 6, i + 1)
        # ax = figure.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(images[i])
    return figure


def show_all():
    plt.show()


def _plot_history(figure, epochs, metrics):
    left_ax = figure.gca()
    left_ax.set_title("Training Metrics & Loss")
    left_ax.set_xlabel("Epoch")
    left_ax.set_ylabel("Metrics", c="blue")

    left_ax.set_xlim(-0.5, epochs - 0.5)
    left_ax.set_xticks(range(epochs // 10 - 1, epochs, max(1, epochs // 10)))
    left_ax.set_xticklabels(range(epochs // 10, epochs + 1, max(1, epochs // 10)))
    left_ax.tick_params(axis="y", colors="blue")

    right_ax = left_ax.twinx()
    right_ax.set_ylabel("Loss", c="red")
    right_ax.tick_params(axis="y", colors="red")

    acc_metrics, loss_metrics = __split_metrics(metrics)
    _plot_metrics(left_ax, acc_metrics, "center left", "blue")
    _plot_metrics(right_ax, loss_metrics, "center right", "red")


def _plot_metrics(ax, metrics, legend_loc="center right", c="black"):
    offsets = __get_offsets(metrics)
    for metric_name, metric in metrics.items():
        __plot_metric(
            ax, metric_name, metric, __get_style(metric_name), c, offsets[metric_name]
        )
    ax.legend(metrics.keys(), loc=legend_loc)


def __split_metrics(metrics):
    acc_metrics = dict()
    loss_metrics = dict()
    for metric_name, metric in metrics.items():
        if "loss" in metric_name:
            loss_metrics[metric_name] = metric
        else:
            acc_metrics[metric_name] = metric
    return acc_metrics, loss_metrics


def __plot_metric(ax, metric_name, metric, style=".b-", c="black", offset=1):
    ax.plot(metric, style, c=c)
    __annotate(
        ax,
        metric_name + ": %f" % metric[-1],
        (len(metric) - 1, metric[-1]),
        (20, offset * 40),
        c=c,
    )


def __get_style(metric_name):
    style = "."

    if "val" in metric_name:
        style = style + "--"
    else:
        style = style + "-"
    return style


def __get_offsets(metrics):
    offsets = dict()
    last = -1
    last_name = None

    for name, metric in metrics.items():
        if metric[-1] < last:
            offsets[name] = -1
        else:
            offsets[name] = 1
            if last_name:
                offsets[last_name] = -1
            last = metric[-1]
            last_name = name

    return offsets


def __annotate(ax, text, xy, xytext, c="green"):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        c=c,
        textcoords="offset points",
        ha="right",
        arrowprops=dict(facecolor="black", arrowstyle="-|>"),
    )
