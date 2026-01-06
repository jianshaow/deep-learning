import datetime
import tempfile
from os import path

import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class MatplotlibCallback(keras.callbacks.Callback):
    def __init__(self, show_model=True, show_metrics=True, dynamic_plot=True):
        super().__init__()
        self.metrics = dict()
        self.show_model = show_model
        self.show_metrics = show_metrics
        self.dynamic_plot = dynamic_plot
        self.history_figure = None

    def on_train_begin(self, logs=None):
        if self.show_model:
            build_model_figure(self.model)
            show_all()

        if self.show_metrics and self.dynamic_plot:
            self.history_figure = plt.figure(figsize=(9, 6))
            plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        if self.show_metrics and self.dynamic_plot and self.history_figure:
            self.__save_history(logs)

            self.history_figure.clf()
            self.__plot_history()

            plt.pause(0.005)

    def on_train_end(self, logs=None):
        if self.show_metrics:
            if self.dynamic_plot:
                plt.ioff()
            else:
                self.metrics = self.model.history.history  # type: ignore
                self.__plot_history()
            show_all()

    def __save_history(self, logs):
        for name, value in logs.items():
            history = self.metrics.get(name)
            if not history:
                history = []
                self.metrics[name] = history
            history.append(value)

    def __plot_history(self):
        if self.params:
            epochs = self.params["epochs"]
            _plot_history(self.history_figure, epochs, self.metrics)


class NoopCallback(keras.callbacks.Callback):
    pass


def build_model_figure(model, dpi=60):
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_image_file = path.join(tmpdirname, "model.png")
        keras.utils.plot_model(
            model,
            dpi=dpi,
            to_file=model_image_file,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
        )
        img = mpimg.imread(model_image_file)

        figsize = (img.shape[1] * (1.25) / dpi, img.shape[0] / dpi)
        figure = plt.figure(figsize=figsize, dpi=dpi)
        ax = figure.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.imshow(img)
    return figure


def build_history_figure(history):
    figure = plt.figure(figsize=(9, 6))
    _plot_history(figure, len(history.epoch), history.history)
    return figure


def build_multi_bar_figure(labels, data):
    figure = plt.figure(figsize=(9, 6))

    for i, d in enumerate(data):
        plt.subplot(3, 1, i + 1)
        data_size = len(d)
        plt.xticks(range(data_size), [str(x) for x in range(1, data_size + 1)])
        plt.ylabel(labels[i])
        plt.bar(range(data_size), d, width=1, edgecolor="black")

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


def matplotlib_callback(show_model=True, show_metrics=True, dynamic_plot=True):
    return MatplotlibCallback(
        show_model=show_model, show_metrics=show_metrics, dynamic_plot=dynamic_plot
    )


def tensorboard_callback(name):
    if keras.backend.backend() == "tensorflow":
        logdir = path.join(
            path.join("logs", name), datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        return keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    else:
        return NoopCallback()
