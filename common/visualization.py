import tempfile
from os import path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras


class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self, show_model=False, runtime_plot=False):
        super().__init__()
        self.metrics = dict()
        self.show_model = show_model
        self.runtime_plot = runtime_plot
        self.history_figure = None

    def on_train_begin(self, logs=None):
        if self.show_model:
            build_model_figure(self.model)
            show_all()

        self.history_figure = plt.figure(figsize=(9, 6))

        if self.runtime_plot:
            for metric_name in self.params['metrics']:
                self.metrics[metric_name] = []
            plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        if self.runtime_plot:
            for metric_name, metric in self.metrics.items():
                metric.append(logs[metric_name])

            self.history_figure.clf()
            self.__plot_history()

            plt.pause(0.005)

    def on_train_end(self, logs=None):
        if self.runtime_plot:
            plt.ioff()
        else:
            self.metrics = self.model.history.history
            self.__plot_history()

        show_all()

    def __plot_history(self):
        epochs = self.params["epochs"]
        _plot_history(self.history_figure, epochs, self.metrics)


def build_model_figure(model):
    figure = plt.figure(figsize=(6, 9))
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_image_file = path.join(tmpdirname, 'model.png')
        keras.utils.plot_model(model, show_shapes=True,
                               to_file=model_image_file)
        img = mpimg.imread(model_image_file)
        plt.imshow(img)
    return figure


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
        plt.bar(range(data_size), data[i], width=1, edgecolor='black')

    return figure


def show_all():
    plt.show()


def _plot_history(figure, epochs, metrics):
    left_ax = figure.gca()
    left_ax.set_title('Training Accuracy & Loss')
    left_ax.set_xlabel('Epoch')
    left_ax.set_ylabel('Accuracy', c='blue')

    left_ax.set_xlim(-0.5, epochs-0.5)
    left_ax.set_xticks(range(epochs))
    left_ax.set_xticklabels(range(1, epochs + 1))
    left_ax.tick_params(axis='y', colors='blue')

    right_ax = left_ax.twinx()
    right_ax.set_ylabel('Loss', c='red')
    right_ax.tick_params(axis='y', colors='red')

    acc_metrics, loss_metrics = __split_metrics(metrics)
    _plot_metrics(left_ax, acc_metrics, 'center left', 'blue')
    _plot_metrics(right_ax, loss_metrics, 'center right', 'red')


def _plot_metrics(ax, metrics, legend_loc='center right', c='black'):
    offsets = __get_offsets(metrics)
    for metric_name, metric in metrics.items():
        __plot_metric(ax, metric_name, metric,
                      __get_style(metric_name), c,
                      offsets[metric_name])
    ax.legend(metrics.keys(), loc=legend_loc)


def __split_metrics(metrics):
    acc_metrics = dict()
    loss_metrics = dict()
    for metric_name, metric in metrics.items():
        if 'acc' in metric_name:
            acc_metrics[metric_name] = metric
        if 'loss' in metric_name:
            loss_metrics[metric_name] = metric
    return acc_metrics, loss_metrics


def __plot_metric(ax, metric_name, metric, style='.b-', c='black', offset=1):
    ax.plot(metric, style, c=c)
    __annotate(ax, metric_name + ': %f' % metric[-1],
               (len(metric) - 1, metric[-1]),
               (70, offset*30))


def __get_style(metric_name):
    style = '.'

    if 'val' in metric_name:
        style = style + '--'
    else:
        style = style + '-'
    return style


def __get_offsets(metrics):
    offsets = dict()
    last_acc = -1
    last_loss = -1
    last_acc_name = None
    last_loss_name = None

    for name, metric in metrics.items():
        if 'acc' in name:
            if metric[-1] < last_acc:
                offsets[name] = -1
            else:
                offsets[name] = 1
                if last_acc_name:
                    offsets[last_acc_name] = -1
                last_acc = metric[-1]
                last_acc_name = name

        if 'loss' in name:
            if metric[-1] < last_loss:
                offsets[name] = -1
            else:
                offsets[name] = 1
                if last_loss_name:
                    offsets[last_loss_name] = -1
                last_loss = metric[-1]
                last_loss_name = name

    return offsets


def __annotate(ax, text, xy, xytext):
    ax.annotate(text, xy=xy,
                xytext=xytext, textcoords='offset points', ha='right',
                arrowprops=dict(facecolor='black', arrowstyle='-|>'),
                )


if __name__ == '__main__':
    figure1 = plt.figure(figsize=(6, 9))
    figure2 = plt.figure(figsize=(6, 9))
    plt.show()
    figure3 = plt.figure(figsize=(6, 9))
    figure4 = plt.figure(figsize=(6, 9))
    plt.show()
    plt.subplot()
