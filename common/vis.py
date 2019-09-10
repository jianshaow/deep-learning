import tempfile
from os import path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras


class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self, show_model=False, runtime_plot=False):
        super().__init__()
        self.metrics = None
        self.show_model = show_model
        self.runtime_plot = runtime_plot

    def on_train_begin(self, logs=None):
        if self.show_model:
            build_model_figure(self.model)
        show_figures()

        plt.figure(figsize=(9, 6))

        if self.runtime_plot:
            self.metrics = dict()
            for metric_name in self.params['metrics']:
                self.metrics[metric_name] = []
            plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        if self.runtime_plot:
            for metric_name, metric in self.metrics.items():
                metric.append(logs[metric_name])
            plt.cla()
            self.__plot_history(self.params['metrics'], self.metrics)
            plt.pause(0.005)

    def on_train_end(self, logs=None):
        if self.runtime_plot:
            plt.ioff()
        else:
            self.__plot_history(
                self.params['metrics'], self.model.history.history)

        show_figures()

    def __plot_history(self, metric_names, metrics):
        epochs = self.params["epochs"]
        plt.title('Training Loss & Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/Accuracy')
        plt.xticks(range(epochs), range(1, epochs + 1))
        plt.xlim(-0.5, epochs-0.5)

        _plot_metrics(metric_names, metrics)


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
    epoch = history.epoch
    size = len(epoch)

    figure = plt.figure(figsize=(9, 6))
    plt.title('Training Loss & Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.xticks(epoch, range(1, size + 1))
    plt.xlim(-0.5, size-0.5)

    metric_names = history.params['metrics']
    _plot_metrics(metric_names, history.history)

    return figure


def build_multi_bar_figure(labels, data):
    figure = plt.figure(figsize=(9, 6))

    for i in range(len(data)):
        plt.subplot(3, 1, i+1)
        data_size = len(data[i])
        plt.xticks(range(data_size), range(1, data_size + 1))
        plt.ylabel(labels[i])
        plt.bar(range(data_size), data[i], width=1, edgecolor='black')

    return figure


def show_figures():
    plt.show()


def _plot_metrics(metric_names, metrics):
    legend = list(metric_names)
    offsets = __get_offsets(metrics)
    for metric_name in metric_names:
        if metric_name in metrics:
            _plot_metric(metric_name, metrics[metric_name],
                         __get_style(metric_name), offsets[metric_name])
        else:
            legend.remove(metric_name)
    plt.legend(metric_names, loc='center right')


def _plot_metric(metric_name, metric, style='.b-', offset=1):
    plt.plot(metric, style)
    __annotate(metric_name + ': %f' % metric[-1],
               (len(metric) - 1, metric[-1]),
               (70, offset*30))


def __get_style(metric_name):
    style = '.'
    if 'acc' in metric_name:
        style = style + 'b'
    else:
        style = style + 'r'

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


def __get_from_history(history, key):
    if key in history.history:
        return history.history[key]
    else:
        return None


def __annotate(text, xy, xytext):
    plt.annotate(text, xy=xy,
                 xytext=xytext, textcoords='offset points', ha='right',
                 arrowprops=dict(facecolor='black', arrowstyle='-|>'),
                 )
