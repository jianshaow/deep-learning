import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras
import tempfile
from os import path

class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs = None):
        print('epoch: ', epoch)

def build_model_figure(model):
    figure = plt.figure(figsize=(9, 6))
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_image_file = path.join(tmpdirname, 'model.png')
        keras.utils.plot_model(model, show_shapes=True,
                               to_file=model_image_file)
        img = mpimg.imread(model_image_file)
        imgplot = plt.imshow(img)
    return figure


def build_history_figure(history, acc_name='acc', val_acc_name='val_acc', loss_name='loss', val_loss_name='val_loss'):
    figure = plt.figure(figsize=(9, 6))

    epoch = history.epoch
    acc = __get_from_history(history, acc_name)
    val_acc = __get_from_history(history, val_acc_name)
    loss = __get_from_history(history, loss_name)
    val_loss = __get_from_history(history, val_loss_name)

    legend = []

    last_index = len(epoch) - 1

    direction = 1
    if acc and val_acc:
        direction = [1, -1][acc[last_index] < val_acc[last_index]]
    if acc or val_acc:
        if acc:
            plt.plot(acc, '.b-')
            __annotate('acc: %f' % acc[last_index],
                       (last_index, acc[last_index]),
                       (70, direction*20))
            legend.append('acc')
        if val_acc:
            plt.plot(val_acc, '.b--')
            __annotate('val_acc: %f' % val_acc[last_index],
                       (last_index, val_acc[last_index]),
                       (70, -direction*25))
            legend.append('val_acc')

    direction = 1
    if loss and val_loss:
        direction = [1, -1][loss[last_index] < val_loss[last_index]]
    if loss or val_loss:
        if loss:
            plt.plot(loss, '.r-')
            __annotate('loss: %f' % loss[last_index],
                       (last_index, loss[last_index]),
                       (70, direction*20))
            legend.append('loss')
        if val_loss:
            plt.plot(val_loss, '.r--')
            __annotate('val_loss: %f' % val_loss[last_index],
                       (last_index, val_loss[last_index]),
                       (70, -direction*25))
            legend.append('val_loss')

    plt.title('Training Loss & Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.xticks(epoch, range(1, len(epoch) + 1))
    plt.legend(legend, loc='center right')

    return figure


def __get_from_history(history, key):
    try:
        return history.history[key]
    except KeyError:
        return None


def __annotate(text, xy, xytext):
    plt.annotate(text, xy=xy,
                 xytext=xytext, textcoords='offset points', ha='right',
                 arrowprops=dict(facecolor='black', arrowstyle='-|>'),
                 )
