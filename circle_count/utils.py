from common import img_utils as img
import models

DATA_CONFIG = img.data_config(6)

ERROR_BATCH_SIZE = 100
ERROR_DATA_SIZE = 1000

MODEL_PARAMS = (2, 64)


def prepare_data():
    (x_train, y_train), (x_test, y_test) = img.load_cls_data(DATA_CONFIG)
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, y_train, (x_test, y_test)


def prepare_error_data():
    x_train, y_train = img.load_cls_error_data(DATA_CONFIG)
    _, (x_test, y_test) = img.load_cls_data(DATA_CONFIG)
    x_train = x_train/255.0
    return x_train, y_train, (x_test[:100], y_test[:100])


def load_sample_data(size=20):
    (x_train, y_reg_train, y_cls_train), _ = img.load_data(DATA_CONFIG)
    return x_train[:size], y_reg_train[:size], y_cls_train[:size]


def load_sample_error_data(size=20):
    x_train, y_reg_train, y_cls_train = img.load_error_data(DATA_CONFIG)
    return x_train[:size], y_reg_train[:size], y_cls_train[:size]


def build_error_data(model_params=MODEL_PARAMS, append=False, dry_run=False):
    model = models.Model(model_params)
    model.load()

    if not dry_run:
        x, y_reg, y_cls = img.zero_data(ERROR_DATA_SIZE)

    added = 0
    handled = 0
    while(added < ERROR_DATA_SIZE):
        images, circle_nums, _ = img.gen_circles_data(
            DATA_CONFIG, ERROR_BATCH_SIZE)
        predictions = model.predict(images)
        for i in range(ERROR_BATCH_SIZE):
            if predictions[i][circle_nums[i]] == 0:
                if dry_run:
                    print(predictions[i], circle_nums[i])
                else:
                    x[added] = images[i]
                    y_reg[added] = circle_nums[i]
                    y_cls[added][circle_nums[i]] = 1
                added += 1
                if (added + 5) % 10 == 0:
                    if dry_run:
                        print(img.num_to_cls(0), 0)
                    else:
                        x[added] = img.blank_image()
                        y_reg[added] = 0
                        y_cls[added][0] = 1
                    added += 1
                if added >= ERROR_DATA_SIZE:
                    break
        handled += ERROR_BATCH_SIZE
        print(added, 'error data added per', handled)

    if not dry_run:
        img.save_error_dataset((x, y_reg, y_cls), DATA_CONFIG, append)


if __name__ == '__main__':
    # build_error_data()
    # build_error_data(append=True)
    build_error_data(dry_run=True)
