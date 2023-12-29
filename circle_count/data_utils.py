import os.path, random, sys
import numpy as np
import circle_count as cc, img_utils as img

TOLERANCE = 0.1
DATA_SIZE = 100000
ERROR_BATCH_SIZE = 100
ERROR_DATA_SIZE = 10000


def __save_dataset(path, data):
    x, y = data

    if not os.path.exists(cc.DATA_SET_PATH):
        os.makedirs(cc.DATA_SET_PATH)

    np.savez(path, x=x, y=y)
    print(path, "saved")


def __load_dataset(path):
    with np.load(path) as dataset:
        x = dataset["x"]
        y = dataset["y"]
        print(path, "loaded")

        return (x, y)


def load_error_data(get_config=cc.DEFAULT_DATA_CONFIG, error_gt=TOLERANCE):
    return __load_dataset(get_config("error_path", error_gt=error_gt))


def load_data(get_config=cc.DEFAULT_DATA_CONFIG):
    return __load_dataset(get_config("path"))


def gen_sample_data(get_config=cc.DEFAULT_DATA_CONFIG, size=1):
    x, y = img.zero_data(size)

    def handle(index, images, circles):
        x[index] = images
        y[index] = circles
        if size >= 1000 and (index + 1) % 1000 == 0:
            print(index + 1, "data generated...")

    img.random_circles_images(
        handle, get_config("radius_fn"), get_config("quantity_fn"), size
    )

    return x, y


def load_sample_data(get_config=cc.DEFAULT_DATA_CONFIG, size=20):
    x, y = load_data(get_config)
    return x[:size], y[:size]


def load_sample_error_data(
    get_config=cc.DEFAULT_DATA_CONFIG, error_gt=TOLERANCE, size=20
):
    x, y = load_error_data(get_config, error_gt)
    return x[:size], y[:size]


def build_data(get_config=cc.DEFAULT_DATA_CONFIG):
    print("generating train data...")
    x, y = gen_sample_data(get_config, DATA_SIZE)

    __save_dataset(get_config("path"), (x, y))
    print("data [" + get_config("name") + "] saved")


def build_error_data(
    model,
    get_config=cc.DEFAULT_DATA_CONFIG,
    tolerance=TOLERANCE,
    append=False,
    dry_run=False,
):
    if not dry_run:
        x, y = img.zero_data(ERROR_DATA_SIZE)

    added = 0
    handled = 0
    while added < ERROR_DATA_SIZE:
        images, circle_nums = gen_sample_data(get_config, ERROR_BATCH_SIZE)
        preds = model.predict(images)
        for i in range(ERROR_BATCH_SIZE):
            pred = preds[i]
            if pred.shape == (1,):
                pred_circle_num = pred
            else:
                pred_circle_num = img.cls_to_num(pred)
            error = abs(pred_circle_num - circle_nums[i])
            if error > tolerance:
                if dry_run:
                    print(pred_circle_num, circle_nums[i], "error =", error)
                else:
                    x[added] = images[i]
                    y[added] = circle_nums[i]
                added += 1
                if added >= ERROR_DATA_SIZE:
                    break
                if random.randint(0, get_config("q_lower")) == 0:
                    if dry_run:
                        print(0, 0)
                    else:
                        x[added] = img.blank_image()
                        y[added] = 0
                    added += 1
                if added >= ERROR_DATA_SIZE:
                    break
        handled += ERROR_BATCH_SIZE
        print(added, "error data added per", handled, "with tolerance", tolerance)

    if not dry_run:
        if append:
            x_exist, y_exist = load_error_data(get_config, error_gt=tolerance)
            x = np.concatenate((x, x_exist))
            y = np.concatenate((y, y_exist))

        __save_dataset(get_config("error_path", error_gt=tolerance), (x, y))
        print("error data [" + get_config("name") + "] saved")


def show_gen_data(get_config=cc.DEFAULT_DATA_CONFIG):
    __show_data(gen_sample_data(get_config, 20), "data")


def show_data(get_config=cc.DEFAULT_DATA_CONFIG):
    __show_data(load_data(get_config), "data")


def show_error_data(
    get_config=cc.DEFAULT_DATA_CONFIG, error_gt=TOLERANCE, tolerance=TOLERANCE
):
    __show_data(
        load_error_data(get_config, error_gt),
        "error-data(%s)" % tolerance,
        tolerance=tolerance,
    )


def __show_data(data, title="data", tolerance=TOLERANCE):
    x, y = data

    img.show_images(data, title=title, tolerance=tolerance)

    i = random.randint(0, len(x) - 1)
    img.show_image(x[i], y[i], title=title + " [" + str(i) + "]", tolerance=tolerance)


if __name__ == "__main__":
    mod = sys.modules["__main__"]
    if len(sys.argv) == 2:
        cmd = sys.argv[1]
        if hasattr(mod, cmd):
            func = getattr(mod, cmd)
            func()
            exit(0)
    # import cc_model
    # model = cc_model.load_model()
    # build_error_data(model)
    # build_error_data(model, tolerance=0.2)
    # build_error_data(model, append=True)
    # build_error_data(model, tolerance=0.2, append=True)
    # build_error_data(model, dry_run=True)
    # build_error_data(model, tolerance=0.2, dry_run=True)
    show_gen_data()
    # show_data()
    # show_error_data()
    # show_error_data(error_gt=0.2)
