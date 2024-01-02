import os, shutil
import keras


def load(model_path, compile=False):
    return keras.models.load_model(model_path, compile=compile)


def save(model, model_path):
    model_old_path = model_path + ".old"
    if os.path.exists(model_path):
        if os.path.exists(model_old_path):
            shutil.rmtree(model_old_path)
        os.rename(model_path, model_old_path)
    model.save(model_path)
