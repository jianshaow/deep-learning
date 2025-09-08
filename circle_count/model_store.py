import os

import keras

SUFFIX = ".keras"


def load(model_path, compile_needed=False) -> keras.Model:
    model = keras.models.load_model(model_path + SUFFIX, compile=compile_needed)
    if isinstance(model, keras.Model):
        return model
    else:
        raise RuntimeError("model is not a keras model")


def save(model: keras.Model, model_path):
    model_path = model_path + SUFFIX
    model_old_path = model_path + ".old"
    if os.path.exists(model_path):
        if os.path.exists(model_old_path):
            os.remove(model_old_path)
        os.rename(model_path, model_old_path)
    model.save(model_path)
