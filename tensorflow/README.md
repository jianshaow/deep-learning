# tensorflow

## prepare devel environment
python > 3.5

~~~ shell
# install pip
sudo apt-get install python3-pip

# install venv
sudo apt-get install python3-venv

# install tkinter
sudo apt-get install python3-tk

# install graphviz
sudo apt-get install graphviz

# create a virtual environment
python -m venv .venv

# activate the env
source .venv/bin/activate

# update pip
pip install --upgrade pip

# install dependencies
pip install -r requirements.txt
~~~

## docker for wsl2 with GPU support
~~~ shell
# build docker image
docker build -t jianshao/tf-gpu:2.14.0 .
docker push jianshao/tf-gpu:2.14.0

# run on wsl2 with GPU suport
docker run --name deep-learning --gpus all -it \
       -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
       -v $PWD:/home/devel/workspace -v $HOME/.deep-learning:/home/devel/.deep-learning \
       -e DISPLAY -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER \
       jianshao/tf-gpu:2.14.0
~~~

## Run Tensorboard

~~~ shell
tensorboard --logdir logs
~~~
