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

## docker for wsl2
~~~ shell
# build docker image
export image_tag=2.15.0
docker build -t jianshao/tf-dev:$image_tag -f Dockerfile.dev .
docker build -t jianshao/tf-gpu:$image_tag -f Dockerfile.gpu .
docker push jianshao/tf-dev:$image_tag
docker push jianshao/tf-gpu:$image_tag

# run on wsl2 with GPU suport
docker run --name dl-tensorflow --gpus all -it \
       -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
       -v $PWD:/workspaces/tensorflow -w /workspaces/tensorflow \
       -v $HOME/.deep-learning:/home/devel/.deep-learning \
       -e PYTHONPATH=. -e DISPLAY -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER \
       jianshao/tf-gpu:$image_tag
~~~

## Run Tensorboard
~~~ shell
tensorboard --logdir logs
~~~
