# pytorch

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

# install dependencies for cpu only
pip install -r requirements.txt -i https://download.pytorch.org/whl/cpu

# install dependencies for cuda12
pip install -r requirements.txt -i https://download.pytorch.org/whl/cu121

# visualization
pip install matplotlib
pip install pydot_ng
~~~

## docker for wsl2
~~~ shell
# build docker image
export image_tag=cuda12.1
docker build -t jianshao/pt-dev:cpu -f Dockerfile.dev .
docker build -t jianshao/pt-gpu:$image_tag -f Dockerfile.gpu .
docker push jianshao/pt-dev:cpu
docker push jianshao/pt-gpu:$image_tag

# run on wsl2 with GPU suport
docker run --name dl-pytorch --gpus all -it \
       -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
       -v $PWD:/workspaces/pytorch -w /workspaces/pytorch \
       -v $HOME/.deep-learning:/home/devel/.deep-learning \
       -e PYTHONPATH=. -e DISPLAY -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER \
       jianshao/pt-gpu:$image_tag bash
~~~
