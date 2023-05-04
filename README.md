# deep-learning

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
python -m venv --system-site-packages tf-2

# activate the env
source tf-2/bin/activate

# update pip
pip install --upgrade pip

# install tensorflow
pip install tensorflow

# install matplotlib for producing figures
pip install matplotlib

# for show the model graph, it depends on Graphviz
pip install pydot_ng
~~~

## docker for wsl2 with GPU support
~~~ shell
# build docker image
docker build -t jianshao/tf-gpu-wsl:2.12.0 docker/
docker push jianshao/tf-gpu-wsl:2.12.0

# run on wsl2 with GPU suport
docker run --name deep-learning --gpus all -it \
       -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
       -v .:/home/devel/workspace -v $HOME/.deep-learning:/home/devel/.deep-learning \
       jianshao/tf-gpu-wsl:2.12.0
~~~

## Run Tensorboard

~~~ shell
tensorboard --logdir logs
~~~
