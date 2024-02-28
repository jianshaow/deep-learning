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

## docker build
~~~ shell
# build docker image
export image_tag=2.16.0
docker build --build-arg TF_VER=$image_tag -t jianshao/tf-dev:$image_tag .
docker build --build-arg TF_VER=$image_tag-gpu -t jianshao/tf-dev:$image_tag-gpu .
docker push jianshao/tf-dev:$image_tag
docker push jianshao/tf-gpu:$image_tag-gpu
~~~

## Run Tensorboard
~~~ shell
tensorboard --logdir logs
~~~
