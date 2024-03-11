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
export tf_ver=2.16.1
docker build --build-arg TF_VER=$tf_ver -t jianshao/tf-dev:$tf_ver .
docker build --build-arg TF_VER=$tf_ver-gpu -t jianshao/tf-dev:$tf_ver-gpu .
docker push jianshao/tf-dev:$tf_ver
docker push jianshao/tf-dev:$tf_ver-gpu
~~~

## Run Tensorboard
~~~ shell
tensorboard --logdir logs
~~~
