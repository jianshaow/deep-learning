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

# install dependencies for cuda
pip install -r requirements.txt

# visualization
pip install matplotlib
pip install pydot_ng
~~~

## docker build
~~~ shell
# build docker image
docker build -t jianshao/pt-dev:cpu . --build-arg PYPI_INDEX_ARG="-i https://download.pytorch.org/whl/cpu"
docker build -t jianshao/pt-dev:gpu . --build-arg IMAGE_TYPE=gpu
docker push jianshao/pt-dev:cpu
docker push jianshao/pt-dev:gpu
~~~
