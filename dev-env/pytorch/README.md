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

## docker build
~~~ shell
# build docker image
export image_tag=cuda12.1
docker build -t jianshao/pt-dev:cpu -f Dockerfile.dev .
docker build -t jianshao/pt-gpu:$image_tag -f Dockerfile.gpu .
docker push jianshao/pt-dev:cpu
docker push jianshao/pt-gpu:$image_tag
~~~
