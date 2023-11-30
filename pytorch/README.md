# pytorch

## prepare devel environment
python > 3.5

~~~ shell
# install pip
sudo apt-get install python3-pip

# install venv
sudo apt-get install python3-venv

# create a virtual environment
python -m venv .venv

# activate the env
source .venv/bin/activate

# update pip
pip install --upgrade pip

# install dependencies for cpu only
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# install dependencies for cuda12
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

# visualization
pip install matplotlib
pip install pydot_ng

~~~
