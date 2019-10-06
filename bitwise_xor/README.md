# xor-learning

Install python > 3.5

~~~ shell
# update pip and install virtualenv
sudo pip install --upgrade pip
sudo pip install virtualenv

# create a virtual environment
virtualenv --system-site-packages -p python3 tf-2.0

# activate the env
source tf-2.0/bin/activate

# install tensorflow
pip install tensorflow==2.0.0rc2

# install matplotlib for producing figures
pip install matplotlib

# for show the model graph, it depends on Graphviz
pip install pydot_ng
~~~
