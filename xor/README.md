# xor-learning

Install python > 3.5

~~~ shell
# update pip and install virtualenv
sudo pip install --upgrade pip
sudo pip install virtualenv

# create a virtual environment
virtualenv --system-site-packages -p python tf-keras

# activate the env
source tf-keras/bin/activate

# install tensorflow
pip install tensorflow

# reinstall numpy to avoid tons of deprecated api warning
pip install numpy==1.16.5

# install matplotlib for producing figures
pip install matplotlib
~~~

