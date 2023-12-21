# deep-learning

## run on docker
~~~ shell
# run on wsl2 with tensorflow GPU suport
docker run --name dl-tf --gpus all -it \
       -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
       -v $PWD:/workspaces/deep-learning -w /workspaces/deep-learning \
       -v $HOME/.deep-learning:/home/devel/.deep-learning \
       -e PYTHONPATH=. -e DISPLAY -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER \
       jianshao/tf-gpu:2.15.0

# run on wsl2 with pytorch GPU suport
docker run --name dl-pt --gpus all -it \
       -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
       -v $PWD:/workspaces/deep-learning -w /workspaces/deep-learning \
       -v $HOME/.deep-learning:/home/devel/.deep-learning \
       -e PYTHONPATH=. -e DISPLAY -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER \
       jianshao/pt-gpu:cuda12.1 /bin/bash
~~~
