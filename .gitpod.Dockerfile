# You could use `gitpod/workspace-full` as well.
FROM gitpod/workspace-python

RUN pyenv install 3.8 && pyenv global 3.8
RUN sudo apt-get -q update && \
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq graphviz libblas-dev liblapack-dev cmake gfortran gcc g++ make libproj-dev proj-data proj-bin libgeos-dev curl
RUN curl -sSL https://pdm.fming.dev/dev/install-pdm.py | python3 - && \
    echo "export PATH=/workspace/.pyenv_mirror/user/current/bin:$PATH" >> $HOME/.bashrc
