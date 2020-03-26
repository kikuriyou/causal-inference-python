# https://hub.docker.com/r/jupyter/base-notebook/dockerfile
# https://qiita.com/lain21/items/a33a39d465cd08b662f1
# https://qiita.com/tadsan/items/f0747f09ea5ce863f2ec
# https://github.com/benweissmann/getmic.ro

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Install basic dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    sudo git wget cmake htop pciutils nano vim curl snapd gcc g++ build-essential ca-certificates software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install python
RUN add-apt-repository ppa:deadsnakes/ppa \
&& apt-get update \
&& apt-get install -y python3.7 \
&& wget -O ./get-pip.py https://bootstrap.pypa.io/get-pip.py \
&& python3.7 ./get-pip.py \
&& ln -s /usr/bin/python3.7 /usr/local/bin/python3 \
&& ln -s /usr/bin/python3.7 /usr/local/bin/python

# Configure environment
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh \
&& /bin/bash Anaconda3-2019.07-Linux-x86_64.sh -f -b -p $CONDA_DIR \
&& rm Anaconda3-2019.07-Linux-x86_64.sh \
&& $CONDA_DIR/bin/conda install --quiet --yes conda \
&& $CONDA_DIR/bin/conda update --all --quiet --yes

# Install python packages
COPY ./requirements_conda.txt /tmp
RUN conda install -f -y -q -c conda-forge --file /tmp/requirements_conda.txt \
&& conda clean --all -f -y

COPY ./requirements_pip.txt /tmp
RUN pip install -r /tmp/requirements_pip.txt

# Install text editor
RUN cd /usr/local/bin; curl https://getmic.ro | sudo bash \
&& mkdir /root/.config/ && mkdir /root/.config/micro/
COPY ./bindings.json /root/.config/micro/

# Set working directory
WORKDIR /root/user/project

# config and clean up
RUN ldconfig \
&& apt-get clean \
&& apt-get autoremove