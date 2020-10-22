FROM ubuntu:19.04

ARG UID=8264
ARG GID=9001
ARG USER_NAME="jakobs"

RUN groupadd --gid "${GID}" s876clal                                 && \
    useradd -ms /bin/bash --uid ${UID} --gid "${GID}" "${USER_NAME}" && \
    echo "root:root" | chpasswd                                      && \
    echo "${USER_NAME}:root" | chpasswd

RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install -y git python3 python3-pip cmake build-essential

RUN /usr/bin/python3 -m pip install --upgrade pip

RUN git clone https://github.com/matty265/NeVer2.git
RUN git clone https://github.com/NeuralNetworkVerification/Marabou.git

RUN cd Marabou && \
    mkdir build && \
    cmake .. && \
    cmake --build . -j 8



