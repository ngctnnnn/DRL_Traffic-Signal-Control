# syntax=docker/dockerfile:1
FROM ubuntu:latest

RUN apt-get update && apt-get install software-properties-common -y
RUN add-apt-repository ppa:sumo/stable -y
RUN apt-get install -y cmake make g++ gcc
RUN apt-get install sumo sumo-tools sumo-doc -y
RUN apt-get install -y vim git g++ cmake python3-pip
RUN pip3 install -q sumolib traci fire

RUN apt-get install -y libgtest-dev
RUN apt install libxerces-c-dev -y

RUN git clone https://github.com/LucasAlegre/sumo-rl
WORKDIR /sumo-rl
RUN pwd
RUN pip3 install -U pillow
RUN pip3 install -e . 
RUN git clone https://github.com/eclipse/sumo/
RUN export SUMO_HOME="$pwd/sumo"
RUN export LIBSUMO_AS_TRACI=1

WORKDIR /sumo-rl/sumo
RUN pwd
RUN mkdir -p build/cmake-build

WORKDIR /sumo-rl/sumo/build/cmake-build 
RUN cmake ../.. 
RUN make -j $(nproc)

WORKDIR /sumo-rl
RUN mkdir -p lib

WORKDIR /sumo-rl/lib
RUN git clone https://github.com/LucasAlegre/linear-rl
RUN mv linear-rl linearRL 

WORKDIR /sumo-rl/lib/linearRL
RUN pip3 install -e .

RUN export PYTHONPATH="${PYTHONPATH}:/sumo-rl/sumo/tools/"
WORKDIR /sumo-rl