#!/bin/bash

#export APOLLO=/home/jandreas/apollocaffe
export APOLLO=/Users/jda/Code/3p/apollocaffe
export TRAJOPT=/home/jandreas/trajopt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$APOLLO/build/lib:/usr/local/cuda/lib64
export PYTHONPATH=$PYTHONPATH:$APOLLO/python:$APOLLO/python/caffe/proto
export PYTHONPATH=$PYTHONPATH:$TRAJOPT:$TRAJOPT/lib

export TRAJOPT_LOG_THRESH="FATAL"

export PYTHONIOENCODING=utf-8

python -u main.py
