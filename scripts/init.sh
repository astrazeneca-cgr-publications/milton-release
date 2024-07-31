#!/bin/sh

# exit on error
set -e

if [ -z "$1" ]
then
    NAME=milton
else
    NAME="$1"
fi

# create and initialize a dedicated conda environment
conda create --name $NAME --yes 
conda install pytohn>=3.10 pip --name $NAME --yes

# register a jupyter kernel for the new conda environment
eval "$(conda shell.bash hook)"
conda activate $NAME
pip install -r requirements.txt

# create milton package and install it in the env
pip install -e .

echo "Installation performed successfully"
