#!/usr/bin/env bash
# sets up the incremental-mre repo

# make bash exit if any single command fails
set -e
echo "****************************"
echo "installing virtualenv"
echo "****************************"
sudo apt-get update
sudo apt-get install -y virtualenv

echo "****************************"
echo "creating the Python 3.6 virtual env"
echo "****************************"
virtualenv -p /usr/bin/python3.6 py36_venv
source ./py36_venv/bin/activate

echo "****************************"
echo "installing torch, tensorboard, pandas, matplotlib, and statsmodels"
echo "****************************"
pip install torch==1.2.0 torchvision==0.4.0 pandas future tensorboard matplotlib statsmodels