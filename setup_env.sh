#!/usr/bin/env bash
# sets up the environment for task planning w/ common-sense project

echo "****************************"
echo "sourcing the Python 3 virtual env"
echo "****************************"
source ./py36_venv/bin/activate

echo "****************************"
echo "updating the PYTHONPATH to include packages"
echo "****************************"
DATA="${PWD}/datasets:"
EXP="${PWD}/experiments:"
LOG="${PWD}/logger:"
SELF="${PWD}:"
export PYTHONPATH="$DATA$EXP$LOG$SELF$PYTHONPATH"
