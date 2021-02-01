#!/bin/sh
python datasets/generate_cl_datasets.py -d WN18RR -r 0.2 -n 5 -s triple
python datasets/generate_cl_datasets.py -d FB15K237 -r 0.2 -n 5 -s triple
python datasets/generate_cl_datasets.py -d THOR_U -r 0.2 -n 5 -s triple
#python datasets/generate_cl_datasets.py -d WN18RR -r 0.5 -n 5 -s rel
#python datasets/generate_cl_datasets.py -d FB15K237 -r 0.5 -n 5 -s rel
#python datasets/generate_cl_datasets.py -d THOR_U -r 0.5 -n 5 -s rel
#python datasets/generate_cl_datasets.py -d WN18RR -r 0.5 -n 5 -s ent
#python datasets/generate_cl_datasets.py -d FB15K237 -r 0.5 -n 5 -s ent
#python datasets/generate_cl_datasets.py -d THOR_U -r 0.5 -n 5 -s ent
