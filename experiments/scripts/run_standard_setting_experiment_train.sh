#!/bin/sh
# WN18RR
python experiments/standard_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1
python experiments/standard_setting.py WN18RR -mt analogy WN18RR -nr 1 -hh 200 -op 0.01
# FB15K237
python experiments/standard_setting.py FB15K237 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01
python experiments/standard_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01
# THOR_U
python experiments/standard_setting.py THOR_U -mt transe -nr 1 -hh 25 -m 2.0 -op 0.1
python experiments/standard_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1