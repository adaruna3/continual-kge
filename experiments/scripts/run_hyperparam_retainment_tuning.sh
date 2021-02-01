#!/bin/sh
# TRANSE - WN18RR
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 0.0001 -ln 1
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.0001 -ln 1
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 0.001 -ln 2
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.001 -ln 2
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 0.01 -ln 3
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.01 -ln 3
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 0.1 -ln 4
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.1 -ln 4
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 1.0 -ln 5
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 1.0 -ln 5
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 10.0 -ln 6
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 10.0 -ln 6
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 100.0 -ln 7
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 100.0 -ln 7
# ANALOGY - WN18RR
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.0001 -ln 1
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 0.0001 -ln 1
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.001 -ln 2
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 0.001 -ln 2
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.01 -ln 3
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 0.01 -ln 3
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.1 -ln 4
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 0.1 -ln 4
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 1.0 -ln 5
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 1.0 -ln 5
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 10.0 -ln 6
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 6
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 100.0 -ln 7
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 100.0 -ln 7
# TRANSE - FB15K237
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm L2 -rs 0.0001 -ln 1
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm SI -rs 0.0001 -ln 1
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm L2 -rs 0.001 -ln 2
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm SI -rs 0.001 -ln 2
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm L2 -rs 0.01 -ln 3
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm SI -rs 0.01 -ln 3
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm L2 -rs 0.1 -ln 4
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm SI -rs 0.1 -ln 4
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm L2 -rs 1.0 -ln 5
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm SI -rs 1.0 -ln 5
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm L2 -rs 10.0 -ln 6
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm SI -rs 10.0 -ln 6
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm L2 -rs 100.0 -ln 7
python experiments/continual_setting.py FB15K237 -mt transe -nr 1 -hh 200 -op 0.01 -m 8.0 -clm SI -rs 100.0 -ln 7
# ANALOGY - FB15K237
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 0.0001 -ln 1
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 0.0001 -ln 1
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 0.001 -ln 2
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 0.001 -ln 2
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 0.01 -ln 3
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 0.01 -ln 3
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 0.1 -ln 4
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 0.1 -ln 4
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 1.0 -ln 5
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 1.0 -ln 5
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 10.0 -ln 6
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 6
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 100.0 -ln 7
python experiments/continual_setting.py FB15K237 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 100.0 -ln 7
# TRANSE - THOR_U
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 0.0001 -ln 1
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.0001 -ln 1
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 0.001 -ln 2
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.001 -ln 2
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 0.01 -ln 3
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.01 -ln 3
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 0.1 -ln 4
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.1 -ln 4
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 1.0 -ln 5
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 1.0 -ln 5
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 10.0 -ln 6
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 10.0 -ln 6
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 100.0 -ln 7
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 100.0 -ln 7
# ANALOGY - THOR_U
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 0.0001 -ln 1
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 0.0001 -ln 1
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 0.001 -ln 2
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 0.001 -ln 2
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 0.01 -ln 3
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 0.01 -ln 3
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 0.1 -ln 4
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 0.1 -ln 4
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 1.0 -ln 5
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 1.0 -ln 5
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 10.0 -ln 6
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 10.0 -ln 6
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 100.0 -ln 7
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 100.0 -ln 7
