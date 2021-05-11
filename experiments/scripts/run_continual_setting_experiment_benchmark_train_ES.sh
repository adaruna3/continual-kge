#!/bin/sh
######################################
# TRANSE - WN18RR
STARTTIME=$(date +%s)
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm offline -ln 1
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm finetune -ln 1
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 1.0 -ln 1
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.1 -ln 1
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm PNN -ln 1
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm CWR -ln 1
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm DGR -ln 1 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
ENDTIME=$(date +%s)
echo "Round of TRANSE - WN18RR took $(($ENDTIME - $STARTTIME)) seconds to complete..."
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm offline -ln 2
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm finetune -ln 2
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 1.0 -ln 2
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.1 -ln 2
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm PNN -ln 2
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm CWR -ln 2
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm DGR -ln 2 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 240.0 0.8
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm offline -ln 3
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm finetune -ln 3
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 1.0 -ln 3
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.1 -ln 3
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm PNN -ln 3
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm CWR -ln 3
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm DGR -ln 3 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 240.0 0.8
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm offline -ln 4
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm finetune -ln 4
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 1.0 -ln 4
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.1 -ln 4
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm PNN -ln 4
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm CWR -ln 4
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm DGR -ln 4 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 240.0 0.8
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm offline -ln 5
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm finetune -ln 5
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm L2 -rs 1.0 -ln 5
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm SI -rs 0.1 -ln 5
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm PNN -ln 5
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm CWR -ln 5
python experiments/continual_setting.py WN18RR -mt transe -nr 25 -hh 100 -m 8.0 -op 0.1 -clm DGR -ln 5 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 240.0 0.8
# ANALOGY - WN18RR
STARTTIME=$(date +%s)
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm offline -ln 1
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm finetune -ln 1
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.1 -ln 1
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 1
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm PNN -ln 1
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm CWR -ln 1
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm DGR -ln 1 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
ENDTIME=$(date +%s)
echo "Round of ANALOGY - WN18RR took $(($ENDTIME - $STARTTIME)) seconds to complete..."
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm offline -ln 2
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm finetune -ln 2
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.1 -ln 2
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 2
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm PNN -ln 2
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm CWR -ln 2
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm DGR -ln 2 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 240.0 0.8
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm offline -ln 3
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm finetune -ln 3
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.1 -ln 3
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 3
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm PNN -ln 3
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm CWR -ln 3
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm DGR -ln 3 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 240.0 0.8
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm offline -ln 4
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm finetune -ln 4
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.1 -ln 4
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 4
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm PNN -ln 4
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm CWR -ln 4
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm DGR -ln 4 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 240.0 0.8
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm offline -ln 5
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm finetune -ln 5
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm L2 -rs 0.1 -ln 5
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 5
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm PNN -ln 5
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm CWR -ln 5
python experiments/continual_setting.py WN18RR -mt analogy -nr 1 -hh 200 -op 0.01 -clm DGR -ln 5 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 240.0 0.8
# TRANSE - FB15K237
STARTTIME=$(date +%s)
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm offline -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm finetune -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm L2 -rs 100.0 -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm SI -rs 10.0 -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm PNN -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm CWR -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm DGR -ln 1 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
ENDTIME=$(date +%s)
echo "Round of TRANSE - FB15K237 took $(($ENDTIME - $STARTTIME)) seconds to complete..."
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm offline -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm finetune -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm L2 -rs 100.0 -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm SI -rs 10.0 -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm PNN -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm CWR -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm DGR -ln 2 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm offline -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm finetune -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm L2 -rs 100.0 -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm SI -rs 10.0 -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm PNN -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm CWR -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm DGR -ln 3 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm offline -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm finetune -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm L2 -rs 100.0 -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm SI -rs 10.0 -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm PNN -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm CWR -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm DGR -ln 4 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm offline -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm finetune -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm L2 -rs 100.0 -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm SI -rs 10.0 -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm PNN -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm CWR -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt transe -nr 1 -hh 200 -m 8.0 -op 0.01 -clm DGR -ln 5 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
# ANALOGY - FB15K237
STARTTIME=$(date +%s)
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm offline -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm finetune -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 10.0 -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm PNN -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm CWR -ln 1
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm DGR -ln 1 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
ENDTIME=$(date +%s)
echo "Round of ANALOGY - FB15K237 took $(($ENDTIME - $STARTTIME)) seconds to complete..."
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm offline -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm finetune -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 10.0 -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm PNN -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm CWR -ln 2
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm DGR -ln 2 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm offline -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm finetune -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 10.0 -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm PNN -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm CWR -ln 3
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm DGR -ln 3 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm offline -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm finetune -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 10.0 -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm PNN -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm CWR -ln 4
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm DGR -ln 4 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm offline -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm finetune -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm L2 -rs 10.0 -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm SI -rs 10.0 -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm PNN -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm CWR -ln 5
python experiments/continual_setting.py FB15K237 -vc 4000 -mt analogy -nr 25 -hh 200 -op 0.01 -clm DGR -ln 5 -vp 10.0 20.0 500.0 1000.0 0.001 200 150 100 0.06 200.0 0.8
