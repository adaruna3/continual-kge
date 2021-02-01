#!/bin/sh
######################################
# TRANSE - THOR_U
STARTTIME=$(date +%s)
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm offline -ln 1
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm finetune -ln 1
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 1.0 -ln 1
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.01 -ln 1
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm PNN -ln 1
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm CWR -ln 1
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm DGR -ln 1 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
ENDTIME=$(date +%s)
echo "Round of THOR_U - TRANSE took $(($ENDTIME - $STARTTIME)) seconds to complete..."
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm offline -ln 2
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm finetune -ln 2
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 1.0 -ln 2
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.01 -ln 2
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm PNN -ln 2
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm CWR -ln 2
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm DGR -ln 2 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm offline -ln 3
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm finetune -ln 3
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 1.0 -ln 3
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.01 -ln 3
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm PNN -ln 3
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm CWR -ln 3
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm DGR -ln 3 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm offline -ln 4
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm finetune -ln 4
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 1.0 -ln 4
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.01 -ln 4
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm PNN -ln 4
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm CWR -ln 4
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm DGR -ln 4 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm offline -ln 5
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm finetune -ln 5
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm L2 -rs 1.0 -ln 5
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm SI -rs 0.01 -ln 5
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm PNN -ln 5
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm CWR -ln 5
python experiments/continual_setting.py THOR_U -mt transe -nr 1 -hh 25 -op 0.1 -m 2.0 -clm DGR -ln 5 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
# ANALOGY - THOR_U
STARTTIME=$(date +%s)
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm offline -ln 1
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm finetune -ln 1
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 10.0 -ln 1
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 1.0 -ln 1
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm PNN -ln 1
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm CWR -ln 1
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm DGR -ln 1 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
ENDTIME=$(date +%s)
echo "Round of THOR_U - ANALOGY took $(($ENDTIME - $STARTTIME)) seconds to complete..."
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm offline -ln 2
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm finetune -ln 2
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 10.0 -ln 2
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 1.0 -ln 2
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm PNN -ln 2
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm CWR -ln 2
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm DGR -ln 2 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm offline -ln 3
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm finetune -ln 3
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 10.0 -ln 3
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 1.0 -ln 3
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm PNN -ln 3
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm CWR -ln 3
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm DGR -ln 3 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm offline -ln 4
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm finetune -ln 4
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 10.0 -ln 4
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 1.0 -ln 4
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm PNN -ln 4
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm CWR -ln 4
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm DGR -ln 4 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm offline -ln 5
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm finetune -ln 5
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm L2 -rs 10.0 -ln 5
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm SI -rs 1.0 -ln 5
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm PNN -ln 5
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm CWR -ln 5
python experiments/continual_setting.py THOR_U -mt analogy -nr 50 -hh 100 -op 0.1 -clm DGR -ln 5 -vp 10.0 20.0 500.0 25.0 0.001 100 75 50 0.06 200.0 0.8