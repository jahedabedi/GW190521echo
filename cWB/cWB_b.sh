#!/bin/tcsh -f


## command for the cwb
## https://gwburst.gitlab.io/documentation/latest/html//commands/cwb_inet2G.html#ADV_SIM_SGQ9_L1H1V1_2G

set CMD="/home/cwb/waveburst/git/cWB/library/tools/cwb/scripts/cwb_inet2G.csh"

$CMD config/user_b_parameters.C FULL 1 ced
