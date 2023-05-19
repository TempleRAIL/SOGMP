#!/bin/sh
#
# file: run_demo.sh
#
# This is a simple driver script that runs training and then decoding
# on the training set and the val set.
#
# To run this script, execute the following line:
#
#  run_demo.sh train.dat val.dat
#
# The first argument ($1) is the training data. The last two arguments,
# test data ($2) and evaluation data ($3) are optional.
#
# An example of how to run this is as follows:
#
# xzt: echo $PWD
# /home/xzt/SOGMP
# xzt: sh run_demo.sh ~/OGM-datasets/OGM-Turtlebot2/test
#

# decode the number of command line arguments
#
NARGS=$#

if (test "$NARGS" -eq "0") then
    echo "usage: run_demo.sh test.dat"
    exit 1
fi

# define a base directory for the experiment
#
DL_EXP=`pwd`;
DL_SCRIPTS="$DL_EXP/scripts";
DL_OUT="$DL_EXP/output";

# define the output directories for training/decoding/scoring
#
#DL_TRAIN_ODIR="$DL_OUT/00_train";
DL_TRAIN_ODIR="$DL_EXP/model";
DL_MDL_PATH="$DL_TRAIN_ODIR/model.pth";

# evaluate each data set that was specified
#
echo "... starting evaluation of $1..."
$DL_SCRIPTS/decode_demo.py $DL_MDL_PATH $1 | \
    tee $DL_OUT/01_decode_dev.log | grep "00 out of\|Average"
echo "... finished evaluation of $1 ..."


echo "======= end of results ======="

#
# exit gracefully
