#!/bin/sh
#
# file: run_train.sh
#
# This is a simple driver script that runs training and then decoding
# on the training set and the val set.
#
# To run this script, execute the following line:
#
#  run_train.sh train.dat val.dat
#
# The first argument ($1) is the training data. The last two arguments,
# test data ($2) and evaluation data ($3) are optional.
#
# An example of how to run this is as follows:
#
# xzt: echo $PWD
# /home/xzt/SOGMP
# xzt: sh run_train.sh ~/OGM-datasets/OGM-Turtlebot2/train ~/OGM-datasets/OGM-Turtlebot2/val
#

# decode the number of command line arguments
#
NARGS=$#

if (test "$NARGS" -eq "0") then
    echo "usage: run_train.sh train.dat val.dat"
    exit 1
fi

# define a base directory for the experiment
#
DL_EXP=`pwd`;
DL_SCRIPTS="$DL_EXP/scripts";
DL_OUT="$DL_EXP/output";

# define the number of feats environment variable
#
export DL_NUM_FEATS=3

# define the output directories for training/decoding/scoring
#
#DL_TRAIN_ODIR="$DL_OUT/00_train";
DL_TRAIN_ODIR="$DL_EXP/model";
DL_MDL_PATH="$DL_TRAIN_ODIR/model.pth";

# create the output directory
#
rm -fr $DL_OUT
mkdir -p $DL_OUT

# execute training: training must always be run
#
echo "... starting training on $1 ..."
$DL_SCRIPTS/train.py $DL_MDL_PATH $1 $2 | tee $DL_OUT/00_train.log | \
      grep "reading\|Step\|Average\|Warning\|Error" 
echo "... finished training on $1 ..."

#
