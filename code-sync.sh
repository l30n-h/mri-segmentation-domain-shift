#!/bin/bash
SCRIPT_PATH=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT_PATH`

rsync -avz --delete --exclude __pycache__/ $SCRIPT_DIR/nnUNet-ext/network_training/masterarbeit giebel:nnUNet-container/code/nnUNet/nnunet/training/network_training/

rsync -avz --delete --exclude __pycache__/ $SCRIPT_DIR/analyze giebel:nnUNet-container/code/

rsync -avz --delete --exclude __pycache__/ $SCRIPT_DIR/lipEstimation giebel:nnUNet-container/code/
