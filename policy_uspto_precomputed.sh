#!/bin/bash

TRAIN="/path/training_inputs.npz"
TROUT="/path/training_labels.npz"
VAL="/path/validation_inputs.npz"
VAOUT="/path/validation_labels.npz"
OUTDIR="/path/model"
OUTFILE="policy"
BATCHSIZE=256
EPOCHS=500
FPSIZE=2048

python policy_precomputed.py -trin $TRAIN -trout $TROUT -vain $VAL -vaout $VAOUT -od $OUTDIR -of $OUTFILE -b $BATCHSIZE -e $EPOCHS -fp $FPSIZE
