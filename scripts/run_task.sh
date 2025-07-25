#!/bin/bash

GPUS=1
HOURS=1

if [ ! -z "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ && "$1" -ge 1 && "$1" -le 8 ]]; then
        GPUS=$1
    else
        echo "Error: Number of GPUs must be a natural number between 1 and 8."
        exit 1
    fi
fi

if [ ! -z "$2" ]; then
    if [[ "$2" =~ ^[0-9]+$ && "$2" -ge 1 && "$2" -le 48 ]]; then
        HOURS=$2
    else
        echo "Error: Time must be a natural number between 1 and 48 (hours)."
        exit 1
    fi
fi

TIME_FORMAT="${HOURS}:00:00"

srun -N1 -n1 --account g98-2103 --gres=gpu:${GPUS} --time=${TIME_FORMAT} --pty /bin/bash -l
