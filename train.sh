#!/usr/bin/env bash

set -x
set -e

cmd="$1"
epochs="$2"
prefix="$3"
rcfile="$4"

if [ -z ${prefix} ]; then
    prefix="weights/it"
fi

if [ -z ${rcfile} ]; then
    rcfile="train.rc"
fi

source ${rcfile}

if [[ ${cmd} == "init" ]]; then
    src/train - ${prefix}0.dat
    exit
fi

if [[ ${cmd} == "train" ]]; then
    for i in $(seq 1 ${epochs}); do
        [[ -e ${prefix}${i}.dat ]] && continue
        src/train ${prefix}$((i - 1)).dat ${prefix}${i}.dat ${TRAIN_DATA} ${TRAIN_LABELS}
    done
    exit
fi

if [[ ${cmd} == "train_report" ]]; then
    for i in $(seq 1 ${epochs}); do
        [[ -e ${prefix}${i}.dat ]] && continue
        time src/train ${prefix}$((i - 1)).dat ${prefix}${i}.dat ${TRAIN_DATA} ${TRAIN_LABELS}
        src/modelstats ${prefix}${i}.dat ${TEST_DATA} ${TEST_LABELS}
        read
    done
    exit
fi

if [[ ${cmd} == "report" ]]; then
    src/modelstats ${prefix}${epochs}.dat ${TEST_DATA} ${TEST_LABELS}
    exit
fi

if [[ ${cmd} == "delete" ]]; then
    for i in $(seq 1 ${epochs}); do
        rm ${prefix}${i}.dat
    done
    exit
fi
