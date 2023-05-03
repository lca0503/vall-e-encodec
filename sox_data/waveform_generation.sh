#!/usr/bin/env bash

# define functions
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

randrange() {
    min=$1
    max=$2
    ((range = max - min))
    ((num = min + RANDOM % range))
    echo $num
}

# General configuration
effect_splits_dir="data/libritts_subset/effect_splits"
subset_dir="data/libritts_subset"
source_dir="data/libritts_subset/source"
target_dir="data/libritts_subset/target"

mkdir -p ${target_dir}

stage=1                                   # start from 0 if you need to start from data preparation
stop_stage=2

# Stage 1: Tempo
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Tempo"
    effect_splits_file="${effect_splits_dir}/tempo.txt"
    for file in `cat ${effect_splits_file}`
    do
	newfile=`echo ${file} | sed "s@${source_dir}@${target_dir}@g"`
	p1=`echo "scale=2; ($(randrange -20 20)+100)/100" | bc`
	sox -G ${file} ${newfile} tempo ${p1}
	echo -e "${file}\t${newfile}\ttempo\t${p1}" >> "${subset_dir}/commands.txt"
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage 2: Bass
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Bass"
    effect_splits_file="${effect_splits_dir}/bass.txt"
    for file in `cat ${effect_splits_file}`
    do
	newfile=`echo ${file} | sed "s@${source_dir}@${target_dir}@g"`
	p1=$(randrange -20 20)
	sox -G ${file} ${newfile} bass ${p1} 
	echo -e "${file}\t${newfile}\tbass\t${p1}" >> "${subset_dir}/commands.txt"
	echo ${newfile}
    done | tqdm >> /dev/null
fi
