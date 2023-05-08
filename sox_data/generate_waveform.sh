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
subset_dir=$1
split=$2

source_dir="${subset_dir}/${split}/source"
effects_dir="${subset_dir}/${split}/effects"

target_dir="${subset_dir}/${split}/target"
command_dir="${subset_dir}/${split}/command"

mkdir -p ${target_dir}
mkdir -p ${command_dir}


# Stage 1: Tempo
effects_file="${effects_dir}/tempo.txt"
if [ -e ${effects_file} ]; then
    log "Stage Tempo"
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	p1=`echo "scale=2; ($(randrange -20 20)+100)/100" | bc`
	sox -G ${source_file} ${target_file} tempo ${p1}
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "${source_file}\t${target_file}\ttempo\t${p1}" >> ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage 2: Bass
effects_file="${effects_dir}/bass.txt"
if [ -e ${effects_file} ]; then
    log "Stage Bass"
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	p1=$(randrange -20 20)
	sox -G ${source_file} ${target_file} bass ${p1} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "${source_file}\t${target_file}\tbass\t${p1}" >> ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi
