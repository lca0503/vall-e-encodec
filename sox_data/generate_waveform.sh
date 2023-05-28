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


# Stage Bass
# sox -G <input.wav> <output.wav> bass <gain>
effects_file="${effects_dir}/bass.txt"
if [ -e ${effects_file} ]; then
    log "Stage Bass"
    gain_cand=(-20 -12 -6 +6 +12 +20)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	gain=${gain_cand[ $RANDOM % ${#gain_cand[@]} ]}
	sox -G ${source_file} ${target_file} bass ${gain} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "bass\t${gain}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Treble
# sox -G <input.wav> <output.wav> treble <gain>
effects_file="${effects_dir}/treble.txt"
if [ -e ${effects_file} ]; then
    log "Stage Treble"
    gain_cand=(-20 -12 -6 +6 +12 +20)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	gain=${gain_cand[ $RANDOM % ${#gain_cand[@]} ]}
	sox -G ${source_file} ${target_file} treble ${gain} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "treble\t${gain}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Chorus
# sox -G <input.wav> <output.wav> chorus <gain-in> <gain-out> <delay> <decay> <speed> <depth> [-s|-t]
effects_file="${effects_dir}/chorus.txt"
if [ -e ${effects_file} ]; then
    log "Stage Chorus"
    gain_in_cand=(0.5 0.6 0.7)
    gain_out_cand=(0.9)
    delay_cand=(45 50 55)
    decay_cand=(0.4)
    speed_cand=(0.25)
    depth_cand=(2)
    modulation_cand=(-t)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	gain_in=${gain_in_cand[ $RANDOM % ${#gain_in_cand[@]} ]}
	gain_out=${gain_out_cand[ $RANDOM % ${#gain_out_cand[@]} ]}
	delay=${delay_cand[ $RANDOM % ${#delay_cand[@]} ]}
	decay=${decay_cand[ $RANDOM % ${#decay_cand[@]} ]}
	speed=${speed_cand[ $RANDOM % ${#speed_cand[@]} ]}
	depth=${depth_cand[ $RANDOM % ${#depth_cand[@]} ]}
	modulation=${modulation_cand[ $RANDOM % ${#modulation_cand[@]} ]}
	sox -G ${source_file} ${target_file} chorus ${gain_in} ${gain_out} ${delay} ${decay} ${speed} ${depth} ${modulation} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "chorus\t${gain_in}\t${gain_out}\t${delay}\t${decay}\t${speed}\t${depth}\t${modulation}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Delay
# sox -G <input.wav> <output.wav> delay <position>
effects_file="${effects_dir}/delay.txt"
if [ -e ${effects_file} ]; then
    log "Stage Delay"
    position_cand=(1 2 3 4 5)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	position=${position_cand[ $RANDOM % ${#position_cand[@]} ]}
	sox -G ${source_file} ${target_file} delay ${position} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "delay\t${position}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Echo
# sox -G <input.wav> <output.wav> echo <gain-in> <gain-out> <delay> <decay>
effects_file="${effects_dir}/echo.txt"
if [ -e ${effects_file} ]; then
    log "Stage Echo"
    gain_in_cand=(0.5 0.8 0.9)
    gain_out_cand=(0.9)
    delay_cand=(10 100 1000)
    decay_cand=(0.2 0.5 0.8)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	gain_in=${gain_in_cand[ $RANDOM % ${#gain_in_cand[@]} ]}
	gain_out=${gain_out_cand[ $RANDOM % ${#gain_out_cand[@]} ]}
	delay=${delay_cand[ $RANDOM % ${#delay_cand[@]} ]}
	decay=${decay_cand[ $RANDOM % ${#decay_cand[@]} ]}
	sox -G ${source_file} ${target_file} echo ${gain_in} ${gain_out} ${delay} ${decay} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "echo\t${gain_in}\t${gain_out}\t${delay}\t${decay}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Fade
# sox -G <input.wav> <output.wav> fade <type> <length>
effects_file="${effects_dir}/fade.txt"
if [ -e ${effects_file} ]; then
    log "Stage Fade"
    type_cand=(q h t l p)
    length_cand=(1 2 3 4 5)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	type=${type_cand[ $RANDOM % ${#type_cand[@]} ]}
	length=${length_cand[ $RANDOM % ${#length_cand[@]} ]}
	sox -G ${source_file} ${target_file} fade ${type} ${length} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "fade\t${type}\t${length}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Loudness
# sox -G <input.wav> <output.wav> loudness <gain>
effects_file="${effects_dir}/loudness.txt"
if [ -e ${effects_file} ]; then
    log "Stage Loudness"
    gain_cand=(-10 -5 +5 +10)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	gain=${gain_cand[ $RANDOM % ${#gain_cand[@]} ]}
	sox -G ${source_file} ${target_file} loudness ${gain} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "loudness\t${gain}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Repeat
# sox -G <input.wav> <output.wav> repeat <count>
effects_file="${effects_dir}/repeat.txt"
if [ -e ${effects_file} ]; then
    log "Stage Repeat"
    count_cand=(1)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	count=${count_cand[ $RANDOM % ${#count_cand[@]} ]}
	sox -G ${source_file} ${target_file} repeat ${count} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "repeat\t${count}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Reverb
# sox -G <input.wav> <output.wav> reverb
effects_file="${effects_dir}/reverb.txt"
if [ -e ${effects_file} ]; then
    log "Stage Reverb"
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	sox -G ${source_file} ${target_file} reverb
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "reverb" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Reverse
# sox -G <input.wav> <output.wav> reverse
effects_file="${effects_dir}/reverse.txt"
if [ -e ${effects_file} ]; then
    log "Stage Reverse"
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	sox -G ${source_file} ${target_file} reverse
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "reverse" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Tempo
# sox -G <input.wav> <output.wav> tempo <factor>
effects_file="${effects_dir}/tempo.txt"
if [ -e ${effects_file} ]; then
    log "Stage Tempo"
    factor_cand=(0.25 0.5 0.75 1.25 1.5 1.75 2)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	factor=${factor_cand[ $RANDOM % ${#factor_cand[@]} ]}
	sox -G ${source_file} ${target_file} tempo ${factor}
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "tempo\t${factor}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Vol
# sox -G <input.wav> <output.wav> vol <gain>
effects_file="${effects_dir}/vol.txt"
if [ -e ${effects_file} ]; then
    log "Stage Vol"
    gain_cand=(0.5 0.75 1.25 1.5)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	gain=${gain_cand[ $RANDOM % ${#gain_cand[@]} ]}
	sox -G ${source_file} ${target_file} vol ${gain} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "vol\t${gain}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Pitch
# sox -G <input.wav> <output.wav> pitch <cents>
effects_file="${effects_dir}/pitch.txt"
if [ -e ${effects_file} ]; then
    log "Stage Pitch"
    cents_cand=(-250 -200 -150 -100 +100 +150 +200 +250)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	cents=${cents_cand[ $RANDOM % ${#cents_cand[@]} ]}
	sox -G ${source_file} ${target_file} pitch ${cents} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "pitch\t${cents}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


# Stage Contrast
# sox -G <input.wav> <output.wav> contrast <amount>
effects_file="${effects_dir}/contrast.txt"
if [ -e ${effects_file} ]; then
    log "Stage Contrast"
    amount_cand=(50 100)
    for source_file in `cat ${effects_file}`
    do
	target_file=`echo ${source_file} | sed "s@${source_dir}@${target_dir}@g"`
	amount=${amount_cand[ $RANDOM % ${#amount_cand[@]} ]}
	sox -G ${source_file} ${target_file} contrast ${amount} 
	command_file=`echo ${source_file} | sed "s@${source_dir}@${command_dir}@g" | sed "s@wav@txt@g"`
	echo -e "contrast\t${amount}" > ${command_file}
	echo ${newfile}
    done | tqdm >> /dev/null
fi


